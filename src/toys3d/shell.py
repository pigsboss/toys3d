import sys
import os

# Ensure src directory is on the path so that 'toys3d' can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_parent = os.path.dirname(_project_root)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)

import numpy as np
import trimesh

from toys3d.geometrics import (
    compute_mesh_stats,
    analyze_mesh_defects,
    repair_mesh_by_removing_duplicates,
    repair_nonmanifold_edges,
    fill_small_holes,
    extract_boundary_loops,
    compute_g1_deviation,
    build_face_adjacency,
    estimate_shell_thickness,
    segment_plates_by_smoothness,
)


# ------------------------------------------------------------------
#  辅助可视化
# ------------------------------------------------------------------

def add_axes_to_scene(scene, origin, u_x, u_y, u_z, length=0.3, radius=0.01):
    """在场景中添加红、绿、蓝三根坐标轴。"""
    def add_arrow(o, d, color):
        cyl = trimesh.creation.cylinder(radius=radius, segment=[o, o + d * length])
        cyl.visual.face_colors = color
        scene.add_geometry(cyl)
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius * 3)
        sphere.apply_translation(o + d * length)
        sphere.visual.face_colors = color
        scene.add_geometry(sphere)

    add_arrow(origin, u_x, [255, 0, 0, 255])
    add_arrow(origin, u_y, [0, 255, 0, 255])
    add_arrow(origin, u_z, [0, 0, 255, 255])


def visualize_thickness(mesh, thickness, mask=None):
    """
    用伪色彩显示厚度场。
    thickness : (N,) ndarray
    mask : (N,) bool or None  只显示可靠的面片
    """
    scene = trimesh.Scene()
    vis = mesh.copy()
    N = len(vis.faces)

    colors = np.full((N, 4), 200, dtype=np.uint8)
    colors[:, 3] = 255

    if mask is None:
        mask = np.ones(N, dtype=bool)

    valid = mask & np.isfinite(thickness)
    if np.any(valid):
        t_vals = thickness[valid]
        t_min = t_vals.min()
        t_max = t_vals.max()
        if t_max - t_min > 1e-12:
            t_norm = (t_vals - t_min) / (t_max - t_min)
        else:
            t_norm = np.zeros_like(t_vals)
        # 热力图：蓝(冷) -> 红(热)
        r = (t_norm * 255).astype(np.uint8)
        g = (np.where(t_norm < 0.5, 2 * t_norm * 255, 255 - 2 * (t_norm - 0.5) * 255)).astype(np.uint8)
        b = (255 - r).astype(np.uint8)
        indices = np.where(valid)[0]
        colors[indices, 0] = r
        colors[indices, 1] = g
        colors[indices, 2] = b
    # 非有效区域设为灰色
    invalid = ~valid
    colors[invalid] = [180, 180, 180, 255]

    vis.visual.face_colors = colors
    scene.add_geometry(vis)
    origin = mesh.bounding_box.centroid
    max_ext = mesh.bounding_box.extents.max()
    add_axes_to_scene(scene, origin,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)
    return scene


def visualize_plate_labels(mesh, labels):
    """
    用色调显示薄板编号。
    labels : (N,) ndarray, int   -1 表示未归类（灰色）
    """
    scene = trimesh.Scene()
    vis = mesh.copy()
    N = len(vis.faces)
    uniq = np.unique(labels)
    uniq = uniq[uniq >= 0]
    palette = np.array([
        [255, 0, 0, 255],   # 红
        [0, 255, 0, 255],   # 绿
        [0, 0, 255, 255],   # 蓝
        [255, 255, 0, 255], # 黄
        [255, 0, 255, 255], # 紫
        [0, 255, 255, 255], # 青
        [255, 128, 0, 255], # 橙
        [128, 0, 255, 255], # 粉
        [0, 128, 128, 255], # 深青
        [128, 128, 0, 255],
    ], dtype=np.uint8)

    colors = np.full((N, 4), 200, dtype=np.uint8)
    colors[:, 3] = 255
    for i, lbl in enumerate(uniq):
        col = palette[i % len(palette)]
        colors[labels == lbl] = col
    colors[labels == -1] = [180, 180, 180, 255]

    vis.visual.face_colors = colors
    scene.add_geometry(vis)
    origin = mesh.bounding_box.centroid
    max_ext = mesh.bounding_box.extents.max()
    add_axes_to_scene(scene, origin,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)
    return scene


# ------------------------------------------------------------------
#  主流程
# ------------------------------------------------------------------

def process_shell(mesh, num_passes=0, repair_mode=False, angle_threshold_deg=30.0,
                  min_faces=30):
    """
    薄板分割与厚度分析主函数。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    num_passes : 0 | 1 | 2
    repair_mode : bool
    angle_threshold_deg : float
    min_faces : int

    Returns
    -------
    scene : trimesh.Scene (厚度场或薄板分割可视化)
    thickness : (N,) ndarray
    labels : (N,) ndarray
    stats : dict
    """
    stats = compute_mesh_stats(mesh)
    print("Shell thickness analysis:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 0. 缺陷分析
    defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    print("\nDefect analysis:")
    print(f"  open edges: {defect_stats['open_edges']}")
    print(f"  nonmanifold edges: {defect_stats['nonmanifold_edges']}")
    print(f"  watertight (no open edges): {defect_stats['watertight_by_count']}")

    # 1. 修复（可选）
    if repair_mode and (defect_stats['open_edges'] > 0 or defect_stats['nonmanifold_edges'] > 0):
        print("\n[Repair mode] Attempting to fix mesh...")
        max_repair_iter = 5
        for it in range(max_repair_iter):
            print(f"\n  [Repair iter {it}]")
            mesh = repair_mesh_by_removing_duplicates(mesh)
            mesh = repair_nonmanifold_edges(mesh)
            mesh = fill_small_holes(mesh)
            defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
            print(f"  After iter {it}: open_edges={defect_stats['open_edges']}, "
                  f"nonmanifold_edges={defect_stats['nonmanifold_edges']}")
            if defect_stats['open_edges'] == 0 and defect_stats['nonmanifold_edges'] == 0:
                print("  Mesh fully repaired.")
                break
        stats = compute_mesh_stats(mesh)
        print("\nAfter repair:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    # 2. 厚度估计
    print("\nEstimating shell thickness (ray‑based)...")
    thickness, reliability = estimate_shell_thickness(mesh)
    median_th = np.nanmedian(thickness[reliability]) if np.any(reliability) else np.nan
    print(f"  Median thickness: {median_th:.4f}")
    print(f"  Reliable faces: {np.sum(reliability)} / {len(mesh.faces)}")

    # 3. 薄板分割
    print("\nSegmenting thin plates by smoothness...")
    labels = segment_plates_by_smoothness(mesh,
                                          angle_threshold_deg=angle_threshold_deg,
                                          min_faces=min_faces)
    num_plates = len(np.unique(labels[labels >= 0]))
    print(f"  Number of plates: {num_plates}")

    # 4. 构建统计信息
    plate_thicknesses = {}
    for lbl in np.unique(labels):
        if lbl < 0:
            continue
        mask = (labels == lbl) & reliability
        if np.any(mask):
            plate_thicknesses[int(lbl)] = {
                'thickness_mean': float(np.mean(thickness[mask])),
                'thickness_std': float(np.std(thickness[mask])),
                'face_count': int(np.sum(mask)),
            }

    shell_stats = {
        'median_thickness': median_th,
        'reliable_ratio': float(np.sum(reliability) / len(mesh.faces)),
        'num_plates': num_plates,
        'plate_thicknesses': plate_thicknesses,
    }
    print("\nShell statistics:")
    for k, v in shell_stats.items():
        if k != 'plate_thicknesses':
            print(f"  {k}: {v}")

    # 5. 可视化
    if num_passes == 0:
        # 显示厚度场
        scene = visualize_thickness(mesh, thickness, reliability)
    else:
        # 显示薄板分割
        scene = visualize_plate_labels(mesh, labels)

    return scene, thickness, labels, shell_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="薄板厚度分析与光滑分割。")
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("--num-passes", type=int, default=0, choices=[0, 1, 2],
                        help="处理阶段：0=厚度场可视化，1=薄板分割可视化 (默认0)")
    parser.add_argument("--repair", action="store_true",
                        help="尝试自动修复网格")
    parser.add_argument("--angle-threshold", type=float, default=30.0,
                        help="分割用二面角阈值（度） (默认30)")
    parser.add_argument("--min-faces", type=int, default=30,
                        help="分割最小面片数 (默认30)")
    parser.add_argument("--show", action="store_true", help="显示可视化窗口")
    args = parser.parse_args()

    mesh = trimesh.load(args.input_file)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")
    print(f"Loading model: {args.input_file}")

    scene, thickness, labels, shell_stats = process_shell(
        mesh,
        num_passes=args.num_passes,
        repair_mode=args.repair,
        angle_threshold_deg=args.angle_threshold,
        min_faces=args.min_faces,
    )

    # 打印各薄板平均厚度
    for lbl, info in shell_stats['plate_thicknesses'].items():
        print(f"  Plate {lbl}: mean thickness={info['thickness_mean']:.4f}, "
              f"std={info['thickness_std']:.4f}, faces={info['face_count']}")

    if args.show:
        try:
            os.environ['TRIMESH_DEFAULT_VIEWER'] = 'vedo'
            scene.show()
        except Exception as e:
            print(f"\n[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
