import sys
import os
import time

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
    estimate_shell_thickness,
    segment_plates_by_smoothness,
    detect_thin_regions,
    compute_wall_thickness_statistics,
    extract_plate_boundary_loops,
    classify_edge_regularity,
)


# ------------------------------------------------------------------
#  Timer 类用于记录耗时
# ------------------------------------------------------------------
class Timer:
    """Simple timer context manager for logging elapsed time."""
    def __init__(self, label=''):
        self.label = label
        self.start = None
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[Timer] {self.label} took {elapsed:.3f}s")


# ------------------------------------------------------------------
#  可视化辅助
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


def _jet_colormap(t):
    """简化的 jet 伪彩色映射，输入 t 在 [0,1]。"""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    r = np.clip(1.5 - 4.0 * np.abs(t - 0.75), 0.0, 1.0)
    g = np.clip(1.5 - 4.0 * np.abs(t - 0.5), 0.0, 1.0)
    b = np.clip(1.5 - 4.0 * np.abs(t - 0.25), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def visualize_thickness(mesh, thickness, reliability=None):
    """用伪彩色显示厚度场，不可靠/无效区域显示为灰色。"""
    scene = trimesh.Scene()
    vis = mesh.copy()
    N = len(vis.faces)

    colors = np.full((N, 4), 180, dtype=np.uint8)
    colors[:, 3] = 255

    if reliability is None:
        reliability = np.isfinite(thickness)
    valid = reliability & np.isfinite(thickness)

    if np.any(valid):
        t = thickness[valid]
        t_min, t_max = np.percentile(t, [2, 98])
        rng = t_max - t_min
        if rng < 1e-12:
            rng = 1.0
        norm = np.clip((t - t_min) / rng, 0.0, 1.0)
        rgb = _jet_colormap(norm)
        colors[np.where(valid)[0], :3] = rgb

    colors[~valid] = [180, 180, 180, 255]
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


def visualize_plates(mesh, labels):
    """用不同颜色显示薄板分割结果，未归类面片为灰色。"""
    scene = trimesh.Scene()
    vis = mesh.copy()
    N = len(vis.faces)

    rng = np.random.default_rng(42)
    palette = np.column_stack([
        rng.integers(60, 255, size=20),
        rng.integers(60, 255, size=20),
        rng.integers(60, 255, size=20),
        np.full(20, 255, dtype=np.uint8),
    ])

    colors = np.full((N, 4), 180, dtype=np.uint8)
    colors[:, 3] = 255

    unique = np.unique(labels)
    for i, lbl in enumerate(unique):
        if lbl < 0:
            continue
        col = palette[i % len(palette)]
        colors[labels == lbl] = col
    colors[labels < 0] = [180, 180, 180, 255]

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


def visualize_plate_centers(mesh, labels, alpha=80):
    """
    高亮显示每个薄板的中心面（局部拟合平面上的圆盘），
    并以半透明方式显示原始网格。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    labels : (N,) ndarray
        薄板标签。
    alpha : int
        原始网格透明度（0-255）。

    Returns
    -------
    scene : trimesh.Scene
    """
    scene = trimesh.Scene()

    # 原始网格半透明
    base = mesh.copy()
    N = len(base.faces)
    colors = np.full((N, 4), 200, dtype=np.uint8)
    colors[:, 3] = alpha
    base.visual.face_colors = colors
    scene.add_geometry(base)

    # 随机调色板
    rng = np.random.default_rng(42)
    n_plates = int(max(0, labels.max()) + 1)
    palette = np.column_stack([
        rng.integers(80, 255, size=n_plates),
        rng.integers(80, 255, size=n_plates),
        rng.integers(80, 255, size=n_plates),
        np.full(n_plates, 255, dtype=np.uint8),
    ])

    # 为每个薄板生成中心面圆盘
    for lbl in range(n_plates):
        mask = labels == lbl
        n_faces = int(np.sum(mask))
        if n_faces < 3:
            continue

        centers = mesh.triangles_center[mask]
        mean_pt = centers.mean(axis=0)

        # 用 PCA 拟合局部平面
        centered = centers - mean_pt
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]

        # 确保法向与薄板平均法向一致
        face_normals = mesh.face_normals[mask]
        avg_normal = face_normals.mean(axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-12)
        if np.dot(normal, avg_normal) < 0:
            normal = -normal

        # 在平面内构造两个正交基
        if abs(normal[2]) < 0.9:
            u = np.cross([0.0, 0.0, 1.0], normal)
        else:
            u = np.cross([1.0, 0.0, 0.0], normal)
        un = np.linalg.norm(u)
        if un < 1e-12:
            continue
        u = u / un
        v = np.cross(normal, u)

        # 圆盘半径：薄板投影到平面的包围半径
        proj_u = centered @ u
        proj_v = centered @ v
        sq = proj_u * proj_u + proj_v * proj_v
        radius = float(np.sqrt(np.maximum(sq, 0).max()))

        if radius < 1e-6:
            continue

        # 创建圆盘并变换到平面位置
        disk = trimesh.creation.cylinder(radius=radius, height=0.01 * radius, sections=32)
        R = np.column_stack([u, v, normal])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = mean_pt
        disk.apply_transform(T)
        disk.visual.face_colors = palette[lbl % len(palette)]

        scene.add_geometry(disk)

    # 添加坐标轴
    origin = mesh.bounding_box.centroid
    max_ext = mesh.bounding_box.extents.max()
    add_axes_to_scene(scene, origin,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)
    return scene


def visualize_boundaries(mesh, labels, scale, line_tol, circle_tol, spline_tol):
    """
    按边界环规律性着色：直线=绿，圆弧=蓝，样条=黄，不规则=红。
    同时返回被判为不规则的边界环列表。
    同时输出每个薄板的边界环数量。
    """
    scene = trimesh.Scene()
    vis = mesh.copy()
    N = len(vis.faces)

    colors = np.full((N, 4), 200, dtype=np.uint8)
    colors[:, 3] = 255

    edge_map = {}
    for fi, face in enumerate(mesh.faces):
        for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            key = tuple(sorted((int(a), int(b))))
            edge_map.setdefault(key, []).append(fi)

    irregular_loops = []
    for lbl in np.unique(labels):
        if lbl < 0:
            continue
        loops = extract_plate_boundary_loops(mesh, labels == lbl)
        # 输出每个薄板的边界环数量
        print(f"  Plate {lbl}: {len(loops)} boundary loops")
        for loop in loops:
            pts = mesh.vertices[np.asarray(loop, dtype=int)]
            ctype, score = classify_edge_regularity(
                pts, scale=scale,
                line_tol=line_tol, circle_tol=circle_tol, spline_tol=spline_tol
            )
            if ctype == 'line':
                col = [0, 255, 0, 255]
            elif ctype == 'circle':
                col = [0, 0, 255, 255]
            elif ctype == 'spline':
                col = [255, 255, 0, 255]
            else:
                col = [255, 0, 0, 255]
                irregular_loops.append(loop)

            for i in range(len(loop)):
                key = tuple(sorted((int(loop[i]), int(loop[(i + 1) % len(loop)]))))
                for fi in edge_map.get(key, []):
                    colors[fi] = col

    vis.visual.face_colors = colors
    scene.add_geometry(vis)
    origin = mesh.bounding_box.centroid
    max_ext = mesh.bounding_box.extents.max()
    add_axes_to_scene(scene, origin,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)
    return scene, irregular_loops


# ------------------------------------------------------------------
#  几何辅助
# ------------------------------------------------------------------

def _loop_geometry(mesh, loop):
    """计算边界环的周长与投影面积。"""
    pts = mesh.vertices[np.asarray(loop, dtype=int)]
    n = len(pts)

    perim = 0.0
    for i in range(n):
        perim += np.linalg.norm(pts[(i + 1) % n] - pts[i])

    # Newell 法求最佳投影平面法向
    normal = np.zeros(3, dtype=np.float64)
    for i in range(n):
        normal += np.cross(pts[i], pts[(i + 1) % n])
    nlen = np.linalg.norm(normal)
    if nlen < 1e-12:
        return perim, 0.0
    normal /= nlen

    if abs(normal[2]) < 0.9:
        u = np.cross([0.0, 0.0, 1.0], normal)
    else:
        u = np.cross([1.0, 0.0, 0.0], normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    proj = np.column_stack([pts @ u, pts @ v])
    area = 0.5 * abs(
        np.sum(proj[:, 0] * np.roll(proj[:, 1], -1)
               - np.roll(proj[:, 0], -1) * proj[:, 1])
    )
    return perim, area


def _faces_adjacent_to_loops(mesh, loops):
    """返回与任意边界环共享边的所有面片掩码。"""
    edge_map = {}
    for fi, face in enumerate(mesh.faces):
        for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            key = tuple(sorted((int(a), int(b))))
            edge_map.setdefault(key, []).append(fi)

    mask = np.zeros(len(mesh.faces), dtype=bool)
    for loop in loops:
        loop_arr = np.asarray(loop, dtype=int)
        for i in range(len(loop_arr)):
            key = tuple(sorted((int(loop_arr[i]), int(loop_arr[(i + 1) % len(loop_arr)]))))
            for fi in edge_map.get(key, []):
                mask[fi] = True
    return mask


def _repair_loop(mesh, max_iter=5):
    """通用修复循环：去重/退化面、去非流形边、补洞。"""
    for it in range(max_iter):
        mesh = repair_mesh_by_removing_duplicates(mesh)
        mesh = repair_nonmanifold_edges(mesh, verbose=(it == 0))
        mesh = fill_small_holes(mesh, max_loop_edges=50, verbose=(it == 0))

        defect_stats, _, _ = analyze_mesh_defects(mesh)
        if defect_stats['open_edges'] == 0 and defect_stats['nonmanifold_edges'] == 0:
            break
    return mesh


# ------------------------------------------------------------------
#  主处理流程
# ------------------------------------------------------------------

def process_shell(mesh, num_passes=0, repair_mode=False,
                  angle_threshold_deg=30.0, min_faces=30,
                  thin_mode='adaptive', thin_threshold=0.1,
                  line_tol=0.05, circle_tol=0.05, spline_tol=0.1,
                  irregular_perimeter_ratio=0.05,
                  thickness_grid_size=128,
                  vis_mode='plates', vis_alpha=80):
    """
    薄壳扫描网格处理主函数。

    Returns
    -------
    scene : trimesh.Scene
        当前阶段主可视化场景。
    world_scene : trimesh.Scene
        厚度场可视化场景（始终基于当前处理后的网格）。
    processed_mesh : trimesh.Trimesh
        处理后的网格。
    all_stats : dict
        统计信息。
    """
    # 0. 基础统计
    with Timer("Initial stats/defect analysis"):
        stats = compute_mesh_stats(mesh)
        print("Yo, mesh stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
        print("\nMesh defect analysis:")
        print(f"  open edges: {defect_stats['open_edges']}")
        print(f"  nonmanifold edges: {defect_stats['nonmanifold_edges']}")
        print(f"  watertight (no open edges): {defect_stats['watertight_by_count']}")

    # 可选修复
    if repair_mode and (defect_stats['open_edges'] > 0 or defect_stats['nonmanifold_edges'] > 0):
        with Timer("Mesh repair"):
            print("\n[Repair mode] Attempting to fix mesh...")
            mesh = _repair_loop(mesh)
            stats = compute_mesh_stats(mesh)
            print("\nAfter repair:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
            print(f"  After repair defects: open_edges={defect_stats['open_edges']}, "
                  f"nonmanifold_edges={defect_stats['nonmanifold_edges']}")

    # 首次厚度估计与薄板分割（后续各阶段都会复用）
    def analyze_current(m):
        with Timer("  thickness estimation"):
            thickness, reliability = estimate_shell_thickness(m, grid_size=thickness_grid_size)
        with Timer("  thickness statistics"):
            th_stats = compute_wall_thickness_statistics(thickness, reliability)
        with Timer("  plate segmentation"):
            labels = segment_plates_by_smoothness(
                m, angle_threshold_deg=angle_threshold_deg, min_faces=min_faces
            )
        return thickness, reliability, th_stats, labels

    with Timer("First thickness/segmentation analysis"):
        thickness, reliability, th_stats, labels = analyze_current(mesh)

    print("\n[Shell analysis]")
    print(f"  Median thickness: {th_stats['median']:.4f}")
    print(f"  Reliable faces: {th_stats['reliable_count']} / {len(mesh.faces)} "
          f"({th_stats['reliable_ratio']:.2%})")
    print(f"  Number of plates: {th_stats.get('num_plates', int(labels.max()) + 1)}")

    # Pass 0: 检测可视化
    if num_passes == 0:
        with Timer("pass 0 visualization"):
            if vis_mode == 'thickness':
                world_scene = visualize_thickness(mesh, thickness, reliability)
                scene = visualize_plates(mesh, labels)
            elif vis_mode == 'centers':
                world_scene = visualize_plate_centers(mesh, labels, alpha=vis_alpha)
                scene = world_scene
            else:  # 'plates'
                world_scene = visualize_thickness(mesh, thickness, reliability)
                scene = visualize_plates(mesh, labels)
        return scene, world_scene, mesh.copy(), {**stats, **th_stats}

    # Pass 1: 删除过薄 / 不可靠区域
    print("\n[Pass 1] Removing thin / unreliable regions...")
    thin_mask = detect_thin_regions(
        thickness, mode=thin_mode, threshold=thin_threshold,
        fallback_median=th_stats['median']
    )
    unreliable_mask = ~reliability
    remove_mask = thin_mask | unreliable_mask

    print(f"  Removing {np.sum(remove_mask)} faces "
          f"({np.sum(thin_mask)} thin + {np.sum(unreliable_mask)} unreliable)")

    if np.all(remove_mask):
        raise ValueError("All faces would be removed. Check thin-threshold or mesh quality.")

    if np.any(remove_mask):
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[~remove_mask],
            process=False
        )
        mesh.remove_unreferenced_vertices()
        mesh = fill_small_holes(mesh, max_loop_edges=100, verbose=False)

    with Timer("Second thickness/segmentation analysis (after cleanup)"):
        thickness, reliability, th_stats, labels = analyze_current(mesh)
    print(f"  After cleanup: {mesh.faces.shape[0]} faces, median thickness: {th_stats['median']:.4f}")

    if num_passes == 1:
        world_scene = visualize_thickness(mesh, thickness, reliability)
        scene = visualize_plates(mesh, labels)
        return scene, world_scene, mesh.copy(), {**stats, **th_stats}

    # Pass 2: 基于边界环规律性精化
    print("\n[Pass 2] Refining by boundary regularity...")
    scale = float(np.linalg.norm(mesh.bounding_box.extents))
    if scale < 1e-12:
        scale = 1.0

    # 先可视化并收集不规则小边界环
    with Timer("visualize_boundaries (pass 2)"):
        scene, irregular_loops = visualize_boundaries(
            mesh, labels, scale=scale,
            line_tol=line_tol, circle_tol=circle_tol, spline_tol=spline_tol
        )

    print(f"  Found {len(irregular_loops)} irregular small boundary loops")

    if irregular_loops:
        remove_mask = _faces_adjacent_to_loops(mesh, irregular_loops)
        print(f"  Removing {np.sum(remove_mask)} faces adjacent to irregular loops")
        if np.any(remove_mask) and not np.all(remove_mask):
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces[~remove_mask],
                process=False
            )
            mesh.remove_unreferenced_vertices()
            mesh = fill_small_holes(mesh, max_loop_edges=100, verbose=False)

    # 重新分析用于最终可视化
    with Timer("Third thickness/segmentation analysis (after refinement)"):
        thickness, reliability, th_stats, labels = analyze_current(mesh)
    print(f"  After refinement: {mesh.faces.shape[0]} faces, "
          f"median thickness: {th_stats['median']:.4f}")

    world_scene = visualize_thickness(mesh, thickness, reliability)
    scene, _ = visualize_boundaries(
        mesh, labels, scale=scale,
        line_tol=line_tol, circle_tol=circle_tol, spline_tol=spline_tol
    )

    return scene, world_scene, mesh.copy(), {**stats, **th_stats}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="薄壳扫描网格处理：厚度分析、薄板分割、去薄区、边界规律性精化。"
    )
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("--output", help="保存清理后的网格路径 (可选)")
    parser.add_argument("--num-passes", type=int, default=0, choices=[0, 1, 2],
                        help="处理阶段：0=检测，1=去薄区，2=边界规律性精化（默认0）")
    parser.add_argument("--repair", action="store_true",
                        help="尝试自动修复网格拓扑缺陷")
    parser.add_argument("--angle-threshold", type=float, default=30.0,
                        help="薄板分割二面角阈值（度，默认30）")
    parser.add_argument("--min-faces", type=int, default=30,
                        help="薄板最小面片数（默认30）")
    parser.add_argument("--thin-mode", type=str, default='adaptive',
                        choices=['adaptive', 'absolute'],
                        help="薄区阈值模式：adaptive=厚度中位数比例，absolute=绝对值")
    parser.add_argument("--thin-threshold", type=float, default=0.1,
                        help="薄区阈值（adaptive 默认 0.1*median；absolute 为绝对厚度）")
    parser.add_argument("--line-tol", type=float, default=0.05,
                        help="边界环直线拟合归一化误差阈值")
    parser.add_argument("--circle-tol", type=float, default=0.05,
                        help="边界环圆弧拟合归一化误差阈值")
    parser.add_argument("--spline-tol", type=float, default=0.1,
                        help="边界环样条拟合归一化误差阈值")
    parser.add_argument("--irregular-perimeter-ratio", type=float, default=0.05,
                        help="不规则边界环周长小于该比例*包围盒对角线时会被删除")
    parser.add_argument("--thickness-grid-size", type=int, default=128,
                        help="体素分辨率用于厚度估计（默认128）")
    parser.add_argument("--vis-mode", type=str, default='plates',
                        choices=['plates', 'centers', 'thickness'],
                        help="Pass 0 可视化模式：plates=薄板着色, centers=中心面高亮, thickness=厚度场")
    parser.add_argument("--vis-alpha", type=int, default=80,
                        help="原始网格透明度（仅 centers 模式，0-255，默认80）")
    parser.add_argument("--show", action="store_true", help="显示可视化窗口")
    args = parser.parse_args()

    mesh = trimesh.load(args.input_file)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")
    print(f"Yo, loading model: {args.input_file}")

    scene, world_scene, processed_mesh, all_stats = process_shell(
        mesh,
        num_passes=args.num_passes,
        repair_mode=args.repair,
        angle_threshold_deg=args.angle_threshold,
        min_faces=args.min_faces,
        thin_mode=args.thin_mode,
        thin_threshold=args.thin_threshold,
        line_tol=args.line_tol,
        circle_tol=args.circle_tol,
        spline_tol=args.spline_tol,
        irregular_perimeter_ratio=args.irregular_perimeter_ratio,
        thickness_grid_size=args.thickness_grid_size,
        vis_mode=args.vis_mode,
        vis_alpha=args.vis_alpha,
    )

    if args.output:
        processed_mesh.export(args.output)
        print(f"Processed mesh saved to {args.output}")

    if args.show:
        try:
            os.environ['TRIMESH_DEFAULT_VIEWER'] = 'vedo'
            world_scene.show()
        except Exception as e:
            print(f"\n[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
