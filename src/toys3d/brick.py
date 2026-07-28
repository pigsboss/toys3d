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
    build_box_aligned_frame_voxel,
    build_box_aligned_frame_mesh,
)


# ------------------------------------------------------------------
#  辅助可视化函数
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


def colorize_defects(mesh, open_face_mask, nonmanifold_face_mask):
    """缺陷面片着色。"""
    labels = np.zeros(len(mesh.faces), dtype=int)
    labels[open_face_mask & ~nonmanifold_face_mask] = 1
    labels[~open_face_mask & nonmanifold_face_mask] = 2
    labels[open_face_mask & nonmanifold_face_mask] = 3

    palette = np.array([
        [200, 200, 200, 255],
        [255, 0, 0, 255],
        [255, 255, 0, 255],
        [160, 32, 240, 255],
    ], dtype=np.uint8)
    mesh.visual.face_colors = palette[labels]
    return labels


def colorize_by_box_faces(mesh, axes, origin, threshold=1e-3):
    """
    根据面片中心在长方体局部坐标中的位置，将其归为 6 个面之一。
    0=左(-x), 1=右(+x), 2=前(-y), 3=后(+y), 4=底(-z), 5=顶(+z), 6=中性
    """
    centers = mesh.triangles_center
    local = (centers - origin) @ axes.T
    abs_local = np.abs(local)
    max_dim = np.argmax(abs_local, axis=1)
    sign = np.sign(local[np.arange(len(local)), max_dim])

    labels = np.full(len(centers), 6, dtype=int)
    for dim in range(3):
        mask = max_dim == dim
        neg = mask & (sign < 0)
        pos = mask & (sign > 0)
        labels[neg] = 2 * dim
        labels[pos] = 2 * dim + 1

    palette = np.array([
        [200, 50, 50, 255],    # 0 -x 红
        [255, 100, 100, 255],  # 1 +x 浅红
        [50, 200, 50, 255],    # 2 -y 绿
        [100, 255, 100, 255],  # 3 +y 浅绿
        [50, 50, 200, 255],    # 4 -z 蓝
        [100, 100, 255, 255],  # 5 +z 浅蓝
        [180, 180, 180, 255],  # 6 中性 灰
    ], dtype=np.uint8)
    mesh.visual.face_colors = palette[labels]
    return labels


def create_oriented_box(origin, axes, extents, color=[255, 165, 0, 60]):
    """创建半透明的 OBB/长方体盒子用于可视化。"""
    box = trimesh.creation.box(extents=extents)
    R = axes.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    box.apply_transform(T)
    box.visual.face_colors = color
    return box


def build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask):
    """仅检测模式：缺陷可视化。"""
    scene = trimesh.Scene()
    vis_mesh = mesh.copy()
    colorize_defects(vis_mesh, open_face_mask, nonmanifold_face_mask)
    scene.add_geometry(vis_mesh)
    origin = mesh.bounding_box.centroid
    max_ext = mesh.bounding_box.extents.max()
    add_axes_to_scene(scene, origin=origin,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)
    return scene


def build_initial_visualization(mesh, axes, origin, extents=None):
    """初步处理可视化：原始网格 + 候选轴 + OBB 盒子。"""
    scene = trimesh.Scene()
    vis_mesh = mesh.copy()

    vis_mesh.visual.face_colors = [200, 200, 200, 255]

    scene.add_geometry(vis_mesh)

    max_ext = mesh.bounding_box.extents.max()
    L = max_ext * 0.6

    for i, color in enumerate([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]):
        axis = axes[i]
        p1 = origin - L * axis
        p2 = origin + L * axis
        cyl = trimesh.creation.cylinder(0.008 * max_ext, segment=[p1, p2])
        cyl.visual.face_colors = color
        scene.add_geometry(cyl)

    if extents is not None:
        box = create_oriented_box(origin, axes, extents)
        scene.add_geometry(box)

    add_axes_to_scene(scene, origin=mesh.bounding_box.centroid,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)
    return scene


def build_final_visualization(mesh_transformed, axes, origin, extents):
    """精细处理可视化：已变换网格 + 局部坐标架 + 拟合长方体盒子。"""
    scene = trimesh.Scene()

    colorize_by_box_faces(mesh_transformed, np.eye(3), np.zeros(3))
    scene.add_geometry(mesh_transformed)

    box = create_oriented_box(np.zeros(3), np.eye(3), extents, color=[255, 165, 0, 40])
    scene.add_geometry(box)

    add_axes_to_scene(scene, origin=np.zeros(3),
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max(extents) * 0.6)
    return scene


# ------------------------------------------------------------------
#  主处理流程
# ------------------------------------------------------------------

def process_brick(mesh, method='voxel', num_passes=2, repair_mode=False,
                  grid_size=128, shell_depths=None):
    """
    处理长方体扫描网格，建立局部正交坐标系。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    method : 'voxel' | 'mesh'
    num_passes : 0 | 1 | 2
    repair_mode : bool
    grid_size : int
    shell_depths : tuple of 3 tuples
        ((x_neg, x_pos), (y_neg, y_pos), (z_neg, z_pos))

    Returns
    -------
    scene : trimesh.Scene
        最终或中间可视化场景。
    world_scene : trimesh.Scene
        世界坐标系下的可视化场景。
    transformed_mesh : trimesh.Trimesh
        已变换到新坐标系的原始网格（不含拟合盒子）。
    stats : dict
        统计信息。
    """
    # 0. 基本统计
    stats = compute_mesh_stats(mesh)
    print("Sup, mesh stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 缺陷分析
    defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    print("\nMesh defect analysis:")
    print(f"  open edges: {defect_stats['open_edges']}")
    print(f"  nonmanifold edges: {defect_stats['nonmanifold_edges']}")
    print(f"  watertight (no open edges): {defect_stats['watertight_by_count']}")

    # 修复
    if repair_mode and (defect_stats['open_edges'] > 0 or defect_stats['nonmanifold_edges'] > 0):
        print("\n[Repair mode] Attempting to fix mesh...")
        mesh = repair_mesh_by_removing_duplicates(mesh)
        mesh = repair_nonmanifold_edges(mesh)
        mesh = fill_small_holes(mesh)
        stats = compute_mesh_stats(mesh)
        print("After repair:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # 重新计算缺陷统计和掩码，使其与修复后的网格面数匹配
        defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
        print(f"  After repair defects: open_edges={defect_stats['open_edges']}, "
              f"nonmanifold_edges={defect_stats['nonmanifold_edges']}")

    # num-passes 0: 仅检测/修复
    if num_passes == 0:
        world_scene = build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask)
        scene = trimesh.Scene(mesh.copy())
        colorize_defects(scene.geometry[list(scene.geometry.keys())[0]],
                         open_face_mask, nonmanifold_face_mask)
        return scene, world_scene, mesh.copy(), stats

    # num-passes 1: 初步处理
    if num_passes == 1:
        if method == 'voxel':
            print(f"\n[Pass 1] Initial OBB estimation using voxel PCA (grid_size={grid_size})")
            T_w2l, T_l2w, ux, uy, uz, origin, extents, fit = build_box_aligned_frame_voxel(
                mesh, grid_size=grid_size, optimize=False
            )
        else:
            print("\n[Pass 1] Initial frame estimation using mesh-based planar detection + 2D OBB")
            T_w2l, T_l2w, ux, uy, uz, origin, extents, fit = build_box_aligned_frame_mesh(
                mesh, distance_thr_ratio=0.02, shell_depths=shell_depths, refine=False
            )

        axes = np.vstack([ux, uy, uz])
        print(f"  origin  = {origin.round(4)}")
        print(f"  extents = {extents.round(4)}")
        print(f"  fit     = {fit}")

        mesh_transformed = mesh.copy()
        mesh_transformed.apply_transform(T_w2l)

        world_scene = build_initial_visualization(mesh, axes, origin, extents=extents)
        scene = build_final_visualization(mesh_transformed, axes, origin, extents)
        return scene, world_scene, mesh_transformed, stats

    # num-passes 2: 精细处理
    if num_passes == 2:
        if method == 'voxel':
            print(f"\n[Pass 2] Refined OBB estimation using voxel optimization (grid_size={grid_size})")
            T_w2l, T_l2w, ux, uy, uz, origin, extents, fit = build_box_aligned_frame_voxel(
                mesh, grid_size=grid_size, optimize=True
            )
        else:
            print("\n[Pass 2] Refined frame estimation using mesh-based planar detection + side-band RANSAC")
            T_w2l, T_l2w, ux, uy, uz, origin, extents, fit = build_box_aligned_frame_mesh(
                mesh, distance_thr_ratio=0.02, shell_depths=shell_depths, refine=True
            )

        axes = np.vstack([ux, uy, uz])
        print(f"  origin  = {origin.round(4)}")
        print(f"  extents = {extents.round(4)}")
        print(f"  fit     = {fit}")

        mesh_transformed = mesh.copy()
        mesh_transformed.apply_transform(T_w2l)

        world_scene = build_initial_visualization(mesh, axes, origin, extents=extents)
        scene = build_final_visualization(mesh_transformed, axes, origin, extents)
        return scene, world_scene, mesh_transformed, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="处理长方体扫描网格（音箱/手机/砖块），建立正交坐标系。")
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("--output", help="保存处理后的网格路径 (可选)")
    parser.add_argument("--method", type=str, default='voxel', choices=['voxel', 'mesh'],
                        help="处理方法：voxel=体素化OBB，mesh=基于网格几何的平面检测+二维OBB（默认voxel）")
    parser.add_argument("--num-passes", type=int, default=2, choices=[0, 1, 2],
                        help="处理阶段：0=检测/修复，1=初步处理，2=精细处理（默认2）")
    parser.add_argument("--grid-size", type=int, default=128,
                        help="体素化分辨率（默认128）")
    parser.add_argument("--shell", type=float, nargs='+', default=None,
                        help="壳厚度比率，支持：1 个值 s（各方向统一）、"
                             "3 个值 x y z（正负对称）、"
                             "6 个值 xn xp yn yp zn zp（各方向独立）")
    parser.add_argument("--repair", action="store_true",
                        help="尝试自动修复网格")
    parser.add_argument("--show", action="store_true", help="显示可视化窗口")
    args = parser.parse_args()

    mesh = trimesh.load(args.input_file)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")
    print(f"Sup, loading model: {args.input_file}")

    # 解析 --shell 参数
    shell_depths = None
    if args.shell is not None:
        s = args.shell
        if len(s) == 1:
            v = s[0]
            shell_depths = ((v, v), (v, v), (v, v))
        elif len(s) == 3:
            shell_depths = ((s[0], s[0]), (s[1], s[1]), (s[2], s[2]))
        elif len(s) == 6:
            shell_depths = ((s[0], s[1]), (s[2], s[3]), (s[4], s[5]))
        else:
            raise ValueError("--shell 需要 1、3 或 6 个数值")

        # 校验：各方向正负壳厚度之和应小于 1.0，避免壳带重叠
        for i, (neg, pos) in enumerate(shell_depths):
            if neg < 0 or pos < 0:
                raise ValueError(f"--shell 分量必须非负，axis={i}")
            if neg + pos >= 1.0:
                raise ValueError(
                    f"--shell 轴 {i} 正负分量之和 {neg + pos} 必须小于 1.0")

    scene, world_scene, transformed_mesh, stats = process_brick(
        mesh,
        method=args.method,
        num_passes=args.num_passes,
        repair_mode=args.repair,
        grid_size=args.grid_size,
        shell_depths=shell_depths
    )

    if args.output:
        transformed_mesh.export(args.output)
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
