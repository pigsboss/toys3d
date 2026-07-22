import sys
import os

# Ensure src directory is on the path so that 'toys3d' can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up: src/
_src_parent = os.path.dirname(_project_root)  # parent of src, i.e., the project root
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)

import numpy as np
import trimesh

from toys3d.geometrics import (
    plucker_design_matrix,
    axis_from_plucker,
    orthogonalize_axes,
    segment_tubular_regions,
    line_line_distance_and_midpoint,
    kmeans_1d,
    sample_axial_section_areas,
    suggest_slice_spacing,
    compute_cross_section_area,
)

# ------------------------- 辅助函数 -------------------------

def closest_points_on_lines(dir1, pt1, dir2, pt2):
    """Return the two closest points of two infinite lines and their midpoint."""
    d1 = dir1 / np.linalg.norm(dir1)
    d2 = dir2 / np.linalg.norm(dir2)
    w = pt1 - pt2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w)
    e = np.dot(d2, w)
    denom = a * c - b * b
    if abs(denom) < 1e-9:          # parallel
        sc = 0.0
        tc = (b > c and d / b) or (e / c)
    else:
        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom
    p1 = pt1 + sc * d1
    p2 = pt2 + tc * d2
    midpoint = (p1 + p2) / 2.0
    return p1, p2, midpoint

def compute_mesh_stats(mesh):
    """返回网格基本统计信息字典。"""
    stats = {}
    stats['vertices'] = mesh.vertices.shape[0]
    stats['faces'] = mesh.faces.shape[0]
    stats['edges'] = mesh.edges_unique.shape[0]
    # 边界边数 = 非重复边数 - 出现在两个面中的边数（mesh.edges 为面边对出现的索引）
    stats['boundary_edges'] = mesh.edges_unique.shape[0] - mesh.edges.shape[0]
    stats['is_watertight'] = mesh.is_watertight
    return stats

def symmetry_score(mesh, region_mask, axis_dir, axis_point, midpoint):
    """
    计算管状区域相对于公垂线中点的对称性得分（0~1，越高越对称）。
    原理：核心面片中心投影到轴线上，比较中点两侧的面积占比。
    """
    indices = np.where(region_mask)[0]
    if len(indices) < 5:
        return 0.0
    centers = mesh.triangles_center[indices]
    areas = mesh.area_faces[indices]
    vecs = centers - midpoint
    proj = np.dot(vecs, axis_dir)
    pos_mask = proj >= 0
    neg_mask = ~pos_mask
    pos_area = areas[pos_mask].sum()
    neg_area = areas[neg_mask].sum()
    total_area = pos_area + neg_area
    if total_area == 0:
        return 0.0
    ratio_diff = abs(pos_area - neg_area) / total_area
    return 1.0 - ratio_diff

def rough_axis_from_mask(mesh, mask):
    """从面片掩码快速拟合轴线（普吕克），返回 direction, point。"""
    c = mesh.triangles_center[mask]
    n = mesh.face_normals[mask]
    a = mesh.area_faces[mask]
    C, _, ref = plucker_design_matrix(c, n, a)
    d, b = axis_from_plucker(C, ref_point=ref)
    return d, b

def iterative_refine_axis_and_core(mesh, region_mask, init_dir, init_pt,
                                   n_clusters=3, n_iter=2, min_core_faces=50):
    """
    利用横截面积聚类迭代精炼轴线与核心区域。
    ...（完整函数体见下方）
    """
    # 函数体如下（请完整复制）
    # ensure 1D integer array (handles both array and scalar inputs)
    region_idx = np.flatnonzero(region_mask)
    if len(region_idx) < min_core_faces:
        core = region_mask.copy()
        return core, np.zeros_like(region_mask), init_dir, init_pt

    submesh = mesh.submesh(region_idx)[0]
    dir_u = init_dir / np.linalg.norm(init_dir)
    pt = np.asarray(init_pt, dtype=np.float64)

    current_dir = dir_u
    current_pt = pt
    best_core = region_mask.copy()

    for iteration in range(n_iter):
        spacing = suggest_slice_spacing(submesh, current_dir, min_count=15, max_count=200)
        verts = submesh.vertices
        proj = np.dot(verts - current_pt, current_dir)
        d_min = proj.min()
        d_max = proj.max()
        if d_max - d_min < spacing:
            break
        distances = np.arange(d_min, d_max, spacing)
        dists, areas = sample_axial_section_areas(submesh, current_dir, current_pt, distances)

        valid = areas > 1e-9
        if np.sum(valid) < n_clusters:
            break
        areas_valid = areas[valid]
        dists_valid = dists[valid]

        try:
            labels_km, centers = kmeans_1d(areas_valid, n_clusters)
        except Exception:
            break

        min_label = 0  # 中心升序，0最小
        core_distances = dists_valid[labels_km == min_label]
        if len(core_distances) == 0:
            break
        core_distances = np.sort(core_distances)

        centers_region = mesh.triangles_center[region_idx]
        proj_region = np.dot(centers_region - current_pt, current_dir)

        idx = np.searchsorted(core_distances, proj_region, side='left')
        idx = np.clip(idx, 0, len(core_distances) - 1)
        left = core_distances[idx]
        diff = np.abs(proj_region - left)
        if len(core_distances) > 1:
            idx_right = np.clip(idx + 1, 0, len(core_distances) - 1)
            diff = np.minimum(diff, np.abs(proj_region - core_distances[idx_right]))
        core_in_region = diff <= spacing * 0.6

        core_global = np.zeros(mesh.faces.shape[0], dtype=bool)
        core_global[region_idx[core_in_region]] = True
        if np.sum(core_global) < min_core_faces:
            break

        best_core = core_global

        core_indices = np.where(core_global)[0]
        if len(core_indices) < min_core_faces:
            break

        centroids = mesh.triangles_center[core_indices]
        norms = mesh.face_normals[core_indices]
        ars = mesh.area_faces[core_indices]
        C, _, ref = plucker_design_matrix(centroids, norms, ars)
        new_dir, new_pt = axis_from_plucker(C, ref_point=ref)
        current_dir = new_dir
        current_pt = new_pt

    noncore = region_mask & ~best_core

    if np.sum(best_core) < min_core_faces:
        best_core = region_mask.copy()
        noncore = np.zeros_like(noncore)
        current_dir = init_dir
        current_pt = init_pt

    return best_core, noncore, current_dir, current_pt

def colorize_mesh(mesh, final_labels):
    """
    根据最终标签设置面片颜色。
    0: 灰色, 1: 深蓝, 2: 浅蓝, 3: 深红, 4: 浅红
    """
    palette = np.array([
        [180, 180, 180, 255],   # 0 未分类 灰
        [0, 100, 200, 255],     # 1 把横核心 深蓝
        [135, 206, 250, 255],   # 2 把横残余 浅蓝
        [200, 50, 50, 255],     # 3 把立核心 深红
        [250, 128, 114, 255]    # 4 把立残余 浅红
    ], dtype=np.uint8)
    mesh.visual.face_colors = palette[final_labels]

def add_axes_to_scene(scene, origin, u_x, u_y, u_z, length=0.3, radius=0.01):
    """在场景中添加红、绿、蓝三根坐标轴箭头（圆柱+小球示意）"""
    def add_arrow(o, d, color):
        # 圆柱体作为轴线
        cyl = trimesh.creation.cylinder(radius=radius, segment=[o, o + d*length])
        cyl.visual.face_colors = color
        scene.add_geometry(cyl)
        # 小球作为箭头尖端
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius*3)
        sphere.apply_translation(o + d*length)
        sphere.visual.face_colors = color
        scene.add_geometry(sphere)

    add_arrow(origin, u_x, [255,0,0,255])    # X 红
    add_arrow(origin, u_y, [0,255,0,255])    # Y 绿
    add_arrow(origin, u_z, [0,0,255,255])    # Z 蓝

def build_world_visualization(mesh, final_labels, dir_bar, point_bar, dir_stem, point_stem):
    """
    Build a trimesh.Scene in original world coordinates containing:
    - coloured input mesh
    - bar axis line (blue)
    - stem axis line (red)
    - common perpendicular segment (green)
    - midpoint sphere (yellow)
    - orthonormal frame: X = stem, Y = bar, origin = midpoint
    """
    scene = trimesh.Scene()

    # Coloured mesh
    vis_mesh = mesh.copy()
    colorize_mesh(vis_mesh, final_labels)
    scene.add_geometry(vis_mesh)

    # Scale from bounding box
    extents = mesh.bounding_box.extents
    max_ext = extents.max()
    L_axis = max_ext * 1.5
    frame_len = max_ext * 0.6

    # Bar axis (blue)
    pbar1 = point_bar - L_axis * dir_bar
    pbar2 = point_bar + L_axis * dir_bar
    cyl_bar = trimesh.creation.cylinder(0.005 * max_ext, segment=[pbar1, pbar2])
    cyl_bar.visual.face_colors = [0, 100, 200, 255]
    scene.add_geometry(cyl_bar)

    # Stem axis (red)
    pstem1 = point_stem - L_axis * dir_stem
    pstem2 = point_stem + L_axis * dir_stem
    cyl_stem = trimesh.creation.cylinder(0.005 * max_ext, segment=[pstem1, pstem2])
    cyl_stem.visual.face_colors = [200, 50, 50, 255]
    scene.add_geometry(cyl_stem)

    # Closest points and common perpendicular
    p_close_bar, p_close_stem, midpoint = closest_points_on_lines(dir_bar, point_bar, dir_stem, point_stem)

    cyl_perp = trimesh.creation.cylinder(0.008 * max_ext, segment=[p_close_bar, p_close_stem])
    cyl_perp.visual.face_colors = [0, 180, 0, 255]
    scene.add_geometry(cyl_perp)

    sph_mid = trimesh.creation.icosphere(subdivisions=2, radius=0.015 * max_ext)
    sph_mid.apply_translation(midpoint)
    sph_mid.visual.face_colors = [255, 255, 0, 255]
    scene.add_geometry(sph_mid)

    # Orthogonal frame: X = stem, Y = bar (orthogonalised), origin = midpoint
    x_raw = dir_stem / np.linalg.norm(dir_stem)
    y_raw = dir_bar / np.linalg.norm(dir_bar)
    y_ortho = y_raw - np.dot(y_raw, x_raw) * x_raw
    if np.linalg.norm(y_ortho) < 1e-6:
        y_ortho = np.array([1,0,0]) if abs(x_raw[1]) < 0.9 else np.array([0,1,0])
        y_ortho = y_ortho - np.dot(y_ortho, x_raw) * x_raw
    y_ortho /= np.linalg.norm(y_ortho)
    z_ortho = np.cross(x_raw, y_ortho)

    add_axes_to_scene(scene, midpoint, x_raw, y_ortho, z_ortho, length=frame_len)

    return scene

# ------------------------- 主流程函数 -------------------------

def process_handlebar(mesh, 
                      ransac_threshold=0.1,
                      region_label_bar=None,
                      region_label_stem=None):
    """
    对一体把进行完整处理，返回可视化场景。

    参数
    ----
    mesh : trimesh.Trimesh
    ransac_threshold : float
        RANSAC 管状区域分割的阈值。
    region_label_bar, region_label_stem : int or None
        手动指定 RANSAC 输出中哪个标签对应把横/把立。
        若为 None，则使用对称性自动判断。

    返回
    -------
    scene_transformed : trimesh.Scene
        变换后网格和坐标轴的可视化场景。
    world_scene : trimesh.Scene
        原始世界坐标系下的可视化场景（含轴线、公垂线段、中点、坐标架）。
    stats : dict
        统计信息。
    """
    # 1. 基本统计
    stats = compute_mesh_stats(mesh)
    print("Hey, mesh stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 2. 准备数据
    centers = mesh.triangles_center
    normals = mesh.face_normals
    areas = mesh.area_faces
    N = mesh.faces.shape[0]

    # 3. RANSAC 管状分割
    labels_ransac, axes_ransac = segment_tubular_regions(
        normals, areas=areas, threshold=ransac_threshold,
        min_faces=50, max_regions=2, max_iterations=2000
    )
    unique_labels = np.unique(labels_ransac)
    print(f"Hey, RANSAC found regions: {unique_labels}, axes count: {len(axes_ransac)}")

    if len(axes_ransac) < 2:
        raise RuntimeError("未能检测到两个主要管状区域，请检查阈值或模型方向。")

    # 获取各区域面积（用于后续对称性比较）
    area_sums = []
    for lbl in range(1, len(axes_ransac)+1):
        area_sums.append((areas[labels_ransac == lbl].sum(), lbl))
    area_sums.sort(reverse=True)

    # 4. 基于对称性识别把横/把立
    if region_label_bar is None:
        dir1, pt1 = rough_axis_from_mask(mesh, labels_ransac == area_sums[0][1])
        dir2, pt2 = rough_axis_from_mask(mesh, labels_ransac == area_sums[1][1])
        _, midpoint = line_line_distance_and_midpoint(dir1, pt1, dir2, pt2)

        sym1 = symmetry_score(mesh, labels_ransac == area_sums[0][1], dir1, pt1, midpoint)
        sym2 = symmetry_score(mesh, labels_ransac == area_sums[1][1], dir2, pt2, midpoint)
        print(f"Symmetry scores - region {area_sums[0][1]}: {sym1:.3f}, region {area_sums[1][1]}: {sym2:.3f}")

        if sym1 >= sym2:
            label_bar = area_sums[0][1]
            label_stem = area_sums[1][1]
        else:
            label_bar = area_sums[1][1]
            label_stem = area_sums[0][1]
    else:
        label_bar = region_label_bar
        label_stem = region_label_stem

    print(f"Identification result -> bar: region {label_bar}, stem: region {label_stem}")

    # 5. 截面聚类精炼区域与轴线
    mask_bar = labels_ransac == label_bar
    mask_stem = labels_ransac == label_stem

    init_dir_bar, init_pt_bar = rough_axis_from_mask(mesh, mask_bar)
    init_dir_stem, init_pt_stem = rough_axis_from_mask(mesh, mask_stem)

    core_bar, noncore_bar, dir_bar, point_bar = iterative_refine_axis_and_core(
        mesh, mask_bar, init_dir_bar, init_pt_bar,
        n_clusters=3, n_iter=2, min_core_faces=50
    )
    core_stem, noncore_stem, dir_stem, point_stem = iterative_refine_axis_and_core(
        mesh, mask_stem, init_dir_stem, init_pt_stem,
        n_clusters=3, n_iter=2, min_core_faces=50
    )

    # 置信度 = 核心区域总面积
    conf_bar = np.sum(mesh.area_faces[core_bar]) if np.sum(core_bar) > 0 else 0.0
    conf_stem = np.sum(mesh.area_faces[core_stem]) if np.sum(core_stem) > 0 else 0.0

    # 6. 构建最终标签
    final_labels = np.zeros(N, dtype=int)
    final_labels[core_bar] = 1
    final_labels[noncore_bar] = 2
    final_labels[core_stem] = 3
    final_labels[noncore_stem] = 4

    # Build world-coordinate visualisation (before any transformation)
    world_scene = build_world_visualization(mesh, final_labels,
                                            dir_bar, point_bar,
                                            dir_stem, point_stem)

    # 7. 正交坐标系构建
    T_w2l, T_l2w, u_x, u_y, u_z, origin = orthogonalize_axes(
        dir_x=dir_bar, point_x=point_bar, weight_x=conf_bar,
        dir_y=dir_stem, point_y=point_stem, weight_y=conf_stem
    )

    # 8. 变换网格到新坐标系
    mesh_transformed = mesh.copy()
    mesh_transformed.apply_transform(T_w2l)

    # 9. 着色
    colorize_mesh(mesh_transformed, final_labels)

    # 10. 场景构建
    scene = trimesh.Scene(mesh_transformed)
    # 在新坐标系原点添加坐标轴（变换后原点在局部空间为 (0,0,0)）
    add_axes_to_scene(scene, origin=np.zeros(3), u_x=u_x, u_y=u_y, u_z=u_z)

    # 统计最终区域信息
    print("\nRegion statistics:")
    for i, name in enumerate(['Unassigned', 'Bar core', 'Bar non-core', 'Stem core', 'Stem non-core']):
        count = np.sum(final_labels == i)
        print(f"  {name}: {count} triangles")

    return scene, world_scene, stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description="处理公路车把网格，分割把横/把立，建立正交坐标系。")
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("--output", help="保存处理后的网格路径 (可选)")
    parser.add_argument("--ransac_thr", type=float, default=0.1, help="RANSAC 阈值 (默认 0.1)")
    parser.add_argument("--show", action="store_true", help="显示可视化窗口")
    args = parser.parse_args()

    # 加载网格
    mesh = trimesh.load(args.input_file)
    if not isinstance(mesh, trimesh.Trimesh):
        # 如果是场景，尝试合并所有几何体
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")
    print(f"Hey, loading model: {args.input_file}")

    # 处理
    transformed_scene, world_scene, stats = process_handlebar(mesh,
                                     ransac_threshold=args.ransac_thr)

    # 保存输出
    if args.output:
        # 从场景中提取着色后的网格并保存
        out_mesh = transformed_scene.dump(concatenate=True)
        out_mesh.export(args.output)
        print(f"Processed mesh saved to {args.output}")

    # 可视化
    if args.show:
        # 使用 vedo 避免关闭窗口退出问题
        os.environ['TRIMESH_DEFAULT_VIEWER'] = 'vedo'
        world_scene.show()

if __name__ == "__main__":
    main()
