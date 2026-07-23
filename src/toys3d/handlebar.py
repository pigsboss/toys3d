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
    point_line_distance,
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

def partition_four_regions_local(mesh, mask_bar, mask_stem,
                                 local_coords,
                                 bar_y_margin,
                                 stem_x_margin):
    """
    在正交局部坐标系中按轴向区间划分四区域。

    坐标约定：
    local_coords[:, 0] = x（把立方向，+x 指向车身头管，即车头到车尾）
    local_coords[:, 1] = y（把横方向）
    local_coords[:, 2] = z（右手定则）

    分区规则：
    - 把横：排除 |y| 较小的中央部分（junction 附近）
    - 把立：排除 x 较小的前段（junction 附近），保留 x 较大的后段（头管侧）
    """
    N = mesh.faces.shape[0]

    # 把横：在 y 方向上关于 junction 对称，排除 |y| 较小的中央部分
    bar_transition = mask_bar & (np.abs(local_coords[:, 1]) <= bar_y_margin)

    # 把立：x 正方向指向头管，排除 x 较小的前段（junction 附近）
    stem_transition = mask_stem & (local_coords[:, 0] <= stem_x_margin)

    transition_mask = bar_transition | stem_transition

    bar_core = mask_bar & ~bar_transition
    stem_core = mask_stem & ~stem_transition

    residual_mask = ~(mask_bar | mask_stem)

    return bar_core, stem_core, transition_mask, residual_mask

def colorize_mesh(mesh, final_labels):
    """
    根据最终标签设置面片颜色。
    0: 灰色, 1: 深蓝, 2: 深红, 3: 橙色
    """
    palette = np.array([
        [180, 180, 180, 255],   # 0 残余/未分类 灰
        [0, 100, 200, 255],     # 1 把横核心 深蓝
        [200, 50, 50, 255],     # 2 把立核心 深红
        [255, 165, 0, 255],     # 3 过渡区 橙
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


# ----------------------------------------------------------------------
#  新辅助函数
# ----------------------------------------------------------------------
def orient_stem_x(u_x, stem_mask, mesh, origin):
    """Oriente u_x so that majority of stem area lies in +x direction."""
    stem_centers = mesh.triangles_center[stem_mask]
    stem_proj = np.dot(stem_centers - origin, u_x)
    pos_area = np.sum(mesh.area_faces[stem_mask][stem_proj > 0])
    neg_area = np.sum(mesh.area_faces[stem_mask][stem_proj < 0])
    if neg_area > pos_area:
        u_x = -u_x
    return u_x

def pass2_t_shape_partition(mesh, mask_bar, mask_stem, u_x, u_y, u_z, origin):
    """
    第二阶段：纯矩形 T 字形约束分区。

    坐标约定：x=把立（+x 指向头管），y=把横，z=x×y。

    估算：
    - d = 把横沿 x 方向尺寸（95% 范围）
    - w = 把立沿 y 方向尺寸（95% 范围）
    - x_min = 把横区域顶点的最小 x 坐标

    分区：
    - 把横过渡区 = mask_bar & (|y| <= 0.5*w)
    - 把立过渡区 = mask_stem & (x <= x_min + d)
    """
    N = mesh.faces.shape[0]
    R = np.column_stack([u_x, u_y, u_z])
    local_coords = (mesh.triangles_center - origin) @ R

    bar_x = local_coords[mask_bar, 0]
    d = np.percentile(bar_x, 95) - np.percentile(bar_x, 5)
    if not np.isfinite(d) or d <= 0:
        d = 30.0
    d = max(d, 1.0)

    stem_y = local_coords[mask_stem, 1]
    w = np.percentile(stem_y, 95) - np.percentile(stem_y, 5)
    if not np.isfinite(w) or w <= 0:
        w = 40.0
    w = max(w, 1.0)

    bar_verts_mask = np.zeros(len(mesh.vertices), dtype=bool)
    bar_faces = mesh.faces[mask_bar]
    if bar_faces.size > 0:
        bar_verts_mask[bar_faces.ravel()] = True
    local_verts = (mesh.vertices - origin) @ R
    if np.any(bar_verts_mask):
        x_min = local_verts[bar_verts_mask, 0].min()
    else:
        x_min = -0.5 * d

    bar_overlap = np.abs(local_coords[:, 1]) <= 0.5 * w
    stem_overlap = local_coords[:, 0] <= (x_min + d)

    bar_core = mask_bar & ~bar_overlap
    stem_core = mask_stem & ~stem_overlap
    transition_mask = (mask_bar & bar_overlap) | (mask_stem & stem_overlap)
    residual_mask = ~(mask_bar | mask_stem)

    return bar_core, stem_core, transition_mask, residual_mask, d, w

def pass3_refine_by_area(mesh, core_mask, axis_dir, axis_point, dim_est):
    """
    基于轴向距离剔除离群面片的简单精化。
    保留距中位数 1.5*dim_est 范围内的面片，其余转为过渡区。
    """
    N = len(mesh.faces)
    if np.sum(core_mask) < 10:
        return core_mask, np.zeros(N, dtype=bool)
    centers = mesh.triangles_center[core_mask]
    vec = centers - axis_point
    proj = np.dot(vec, axis_dir)
    median = np.median(proj)
    lo = median - 1.5 * dim_est
    hi = median + 1.5 * dim_est
    keep = (proj >= lo) & (proj <= hi)
    global_indices = np.where(core_mask)[0]
    global_keep = np.zeros(N, dtype=bool)
    global_keep[global_indices[keep]] = True
    trans = core_mask & ~global_keep
    return global_keep, trans

def _axes_are_close(ax1, ax2, angle_tol=0.99, dist_tol=1e-3):
    """判断两组轴线是否在方向和位置上足够接近。"""
    d1_bar, p1_bar = ax1['bar']
    d2_bar, p2_bar = ax2['bar']
    d1_stem, p1_stem = ax1['stem']
    d2_stem, p2_stem = ax2['stem']

    bar_dot = abs(np.dot(d1_bar / np.linalg.norm(d1_bar),
                         d2_bar / np.linalg.norm(d2_bar)))
    stem_dot = abs(np.dot(d1_stem / np.linalg.norm(d1_stem),
                          d2_stem / np.linalg.norm(d2_stem)))
    bar_dist = np.linalg.norm(p1_bar - p2_bar)
    stem_dist = np.linalg.norm(p1_stem - p2_stem)

    return (bar_dot > angle_tol and stem_dot > angle_tol and
            bar_dist < dist_tol and stem_dist < dist_tol)

def _add_dashed_line(scene, p1, p2, color, radius, segments=20):
    """用多个短圆柱近似虚线。"""
    t = np.linspace(0, 1, segments * 2 + 1)
    for i in range(0, 2 * segments, 2):
        a = p1 + t[i] * (p2 - p1)
        b = p1 + t[i + 1] * (p2 - p1)
        cyl = trimesh.creation.cylinder(radius, segment=[a, b])
        cyl.visual.face_colors = color
        scene.add_geometry(cyl)

def _add_axis_pair(scene, axes, L_axis, max_ext,
                   bar_color, stem_color,
                   radius_factor, dashed):
    """添加一对把横/把立轴线，可选虚线效果。"""
    dir_bar, point_bar = axes['bar']
    dir_stem, point_stem = axes['stem']

    # 把横轴线
    p1 = point_bar - L_axis * dir_bar
    p2 = point_bar + L_axis * dir_bar
    if dashed:
        _add_dashed_line(scene, p1, p2, bar_color, radius_factor * max_ext)
    else:
        cyl = trimesh.creation.cylinder(radius_factor * max_ext, segment=[p1, p2])
        cyl.visual.face_colors = bar_color
        scene.add_geometry(cyl)

    # 把立轴线
    p1 = point_stem - L_axis * dir_stem
    p2 = point_stem + L_axis * dir_stem
    if dashed:
        _add_dashed_line(scene, p1, p2, stem_color, radius_factor * max_ext)
    else:
        cyl = trimesh.creation.cylinder(radius_factor * max_ext, segment=[p1, p2])
        cyl.visual.face_colors = stem_color
        scene.add_geometry(cyl)

def build_world_visualization(mesh, final_labels,
                              axes_pass1, axes_pass2, axes_pass3,
                              max_ext, num_passes=3):
    """
    在世界坐标系中可视化：
    - 着色网格（最终分区结果）
    - 第一/二/三阶段估计的把横、把立轴线
    """
    scene = trimesh.Scene()

    # 着色网格
    vis_mesh = mesh.copy()
    colorize_mesh(vis_mesh, final_labels)
    scene.add_geometry(vis_mesh)

    L_axis = max_ext * 1.5
    frame_len = max_ext * 0.6

    # Pass 1: 虚线/半透明
    _add_axis_pair(scene, axes_pass1, L_axis, max_ext,
                   bar_color=[0, 100, 200, 120],
                   stem_color=[200, 50, 50, 120],
                   radius_factor=0.004, dashed=True)

    # Pass 2: 点划线/中等透明
    if num_passes >= 2 and not _axes_are_close(axes_pass1, axes_pass2):
        _add_axis_pair(scene, axes_pass2, L_axis, max_ext,
                       bar_color=[0, 150, 255, 180],
                       stem_color=[255, 80, 80, 180],
                       radius_factor=0.006, dashed=True)

    # Pass 3: 实线/不透明
    if num_passes >= 3 and not _axes_are_close(axes_pass2, axes_pass3):
        _add_axis_pair(scene, axes_pass3, L_axis, max_ext,
                       bar_color=[0, 100, 200, 255],
                       stem_color=[200, 50, 50, 255],
                       radius_factor=0.008, dashed=False)

    # 公垂线段与中点（基于最终轴线）
    dir_bar, point_bar = axes_pass3['bar']
    dir_stem, point_stem = axes_pass3['stem']
    p_close_bar, p_close_stem, midpoint = closest_points_on_lines(
        dir_bar, point_bar, dir_stem, point_stem
    )

    cyl_perp = trimesh.creation.cylinder(0.008 * max_ext, segment=[p_close_bar, p_close_stem])
    cyl_perp.visual.face_colors = [0, 180, 0, 255]
    scene.add_geometry(cyl_perp)

    sph_mid = trimesh.creation.icosphere(subdivisions=2, radius=0.015 * max_ext)
    sph_mid.apply_translation(midpoint)
    sph_mid.visual.face_colors = [255, 255, 0, 255]
    scene.add_geometry(sph_mid)

    # 最终坐标架
    x_raw = dir_stem / np.linalg.norm(dir_stem)
    y_raw = dir_bar / np.linalg.norm(dir_bar)
    y_ortho = y_raw - np.dot(y_raw, x_raw) * x_raw
    if np.linalg.norm(y_ortho) < 1e-6:
        y_ortho = np.array([1, 0, 0]) if abs(x_raw[1]) < 0.9 else np.array([0, 1, 0])
        y_ortho = y_ortho - np.dot(y_ortho, x_raw) * x_raw
    y_ortho /= np.linalg.norm(y_ortho)
    z_ortho = np.cross(x_raw, y_ortho)
    add_axes_to_scene(scene, midpoint, x_raw, y_ortho, z_ortho, length=frame_len)

    return scene

# ------------------------- 主流程函数 -------------------------

def process_handlebar(mesh,
                      ransac_threshold=0.1,
                      region_label_bar=None,
                      region_label_stem=None,
                      num_passes=3):
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
    num_passes : {1, 2, 3}
        分区阶段数：
        1 = RANSAC 初始分区
        2 = +T 字形矩形约束
        3 = + 截面积精化

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

    # 5. Pass 1: 初始轴线
    mask_bar = labels_ransac == label_bar
    mask_stem = labels_ransac == label_stem

    init_dir_bar, init_pt_bar = rough_axis_from_mask(mesh, mask_bar)
    init_dir_stem, init_pt_stem = rough_axis_from_mask(mesh, mask_stem)

    axes_pass1 = {
        'bar': (init_dir_bar, init_pt_bar),
        'stem': (init_dir_stem, init_pt_stem)
    }

    if num_passes == 1:
        # 仅第一阶段：直接把 RANSAC 区域作为最终分区
        bar_core = mask_bar
        stem_core = mask_stem
        transition_mask = np.zeros(N, dtype=bool)
        dir_bar, point_bar = init_dir_bar, init_pt_bar
        dir_stem, point_stem = init_dir_stem, init_pt_stem
        axes_pass2 = axes_pass1
        axes_pass3 = axes_pass1
    else:
        # --- Pass 2: T 字形矩形约束分区 ---
        _, midpoint = line_line_distance_and_midpoint(init_dir_bar, init_pt_bar,
                                                      init_dir_stem, init_pt_stem)
        origin = midpoint

        u_x = init_dir_stem / np.linalg.norm(init_dir_stem)
        u_x = orient_stem_x(u_x, mask_stem, mesh, origin)

        y_raw = init_dir_bar - np.dot(init_dir_bar, u_x) * u_x
        if np.linalg.norm(y_raw) < 1e-6:
            y_raw = np.array([0, 1, 0]) if abs(u_x[1]) < 0.9 else np.array([1, 0, 0])
            y_raw = y_raw - np.dot(y_raw, u_x) * u_x
        u_y = y_raw / np.linalg.norm(y_raw)
        u_z = np.cross(u_x, u_y)

        bar_core, stem_core, transition_mask, residual_mask, d_est, w_est = pass2_t_shape_partition(
            mesh, mask_bar, mask_stem, u_x, u_y, u_z, origin
        )

        # 用 Pass 2 核心重新拟合轴线并更新坐标系
        dir_bar_p2, point_bar_p2 = rough_axis_from_mask(mesh, bar_core)
        dir_stem_p2, point_stem_p2 = rough_axis_from_mask(mesh, stem_core)

        axes_pass2 = {
            'bar': (dir_bar_p2, point_bar_p2),
            'stem': (dir_stem_p2, point_stem_p2)
        }

        # 更新原点和坐标轴
        _, midpoint = line_line_distance_and_midpoint(dir_bar_p2, point_bar_p2,
                                                      dir_stem_p2, point_stem_p2)
        origin = midpoint

        u_x = dir_stem_p2 / np.linalg.norm(dir_stem_p2)
        u_x = orient_stem_x(u_x, stem_core, mesh, origin)
        y_raw = dir_bar_p2 - np.dot(dir_bar_p2, u_x) * u_x
        if np.linalg.norm(y_raw) < 1e-6:
            y_raw = np.array([0, 1, 0]) if abs(u_x[1]) < 0.9 else np.array([1, 0, 0])
            y_raw = y_raw - np.dot(y_raw, u_x) * u_x
        u_y = y_raw / np.linalg.norm(y_raw)
        u_z = np.cross(u_x, u_y)

        if num_passes == 2:
            dir_bar, point_bar = dir_bar_p2, point_bar_p2
            dir_stem, point_stem = dir_stem_p2, point_stem_p2
            axes_pass3 = axes_pass2
        else:
            # --- Pass 3: 基于截面积精化 ---
            if np.sum(bar_core) >= 10:
                dir_bar, point_bar = rough_axis_from_mask(mesh, bar_core)
                bar_core, bar_trans_add = pass3_refine_by_area(
                    mesh, bar_core, dir_bar, point_bar, d_est
                )
            else:
                bar_trans_add = np.zeros(N, dtype=bool)
                dir_bar, point_bar = dir_bar_p2, point_bar_p2

            if np.sum(stem_core) >= 10:
                dir_stem, point_stem = rough_axis_from_mask(mesh, stem_core)
                stem_core, stem_trans_add = pass3_refine_by_area(
                    mesh, stem_core, dir_stem, point_stem, w_est
                )
            else:
                stem_trans_add = np.zeros(N, dtype=bool)
                dir_stem, point_stem = dir_stem_p2, point_stem_p2

            transition_mask = transition_mask | bar_trans_add | stem_trans_add

            # 用精化后的核心重新拟合最终轴线
            if np.sum(bar_core) >= 10:
                dir_bar, point_bar = rough_axis_from_mask(mesh, bar_core)
            if np.sum(stem_core) >= 10:
                dir_stem, point_stem = rough_axis_from_mask(mesh, stem_core)

            axes_pass3 = {
                'bar': (dir_bar, point_bar),
                'stem': (dir_stem, point_stem)
            }

    conf_bar = np.sum(mesh.area_faces[bar_core]) if np.sum(bar_core) > 0 else 0.0
    conf_stem = np.sum(mesh.area_faces[stem_core]) if np.sum(stem_core) > 0 else 0.0

    # 6. 构建最终标签
    final_labels = np.zeros(N, dtype=int)
    final_labels[bar_core] = 1
    final_labels[stem_core] = 2
    final_labels[transition_mask] = 3
    # 0 = residual

    # Build world-coordinate visualisation (before any transformation)
    max_ext = mesh.bounding_box.extents.max()
    world_scene = build_world_visualization(mesh, final_labels,
                                            axes_pass1, axes_pass2, axes_pass3,
                                            max_ext, num_passes=num_passes)

    # 7. 正交坐标系构建（x=把立，y=把横）
    T_w2l, T_l2w, u_x, u_y, u_z, origin = orthogonalize_axes(
        dir_x=dir_stem, point_x=point_stem, weight_x=conf_stem,
        dir_y=dir_bar, point_y=point_bar, weight_y=conf_bar
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
    for i, name in enumerate(['Residual', 'Bar core', 'Stem core', 'Transition']):
        count = np.sum(final_labels == i)
        print(f"  {name}: {count} triangles")

    return scene, world_scene, stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description="处理公路车把网格，分割把横/把立，建立正交坐标系。")
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("--output", help="保存处理后的网格路径 (可选)")
    parser.add_argument("--ransac_thr", type=float, default=0.1, help="RANSAC 阈值 (默认 0.1)")
    parser.add_argument("--num-passes", type=int, default=3, choices=[1, 2, 3],
                        help="分区阶段数：1=RANSAC初始分区，2=+T字形约束分区，3=+截面积精化（默认3）")
    parser.add_argument("--show", action="store_true", help="显示可视化窗口")
    args = parser.parse_args()

    # 加载网格
    mesh = trimesh.load(args.input_file)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")
    print(f"Hey, loading model: {args.input_file}")

    # 处理
    transformed_scene, world_scene, stats = process_handlebar(
        mesh,
        ransac_threshold=args.ransac_thr,
        num_passes=args.num_passes
    )

    # 保存输出
    if args.output:
        out_mesh = transformed_scene.dump(concatenate=True)
        out_mesh.export(args.output)
        print(f"Processed mesh saved to {args.output}")

    # 可视化
    if args.show:
        os.environ['TRIMESH_DEFAULT_VIEWER'] = 'vedo'
        world_scene.show()

if __name__ == "__main__":
    main()
