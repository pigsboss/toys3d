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
    line_line_distance_and_midpoint
)

# ------------------------- 辅助函数 -------------------------

def compute_mesh_stats(mesh):
    """返回网格基本统计信息字典。"""
    stats = {}
    stats['vertices'] = mesh.vertices.shape[0]
    stats['faces'] = mesh.faces.shape[0]
    stats['edges'] = mesh.edges_unique.shape[0]
    # 边界边数 = 非重复边数 - 出现在两个面中的边数（mesh.edges 为面边对出现的索引）
    stats['boundary_edges'] = mesh.edges_unique.shape[0] - mesh.edges.shape[0]
    stats['is_manifold'] = mesh.is_manifold
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

def refine_axis_from_region(mesh, region_mask, deviation_thr=0.15):
    """
    对指定区域的面片进行轴线拟合，并划分 core / residual。

    参数
    ----
    mesh : trimesh.Trimesh
    region_mask : (N,) bool
        属于该区域的三角面片掩码。
    deviation_thr : float
        用于分离核心与残余的点积阈值。

    返回
    -------
    core_mask : (N,) bool （相对于全部面片）
        该区域的核心面片。
    residual_mask : (N,) bool
        该区域的残余面片。
    axis_dir : (3,) ndarray
        拟合得到的核心轴线方向（单位向量）。
    axis_point : (3,) ndarray
        核心轴线上的一点（世界坐标下）。
    confidence : float
        置信度（核心面片的加权总面积）。
    """
    # 区域子集
    centers = mesh.triangles_center[region_mask]
    normals = mesh.face_normals[region_mask]
    areas = mesh.area_faces[region_mask]

    # 普吕克拟合 (使用全部区域面片做初值)
    C, moments, ref = plucker_design_matrix(centers, normals, areas)
    dir_all, base_all = axis_from_plucker(C, ref_point=ref)

    # 偏差计算
    dots = np.abs(np.dot(normals, dir_all))  # 区域内面片与轴线的点积

    # 核心与残余
    core_sub = dots <= deviation_thr
    residual_sub = ~core_sub

    # 如果核心太少，回退，全部视为核心
    if np.sum(core_sub) < 3:
        core_sub = np.ones_like(core_sub, dtype=bool)

    # 用核心面片重新拟合轴线
    core_centers = centers[core_sub]
    core_normals = normals[core_sub]
    core_areas = areas[core_sub]

    C_core, _, ref_core = plucker_design_matrix(core_centers, core_normals, core_areas)
    dir_core, base_core = axis_from_plucker(C_core, ref_point=ref_core)

    # 置信度 = 核心总面积
    confidence = np.sum(core_areas)

    # 构造全局掩码
    global_idx = np.where(region_mask)[0]
    core_mask_global = np.zeros(mesh.faces.shape[0], dtype=bool)
    residual_mask_global = np.zeros(mesh.faces.shape[0], dtype=bool)
    core_mask_global[global_idx[core_sub]] = True
    residual_mask_global[global_idx[residual_sub]] = True

    return core_mask_global, residual_mask_global, dir_core, base_core, confidence

def assign_final_labels(labels_ransac, masks_dict):
    """
    将 RANSAC 区域标签与 core/residual 掩码合并为单一 4 类标签。

    masks_dict 示例:
    {
        'bar': (region_label_of_bar, core_bar, residual_bar),
        'stem': (region_label_of_stem, core_stem, residual_stem)
    }
    最终标签:
        0 : 未分类
        1 : 把横核心
        2 : 把横残余
        3 : 把立核心
        4 : 把立残余
    """
    N = len(labels_ransac)
    final_labels = np.zeros(N, dtype=int)

    # 把横
    label_bar, core_bar, res_bar = masks_dict['bar']
    final_labels[core_bar] = 1
    final_labels[res_bar] = 2

    # 把立
    label_stem, core_stem, res_stem = masks_dict['stem']
    final_labels[core_stem] = 3
    final_labels[res_stem] = 4

    return final_labels

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

# ------------------------- 主流程函数 -------------------------

def process_handlebar(mesh, 
                      ransac_threshold=0.1,
                      deviation_thr=0.15,
                      region_label_bar=None,
                      region_label_stem=None):
    """
    对一体把进行完整处理，返回可视化场景。

    参数
    ----
    mesh : trimesh.Trimesh
    ransac_threshold : float
        RANSAC 管状区域分割的阈值。
    deviation_thr : float
        管状区域内部划分核心/残余的阈值。
    region_label_bar, region_label_stem : int or None
        手动指定 RANSAC 输出中哪个标签对应把横/把立。
        若为 None，则使用对称性自动判断。

    返回
    -------
    scene : trimesh.Scene
        包含变换后网格和坐标轴的可视化场景。
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
        # 对面积最大的两个区域进行快速轴线拟合（用于对称性计算）
        def _rough_axis(mask):
            c = mesh.triangles_center[mask]
            n = mesh.face_normals[mask]
            a = mesh.area_faces[mask]
            C, _, ref = plucker_design_matrix(c, n, a)
            d, b = axis_from_plucker(C, ref_point=ref)
            return d, b

        dir1, pt1 = _rough_axis(labels_ransac == area_sums[0][1])
        dir2, pt2 = _rough_axis(labels_ransac == area_sums[1][1])
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

    # 5. 对每个区域精细轴线并划分核心/残余
    mask_bar = labels_ransac == label_bar
    core_bar, residual_bar, dir_bar, point_bar, conf_bar = refine_axis_from_region(
        mesh, mask_bar, deviation_thr
    )
    mask_stem = labels_ransac == label_stem
    core_stem, residual_stem, dir_stem, point_stem, conf_stem = refine_axis_from_region(
        mesh, mask_stem, deviation_thr
    )

    # 6. 合并最终标签
    masks_dict = {
        'bar': (label_bar, core_bar, residual_bar),
        'stem': (label_stem, core_stem, residual_stem)
    }
    final_labels = assign_final_labels(labels_ransac, masks_dict)

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
    for i, name in enumerate(['Unassigned', 'Bar core', 'Bar residual', 'Stem core', 'Stem residual']):
        count = np.sum(final_labels == i)
        print(f"  {name}: {count} triangles")

    return scene, stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description="处理公路车把网格，分割把横/把立，建立正交坐标系。")
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("--output", help="保存处理后的网格路径 (可选)")
    parser.add_argument("--ransac_thr", type=float, default=0.1, help="RANSAC 阈值 (默认 0.1)")
    parser.add_argument("--dev_thr", type=float, default=0.15, help="核心/残余偏差阈值 (默认 0.15)")
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
    scene, stats = process_handlebar(mesh,
                                     ransac_threshold=args.ransac_thr,
                                     deviation_thr=args.dev_thr)

    # 保存输出
    if args.output:
        # 从场景中提取着色后的网格并保存
        out_mesh = scene.dump(concatenate=True)
        out_mesh.export(args.output)
        print(f"Processed mesh saved to {args.output}")

    # 可视化
    if args.show:
        # 使用 vedo 避免关闭窗口退出问题
        os.environ['TRIMESH_DEFAULT_VIEWER'] = 'vedo'
        scene.show()

if __name__ == "__main__":
    main()
