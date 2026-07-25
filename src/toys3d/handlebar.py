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
    intersect_line_plane,
    kmeans_1d,
    average_antiparallel_directions,
    estimate_symmetry_plane_voxel,
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
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    stats = {}
    stats['vertices'] = mesh.vertices.shape[0]
    stats['faces'] = faces.shape[0]
    stats['edges'] = mesh.edges_unique.shape[0]
    # 手动计算边界边：只属于一个面的边
    edge_face_map = {}
    for face_idx, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(face_idx)
    boundary_edges = sum(1 for faces_list in edge_face_map.values() if len(faces_list) == 1)
    stats['boundary_edges'] = boundary_edges
    stats['is_watertight'] = mesh.is_watertight
    return stats

def analyze_mesh_defects(mesh):
    """
    分析网格拓扑缺陷：开放边、非流形边，以及涉及的面片。

    Returns
    -------
    stats : dict
    open_face_mask : (F,) bool
    nonmanifold_face_mask : (F,) bool
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    edge_face_map = {}
    for face_idx, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(face_idx)

    open_edges = []
    manifold_edges = []
    nonmanifold_edges = []
    for edge, face_list in edge_face_map.items():
        k = len(face_list)
        if k == 1:
            open_edges.append(edge)
        elif k == 2:
            manifold_edges.append(edge)
        else:
            nonmanifold_edges.append(edge)

    open_face_mask = np.zeros(len(faces), dtype=bool)
    nonmanifold_face_mask = np.zeros(len(faces), dtype=bool)

    for edge in open_edges:
        for fi in edge_face_map[edge]:
            open_face_mask[fi] = True
    for edge in nonmanifold_edges:
        for fi in edge_face_map[edge]:
            nonmanifold_face_mask[fi] = True

    stats = {
        'total_faces': len(faces),
        'raw_edges_count': mesh.edges.shape[0],
        'unique_edges_count': len(edge_face_map),
        'open_edges': len(open_edges),
        'manifold_edges': len(manifold_edges),
        'nonmanifold_edges': len(nonmanifold_edges),
        'open_faces': int(np.sum(open_face_mask)),
        'nonmanifold_faces': int(np.sum(nonmanifold_face_mask)),
        'both_defect_faces': int(np.sum(open_face_mask & nonmanifold_face_mask)),
        'watertight_by_count': (len(open_edges) == 0),
    }
    return stats, open_face_mask, nonmanifold_face_mask


def colorize_defects(mesh, open_face_mask, nonmanifold_face_mask):
    """
    缺陷面片着色。
      0: 正常面           -> 浅灰
      1: 仅含开放边       -> 红色
      2: 仅含非流形边     -> 黄色
      3: 同时含两种       -> 紫色
    """
    labels = np.zeros(len(mesh.faces), dtype=int)
    labels[open_face_mask & ~nonmanifold_face_mask] = 1
    labels[~open_face_mask & nonmanifold_face_mask] = 2
    labels[open_face_mask & nonmanifold_face_mask] = 3

    palette = np.array([
        [200, 200, 200, 255],   # 0 normal
        [255,   0,   0, 255],   # 1 open edge
        [255, 255,   0, 255],   # 2 nonmanifold edge
        [160,  32, 240, 255],   # 3 both
    ], dtype=np.uint8)
    mesh.visual.face_colors = palette[labels]
    return labels


def build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask):
    """构建仅显示网格缺陷的世界坐标系场景。"""
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


# -------------------------------------------------------------------
#  新增：对称平面着色及可视化
# -------------------------------------------------------------------
def colorize_by_symmetry_plane(mesh, plane_normal, plane_offset, threshold=1e-3):
    """
    根据对称平面将面片分为左/右/中性三类并着色。

    参数
    ----
    plane_normal : (3,)  对称平面法向（y轴方向）
    plane_offset : float 平面偏移（n·x = offset）
    threshold : float    有向距离绝对值小于该值视为中性面

    返回
    ----
    labels : (F,) int    0=左, 1=右, 2=中性
    """
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    centers = mesh.triangles_center
    dists = centers @ n - plane_offset

    labels = np.zeros(len(centers), dtype=int)
    labels[dists > threshold] = 1   # 右侧
    labels[np.abs(dists) <= threshold] = 2  # 中性

    palette = np.array([
        [0,   100, 200, 255],   # 0 left   -> 深蓝
        [0,   180, 255, 255],   # 1 right  -> 青色
        [180, 180, 180, 255],   # 2 neutral -> 灰
    ], dtype=np.uint8)
    mesh.visual.face_colors = palette[labels]
    return labels


def build_symmetry_visualization(mesh, plane_normal, plane_offset):
    """
    构建对称性分析可视化场景：网格按左右着色 + 半透明镜面。
    """
    scene = trimesh.Scene()

    # 着色后的网格
    vis_mesh = mesh.copy()
    colorize_by_symmetry_plane(vis_mesh, plane_normal, plane_offset)
    scene.add_geometry(vis_mesh)

    # 镜面
    max_ext = mesh.bounding_box.extents.max()
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    center = mesh.bounding_box.centroid
    point_on_plane = center - (np.dot(center, n) - plane_offset) * n

    # 构造镜面局部正交基
    if abs(n[2]) < 0.9:
        mirror_u = np.cross(n, [0, 0, 1])
    else:
        mirror_u = np.cross(n, [1, 0, 0])
    mirror_u = mirror_u / np.linalg.norm(mirror_u)
    mirror_v = np.cross(n, mirror_u)

    span = max_ext * 0.8

    # 创建薄板（局部 x=span, y=span, z=0.01）
    # 薄方向为 z 轴，旋转后 z 轴与法向 n 对齐，使镜面正确垂直于对称平面
    mirror = trimesh.creation.box(extents=[span, span, 0.01])

    # 旋转：局部 z 轴对齐法向 n
    z_axis = np.array([0, 0, 1.0])
    if np.allclose(n, z_axis):
        R = np.eye(3)
    elif np.allclose(n, -z_axis):
        R = np.diag([1, -1, -1])
    else:
        v = np.cross(z_axis, n)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, n)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = point_on_plane
    mirror.apply_transform(T)
    mirror.visual.face_colors = [255, 255, 0, 80]  # 半透明黄
    scene.add_geometry(mirror)

    # 坐标架
    add_axes_to_scene(scene, origin=center,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)

    return scene


def repair_mesh_by_removing_duplicates(mesh):
    """
    通过去除重复/退化面片来修复网格，消除部分非流形边。
    操作后可能仍需检查非流形边是否完全消失。
    """
    print("Applying duplicate face removal...")
    orig_faces = mesh.faces.shape[0]

    # 1. 去除重复面片（保留出现顺序中的第一个）
    unique_faces, unique_inverse = np.unique(mesh.faces, axis=0, return_inverse=True)
    if unique_faces.shape[0] < orig_faces:
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=unique_faces,
            process=True
        )

    # 2. 去除退化（面积为零或极小的）面片
    areas = mesh.area_faces
    non_degenerate = areas > 1e-12
    if np.sum(~non_degenerate) > 0:
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[non_degenerate],
            process=True
        )

    # 3. 清理未引用顶点
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()

    print(f"  Faces before: {orig_faces}, after: {mesh.faces.shape[0]}")
    return mesh


# ----------------------------------------------------------------------
#  新增修复函数：repair_nonmanifold_edges 和 fill_small_holes
# ----------------------------------------------------------------------
def repair_nonmanifold_edges(mesh, max_iterations=10, verbose=True):
    """
    策略2：对每个非流形边，保留法向最一致的两个面，删除其余面片。
    迭代直到没有非流形边（或达到迭代上限）。
    """
    for it in range(max_iterations):
        faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

        # 构建 edge -> faces 映射
        edge_face_map = {}
        for fi, face in enumerate(faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
                key = (a, b) if a < b else (b, a)
                edge_face_map.setdefault(key, []).append(fi)

        nonmanifold = {e: fl for e, fl in edge_face_map.items() if len(fl) > 2}
        if not nonmanifold:
            if verbose:
                print(f"  [Iter {it}] No nonmanifold edges remain.")
            break

        if verbose:
            print(f"  [Iter {it}] {len(nonmanifold)} nonmanifold edges, removing extra faces...")

        normals = mesh.face_normals
        areas = mesh.area_faces
        faces_to_remove = set()

        for edge, fl in nonmanifold.items():
            if verbose:
                va, vb = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
                print(f"    edge {edge} at {va} <-> {vb}, shared by {len(fl)} faces")
                for fi in fl:
                    print(f"      face {fi}: area={areas[fi]:.4f}, normal={normals[fi].round(3)}")

            # 选法向最一致的一对（点积最大；平局时选面积较大的组合）
            best_pair, best_key = None, -np.inf
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    dot = np.dot(normals[fl[i]], normals[fl[j]])
                    score = dot + 1e-6 * min(areas[fl[i]], areas[fl[j]])
                    if score > best_key:
                        best_key = score
                        best_pair = (fl[i], fl[j])

            for fi in fl:
                if fi not in best_pair:
                    faces_to_remove.add(fi)

        keep = np.ones(len(faces), dtype=bool)
        keep[list(faces_to_remove)] = False
        mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=faces[keep], process=False)

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    return mesh


def fill_small_holes(mesh, max_loop_edges=50, verbose=True):
    """
    用质心扇形三角化封闭小边界环。
    只处理边数不超过 max_loop_edges 的环。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

    # 找边界边
    edge_face_map = {}
    for fi, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(fi)
    boundary_edges = [e for e, fl in edge_face_map.items() if len(fl) == 1]

    if not boundary_edges:
        if verbose:
            print("  No boundary edges, nothing to fill.")
        return mesh

    # 把边界边连成环
    adjacency = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    visited = set()
    loops = []
    for start in adjacency:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev, curr = None, start
        while True:
            neighbors = [v for v in adjacency[curr] if v != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            if nxt == start and len(loop) > 2:
                break
            if nxt in visited:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt
        if len(loop) >= 3:
            loops.append(loop)

    # 扇形封闭
    new_vertices = [mesh.vertices]
    new_faces = [faces]
    for loop in loops:
        if len(loop) > max_loop_edges:
            if verbose:
                print(f"  Skipping large boundary loop ({len(loop)} edges).")
            continue
        loop_pts = mesh.vertices[np.array(loop)]
        centroid = loop_pts.mean(axis=0)
        c_idx = sum(len(v) for v in new_vertices)
        new_vertices.append(centroid[None, :])
        tris = []
        for i in range(len(loop)):
            tris.append([c_idx, loop[i], loop[(i + 1) % len(loop)]])
        new_faces.append(np.array(tris))
        if verbose:
            print(f"  Filled boundary loop with {len(loop)} edges.")

    vertices = np.vstack(new_vertices)
    faces = np.vstack(new_faces)
    out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # 统一新面片朝向
    out.fix_normals()
    return out


def symmetry_score(mesh, region_mask, axis_dir, axis_point, n_bins=40):
    """
    基于沿轴线的面积分布形状评价对称性。

    思路：
    - 把区域面片沿轴线投影；
    - 以投影中位数为中心，建立加权直方图；
    - 比较左右两半直方图的相似度（相关系数）。
    - 越接近镜像对称，得分越接近 1。

    优点：不关心重心位置，只关心两侧分布形状是否匹配；
          异常大面片不会整体拉偏结果。
    """
    idx = np.flatnonzero(region_mask)
    if len(idx) < 5:
        return 0.0

    centers = mesh.triangles_center[idx]
    areas = mesh.area_faces[idx]

    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 沿轴线投影，并以中位数为中心
    proj = np.dot(centers - axis_point, axis_dir)
    median = np.median(proj)
    centered = proj - median

    max_abs = np.max(np.abs(centered))
    if max_abs < 1e-6:
        return 1.0

    # 加权直方图
    bin_edges = np.linspace(-max_abs, max_abs, n_bins + 1)
    hist, _ = np.histogram(centered, bins=bin_edges, weights=areas)

    # 分成左右两半，翻转左侧后与右侧比较
    mid = n_bins // 2
    left = hist[:mid][::-1]
    right = hist[mid:2*mid]

    denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
    if denom < 1e-12:
        return 0.0

    corr = np.sum(left * right) / denom
    return float(np.clip(corr, 0.0, 1.0))


# ----------------------------------------------------------------------
#  修订1: rough_axis_from_mask
# ----------------------------------------------------------------------
def rough_axis_from_mask(mesh, mask):
    """从面片掩码快速拟合轴线（普吕克），返回 direction, point。"""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.array([1.0, 0.0, 0.0]), mesh.triangles_center.mean(axis=0)
    valid = np.isfinite(mesh.face_normals[idx]).all(axis=1)
    idx = idx[valid]
    if len(idx) < 3:
        centers = mesh.triangles_center[np.flatnonzero(mask)]
        return np.array([1.0, 0.0, 0.0]), centers.mean(axis=0)
    c = mesh.triangles_center[idx]
    n = mesh.face_normals[idx]
    a = mesh.area_faces[idx]
    C, _, ref = plucker_design_matrix(c, n, a)
    d, b = axis_from_plucker(C, ref_point=ref)
    return d, b

def orient_stem_x(u_x, stem_mask, mesh, origin):
    """Oriente u_x so that majority of stem area lies in +x direction."""
    stem_centers = mesh.triangles_center[stem_mask]
    stem_proj = np.dot(stem_centers - origin, u_x)
    pos_area = np.sum(mesh.area_faces[stem_mask][stem_proj > 0])
    neg_area = np.sum(mesh.area_faces[stem_mask][stem_proj < 0])
    if neg_area > pos_area:
        u_x = -u_x
    return u_x

# ----------------------------------------------------------------------
#  修订2: pass2_t_shape_partition (完整版)
# ----------------------------------------------------------------------
def pass2_t_shape_partition(mesh, mask_bar, mask_stem,
                            init_dir_bar, init_dir_stem,
                            u_x, u_y, u_z, origin,
                            bar_x_size=None,
                            stem_y_size=None,
                            ransac_threshold=0.1):
    """
    第二阶段：全局矩形四区域分区 + 区域内 RANSAC 法向过滤。

    坐标约定：x=把立（+x 指向头管），y=把横，z=x×y。

    矩形分区：
    - 把立 core 候选：x > x_min + d  且  |y| <= 0.5*w
    - 把横 core 候选：x <= x_min + d  且  |y| > 0.5*w
    - 过渡区：        x <= x_min + d  且  |y| <= 0.5*w
    - 残余区：        其余

    然后在把横/把立候选区域内分别做法向过滤：
    - 把横：保留 |n · dir_bar| <= ransac_threshold 的面片
    - 把立：保留 |n · dir_stem| <= ransac_threshold 的面片

    被 RANSAC 过滤掉的面片归入残余区。
    """
    N = mesh.faces.shape[0]
    R = np.column_stack([u_x, u_y, u_z])
    local_coords = (mesh.triangles_center - origin) @ R

    # 有效法向量掩码（排除零法向量/NaN 面片）
    valid_normal = np.isfinite(mesh.face_normals).all(axis=1)

    # 估算 d（把横 x 向尺寸）
    if bar_x_size is not None and bar_x_size > 0:
        d = bar_x_size
    else:
        bar_x = local_coords[mask_bar, 0]
        d = np.percentile(bar_x, 95) - np.percentile(bar_x, 5)
        if not np.isfinite(d) or d <= 0:
            d = 30.0
        d = max(d, 1.0)

    # 估算 w（把立 y 向尺寸）
    if stem_y_size is not None and stem_y_size > 0:
        w = stem_y_size
    else:
        stem_y = local_coords[mask_stem, 1]
        w = np.percentile(stem_y, 95) - np.percentile(stem_y, 5)
        if not np.isfinite(w) or w <= 0:
            w = 40.0
        w = max(w, 1.0)

    # 用面片中心稳健估计 x_min，避免单个极值顶点主导
    local_face_centers = (mesh.triangles_center - origin) @ R
    x_min = np.percentile(local_face_centers[:, 0], 1)

    # 全局矩形分区
    stem_rect = valid_normal & \
                (local_coords[:, 0] > x_min + d) & \
                (np.abs(local_coords[:, 1]) <= 0.5 * w)
    bar_rect = valid_normal & \
               (local_coords[:, 0] <= x_min + d) & \
               (np.abs(local_coords[:, 1]) > 0.5 * w)
    transition_mask = valid_normal & \
                      (local_coords[:, 0] <= x_min + d) & \
                      (np.abs(local_coords[:, 1]) <= 0.5 * w)

    # 区域内 RANSAC 法向过滤
    def filter_by_axis(mask, axis_dir):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return mask
        axis_dir = np.asarray(axis_dir, dtype=np.float64)
        axis_dir = axis_dir / np.linalg.norm(axis_dir)
        dots = np.abs(mesh.face_normals[idx] @ axis_dir)
        keep = dots <= ransac_threshold
        filtered = np.zeros(N, dtype=bool)
        filtered[idx[keep]] = True
        return filtered

    stem_core = filter_by_axis(stem_rect, init_dir_stem)
    bar_core = filter_by_axis(bar_rect, init_dir_bar)

    # 被 RANSAC 过滤掉的面片归入残余区
    filtered_out = (stem_rect & ~stem_core) | (bar_rect & ~bar_core)
    residual_mask = ~(stem_core | bar_core | transition_mask) | filtered_out

    print(f"[Pass2] d (bar x-size) = {d:.3f}")
    print(f"[Pass2] w (stem y-size) = {w:.3f}")
    print(f"[Pass2] x_min = {x_min:.3f}")
    print(f"[Pass2] bar_core = {np.sum(bar_core)}, stem_core = {np.sum(stem_core)}, "
          f"transition = {np.sum(transition_mask)}, residual = {np.sum(residual_mask)}, "
          f"filtered_out = {np.sum(filtered_out)}")

    return bar_core, stem_core, transition_mask, residual_mask, d, w

# ----------------------------------------------------------------------
#  修订3: pass3_refine_by_area (无变化)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
#  新增：三分区识别函数
# ----------------------------------------------------------------------
def identify_three_tubular_regions(mesh, labels_ransac, axes_ransac, areas,
                                   ransac_threshold=0.1):
    """
    从 RANSAC 多区域结果识别把立、左把横、右把横。

    Returns
    -------
    axes : dict
        {'stem': (dir, pt), 'left': (dir, pt), 'right': (dir, pt)}
    mask_stem, mask_left, mask_right : (N,) bool
    """
    unique = np.unique(labels_ransac)
    unique = unique[unique != 0]
    N = mesh.faces.shape[0]

    region_info = []
    for lbl in unique:
        mask = labels_ransac == lbl
        d, p = rough_axis_from_mask(mesh, mask)
        c = mesh.triangles_center[mask]
        area = areas[mask].sum()
        region_info.append({
            'label': lbl,
            'mask': mask,
            'dir': d, 'point': p,
            'area': area,
            'centroid_y': c[:, 1].mean(),
            'y_span': c[:, 1].max() - c[:, 1].min()
        })

    def make_axes(stem, left, right):
        return {
            'stem': (stem['dir'], stem['point']),
            'left': (left['dir'], left['point']),
            'right': (right['dir'], right['point'])
        }

    # ---------- 情况 A：RANSAC 只给出 2 个区域 ----------
    if len(region_info) == 2:
        # y_span 较小的通常是把立
        region_info.sort(key=lambda x: x['y_span'])
        stem_info = region_info[0]
        bar_info = region_info[1]

        bar_mask = bar_info['mask']
        bar_indices = np.flatnonzero(bar_mask)
        y_proj = mesh.triangles_center[bar_mask][:, 1]

        labels_lr, _ = kmeans_1d(y_proj, k=2, rng=np.random.default_rng())
        left_mask = np.zeros(N, dtype=bool)
        right_mask = np.zeros(N, dtype=bool)
        left_mask[bar_indices[labels_lr == 0]] = True
        right_mask[bar_indices[labels_lr == 1]] = True

        # kmeans_1d 标签按中心升序，所以 label=0 对应 y 较小（左）
        # 这里用均值再校验一次
        cy_left = mesh.triangles_center[left_mask][:, 1].mean()
        cy_right = mesh.triangles_center[right_mask][:, 1].mean()
        if cy_left > cy_right:
            left_mask, right_mask = right_mask, left_mask

        d_l, p_l = rough_axis_from_mask(mesh, left_mask)
        d_r, p_r = rough_axis_from_mask(mesh, right_mask)
        return make_axes(stem_info,
                         {'dir': d_l, 'point': p_l, 'mask': left_mask},
                         {'dir': d_r, 'point': p_r, 'mask': right_mask}), \
               stem_info['mask'], left_mask, right_mask

    # ---------- 情况 B：RANSAC 给出 >=3 个区域 ----------
    # 取面积前 3
    region_info.sort(key=lambda x: -x['area'])
    top3 = region_info[:3]

    # 重新用 rough_axis_from_mask 精化
    for info in top3:
        d, p = rough_axis_from_mask(mesh, info['mask'])
        info['dir'] = d
        info['point'] = p
        c = mesh.triangles_center[info['mask']]
        info['centroid_y'] = c[:, 1].mean()

    # 把立：y 中心绝对值最小
    stem_info = min(top3, key=lambda x: abs(x['centroid_y']))
    others = [x for x in top3 if x is not stem_info]

    # 左右按 y 中心排序
    if others[0]['centroid_y'] > others[1]['centroid_y']:
        right_info, left_info = others[0], others[1]
    else:
        left_info, right_info = others[0], others[1]

    # 将未进入 top3 的剩余区域按 y 位置合并到左/右/把立
    if len(region_info) > 3:
        for info in region_info[3:]:
            cy = info['centroid_y']
            if abs(cy) < 0.5 * (abs(left_info['centroid_y']) + abs(right_info['centroid_y'])):
                stem_info['mask'] = stem_info['mask'] | info['mask']
            elif cy > 0:
                right_info['mask'] = right_info['mask'] | info['mask']
            else:
                left_info['mask'] = left_info['mask'] | info['mask']

    return make_axes(stem_info, left_info, right_info), \
           stem_info['mask'], left_info['mask'], right_info['mask']

# ----------------------------------------------------------------------
#  新增：左右把横对称约束
# ----------------------------------------------------------------------
def enforce_bar_symmetry(dir_left, pt_left, dir_right, pt_right,
                         plane_normal=None, plane_offset=0.0):
    """
    强制左右把横关于给定平面对称。
    默认 plane_normal=(0,1,0), plane_offset=0 对应 zx 平面。
    约定：左把横位于平面法向负侧，右把横位于正侧。
    """
    if plane_normal is None:
        plane_normal = np.array([0., 1., 0.])
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    dl = np.asarray(dir_left, dtype=np.float64)
    dr = np.asarray(dir_right, dtype=np.float64)
    pl = np.asarray(pt_left, dtype=np.float64)
    pr = np.asarray(pt_right, dtype=np.float64)

    # 方向符号统一：左负右正
    if np.dot(dl, n) > 0:
        dl = -dl
    if np.dot(dr, n) < 0:
        dr = -dr

    # 投影到对称平面
    dl_n = np.dot(dl, n)
    dr_n = np.dot(dr, n)
    dl_in = dl - dl_n * n
    dr_in = dr - dr_n * n
    d_in = average_antiparallel_directions(dl_in, dr_in)
    d_in = d_in - np.dot(d_in, n) * n
    d_in = d_in / (np.linalg.norm(d_in) + 1e-12)

    y_avg = (abs(dl_n) + abs(dr_n)) / 2.0

    dl_new = d_in - y_avg * n
    dr_new = d_in + y_avg * n
    norm_l = np.linalg.norm(dl_new)
    norm_r = np.linalg.norm(dr_new)
    dl_new = dl_new / (norm_l + 1e-12)
    dr_new = dr_new / (norm_r + 1e-12)

    # 基点对称化
    signed_l = np.dot(pl, n) - plane_offset
    signed_r = np.dot(pr, n) - plane_offset
    pl_in = pl - signed_l * n
    pr_in = pr - signed_r * n
    p_in = (pl_in + pr_in) / 2.0
    half_y = (abs(signed_l) + abs(signed_r)) / 2.0

    pl_new = p_in - half_y * n
    pr_new = p_in + half_y * n

    return dl_new, pl_new, dr_new, pr_new


def _mirror_axis_pair(axis_pair, plane_normal, plane_offset):
    """将一个轴线的方向和基点关于平面对称镜像。"""
    d, p = axis_pair
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    d_m = d - 2.0 * np.dot(d, n) * n
    p_m = p - 2.0 * (np.dot(p, n) - plane_offset) * n
    return d_m, p_m


def _mirror_side_result(result, plane_normal, plane_offset, N):
    """根据一侧结果镜像生成另一侧的占位结果（mask 为空）。"""
    mirrored = {}
    for key in ['stem', 'bar']:
        if result[key] is not None:
            mirrored[key] = _mirror_axis_pair(result[key], plane_normal, plane_offset)
        else:
            mirrored[key] = None
        mask_key = 'stem_mask' if key == 'stem' else 'bar_mask'
        if result[mask_key] is not None:
            mirrored[mask_key] = np.zeros(N, dtype=bool)
        else:
            mirrored[mask_key] = None
    return mirrored


def identify_three_tubular_regions_symmetric(mesh, u_y_sym, offset_y,
                                             ransac_threshold=0.1,
                                             min_faces_ratio=0.05,
                                             tol_percentile=10.0,
                                             rng=None):
    """
    基于已估计的镜面对称平面，对左右子空间分别执行 RANSAC 管状分割，
    再按对称约束合并为 {stem, left, right} 三分区。
    """
    N = mesh.faces.shape[0]
    centers = mesh.triangles_center
    n = np.asarray(u_y_sym, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    signed = centers @ n - offset_y
    abs_signed = np.abs(signed)

    tol = np.percentile(abs_signed, tol_percentile) if tol_percentile > 0 else 0.0
    tol = max(tol, 1e-6)

    left_mask = signed < -tol
    right_mask = signed > tol
    middle_mask = ~left_mask & ~right_mask

    print(f"\n[SymRANSAC] tol={tol:.4f}, left={np.sum(left_mask)}, "
          f"right={np.sum(right_mask)}, middle={np.sum(middle_mask)}")

    if rng is None:
        rng = np.random.default_rng()

    def process_side(side_mask, side_name):
        n_faces = np.sum(side_mask)
        if n_faces < 10:
            print(f"[SymRANSAC] {side_name}: too few faces ({n_faces})")
            return None
        face_indices = np.where(side_mask)[0]
        sub = mesh.submesh(face_indices)[0]
        labels, axes = segment_tubular_regions(
            sub.face_normals, areas=sub.area_faces,
            threshold=ransac_threshold,
            min_faces=max(int(min_faces_ratio * n_faces), 10),
            max_regions=2, max_iterations=1000, rng=rng
        )
        n_regions = len(axes)
        if n_regions == 0:
            print(f"[SymRANSAC] {side_name}: no tubular region found")
            return None

        side_indices = face_indices
        regions = []
        for i in range(1, n_regions + 1):
            sub_idx = np.where(labels == i)[0]
            orig_idx = side_indices[sub_idx]
            mask = np.zeros(N, dtype=bool)
            mask[orig_idx] = True
            d, p = rough_axis_from_mask(mesh, mask)
            regions.append({
                'mask': mask,
                'axis': (d, p),
                'abs_dot_y': abs(np.dot(d, n)),
            })

        if n_regions == 1:
            r = regions[0]
            if r['abs_dot_y'] < 0.5:
                return {'stem': r['axis'], 'bar': None,
                        'stem_mask': r['mask'], 'bar_mask': None}
            else:
                return {'stem': None, 'bar': r['axis'],
                        'stem_mask': None, 'bar_mask': r['mask']}
        else:
            stem_idx = int(np.argmin([r['abs_dot_y'] for r in regions]))
            bar_idx = 1 - stem_idx
            return {
                'stem': regions[stem_idx]['axis'],
                'bar': regions[bar_idx]['axis'],
                'stem_mask': regions[stem_idx]['mask'],
                'bar_mask': regions[bar_idx]['mask'],
            }

    left_res = process_side(left_mask, 'left')
    right_res = process_side(right_mask, 'right')

    if left_res is None and right_res is None:
        raise RuntimeError("左右两侧 RANSAC 均失败")
    if left_res is None:
        left_res = _mirror_side_result(right_res, n, offset_y, N)
    if right_res is None:
        right_res = _mirror_side_result(left_res, n, offset_y, N)

    # 补齐缺失的 stem/bar
    if left_res['stem'] is None:
        left_res['stem'] = right_res['stem']
        left_res['stem_mask'] = right_res['stem_mask'].copy()
    if right_res['stem'] is None:
        right_res['stem'] = left_res['stem']
        right_res['stem_mask'] = left_res['stem_mask'].copy()
    if left_res['bar'] is None:
        left_res['bar'] = _mirror_axis_pair(right_res['bar'], n, offset_y)
        left_res['bar_mask'] = np.zeros(N, dtype=bool)
    if right_res['bar'] is None:
        right_res['bar'] = _mirror_axis_pair(left_res['bar'], n, offset_y)
        right_res['bar_mask'] = np.zeros(N, dtype=bool)

    # 合并把立轴线：投影到对称平面后平均
    d_stem_L, p_stem_L = left_res['stem']
    d_stem_R, p_stem_R = right_res['stem']
    d_stem = average_antiparallel_directions(d_stem_L, d_stem_R)
    d_stem = d_stem - np.dot(d_stem, n) * n
    norm = np.linalg.norm(d_stem)
    if norm < 1e-12:
        d_stem = np.array([1.0, 0.0, 0.0])
        d_stem = d_stem - np.dot(d_stem, n) * n
        d_stem = d_stem / (np.linalg.norm(d_stem) + 1e-12)
    else:
        d_stem = d_stem / norm

    def proj_to_plane(p):
        return p - (np.dot(p, n) - offset_y) * n

    p_stem = (proj_to_plane(p_stem_L) + proj_to_plane(p_stem_R)) / 2.0

    # 合并把横轴线：强制关于对称平面对称
    d_bar_L, p_bar_L = left_res['bar']
    d_bar_R, p_bar_R = right_res['bar']
    d_bar_L, p_bar_L, d_bar_R, p_bar_R = enforce_bar_symmetry(
        d_bar_L, p_bar_L, d_bar_R, p_bar_R,
        plane_normal=n, plane_offset=offset_y
    )

    axes = {
        'stem': (d_stem, p_stem),
        'left': (d_bar_L, p_bar_L),
        'right': (d_bar_R, p_bar_R),
    }
    mask_stem = left_res['stem_mask'] | right_res['stem_mask'] | middle_mask
    mask_left = left_res['bar_mask']
    mask_right = right_res['bar_mask']

    print(f"[SymRANSAC] merged: stem={np.sum(mask_stem)}, "
          f"left={np.sum(mask_left)}, right={np.sum(mask_right)}")
    return axes, mask_stem, mask_left, mask_right


# ----------------------------------------------------------------------
#  新增：三分区第二阶段约束
# ----------------------------------------------------------------------
def pass2_aero_shape_partition(mesh,
                               mask_stem, mask_left, mask_right,
                               init_dir_stem, init_pt_stem,
                               init_dir_left, init_pt_left,
                               init_dir_right, init_pt_right,
                               u_x, u_y, u_z, origin,
                               bar_x_size=None,
                               stem_y_size=None,
                               ransac_threshold=0.1):
    """
    三分区版 T 形约束分区。
    坐标约定：x=把立，y 正方向=左→右，z=x×y。
    """
    N = mesh.faces.shape[0]
    R = np.column_stack([u_x, u_y, u_z])
    local = (mesh.triangles_center - origin) @ R
    valid_normal = np.isfinite(mesh.face_normals).all(axis=1)

    # 估算 d（把横 x 向尺寸）和 w（把立 y 向尺寸）
    if bar_x_size is not None and bar_x_size > 0:
        d = bar_x_size
    else:
        bar_x = local[(mask_left | mask_right), 0]
        d = np.percentile(bar_x, 95) - np.percentile(bar_x, 5)
        if not np.isfinite(d) or d <= 0:
            d = 30.0
        d = max(d, 1.0)

    if stem_y_size is not None and stem_y_size > 0:
        w = stem_y_size
    else:
        stem_y = local[mask_stem, 1]
        w = np.percentile(stem_y, 95) - np.percentile(stem_y, 5)
        if not np.isfinite(w) or w <= 0:
            w = 40.0
        w = max(w, 1.0)

    x_min = np.percentile(local[:, 0], 1)

    # 空间距离辅助：面片到各轴线的距离
    def dist_to_axis_mask(base_mask, axis_dir, axis_pt, max_dist):
        idx = np.flatnonzero(base_mask & valid_normal)
        if len(idx) == 0:
            return np.zeros(N, dtype=bool)
        dists = point_line_distance(mesh.triangles_center[idx], axis_pt, axis_dir)
        keep = dists <= max_dist
        out = np.zeros(N, dtype=bool)
        out[idx[keep]] = True
        return out

    max_dist = max(d, w) * 1.5

    # 矩形候选区
    stem_rect = valid_normal & (local[:, 0] > x_min + d) & (np.abs(local[:, 1]) <= 0.5 * w)
    left_rect = valid_normal & (local[:, 1] < -0.5 * w) & (local[:, 0] <= x_min + d)
    right_rect = valid_normal & (local[:, 1] > 0.5 * w) & (local[:, 0] <= x_min + d)

    # 结合到各自轴线的距离
    stem_rect = dist_to_axis_mask(stem_rect, init_dir_stem, init_pt_stem, max_dist)
    left_rect = dist_to_axis_mask(left_rect, init_dir_left, init_pt_left, max_dist)
    right_rect = dist_to_axis_mask(right_rect, init_dir_right, init_pt_right, max_dist)

    # 法向过滤：保留法向近似垂直于轴线的面片
    def filter_by_axis(mask, axis_dir):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return mask
        a = np.asarray(axis_dir, dtype=np.float64)
        a = a / (np.linalg.norm(a) + 1e-12)
        dots = np.abs(mesh.face_normals[idx] @ a)
        keep = dots <= ransac_threshold
        out = np.zeros(N, dtype=bool)
        out[idx[keep]] = True
        return out

    stem_core = filter_by_axis(stem_rect, init_dir_stem)
    left_core = filter_by_axis(left_rect, init_dir_left)
    right_core = filter_by_axis(right_rect, init_dir_right)

    filtered_out = (stem_rect & ~stem_core) | (left_rect & ~left_core) | (right_rect & ~right_core)
    transition_mask = valid_normal & ~(stem_core | left_core | right_core | filtered_out)
    residual_mask = ~valid_normal | filtered_out

    print(f"[Pass2 Aero] d={d:.3f}, w={w:.3f}, x_min={x_min:.3f}")
    print(f"[Pass2 Aero] left={np.sum(left_core)}, right={np.sum(right_core)}, "
          f"stem={np.sum(stem_core)}, trans={np.sum(transition_mask)}, "
          f"residual={np.sum(residual_mask)}")

    return left_core, right_core, stem_core, transition_mask, residual_mask, d, w

# ----------------------------------------------------------------------
#  _axes_are_close (支持任意键集合)
# ----------------------------------------------------------------------
def _axes_are_close(ax1, ax2, angle_tol=0.99, dist_tol=1e-3):
    """判断两组轴线（键集合相同）是否在方向和位置上足够接近。"""
    if set(ax1.keys()) != set(ax2.keys()):
        return False
    for k in ax1.keys():
        d1, p1 = ax1[k]
        d2, p2 = ax2[k]
        n1 = d1 / (np.linalg.norm(d1) + 1e-12)
        n2 = d2 / (np.linalg.norm(d2) + 1e-12)
        if abs(np.dot(n1, n2)) < angle_tol or np.linalg.norm(p1 - p2) >= dist_tol:
            return False
    return True

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

# ----------------------------------------------------------------------
#  替换 build_world_visualization
# ----------------------------------------------------------------------
def build_world_visualization(mesh, final_labels,
                              axes_pass1, axes_pass2, axes_pass3,
                              max_ext, num_passes=3):
    """
    世界坐标系可视化。
    支持双区域（bar/stem）和三分区气动（left/right/stem）模式。
    """
    from toys3d.geometrics import intersect_line_plane

    scene = trimesh.Scene()

    vis_mesh = mesh.copy()
    colorize_mesh(vis_mesh, final_labels)
    scene.add_geometry(vis_mesh)

    L_axis = max_ext * 1.5
    frame_len = max_ext * 0.6

    is_aero = 'left' in axes_pass3
    if is_aero:
        colors1 = {'left': [0, 100, 200, 120], 'right': [0, 180, 255, 120],
                   'stem': [200, 50, 50, 120]}
        colors2 = {'left': [0, 150, 255, 180], 'right': [0, 200, 255, 180],
                   'stem': [255, 80, 80, 180]}
        colors3 = {'left': [0, 100, 200, 255], 'right': [0, 180, 255, 255],
                   'stem': [200, 50, 50, 255]}
    else:
        colors1 = {'bar': [0, 100, 200, 120], 'stem': [200, 50, 50, 120]}
        colors2 = {'bar': [0, 150, 255, 180], 'stem': [255, 80, 80, 180]}
        colors3 = {'bar': [0, 100, 200, 255], 'stem': [200, 50, 50, 255]}

    def add_axis(axis_pair, color, radius_factor, dashed):
        d, p = axis_pair
        p1 = p - L_axis * d
        p2 = p + L_axis * d
        if dashed:
            _add_dashed_line(scene, p1, p2, color, radius_factor * max_ext)
        else:
            cyl = trimesh.creation.cylinder(radius_factor * max_ext, segment=[p1, p2])
            cyl.visual.face_colors = color
            scene.add_geometry(cyl)

    # Pass 1
    for k, c in colors1.items():
        if k in axes_pass1:
            add_axis(axes_pass1[k], c, 0.004, dashed=True)

    # Pass 2
    if num_passes >= 2 and not _axes_are_close(axes_pass1, axes_pass2):
        for k, c in colors2.items():
            if k in axes_pass2:
                add_axis(axes_pass2[k], c, 0.006, dashed=True)

    # Pass 3
    if num_passes >= 3 and not _axes_are_close(axes_pass2, axes_pass3):
        for k, c in colors3.items():
            if k in axes_pass3:
                add_axis(axes_pass3[k], c, 0.008, dashed=False)

    # 最终坐标架
    dir_stem, point_stem = axes_pass3['stem']
    u_x = dir_stem / (np.linalg.norm(dir_stem) + 1e-12)

    if is_aero:
        dir_left, point_left = axes_pass3['left']
        dir_right, point_right = axes_pass3['right']
        u_y_raw = dir_right - dir_left
        u_y = u_y_raw - np.dot(u_y_raw, u_x) * u_x
        u_y = u_y / (np.linalg.norm(u_y) + 1e-12)
        u_z = np.cross(u_x, u_y)
        mid_lr = (point_left + point_right) / 2.0
        origin = intersect_line_plane(dir_stem, point_stem, mid_lr, u_y)
        if origin is None:
            origin = mid_lr
    else:
        dir_bar, point_bar = axes_pass3['bar']
        p_close_bar, p_close_stem, midpoint = closest_points_on_lines(
            dir_bar, point_bar, dir_stem, point_stem
        )
        origin = midpoint
        y_raw = dir_bar - np.dot(dir_bar, u_x) * u_x
        u_y = y_raw / (np.linalg.norm(y_raw) + 1e-12)
        u_z = np.cross(u_x, u_y)

        cyl_perp = trimesh.creation.cylinder(0.008 * max_ext, segment=[p_close_bar, p_close_stem])
        cyl_perp.visual.face_colors = [0, 180, 0, 255]
        scene.add_geometry(cyl_perp)

    sph_mid = trimesh.creation.icosphere(subdivisions=2, radius=0.015 * max_ext)
    sph_mid.apply_translation(origin)
    sph_mid.visual.face_colors = [255, 255, 0, 255]
    scene.add_geometry(sph_mid)

    add_axes_to_scene(scene, origin=origin, u_x=u_x, u_y=u_y, u_z=u_z, length=frame_len)

    return scene

# ------------------------- 主流程函数 -------------------------

def process_handlebar(mesh,
                      ransac_threshold=0.1,
                      region_label_bar=None,
                      region_label_stem=None,
                      num_passes=3,
                      bar_x_size=None,
                      stem_y_size=None,
                      aero_mode=False,
                      repair_mode=False):
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
    num_passes : {0, 1, 2, 3}
        分区阶段数：
        0 = 仅检测网格缺陷
        1 = RANSAC 初始分区
        2 = +T 字形矩形约束
        3 = + 截面积精化
    bar_x_size : float or None
        把横沿 x 方向尺寸（overwrite自动估算）
    stem_y_size : float or None
        把立沿 y 方向尺寸（overwrite自动估算）
    aero_mode : bool
        启用气动把三分区模式（把立 + 左把横 + 右把横）。
    repair_mode : bool
        启用网格修复模式（去除重复/退化面）。

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

    # 网格缺陷分析
    defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    print("\nMesh defect analysis:")
    print(f"  total unique edges : {defect_stats['unique_edges_count']}")
    print(f"  raw edges count    : {defect_stats['raw_edges_count']}")
    print(f"  open edges         : {defect_stats['open_edges']}")
    print(f"  manifold edges     : {defect_stats['manifold_edges']}")
    print(f"  nonmanifold edges  : {defect_stats['nonmanifold_edges']}")
    print(f"  open faces         : {defect_stats['open_faces']}")
    print(f"  nonmanifold faces  : {defect_stats['nonmanifold_faces']}")
    print(f"  both-defect faces  : {defect_stats['both_defect_faces']}")
    print(f"  watertight (no open edges): {defect_stats['watertight_by_count']}")

    # 修复（如果需要）
    if repair_mode and (defect_stats['open_edges'] > 0 or defect_stats['nonmanifold_edges'] > 0):
        print("\n[Repair mode] Attempting to fix mesh...")

        # 策略1：去重复/退化面
        mesh = repair_mesh_by_removing_duplicates(mesh)

        # 策略2：消除非流形边
        mesh = repair_nonmanifold_edges(mesh)

        # 补洞
        mesh = fill_small_holes(mesh)

        # 修复后重新统计
        stats = compute_mesh_stats(mesh)
        print("After repair:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
        print(f"  total unique edges : {defect_stats['unique_edges_count']}")
        print(f"  raw edges count    : {defect_stats['raw_edges_count']}")
        print(f"  open edges         : {defect_stats['open_edges']}")
        print(f"  manifold edges     : {defect_stats['manifold_edges']}")
        print(f"  nonmanifold edges  : {defect_stats['nonmanifold_edges']}")
        print(f"  open faces         : {defect_stats['open_faces']}")
        print(f"  nonmanifold faces  : {defect_stats['nonmanifold_faces']}")
        print(f"  both-defect faces  : {defect_stats['both_defect_faces']}")
        print(f"  watertight (no open edges): {defect_stats['watertight_by_count']}")

    # 对称性估计（修复后，此时网格应水密）
    if mesh.is_watertight:
        print("\n[Symmetry] Estimating symmetry plane from watertight voxelization...")
        u_y_sym, offset_y, sym_score = estimate_symmetry_plane_voxel(
            mesh, grid_size=96, metric='gradient'
        )
        print(f"  symmetry_score = {sym_score:.6f}  (closer to 0 means more symmetric)")
        print(f"  estimated y    = {u_y_sym.round(4)}")
        print(f"  plane offset   = {offset_y:.4f}")
    else:
        u_y_sym = None
        offset_y = 0.0
        print("\n[Symmetry] Mesh is not watertight; skipping voxel-based symmetry estimation.")

    # 仅检测模式
    if num_passes == 0:
        print("\n[--num-passes 0] Only performing mesh defect inspection.")

        if u_y_sym is not None:
            print("[Symmetry] Displaying symmetry analysis visualization.")
            world_scene = build_symmetry_visualization(mesh, u_y_sym, offset_y)
        else:
            world_scene = build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask)

        mesh_copy = mesh.copy()
        if u_y_sym is not None:
            colorize_by_symmetry_plane(mesh_copy, u_y_sym, offset_y)
        else:
            colorize_defects(mesh_copy, open_face_mask, nonmanifold_face_mask)
        scene = trimesh.Scene(mesh_copy)
        add_axes_to_scene(scene, origin=mesh.bounding_box.centroid,
                          u_x=np.array([1,0,0]), u_y=np.array([0,1,0]), u_z=np.array([0,0,1]),
                          length=mesh.bounding_box.extents.max()*0.5)

        print("\nColor map:")
        if u_y_sym is not None:
            print("  0 = Left side   -> dark blue")
            print("  1 = Right side  -> cyan")
            print("  2 = Neutral     -> gray")
            print("  Mirror plane    -> translucent yellow")
        else:
            print("  0 = Normal face            -> gray")
            print("  1 = Open-edge face          -> red")
            print("  2 = Nonmanifold-edge face   -> yellow")
            print("  3 = Both defects            -> purple")
        return scene, world_scene, stats

    # 2. 准备数据
    areas = mesh.area_faces
    N = mesh.faces.shape[0]

    # 3. RANSAC 管状分割
    labels_ransac, axes_ransac = segment_tubular_regions(
        mesh.face_normals, areas=areas, threshold=ransac_threshold,
        min_faces=50, max_regions=5, max_iterations=2000
    )
    unique_labels = np.unique(labels_ransac)
    print(f"Hey, RANSAC found regions: {unique_labels}, axes count: {len(axes_ransac)}")

    if aero_mode:
        if len(axes_ransac) < 2:
            raise RuntimeError("气动把模式至少需要 2 个管状区域。")

        # 对称平面参数（未估计时退化为世界 y=0 平面）
        sym_n = u_y_sym if u_y_sym is not None else np.array([0., 1., 0.])
        sym_offset = offset_y if u_y_sym is not None else 0.0

        if mesh.is_watertight and u_y_sym is not None:
            print("\n[SymRANSAC] Using symmetry-aware three-region identification.")
            axes_init, mask_stem, mask_left, mask_right = identify_three_tubular_regions_symmetric(
                mesh, u_y_sym, offset_y, ransac_threshold=ransac_threshold
            )
        else:
            print("\n[SymRANSAC] Falling back to global RANSAC three-region identification.")
            axes_init, mask_stem, mask_left, mask_right = identify_three_tubular_regions(
                mesh, labels_ransac, axes_ransac, areas, ransac_threshold
            )
    else:
        if len(axes_ransac) < 2:
            raise RuntimeError("未能检测到两个主要管状区域，请检查阈值或模型方向。")

    # =================================================================
    #  双区域模式（原有逻辑）
    # =================================================================
    if not aero_mode:
        # 基于对称性识别把横/把立
        area_sums = []
        for lbl in range(1, len(axes_ransac)+1):
            area_sums.append((areas[labels_ransac == lbl].sum(), lbl))
        area_sums.sort(reverse=True)

        if region_label_bar is None:
            dir1, pt1 = rough_axis_from_mask(mesh, labels_ransac == area_sums[0][1])
            dir2, pt2 = rough_axis_from_mask(mesh, labels_ransac == area_sums[1][1])

            sym1 = symmetry_score(mesh, labels_ransac == area_sums[0][1], dir1, pt1)
            sym2 = symmetry_score(mesh, labels_ransac == area_sums[1][1], dir2, pt2)
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

        mask_bar = labels_ransac == label_bar
        mask_stem = labels_ransac == label_stem

        init_dir_bar, init_pt_bar = rough_axis_from_mask(mesh, mask_bar)
        init_dir_stem, init_pt_stem = rough_axis_from_mask(mesh, mask_stem)

        axes_pass1 = {'bar': (init_dir_bar, init_pt_bar),
                      'stem': (init_dir_stem, init_pt_stem)}

        if num_passes == 1:
            bar_core = mask_bar
            stem_core = mask_stem
            transition_mask = np.zeros(N, dtype=bool)
            dir_bar, point_bar = init_dir_bar, init_pt_bar
            dir_stem, point_stem = init_dir_stem, init_pt_stem
            axes_pass2 = axes_pass1
            axes_pass3 = axes_pass1
        else:
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
                mesh, mask_bar, mask_stem,
                init_dir_bar, init_dir_stem,
                u_x, u_y, u_z, origin,
                bar_x_size=bar_x_size,
                stem_y_size=stem_y_size,
                ransac_threshold=ransac_threshold
            )

            MIN_CORE_FACES = 10
            if np.sum(bar_core) >= MIN_CORE_FACES:
                dir_bar_p2, point_bar_p2 = rough_axis_from_mask(mesh, bar_core)
            else:
                dir_bar_p2, point_bar_p2 = init_dir_bar, init_pt_bar

            if np.sum(stem_core) >= MIN_CORE_FACES:
                dir_stem_p2, point_stem_p2 = rough_axis_from_mask(mesh, stem_core)
            else:
                dir_stem_p2, point_stem_p2 = init_dir_stem, init_pt_stem

            axes_pass2 = {'bar': (dir_bar_p2, point_bar_p2),
                          'stem': (dir_stem_p2, point_stem_p2)}

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

                if np.sum(bar_core) >= 10:
                    dir_bar, point_bar = rough_axis_from_mask(mesh, bar_core)
                if np.sum(stem_core) >= 10:
                    dir_stem, point_stem = rough_axis_from_mask(mesh, stem_core)

                axes_pass3 = {'bar': (dir_bar, point_bar),
                              'stem': (dir_stem, point_stem)}

        conf_bar = np.sum(areas[bar_core]) if np.sum(bar_core) > 0 else 0.0
        conf_stem = np.sum(areas[stem_core]) if np.sum(stem_core) > 0 else 0.0

        final_labels = np.zeros(N, dtype=int)
        final_labels[bar_core] = 1
        final_labels[stem_core] = 2
        final_labels[transition_mask] = 3

        # 世界坐标系可视化
        max_ext = mesh.bounding_box.extents.max()
        world_scene = build_world_visualization(mesh, final_labels,
                                                axes_pass1, axes_pass2, axes_pass3,
                                                max_ext, num_passes=num_passes)

        # 正交坐标系构建
        T_w2l, T_l2w, u_x, u_y, u_z, origin = orthogonalize_axes(
            dir_x=dir_stem, point_x=point_stem, weight_x=conf_stem,
            dir_y=dir_bar, point_y=point_bar, weight_y=conf_bar
        )

    # =================================================================
    #  气动把三分区模式
    # =================================================================
    else:
        init_dir_stem, init_pt_stem = axes_init['stem']
        init_dir_left, init_pt_left = axes_init['left']
        init_dir_right, init_pt_right = axes_init['right']

        init_dir_left, init_pt_left, init_dir_right, init_pt_right = enforce_bar_symmetry(
            init_dir_left, init_pt_left, init_dir_right, init_pt_right,
            plane_normal=sym_n, plane_offset=sym_offset
        )

        axes_pass1 = {
            'stem': (init_dir_stem, init_pt_stem),
            'left': (init_dir_left, init_pt_left),
            'right': (init_dir_right, init_pt_right)
        }

        if num_passes == 1:
            left_core = mask_left
            right_core = mask_right
            stem_core = mask_stem
            transition_mask = np.zeros(N, dtype=bool)
            dir_stem, point_stem = init_dir_stem, init_pt_stem
            dir_left, point_left = init_dir_left, init_pt_left
            dir_right, point_right = init_dir_right, init_pt_right
            axes_pass2 = axes_pass1
            axes_pass3 = axes_pass1
        else:
            u_x = init_dir_stem / np.linalg.norm(init_dir_stem)
            u_y_raw = init_dir_right - init_dir_left
            u_y = u_y_raw - np.dot(u_y_raw, u_x) * u_x
            if np.linalg.norm(u_y) < 1e-6:
                u_y = np.array([0, 1, 0]) if abs(u_x[1]) < 0.9 else np.array([1, 0, 0])
                u_y = u_y - np.dot(u_y, u_x) * u_x
            u_y = u_y / np.linalg.norm(u_y)
            u_z = np.cross(u_x, u_y)

            mid_lr = (init_pt_left + init_pt_right) / 2.0
            origin = intersect_line_plane(init_dir_stem, init_pt_stem, mid_lr, u_y)
            if origin is None:
                origin = mid_lr

            u_x = orient_stem_x(u_x, mask_stem, mesh, origin)
            u_z = np.cross(u_x, u_y)

            left_core, right_core, stem_core, transition_mask, residual_mask, d_est, w_est = pass2_aero_shape_partition(
                mesh, mask_stem, mask_left, mask_right,
                init_dir_stem, init_pt_stem,
                init_dir_left, init_pt_left,
                init_dir_right, init_pt_right,
                u_x, u_y, u_z, origin,
                bar_x_size=bar_x_size,
                stem_y_size=stem_y_size,
                ransac_threshold=ransac_threshold
            )

            MIN_CORE_FACES = 10
            if np.sum(left_core) >= MIN_CORE_FACES:
                dir_left_p2, point_left_p2 = rough_axis_from_mask(mesh, left_core)
            else:
                dir_left_p2, point_left_p2 = init_dir_left, init_pt_left

            if np.sum(right_core) >= MIN_CORE_FACES:
                dir_right_p2, point_right_p2 = rough_axis_from_mask(mesh, right_core)
            else:
                dir_right_p2, point_right_p2 = init_dir_right, init_pt_right

            if np.sum(stem_core) >= MIN_CORE_FACES:
                dir_stem_p2, point_stem_p2 = rough_axis_from_mask(mesh, stem_core)
            else:
                dir_stem_p2, point_stem_p2 = init_dir_stem, init_pt_stem

            dir_left_p2, point_left_p2, dir_right_p2, point_right_p2 = enforce_bar_symmetry(
                dir_left_p2, point_left_p2, dir_right_p2, point_right_p2,
                plane_normal=sym_n, plane_offset=sym_offset
            )

            axes_pass2 = {
                'stem': (dir_stem_p2, point_stem_p2),
                'left': (dir_left_p2, point_left_p2),
                'right': (dir_right_p2, point_right_p2)
            }

            # 更新坐标系
            u_x = dir_stem_p2 / np.linalg.norm(dir_stem_p2)
            u_y_raw = dir_right_p2 - dir_left_p2
            u_y = u_y_raw - np.dot(u_y_raw, u_x) * u_x
            u_y = u_y / np.linalg.norm(u_y)
            u_z = np.cross(u_x, u_y)

            mid_lr = (point_left_p2 + point_right_p2) / 2.0
            origin = intersect_line_plane(dir_stem_p2, point_stem_p2, mid_lr, u_y)
            if origin is None:
                origin = mid_lr

            u_x = orient_stem_x(u_x, stem_core, mesh, origin)
            u_z = np.cross(u_x, u_y)

            if num_passes == 2:
                dir_stem, point_stem = dir_stem_p2, point_stem_p2
                dir_left, point_left = dir_left_p2, point_left_p2
                dir_right, point_right = dir_right_p2, point_right_p2
                axes_pass3 = axes_pass2
            else:
                if np.sum(left_core) >= 10:
                    dir_left, point_left = rough_axis_from_mask(mesh, left_core)
                    left_core, left_trans = pass3_refine_by_area(
                        mesh, left_core, dir_left, point_left, d_est
                    )
                else:
                    left_trans = np.zeros(N, dtype=bool)
                    dir_left, point_left = dir_left_p2, point_left_p2

                if np.sum(right_core) >= 10:
                    dir_right, point_right = rough_axis_from_mask(mesh, right_core)
                    right_core, right_trans = pass3_refine_by_area(
                        mesh, right_core, dir_right, point_right, d_est
                    )
                else:
                    right_trans = np.zeros(N, dtype=bool)
                    dir_right, point_right = dir_right_p2, point_right_p2

                if np.sum(stem_core) >= 10:
                    dir_stem, point_stem = rough_axis_from_mask(mesh, stem_core)
                    stem_core, stem_trans = pass3_refine_by_area(
                        mesh, stem_core, dir_stem, point_stem, w_est
                    )
                else:
                    stem_trans = np.zeros(N, dtype=bool)
                    dir_stem, point_stem = dir_stem_p2, point_stem_p2

                transition_mask = transition_mask | left_trans | right_trans | stem_trans

                if np.sum(left_core) >= 10:
                    dir_left, point_left = rough_axis_from_mask(mesh, left_core)
                if np.sum(right_core) >= 10:
                    dir_right, point_right = rough_axis_from_mask(mesh, right_core)
                if np.sum(stem_core) >= 10:
                    dir_stem, point_stem = rough_axis_from_mask(mesh, stem_core)

                dir_left, point_left, dir_right, point_right = enforce_bar_symmetry(
                    dir_left, point_left, dir_right, point_right,
                    plane_normal=sym_n, plane_offset=sym_offset
                )

                axes_pass3 = {
                    'stem': (dir_stem, point_stem),
                    'left': (dir_left, point_left),
                    'right': (dir_right, point_right)
                }

        conf_left = np.sum(areas[left_core]) if np.sum(left_core) > 0 else 0.0
        conf_right = np.sum(areas[right_core]) if np.sum(right_core) > 0 else 0.0
        conf_stem = np.sum(areas[stem_core]) if np.sum(stem_core) > 0 else 0.0

        final_labels = np.zeros(N, dtype=int)
        final_labels[left_core] = 1
        final_labels[right_core] = 2
        final_labels[stem_core] = 3
        final_labels[transition_mask] = 4

        # 世界坐标系可视化
        max_ext = mesh.bounding_box.extents.max()
        world_scene = build_world_visualization(mesh, final_labels,
                                                axes_pass1, axes_pass2, axes_pass3,
                                                max_ext, num_passes=num_passes)

        # 正交坐标系构建：x=把立，y=公共把横方向，z=x×y
        u_x = dir_stem / (np.linalg.norm(dir_stem) + 1e-12)
        u_y_raw = dir_right - dir_left
        u_y = u_y_raw - np.dot(u_y_raw, u_x) * u_x
        u_y = u_y / (np.linalg.norm(u_y) + 1e-12)
        u_z = np.cross(u_x, u_y)

        mid_lr = (point_left + point_right) / 2.0
        origin = intersect_line_plane(dir_stem, point_stem, mid_lr, u_y)
        if origin is None:
            origin = mid_lr

        u_x = orient_stem_x(u_x, stem_core, mesh, origin)
        u_z = np.cross(u_x, u_y)

        R = np.column_stack([u_x, u_y, u_z])
        T_w2l = np.eye(4)
        T_w2l[:3, :3] = R.T
        T_w2l[:3, 3] = -R.T @ origin

    # 10. 变换网格到新坐标系
    mesh_transformed = mesh.copy()
    mesh_transformed.apply_transform(T_w2l)

    # 9. 着色
    colorize_mesh(mesh_transformed, final_labels)

    # 10. 场景构建
    scene = trimesh.Scene(mesh_transformed)
    add_axes_to_scene(scene, origin=np.zeros(3), u_x=np.array([1,0,0]),
                      u_y=np.array([0,1,0]), u_z=np.array([0,0,1]))

    # 统计输出
    print("\nRegion statistics (color map):")
    if aero_mode:
        print("  0 = Residual     -> gray")
        print("  1 = Left bar     -> dark blue")
        print("  2 = Right bar    -> cyan")
        print("  3 = Stem core    -> dark red")
        print("  4 = Transition   -> orange")
        names = ['Residual', 'Left bar', 'Right bar', 'Stem core', 'Transition']
    else:
        print("  0 = Residual     -> gray")
        print("  1 = Bar core     -> dark blue")
        print("  2 = Stem core    -> dark red")
        print("  3 = Transition   -> orange")
        names = ['Residual', 'Bar core', 'Stem core', 'Transition']
    for i, name in enumerate(names):
        count = np.sum(final_labels == i)
        print(f"  {name}: {count} triangles")

    return scene, world_scene, stats


def colorize_mesh(mesh, final_labels):
    """
    根据最终标签设置面片颜色。
    双区域模式：0=灰, 1=深蓝（把横）, 2=深红（把立）, 3=橙（过渡）
    三分区气动模式：0=灰, 1=深蓝（左把横）, 2=青色（右把横）, 3=深红（把立）, 4=橙（过渡）
    """
    palette = np.array([
        [180, 180, 180, 255],   # 0 residual
        [0,   100, 200, 255],   # 1 left / bar
        [0,   180, 255, 255],   # 2 right
        [200,  50,  50, 255],   # 3 stem
        [255, 165,   0, 255],   # 4 transition
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
#  辅助函数（已提前定义，此处无额外内容）
# ----------------------------------------------------------------------


def main():
    import argparse
    parser = argparse.ArgumentParser(description="处理公路车把网格，分割把横/把立，建立正交坐标系。")
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("--output", help="保存处理后的网格路径 (可选)")
    parser.add_argument("--ransac_thr", type=float, default=0.1, help="RANSAC 阈值 (默认 0.1)")
    parser.add_argument("--num-passes", type=int, default=3, choices=[0, 1, 2, 3],
                        help="分区阶段数：0=仅检测网格缺陷，1=RANSAC初始分区，2=+T字形约束分区，3=+截面积精化（默认3）")
    parser.add_argument("--bar-x-size", type=float, default=None,
                        help="把横沿 x 方向尺寸 d，用于确定把立过渡区范围（默认自动估算）")
    parser.add_argument("--stem-y-size", type=float, default=None,
                        help="把立沿 y 方向尺寸 w，用于确定把横过渡区半宽（默认自动估算）")
    parser.add_argument("--aero", action="store_true",
                        help="启用气动把三分区模式：把立 + 左把横 + 右把横")
    parser.add_argument("--repair", action="store_true",
                        help="尝试自动修复网格（去除重复/退化面）以消除非流形边")
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
        num_passes=args.num_passes,
        bar_x_size=args.bar_x_size,
        stem_y_size=args.stem_y_size,
        aero_mode=args.aero,
        repair_mode=args.repair
    )

    # 保存输出
    if args.output:
        out_mesh = transformed_scene.dump(concatenate=True)
        out_mesh.export(args.output)
        print(f"Processed mesh saved to {args.output}")

    # 可视化
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
