# src/toys3d/meshinspect.py
import sys
import os

# Ensure src directory is on the path so that 'toys3d' can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_parent = os.path.dirname(_project_root)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)

# If the current working directory contains an `inspect.py`, it would shadow
# the standard library module and cause circular imports (e.g., in NumPy).
if os.path.exists(os.path.join(os.getcwd(), 'inspect.py')):
    if '' in sys.path:
        sys.path.remove('')

import argparse
import colorsys
import json
from pathlib import Path
import numpy as np
import trimesh
from collections import deque, Counter
from scipy.sparse import csr_matrix

import matplotlib
matplotlib.use('Agg')  # 无头模式，避免弹出窗口
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from toys3d.geometrics import (
    compute_mesh_stats,
    analyze_mesh_defects,
    compute_hole_area_stats,
    extract_boundary_loops,
    polygon_area_from_3d_ccw,
    repair_mesh_by_removing_duplicates,
    project_vertices_to_shell,
    weld_small_holes,
    compute_vertex_face_counts,
    compute_face_edge_types,
    compute_face_topology_codes,
    compute_edge_to_faces,
    compute_face_edge_keys,
    compute_face_edge_valences,
    compute_class_neighbor_stats,
    compute_single_face_neighbor_stats,
    get_face_topology_code_and_order,
    code_to_hex,
    hex_to_code,
    save_codes,
    group_faces_by_topology_codes,
    build_hole_diagnosis_data,
    analyze_uncovered_open_edge_components,
    build_manifold_face_adjacency,
    is_manifold_closed_boundary,
    find_minimal_enclosing_manifold_boundary_greedy,
    fit_watertight_patch_from_component,
    expand_face_neighborhood,
    compute_face_distances,
    point_in_polygon_2d,
    extract_intersection_faces_by_vertex_state,
    reconstruct_loop_from_edges,
    generate_initial_seifert_disk,
    build_cotangent_laplacian,
    laplacian_smooth_fixed_boundary,
    compute_curvature_statistics,
    compute_face_area_stats,
    compute_bounding_box_stats,
    compute_volume_if_closed,
)


def compute_face_area_stats(mesh):
    """
    计算三角面片面积的统计量。
    """
    areas = mesh.area_faces
    stats = {}
    if len(areas) == 0:
        stats['count'] = 0
        for key in ['mean', 'min', 'max', 'p1', 'p5', 'p10',
                    'p25', 'p50', 'p75', 'p90', 'p95', 'p99']:
            stats[key] = 0.0
        return stats

    stats['count'] = int(len(areas))
    stats['mean'] = float(np.mean(areas))
    stats['min'] = float(np.min(areas))
    stats['max'] = float(np.max(areas))
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        stats[f'p{p}'] = float(np.percentile(areas, p))
    return stats


def compute_bounding_box_stats(mesh):
    """
    计算包围盒相关统计。

    直接使用 mesh.vertices 计算，避免在超大网格上依赖
    trimesh 的 mesh.bounding_box / mesh.bounds 缓存属性。
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return {
            'min': np.zeros(3),
            'max': np.zeros(3),
            'extents': np.zeros(3),
            'diagonal': 0.0,
            'centroid': np.zeros(3),
        }

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extents = vmax - vmin

    return {
        'min': vmin,
        'max': vmax,
        'extents': extents,
        'diagonal': float(np.linalg.norm(extents)),
        'centroid': (vmin + vmax) / 2.0,
    }


def compute_volume_if_closed(mesh):
    """
    若网格水密，返回体积；否则返回 NaN。
    """
    if mesh.is_watertight:
        return float(mesh.volume)
    return float(np.nan)


def compute_face_distances(mesh, source_mask):
    """
    计算每个面片到源面片集（例如缺陷面）的最短拓扑距离。

    使用 scipy.sparse.csr_matrix 存储面邻接关系，避免为每个面片
    构建 Python list，从而大幅降低内存占用。
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros(0, dtype=np.int32)
    if not np.any(source_mask):
        return np.full(n_faces, np.iinfo(np.int32).max, dtype=np.int32)

    face_adj = mesh.face_adjacency
    rows = np.concatenate([face_adj[:, 0], face_adj[:, 1]])
    cols = np.concatenate([face_adj[:, 1], face_adj[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))

    dist = np.full(n_faces, -1, dtype=np.int32)
    q = deque()

    for i in np.where(source_mask)[0]:
        dist[i] = 0
        q.append(int(i))

    while q:
        cur = q.popleft()
        start = adj.indptr[cur]
        end = adj.indptr[cur + 1]
        for idx in range(start, end):
            nb = adj.indices[idx]
            if dist[nb] == -1:
                dist[nb] = dist[cur] + 1
                q.append(int(nb))

    # 没有路径到达的面片（理论上极少出现）设为最大距离
    dist[dist == -1] = np.iinfo(np.int32).max
    return dist


def build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask):
    """
    生成缺陷可视化网格：
    - 灰色：正常面片
    - 黄色：开放边界附近面片
    - 红色：非流形边附近面片
    - 橙色：同时具有两种缺陷的面片
    """
    vis = mesh.copy()
    N = len(vis.faces)

    colors = np.full((N, 4), 200, dtype=np.uint8)
    colors[:, 3] = 255

    open_only = open_face_mask & ~nonmanifold_face_mask
    nonmanifold_only = nonmanifold_face_mask & ~open_face_mask
    both = open_face_mask & nonmanifold_face_mask

    colors[open_only] = [255, 220, 0, 255]      # 黄
    colors[nonmanifold_only] = [255, 0, 0, 255]  # 红
    colors[both] = [255, 128, 0, 255]            # 橙

    vis.visual.face_colors = colors
    return vis


def build_reliable_visualization(mesh, distances, min_distance):
    """
    根据拓扑距离生成可视化网格。
    - 绿色：可靠（距离 >= min_distance）
    - 黄色：中间状态（0 < 距离 < min_distance）
    - 红色：缺陷面（距离 == 0）
    """
    vis = mesh.copy()
    N = len(vis.faces)

    colors = np.full((N, 4), [255, 0, 0, 255], dtype=np.uint8)  # 默认红（缺陷面）
    intermediate = (distances > 0) & (distances < min_distance)
    reliable = distances >= min_distance
    colors[intermediate] = [255, 220, 0, 255]  # 黄
    colors[reliable] = [0, 200, 0, 255]        # 绿

    vis.visual.face_colors = colors
    return vis


def make_double_sided(mesh, backface_color=None):
    """
    将网格渲染为双面几何，避免薄壳背面被背面剔除而显示为透明。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    backface_color : list or None
        背面子颜色（RGBA）。
        若为 None，则根据每个正面颜色自动生成同色系暗色：
        保持 Hue 不变，Saturation * 0.8，Value * 0.5。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) == 0:
        return mesh.copy()

    double_faces = np.vstack([
        faces,
        faces[:, ::-1],
    ])

    vis = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=double_faces,
        process=False,
    )

    if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors.shape[0] == len(faces):
        colors = np.asarray(mesh.visual.face_colors)
    else:
        colors = np.full((len(faces), 4), [200, 200, 200, 255], dtype=np.uint8)

    if backface_color is None:
        # 根据每个正面颜色自动生成同色系暗色背面颜色
        n = len(colors)
        back_colors = np.empty_like(colors)
        for i in range(n):
            r, g, b, a = colors[i].astype(np.float64) / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            back_s = np.clip(s * 0.8, 0.0, 1.0)
            back_v = np.clip(v * 0.5, 0.0, 1.0)
            br, bg, bb = colorsys.hsv_to_rgb(h, back_s, back_v)
            back_colors[i] = np.array([
                br * 255.0,
                bg * 255.0,
                bb * 255.0,
                a * 255.0,
            ], dtype=np.uint8)
    else:
        back_colors = np.full_like(colors, np.asarray(backface_color, dtype=np.uint8))

    vis.visual.face_colors = np.vstack([colors, back_colors])
    return vis


def set_face_alpha(mesh, alpha):
    """将网格所有面片颜色的 alpha 通道设置为指定透明度。"""
    if hasattr(mesh.visual, 'face_colors') and \
            mesh.visual.face_colors.shape[0] == len(mesh.faces):
        mesh.visual.face_colors[:, 3] = int(np.clip(alpha, 0.0, 1.0) * 255)
    return mesh


def add_wireframe_to_scene(scene, mesh, color=None, radius=None):
    """
    将网格的边以圆柱线段形式加入场景，用于观察三角剖分。

    Parameters
    ----------
    scene : trimesh.Scene
    mesh : trimesh.Trimesh
    color : list or tuple or ndarray or None
        RGBA 颜色，默认纯黑不透明 [0, 0, 0, 255]。
    radius : float or None
        圆柱半径，默认基于包围盒对角线的 0.05%。
    """
    if color is None:
        color = np.array([0, 0, 0, 255], dtype=np.uint8)
    else:
        color = np.asarray(color, dtype=np.uint8)

    edges_unique = mesh.edges_unique
    if len(edges_unique) == 0:
        return

    if radius is None or radius <= 0:
        if len(mesh.vertices) == 0:
            radius = 1e-6
        else:
            vmin = mesh.vertices.min(axis=0)
            vmax = mesh.vertices.max(axis=0)
            diag = float(np.linalg.norm(vmax - vmin))
            radius = max(diag * 0.0005, 1e-6)

    for e in edges_unique:
        v0, v1 = mesh.vertices[e[0]], mesh.vertices[e[1]]
        seg = trimesh.creation.cylinder(
            radius=radius,
            segment=[v0, v1],
            sections=4,
        )
        seg.visual.face_colors = color
        scene.add_geometry(seg)


def _high_saturation_hole_palette():
    """
    返回一组高饱和度、且相互区分的 RGBA 颜色。
    """
    palette = [
        (255,   0,   0, 255),   # 红
        (  0, 255,   0, 255),   # 绿
        (  0, 128, 255, 255),   # 蓝
        (255, 255,   0, 255),   # 黄
        (255,   0, 255, 255),   # 品红
        (  0, 255, 255, 255),   # 青
        (255, 128,   0, 255),   # 橙
        (128,   0, 255, 255),   # 紫
        (  0, 255, 128, 255),   # 春绿
        (255,   0, 128, 255),   # 粉红
    ]
    return [np.array(c, dtype=np.uint8) for c in palette]


def _greedy_color_hole_loops(loops):
    """
    为孔洞边界环分配调色板颜色索引。

    如果两个孔洞共享顶点，则认为它们相邻，应使用不同颜色。
    使用贪心染色。返回 (color_indices, palette)。
    """
    n = len(loops)
    if n == 0:
        return [], _high_saturation_hole_palette()

    vertex_to_loops = {}
    for i, loop in enumerate(loops):
        for v in loop:
            vertex_to_loops.setdefault(int(v), []).append(i)

    adjacency = [set() for _ in range(n)]
    for loop_indices in vertex_to_loops.values():
        if len(loop_indices) <= 1:
            continue
        for i in loop_indices:
            for j in loop_indices:
                if i != j:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

    palette = _high_saturation_hole_palette()
    color_indices = [None] * n

    for i in range(n):
        used = {color_indices[j] for j in adjacency[i]
                if color_indices[j] is not None}
        chosen = None
        for c in range(len(palette)):
            if c not in used:
                chosen = c
                break
        if chosen is None:
            chosen = i % len(palette)
        color_indices[i] = chosen

    return color_indices, palette


def add_hole_boundaries_to_scene(scene, mesh, radius=None,
                                 min_edges=3, min_area=0.0,
                                 verbose=False):
    """
    检测网格中的闭合孔洞边界环，并在场景中用高饱和度颜色绘制。

    只绘制边长和面积均满足阈值的闭合环。
    """
    loops = extract_boundary_loops(mesh)

    filtered_loops = []
    for loop in loops:
        if len(loop) < min_edges:
            continue
        pts = mesh.vertices[np.array(loop)]
        area = polygon_area_from_3d_ccw(pts)
        if area < min_area:
            continue
        filtered_loops.append(loop)

    if not filtered_loops:
        if verbose:
            print("  No closed hole boundary loops matching thresholds.")
        return

    color_indices, palette = _greedy_color_hole_loops(filtered_loops)

    if radius is None or radius <= 0:
        if len(mesh.vertices) == 0:
            radius = 1e-6
        else:
            vmin = mesh.vertices.min(axis=0)
            vmax = mesh.vertices.max(axis=0)
            diag = float(np.linalg.norm(vmax - vmin))
            radius = max(diag * 0.001, 1e-6)

    if verbose:
        print(f"  Drawing {len(filtered_loops)} hole boundary loops "
              f"(filtered from {len(loops)} total):")

    for loop_idx, loop in enumerate(filtered_loops):
        color = palette[color_indices[loop_idx]]
        if verbose:
            print(f"    hole {loop_idx}: {len(loop)} edges, color={color.tolist()}")

        for k in range(len(loop)):
            v0 = mesh.vertices[int(loop[k])]
            v1 = mesh.vertices[int(loop[(k + 1) % len(loop)])]
            seg = trimesh.creation.cylinder(
                radius=radius,
                segment=[v0, v1],
                sections=4,
            )
            seg.visual.face_colors = color
            scene.add_geometry(seg)


def _point_in_polygon_2d(pt, poly):
    """二维射线法判断点是否在多边形内部。"""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1

    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-30) + xi
        ):
            inside = not inside
        j = i

    return inside


def extract_intersection_faces_by_vertex_state(W, N, eps=None):
    """
    基于顶点内外状态，提取邻域网格 N 中与水密包络 W 相交的面片。

    返回:
        face_mask : (len(N.faces),) bool
        face_ids  : 相交面片索引
        vertex_state : (len(N.vertices),) int8
                       0 = inside, 1 = outside, 2 = on_surface
    """
    n_verts = len(N.vertices)
    n_faces = len(N.faces)

    if n_verts == 0 or n_faces == 0:
        return np.zeros(n_faces, dtype=bool), np.array([], dtype=np.int64), np.zeros(n_verts, dtype=np.int8)

    # 1. 顶点在 W 表面上的距离与内外状态
    closest, dist, _ = W.nearest.on_surface(N.vertices)

    # trimesh 的 contains 内部使用射线奇偶判定
    inside = W.contains(N.vertices)

    if eps is None:
        median_w = np.median(W.edges_unique_length) if len(W.edges_unique) else 0.0
        median_n = np.median(N.edges_unique_length) if len(N.edges_unique) else 0.0
        eps = max(median_w, median_n) * 1e-6

    on_surface = dist <= eps
    outside = (~inside) & (~on_surface)
    inside_only = inside & (~on_surface)

    vertex_state = np.zeros(n_verts, dtype=np.int8)
    vertex_state[inside_only] = 0
    vertex_state[outside] = 1
    vertex_state[on_surface] = 2

    # 2. 逐面判定
    face_mask = np.zeros(n_faces, dtype=bool)

    faces = np.asarray(N.faces, dtype=np.int64)
    for fid, tri in enumerate(faces):
        vs = vertex_state[tri]
        has_inside = np.any(vs == 0)
        has_outside = np.any(vs == 1)
        has_surface = np.any(vs == 2)

        if has_surface or (has_inside and has_outside):
            face_mask[fid] = True

    return face_mask, np.where(face_mask)[0], vertex_state


def _reconstruct_loop_from_edges(edge_vertex_pairs):
    """从健康孔洞的无序边集恢复闭合顶点环。"""
    if not edge_vertex_pairs:
        return []

    adj = {}
    for a, b in edge_vertex_pairs:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    start = next(iter(adj))
    loop = [start]
    prev = None
    cur = start

    while True:
        nxts = [v for v in adj[cur] if v != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt

        if len(loop) > len(adj):
            break

    return loop


def _generate_initial_seifert_disk(mesh, loop_vertices):
    """
    以健康孔洞边界环为边界生成初始拓扑圆盘。

    优先使用 SVD 平面投影 + 多边形三角化，得到一个几何更自然的初始盘。
    返回 (disk_mesh, boundary_indices)：
        disk_mesh        : trimesh.Trimesh 或 None
        boundary_indices : 边界顶点在 disk_mesh.vertices 中按 loop 顺序排列的索引
    """
    loop_vertices = [int(v) for v in loop_vertices]
    if len(loop_vertices) < 3:
        return None, []

    pts = np.asarray(mesh.vertices[loop_vertices], dtype=np.float64)

    try:
        from shapely.geometry import Polygon

        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        u = vh[0]
        v = vh[1]

        poly2d = np.column_stack([
            (pts - centroid) @ u,
            (pts - centroid) @ v,
        ])

        polygon = Polygon(poly2d)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        triangulated = trimesh.creation.triangulate_polygon(polygon)
        if triangulated is None:
            raise ValueError("triangulate_polygon returned None")

        tri_vertices_2d, tri_faces = triangulated
        tri_vertices_2d = np.asarray(tri_vertices_2d, dtype=np.float64)
        tri_faces = np.asarray(tri_faces, dtype=np.int64)

        if tri_vertices_2d.ndim != 2 or tri_vertices_2d.shape[1] != 2:
            raise ValueError("triangulate_polygon returned invalid 2D vertices")
        if tri_faces.ndim != 2 or tri_faces.shape[1] != 3 or len(tri_faces) == 0:
            raise ValueError("triangulate_polygon returned empty or invalid faces")

        v3d = centroid + tri_vertices_2d[:, 0:1] * u + tri_vertices_2d[:, 1:2] * v

        # 找到每个原始边界点在三角化结果中的索引，保持 loop 顺序
        boundary_indices = []
        for p2d in poly2d:
            dists = np.linalg.norm(tri_vertices_2d - p2d, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] > 1e-8:
                raise ValueError("boundary point not found in triangulation")
            boundary_indices.append(idx)

        disk = trimesh.Trimesh(
            vertices=v3d,
            faces=tri_faces,
            process=False,
        )

        # 直接使用之前二维三角化得到的边界索引，不再合并顶点
        if len(boundary_indices) != len(poly2d):
            raise ValueError("boundary indices mismatch")

        return disk, boundary_indices

    except Exception as e:
        print(f"  [WARN] 初始 Seifert 圆盘生成失败: {e}")
        return None, []


def _build_cotangent_laplacian(mesh):
    """
    构建当前网格的余切权重 Laplacian 矩阵，返回 scipy.sparse.csr_matrix。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n_vertices = len(vertices)
    if n_vertices == 0:
        return csr_matrix((0, 0))

    # 边 -> 面片索引列表
    edge_to_faces = {}
    for fid, face in enumerate(faces):
        for i in range(3):
            v0 = int(face[i])
            v1 = int(face[(i + 1) % 3])
            key = (v0, v1) if v0 < v1 else (v1, v0)
            edge_to_faces.setdefault(key, []).append(fid)

    row = []
    col = []
    data = []

    # 对每个面计算三个角，将对角的余切值加到对应边上
    for fid, face in enumerate(faces):
        tri = vertices[face]
        a, b, c = tri[0], tri[1], tri[2]

        def angle_at(p, q, r):
            v1 = q - p
            v2 = r - p
            dot = np.dot(v1, v2)
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < 1e-12:
                return 0.0
            cos_angle = np.clip(dot / denom, -1.0, 1.0)
            return float(np.arccos(cos_angle))

        alpha = angle_at(a, b, c)  # 顶点 a
        beta = angle_at(b, c, a)   # 顶点 b
        gamma = angle_at(c, a, b)  # 顶点 c

        # 边 (b,c) 对角 alpha
        edge0 = (int(face[1]), int(face[2])) if face[1] < face[2] else (int(face[2]), int(face[1]))
        # 边 (c,a) 对角 beta
        edge1 = (int(face[2]), int(face[0])) if face[2] < face[0] else (int(face[0]), int(face[2]))
        # 边 (a,b) 对角 gamma
        edge2 = (int(face[0]), int(face[1])) if face[0] < face[1] else (int(face[1]), int(face[0]))

        cot_alpha = 1.0 / np.tan(alpha) if abs(np.tan(alpha)) > 1e-12 else 0.0
        cot_beta = 1.0 / np.tan(beta) if abs(np.tan(beta)) > 1e-12 else 0.0
        cot_gamma = 1.0 / np.tan(gamma) if abs(np.tan(gamma)) > 1e-12 else 0.0

        # 在 Laplacian 中，权重要加到两个对应顶点间的 off-diagonal 项
        # 我们直接向 row/col/data 累加，最后再构建对称矩阵
        def add_weight(e0, e1, w):
            if w == 0:
                return
            row.append(e0)
            col.append(e1)
            data.append(w)
            row.append(e1)
            col.append(e0)
            data.append(w)

        add_weight(edge0[0], edge0[1], cot_alpha)
        add_weight(edge1[0], edge1[1], cot_beta)
        add_weight(edge2[0], edge2[1], cot_gamma)

    if not row:
        return csr_matrix((n_vertices, n_vertices))

    L = csr_matrix(
        (np.array(data, dtype=np.float64), (np.array(row, dtype=np.int64), np.array(col, dtype=np.int64))),
        shape=(n_vertices, n_vertices),
    )

    # 对角线设为负的行和，保证每行和为0
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    L = L - csr_matrix(
        (row_sums, (np.arange(n_vertices), np.arange(n_vertices))),
        shape=(n_vertices, n_vertices),
    )

    return L


def _laplacian_smooth_fixed_boundary(
    mesh,
    boundary_vertex_indices,
    iterations=200,
    step_size=1.0,
    tol=1e-7,
):
    """
    固定边界顶点，内部顶点按离散 Plateau 问题迭代求解：
    每次根据当前几何构建余切 Laplacian，然后求解
        L[interior, interior] * X_int = -L[interior, boundary] * X_boundary
    并对解做松弛更新。
    """
    from scipy.sparse.linalg import spsolve

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = mesh.vertices.copy()
    n_vertices = len(vertices)
    boundary_set = set(int(v) for v in boundary_vertex_indices)

    if n_vertices == 0 or len(faces) == 0:
        return mesh.copy()

    all_indices = np.arange(n_vertices, dtype=np.int64)
    interior_indices = np.array(
        [i for i in all_indices if int(i) not in boundary_set],
        dtype=np.int64,
    )
    boundary_indices = np.array(sorted(boundary_set), dtype=np.int64)

    if len(interior_indices) == 0:
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    step_size = float(np.clip(step_size, 0.0, 1.0))

    for it in range(iterations):
        current = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        L = _build_cotangent_laplacian(current)

        Lint = L[interior_indices, :][:, interior_indices].tocsr()
        Lbnd = L[interior_indices, :][:, boundary_indices].tocsr()

        # 增加极小对角正则项，防止因退化三角形导致奇异
        n_int = len(interior_indices)
        eps_reg = 1e-10
        Lint = Lint + csr_matrix(
            np.eye(n_int, dtype=np.float64) * eps_reg
        )

        rhs = -Lbnd @ vertices[boundary_indices]
        sol = spsolve(Lint, rhs)

        new_vertices = vertices.copy()
        new_vertices[interior_indices] = (
            vertices[interior_indices]
            + step_size * (sol - vertices[interior_indices])
        )

        moves = np.linalg.norm(
            new_vertices[interior_indices] - vertices[interior_indices],
            axis=1,
        )
        max_move = float(np.max(moves)) if len(moves) > 0 else 0.0
        vertices = new_vertices

        if max_move < tol:
            print(f"    Seifert 优化在第 {it+1} 次迭代收敛，最大位移 {max_move:.6e}")
            break

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _compute_curvature_statistics(mesh, boundary_vertex_indices):
    """
    计算 Seifert 曲面内部顶点的离散曲率统计。
    返回字典，包含平均曲率绝对值和高斯曲率近似。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n_vertices = len(vertices)
    boundary_set = set(int(v) for v in boundary_vertex_indices)

    L = _build_cotangent_laplacian(mesh)

    area_faces = mesh.area_faces
    vertex_areas = np.bincount(
        faces.ravel(),
        weights=np.repeat(area_faces, 3),
        minlength=n_vertices,
    ) / 3.0
    vertex_areas[vertex_areas < 1e-12] = 1.0

    Hn = L @ vertices  # (n,3)
    H_mag = np.linalg.norm(Hn, axis=1) / (2.0 * vertex_areas)

    # 高斯曲率：角缺陷 / 顶点面积
    # 先计算各顶点周围角和
    angle_sum = np.zeros(n_vertices, dtype=np.float64)
    for face in faces:
        tri = vertices[face]
        for j in range(3):
            v_idx = int(face[j])
            p = tri[j]
            q = tri[(j + 1) % 3]
            r = tri[(j + 2) % 3]
            v1 = q - p
            v2 = r - p
            dot = np.dot(v1, v2)
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < 1e-12:
                angle = 0.0
            else:
                cos_angle = np.clip(dot / denom, -1.0, 1.0)
                angle = float(np.arccos(cos_angle))
            angle_sum[v_idx] += angle
    K = (2.0 * np.pi - angle_sum) / vertex_areas

    interior_mask = np.array([i not in boundary_set for i in range(n_vertices)], dtype=bool)
    if not np.any(interior_mask):
        interior_mask = np.ones(n_vertices, dtype=bool)  # 退而求其次

    H_int = H_mag[interior_mask]
    K_int = K[interior_mask]

    stats = {
        "mean_abs_mean_curvature": float(np.mean(H_int)),
        "median_abs_mean_curvature": float(np.median(H_int)),
        "max_abs_mean_curvature": float(np.max(H_int)),
        "p95_abs_mean_curvature": float(np.percentile(H_int, 95)),
        "std_abs_mean_curvature": float(np.std(H_int)),
        "mean_gaussian_curvature": float(np.mean(K_int)),
        "median_gaussian_curvature": float(np.median(K_int)),
        "max_gaussian_curvature": float(np.max(K_int)),
        "min_gaussian_curvature": float(np.min(K_int)),
        "area": float(mesh.area),
        "perimeter": float(np.sum(np.linalg.norm(
            vertices[boundary_vertex_indices] -
            np.roll(vertices[boundary_vertex_indices], -1, axis=0), axis=1))),
    }
    return stats


# [remaining code continues exactly as before]
