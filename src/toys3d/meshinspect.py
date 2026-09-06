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


def _print_projected_boundary_diagnostics(
    loops,
    vertex_to_projected,
    vertex_to_tri,
    vertex_to_dist,
    proxy_mesh,
    source_mesh,
    max_report_loops=None,
    max_report_vertices=20,
):
    """
    打印投影孔洞边界包围的代理网格顶点信息。
    """
    from scipy.spatial import cKDTree

    proxy_vertices = np.asarray(proxy_mesh.vertices, dtype=np.float64)
    if len(proxy_vertices) == 0:
        return

    tree = cKDTree(proxy_vertices)

    if max_report_loops is None or max_report_loops <= 0:
        max_report_loops = len(loops)

    report_loops = loops[:max_report_loops]
    print(f"  Projected boundary diagnostics for {len(report_loops)} loops:")

    for loop_idx, loop in enumerate(report_loops):
        ids = np.array(loop, dtype=np.int64)
        pts = np.array([vertex_to_projected[int(v)] for v in ids], dtype=np.float64)
        original_pts = source_mesh.vertices[ids]

        if len(pts) < 3:
            print(f"  [loop {loop_idx}] skipped: only {len(pts)} projected points")
            continue

        tri_ids = [int(vertex_to_tri[int(v)]) for v in ids]
        dists = [float(vertex_to_dist[int(v)]) for v in ids]

        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        u = vh[0]
        v = vh[1]

        poly2d = np.column_stack([
            (pts - centroid) @ u,
            (pts - centroid) @ v,
        ])

        radius = float(np.linalg.norm(pts - centroid, axis=1).max()) + 1e-12
        candidate_indices = tree.query_ball_point(centroid, r=radius)

        if not candidate_indices:
            inside_indices = np.array([], dtype=np.int64)
        else:
            cand_pts = proxy_vertices[candidate_indices]
            cand2d = np.column_stack([
                (cand_pts - centroid) @ u,
                (cand_pts - centroid) @ v,
            ])

            inside_mask = [
                _point_in_polygon_2d(tuple(p), poly2d)
                for p in cand2d
            ]

            inside_indices = np.asarray(candidate_indices, dtype=np.int64)[inside_mask]

        unique_tri_ids = sorted(set(tri_ids))

        input_hole_area = polygon_area_from_3d_ccw(original_pts)
        projected_area = polygon_area_from_3d_ccw(pts)

        if input_hole_area < 1e-12:
            status = "PSEUDO_HOLE"
        elif projected_area < 1e-12:
            status = "PROJECTION_DEGENERATE"
        else:
            status = "REAL_HOLE"

        print(f"  [loop {loop_idx}]")
        print(f"    status={status}")
        print(f"    boundary_edges={len(ids)}")
        print(f"    input_hole_area={input_hole_area:.6f}")
        print(f"    projected_area={projected_area:.6f}")
        print(f"    projection_dist: "
              f"mean={np.mean(dists):.6f}, max={np.max(dists):.6f}")
        print(f"    projected_boundary_triangles={unique_tri_ids}")
        print(f"    enclosed_proxy_vertices={len(inside_indices)}")

        if len(inside_indices) > 0:
            shown = inside_indices[:max_report_vertices]
            print(f"    enclosed_vertex_indices={shown.tolist()}")

            if len(inside_indices) > max_report_vertices:
                print(
                    f"    ... {len(inside_indices) - max_report_vertices} more"
                )


def add_boundary_projection_to_scene(scene, boundary_mesh, proxy_mesh,
                                     radius=None, verbose=False,
                                     print_enclosed_vertices=False,
                                     max_report_loops=None,
                                     max_report_vertices=20):
    """
    将 boundary_mesh 的闭合孔洞边界环投影到 proxy_mesh 表面，
    并在场景中绘制投影线段。
    """
    loops = extract_boundary_loops(boundary_mesh)
    if not loops:
        if verbose:
            print("  No boundary loops to project.")
        return

    try:
        # 收集所有边界环上的唯一顶点
        all_boundary_verts = np.unique(
            np.concatenate([np.array(loop, dtype=np.int64) for loop in loops])
        )
        points = boundary_mesh.vertices[all_boundary_verts]

        projected_points, distances, triangle_indices = project_vertices_to_shell(
            points, proxy_mesh
        )
    except Exception as e:
        if verbose:
            print(f"  Boundary projection failed: {e}")
        return

    # 建立原始顶点索引到投影点的映射
    vertex_to_projected = {
        int(v): projected_points[i]
        for i, v in enumerate(all_boundary_verts)
    }

    vertex_to_tri = {
        int(v): triangle_indices[i]
        for i, v in enumerate(all_boundary_verts)
    }

    vertex_to_dist = {
        int(v): distances[i]
        for i, v in enumerate(all_boundary_verts)
    }

    if radius is None or radius <= 0:
        if len(proxy_mesh.vertices) == 0:
            radius = 1e-6
        else:
            vmin = proxy_mesh.vertices.min(axis=0)
            vmax = proxy_mesh.vertices.max(axis=0)
            diag = float(np.linalg.norm(vmax - vmin))
            radius = max(diag * 0.001, 1e-6)

    color = np.array([0, 255, 255, 255], dtype=np.uint8)  # 青色

    if print_enclosed_vertices:
        _print_projected_boundary_diagnostics(
            loops,
            vertex_to_projected,
            vertex_to_tri,
            vertex_to_dist,
            proxy_mesh,
            source_mesh=boundary_mesh,
            max_report_loops=max_report_loops,
            max_report_vertices=max_report_vertices,
        )

    for loop in loops:
        pts = [vertex_to_projected[int(v)] for v in loop]
        if len(pts) < 2:
            continue

        pts.append(pts[0])  # 闭合环首尾相连
        for i in range(len(pts) - 1):
            seg = trimesh.creation.cylinder(
                radius=radius,
                segment=[pts[i], pts[i + 1]],
                sections=4,
            )
            seg.visual.face_colors = color
            scene.add_geometry(seg)

    if verbose:
        print(f"  Projected {len(loops)} boundary loops onto proxy mesh.")


def load_uncovered_edge_data(data_dir):
    """
    从 hole diagnosis 输出目录加载未覆盖开放边数据。

    返回:
        uncovered_ids : (U,) int64，未覆盖开放边 ID
        all_vertex_pairs : (E,2) int64，所有开放边的顶点对
        categories : (U,) int8，未覆盖开放边的分类
    """
    data_dir = Path(data_dir)
    npz_path = data_dir / "hole_diagnosis_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"未找到 {npz_path}")
    npz = np.load(npz_path)
    uncovered_ids = npz["uncovered_edge_ids"]
    all_vertex_pairs = npz["open_edge_vertex_pairs"]
    categories = npz["uncovered_category"]
    return uncovered_ids, all_vertex_pairs, categories


def add_uncovered_edges_to_scene(scene, mesh, data_dir,
                                 radius=None, verbose=False):
    """
    将 hole diagnosis 中未覆盖的开放边高亮添加到场景。

    分类颜色：
        0: 孤立开放链 -> 蓝色
        1: 悬空开放边 -> 黄色
        2: 分支内部开放边 -> 橙色
        4: 非流形关联开放边 -> 红色
        5: 其他复杂开放边 -> 灰色
    """
    try:
        uncovered_ids, all_vertex_pairs, categories = load_uncovered_edge_data(data_dir)
    except FileNotFoundError as e:
        if verbose:
            print(f"[WARNING] {e}")
        return

    if len(uncovered_ids) == 0:
        if verbose:
            print("没有未覆盖开放边。")
        return

    uncovered_vertex_pairs = all_vertex_pairs[uncovered_ids]

    category_colors = {
        0: (0, 0, 255, 255),       # 孤立开放链 -> 蓝色
        1: (255, 255, 0, 255),     # 悬空开放边 -> 黄色
        2: (255, 128, 0, 255),     # 分支内部开放边 -> 橙色
        4: (255, 0, 0, 255),       # 非流形关联开放边 -> 红色
        5: (128, 128, 128, 255),   # 其他复杂开放边 -> 灰色
    }

    if radius is None or radius <= 0:
        bounds = mesh.bounds
        diag = np.linalg.norm(bounds[1] - bounds[0])
        radius = max(diag * 0.0005, 1e-6)

    if verbose:
        print(f"高亮未覆盖开放边 {len(uncovered_vertex_pairs)} 条")

    for i, (v0, v1) in enumerate(uncovered_vertex_pairs):
        cat = int(categories[i])
        color = category_colors.get(cat, (255, 255, 255, 255))
        seg = trimesh.creation.cylinder(
            radius=radius,
            segment=[mesh.vertices[v0], mesh.vertices[v1]],
            sections=4,
        )
        seg.visual.face_colors = color
        scene.add_geometry(seg)


def load_proxy_mesh(args):
    """根据参数加载代理网格，返回代理网格或 None。"""
    if not args.overlay_proxy:
        return None

    proxy = trimesh.load(args.overlay_proxy, force="mesh")
    if isinstance(proxy, trimesh.Scene):
        proxy = proxy.dump(concatenate=True)

    if len(proxy.faces) == 0:
        print("[WARNING] Proxy mesh is empty; skipping overlay")
        return None

    return proxy


def add_proxy_overlay_to_scene(scene, args, proxy):
    """将代理网格以半透明方式叠加到场景。"""
    if proxy is None:
        return

    # 解析代理颜色
    if args.proxy_color is not None:
        color_components = args.proxy_color
        if len(color_components) == 3:
            alpha = int(np.clip(args.proxy_alpha, 0.0, 1.0) * 255)
            color = [*color_components, alpha]
        else:
            color = color_components
    else:
        base = [128, 180, 255]
        alpha = int(np.clip(args.proxy_alpha, 0.0, 1.0) * 255)
        color = [*base, alpha]

    color_arr = np.array(color, dtype=np.uint8)
    proxy.visual.face_colors = np.tile(color_arr, (len(proxy.faces), 1))

    if args.proxy_double_sided:
        proxy = make_double_sided(proxy, backface_color=None)

    scene.add_geometry(proxy)


def _generate_topology_diagram(code, output_path):
    """
    根据拓扑编码生成三角形的点-线示意图。
    编码格式：A, AB, B, BC, C, CA，每个字段为 uint8 数值。
    顶点元：1 实心，其他空心。
    边元：1 蓝色，2 绿色，3 红色。
    """
    # 将输入统一为整数列表
    if isinstance(code, bytes):
        fields = list(code)
    elif hasattr(code, 'tolist'):
        fields = [int(x) for x in code.tolist()]
    else:
        fields = [int(x) for x in code]

    if len(fields) != 6:
        raise ValueError("code must have exactly 6 fields")

    vA, eAB, vB, eBC, vC, eCA = fields

    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=120)
    pts = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (0.5, np.sqrt(3) / 2)
    }

    edge_styles = {
        1: ('blue', 'solid'),
        2: ('green', 'solid'),
        3: ('red', 'solid')
    }
    for (p1, p2, ecode) in [
        (pts['A'], pts['B'], eAB),
        (pts['B'], pts['C'], eBC),
        (pts['C'], pts['A'], eCA)
    ]:
        color, ls = edge_styles.get(ecode, ('black', 'dashed'))
        line = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                      color=color, linewidth=2, linestyle=ls)
        ax.add_line(line)

    for (pt, vcode) in [(pts['A'], vA), (pts['B'], vB), (pts['C'], vC)]:
        fill = (vcode == 1)
        circle = Circle(pt, radius=0.05, fill=fill,
                        color='black', linewidth=2)
        ax.add_patch(circle)

    height = np.sqrt(3) / 2
    margin_x = 0.2
    margin_y = 0.15
    ax.set_xlim(0 - margin_x, 1 + margin_x)
    ax.set_ylim(0 - margin_y, height + margin_y)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def _generate_component_3d_diagram(component, mesh, output_path):
    """
    为单个未覆盖开放边连通分量生成三维 SVG 图。

    - 边：蓝色线段
    - 端点（度数为1）：绿色圆点
    - 分支点（度数>=3）：红色方块
    - 候选断裂点对：橙色虚线
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    vertex_pairs = component.get('edge_vertex_pairs', [])
    if not vertex_pairs:
        return

    vertices = mesh.vertices
    # 提取组件所有顶点索引，用于确定坐标范围
    involved_vertices = list(set(sum(vertex_pairs, [])))

    v_coords = vertices[involved_vertices]
    vmin = v_coords.min(axis=0)
    vmax = v_coords.max(axis=0)
    center = (vmin + vmax) / 2.0
    max_extent = (vmax - vmin).max()
    extra = max_extent * 0.1 + 1e-12

    fig = plt.figure(figsize=(3.0, 3.0), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制边
    for v0, v1 in vertex_pairs:
        p0 = vertices[v0]
        p1 = vertices[v1]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color='blue', linewidth=0.8, alpha=0.7
        )

    # 端点
    endpoints = component.get('endpoints', [])
    if endpoints:
        ep = vertices[endpoints]
        ax.scatter(ep[:, 0], ep[:, 1], ep[:, 2],
                   c='green', marker='o', s=20, label='Endpoints')

    # 分支点
    branch_vertices = component.get('branch_vertices', [])
    if branch_vertices:
        bv = vertices[branch_vertices]
        ax.scatter(bv[:, 0], bv[:, 1], bv[:, 2],
                   c='red', marker='s', s=30, label='Branch vertices')

    # 候选断裂点对
    candidate_breaks = component.get('candidate_breaks', [])
    for cand in candidate_breaks:
        p0 = vertices[cand['v0']]
        p1 = vertices[cand['v1']]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            '--', color='orange', linewidth=0.8, alpha=0.9
        )

    # 设置坐标轴范围，使图居中
    ax.set_xlim([center[0] - max_extent/2 - extra, center[0] + max_extent/2 + extra])
    ax.set_ylim([center[1] - max_extent/2 - extra, center[1] + max_extent/2 + extra])
    ax.set_zlim([center[2] - max_extent/2 - extra, center[2] + max_extent/2 + extra])

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    if endpoints or branch_vertices:
        ax.legend(loc='upper right', fontsize=6)

    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def _print_boundary_component_diagnostics(mesh, comp, boundary_type, boundary_id, neighborhood_depth):
    """
    打印边界组件的简要诊断信息（用于 extract_component_package）。
    """
    vertices = comp.get("vertices", [])
    edges = comp.get("edge_vertex_pairs", [])
    face_ids = comp.get("face_ids", [])

    print(f"  [{boundary_type} #{boundary_id}] 组件概览 "
          f"(深度 {neighborhood_depth}):")
    print(f"    组件顶点数: {len(vertices)}")
    print(f"    组件边数: {len(edges)}")
    print(f"    关联面数: {len(face_ids)}")
    if comp.get("is_cycle"):
        print("    类型: 闭合健康孔洞环")
    else:
        print("    类型: 未覆盖开放边分量")


def _build_vertex_face_csr(mesh):
    """
    构建 (n_vertices, n_faces) 的 CSR 矩阵，行内存储包含该顶点的面索引。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_vertices = len(mesh.vertices)
    n_faces = len(faces)
    row_idx = faces.ravel()
    col_idx = np.repeat(np.arange(n_faces), 3)
    data = np.ones(3 * n_faces, dtype=np.int8)
    return csr_matrix((data, (row_idx, col_idx)), shape=(n_vertices, n_faces))


def expand_face_neighborhood(mesh, seed_faces, depth):
    """
    从种子面片出发，返回拓扑邻域扩展 depth 层后的面片索引集合。
    depth=0 返回空集合；depth=1 返回 seed_faces 本身；
    depth>=2 依次加入直接邻居、邻居的邻居等。
    """
    if depth <= 0:
        return set()
    seed_faces = set(map(int, seed_faces))
    if depth == 1:
        return seed_faces.copy()

    n_faces = len(mesh.faces)
    adjacency = [[] for _ in range(n_faces)]
    for f0, f1 in mesh.face_adjacency:
        adjacency[int(f0)].append(int(f1))
        adjacency[int(f1)].append(int(f0))

    # 初始层就是种子面片
    current = list(seed_faces)
    visited = set(seed_faces)

    # 已经占用了 depth=1，因此需要再向外扩展 depth-1 层
    for _ in range(depth - 1):
        next_layer = []
        for f in current:
            for nb in adjacency[f]:
                if nb not in visited:
                    visited.add(nb)
                    next_layer.append(nb)
        current = next_layer
        if not current:
            break
    return visited


def load_boundary_component_data(data_dir, boundary_id, boundary_type="uncovered"):
    """
    从 hole diagnosis 输出目录加载指定边界组件或健康孔洞的数据。

    boundary_type:
        "uncovered" : 未覆盖开放边分量
        "healthy"   : 健康孔洞

    返回统一的组件字典，包含边、面、端点、分支点、候选断裂等信息。
    """
    data_dir = Path(data_dir)

    if boundary_type == "uncovered":
        component_json = data_dir / "uncovered_component_analysis.json"
        if not component_json.exists():
            raise FileNotFoundError(f"未找到 {component_json}")
        with open(component_json, "r") as f:
            comp_data = json.load(f)
        components = comp_data.get("components", [])
        if boundary_id < 0 or boundary_id >= len(components):
            raise ValueError(
                f"无效的未覆盖分量 ID: {boundary_id}，共 {len(components)} 个分量"
            )
        comp = components[boundary_id]
        # 确保字段以 list 形式存在
        comp.setdefault("endpoints", [])
        comp.setdefault("branch_vertices", [])
        comp.setdefault("candidate_breaks", [])
        return comp

    elif boundary_type == "healthy":
        npz_path = data_dir / "hole_diagnosis_data.npz"
        json_path = data_dir / "hole_diagnosis.json"
        if not npz_path.exists() or not json_path.exists():
            raise FileNotFoundError(f"未找到 {npz_path} 或 {json_path}")

        npz = np.load(npz_path)
        with open(json_path, "r") as f:
            diag_json = json.load(f)

        healthy_holes = diag_json.get("healthy_holes", [])
        if boundary_id < 0 or boundary_id >= len(healthy_holes):
            raise ValueError(
                f"无效的健康孔洞 ID: {boundary_id}，共 {len(healthy_holes)} 个孔洞"
            )

        hole_vertex_list = healthy_holes[boundary_id]["vertex_indices"]
        hole_ids_per_edge = npz["hole_ids_per_edge"]
        open_edge_vertex_pairs = npz["open_edge_vertex_pairs"]
        open_edge_face_ids = npz["open_edge_face_ids"]

        # 筛选属于该孔洞的开放边
        edge_mask = hole_ids_per_edge == boundary_id
        edge_indices = np.where(edge_mask)[0]
        comp_edges = open_edge_vertex_pairs[edge_indices]
        comp_face_ids = np.unique(open_edge_face_ids[edge_indices])

        # 构造统一结构
        vertices_set = set()
        for v0, v1 in comp_edges:
            vertices_set.add(int(v0))
            vertices_set.add(int(v1))

        component = {
            "component_id": boundary_id,
            "num_edges": int(len(comp_edges)),
            "num_vertices": int(len(vertices_set)),
            "vertices": sorted(vertices_set),
            "edge_vertex_pairs": comp_edges.tolist(),
            "endpoints": [],
            "branch_vertices": [],
            "is_cycle": True,      # 健康孔洞本质上是闭合环
            "face_ids": comp_face_ids.tolist(),
            "open_face_count": int(len(comp_face_ids)),
            "nonmanifold_face_count": 0,
            "candidate_breaks": [],
            "healthy_hole_vertex_indices": hole_vertex_list,
        }
        return component

    else:
        raise ValueError(f"未知的边界类型: {boundary_type}")


def extract_component_package(mesh, args):
    """
    提取指定边界组件/健康孔洞的局部网格，并将重映射后的组件数据保存为 JSON。
    """
    comp = load_boundary_component_data(
        args.boundary_data_dir,
        args.boundary_id,
        args.boundary_type,
    )

    # 估计组件诊断信息
    _print_boundary_component_diagnostics(
        mesh,
        comp,
        args.boundary_type,
        args.boundary_id,
        args.boundary_neighborhood_depth,
    )

    seed_faces = comp.get("face_ids", [])
    expanded = expand_face_neighborhood(
        mesh,
        seed_faces,
        args.boundary_neighborhood_depth,
    )

    if not expanded:
        expanded = set(seed_faces)

    if not expanded:
        print("[ERROR] 没有可提取的面片")
        return

    faces_idx = np.array(sorted(expanded), dtype=np.int64)
    original_faces = np.asarray(mesh.faces, dtype=np.int64)[faces_idx]

    # 局部顶点重映射
    unique_old_vertices = np.unique(original_faces.ravel())
    old_to_new = {
        int(old_v): int(new_v)
        for new_v, old_v in enumerate(unique_old_vertices)
    }

    local_vertices = mesh.vertices[unique_old_vertices]
    local_faces = np.array(
        [
            [old_to_new[int(v)] for v in face]
            for face in original_faces
        ],
        dtype=np.int64,
    )

    local_mesh = trimesh.Trimesh(
        vertices=local_vertices,
        faces=local_faces,
        process=False,
    )

    # 重映射组件内部索引
    comp_new = comp.copy()

    face_idx_to_local = {
        int(old_fid): int(local_fid)
        for local_fid, old_fid in enumerate(faces_idx)
    }

    comp_new["face_ids"] = [
        face_idx_to_local[int(f)]
        for f in comp.get("face_ids", [])
        if int(f) in face_idx_to_local
    ]

    def remap_v(v):
        return old_to_new.get(int(v), -1)

    comp_new["vertices"] = [
        remap_v(v)
        for v in comp.get("vertices", [])
        if remap_v(v) >= 0
    ]

    comp_new["edge_vertex_pairs"] = [
        [remap_v(v0), remap_v(v1)]
        for v0, v1 in comp.get("edge_vertex_pairs", [])
        if remap_v(v0) >= 0 and remap_v(v1) >= 0
    ]

    comp_new["endpoints"] = [
        remap_v(v)
        for v in comp.get("endpoints", [])
        if remap_v(v) >= 0
    ]

    comp_new["branch_vertices"] = [
        remap_v(v)
        for v in comp.get("branch_vertices", [])
        if remap_v(v) >= 0
    ]

    comp_new["candidate_breaks"] = [
        {
            "v0": remap_v(c.get("v0", -1)),
            "v1": remap_v(c.get("v1", -1)),
            "distance": c.get("distance", 0.0),
        }
        for c in comp.get("candidate_breaks", [])
        if remap_v(c.get("v0", -1)) >= 0 and remap_v(c.get("v1", -1)) >= 0
    ]

    if "healthy_hole_vertex_indices" in comp_new:
        comp_new["healthy_hole_vertex_indices"] = [
            remap_v(v)
            for v in comp_new.get("healthy_hole_vertex_indices", [])
            if remap_v(v) >= 0
        ]

    # 输出路径
    input_stem = Path(args.input_file).stem
    ply_path = Path(
        args.component_output
        or f"{input_stem}_component_{args.boundary_id}.ply"
    )
    json_path = Path(
        args.component_json
        or f"{input_stem}_component_{args.boundary_id}.json"
    )

    local_mesh.export(ply_path)
    print(f"局部网格已保存: {ply_path}")

    package_data = {
        "source_file": str(Path(args.input_file).resolve()),
        "boundary_type": args.boundary_type,
        "boundary_id": args.boundary_id,
        "neighborhood_depth": args.boundary_neighborhood_depth,
        "local_vertex_count": int(len(local_vertices)),
        "local_face_count": int(len(local_faces)),
        "component": comp_new,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(package_data, f, indent=2, ensure_ascii=False)
    print(f"组件数据已保存: {json_path}")

    print(
        f"提取完成：局部面片数 {len(local_faces)}，"
        f"局部顶点数 {len(local_vertices)}"
    )


def _generate_initial_seifert_disk(mesh, loop_vertices):
    """
    以健康孔洞边界环为边界生成初始拓扑圆盘。
    使用 SVD 平面投影 + 多边形三角化，避免退化扇形。
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
            raise ValueError("invalid 2D vertices")
        if tri_faces.ndim != 2 or tri_faces.shape[1] != 3 or len(tri_faces) == 0:
            raise ValueError("empty or invalid faces")

        v3d = centroid + tri_vertices_2d[:, 0:1] * u + tri_vertices_2d[:, 1:2] * v

        boundary_indices = []
        for p2d in poly2d:
            dists = np.linalg.norm(tri_vertices_2d - p2d, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] > 1e-8:
                raise ValueError("boundary point not found")
            boundary_indices.append(idx)

        disk = trimesh.Trimesh(
            vertices=v3d,
            faces=tri_faces,
            process=False,
        )

        return disk, boundary_indices

    except Exception as e:
        print(f"  [WARN] 初始 Seifert 圆盘生成失败: {e}")
        return None, []


def _get_component_neighborhood_mesh(mesh, comp, depth):
    """获取指定组件邻域子网格（未双面化）。"""
    face_ids = comp.get("face_ids", [])
    if not face_ids:
        return None

    expanded = expand_face_neighborhood(mesh, face_ids, depth)
    if not expanded:
        expanded = set(face_ids)

    return mesh.submesh([np.array(sorted(expanded), dtype=np.int64)])[0]


def _classify_mesh_edges_and_faces(mesh):
    """
    统计网格中的边与面片缺陷类型。

    返回字典：
        total_edges, open_edges, manifold_edges, nonmanifold_edges
        total_faces, open_faces, manifold_faces, nonmanifold_faces
    """
    defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)

    total_edges = len(mesh.edges_unique)
    open_edges = defect_stats["open_edges"]
    nonmanifold_edges = defect_stats["nonmanifold_edges"]
    manifold_edges = total_edges - open_edges - nonmanifold_edges

    total_faces = len(mesh.faces)
    open_faces = int(open_face_mask.sum())
    nonmanifold_faces = int(nonmanifold_face_mask.sum())
    union_mask = open_face_mask | nonmanifold_face_mask
    manifold_faces = total_faces - int(union_mask.sum())

    return {
        "total_edges": total_edges,
        "open_edges": open_edges,
        "manifold_edges": manifold_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "total_faces": total_faces,
        "open_faces": open_faces,
        "manifold_faces": manifold_faces,
        "nonmanifold_faces": nonmanifold_faces,
    }


def _build_filled_mesh(neighborhood_mesh, seifert_mesh, loop_original_indices,
                       seifert_boundary_indices):
    """
    将邻域网格与 Seifert 曲面合并，使孔洞边界共享同一组顶点。

    loop_original_indices : 孔洞边界在原邻域网格中的顶点索引
    seifert_boundary_indices : Seifert 曲面中对应 loop 顺序的边界顶点索引
    """
    if len(loop_original_indices) != len(seifert_boundary_indices):
        print("  [WARN] Seifert 边界映射长度不一致，跳过填充对比")
        return None

    # Seifert 边界顶点 -> 原邻域网格对应顶点
    seifert_to_orig = {
        int(seifert_boundary_indices[i]): int(loop_original_indices[i])
        for i in range(len(loop_original_indices))
    }

    orig_n = len(neighborhood_mesh.vertices)
    seifert_faces = np.asarray(seifert_mesh.faces, dtype=np.int64)
    remapped_faces = []

    for tri in seifert_faces:
        new_tri = []
        for vid in tri:
            vid = int(vid)
            if vid in seifert_to_orig:
                new_tri.append(seifert_to_orig[vid])
            else:
                new_tri.append(orig_n + vid)
        remapped_faces.append(new_tri)

    remapped_faces = np.array(remapped_faces, dtype=np.int64)
    combined_vertices = np.vstack([neighborhood_mesh.vertices, seifert_mesh.vertices])
    combined_faces = np.vstack([neighborhood_mesh.faces, remapped_faces])

    filled_mesh = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False,
    )
    return filled_mesh


def _edge_tuple(v0, v1):
    v0 = int(v0)
    v1 = int(v1)
    return (v0, v1) if v0 < v1 else (v1, v0)


def _classify_edge_count(cnt):
    if cnt == 1:
        return "open"
    if cnt == 2:
        return "manifold"
    return "nonmanifold"


def _build_edge_tuple_to_faces(mesh, face_indices=None):
    """构建 (min_v, max_v) -> 共享面片索引列表 的完整映射，包含开放边。"""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if face_indices is None:
        face_indices = np.arange(len(faces), dtype=np.int64)

    edge_map = {}
    for fid in face_indices:
        fid = int(fid)
        face = faces[fid]
        for j in range(3):
            v0 = int(face[j])
            v1 = int(face[(j + 1) % 3])
            key = _edge_tuple(v0, v1)
            edge_map.setdefault(key, []).append(fid)
    return edge_map


def _print_seifert_fill_comparison(mesh, comp, seifert_mesh,
                                   loop_original_indices,
                                   seifert_boundary_indices):
    """
    打印 Seifert 曲面填充前后，健康孔洞边界边及新增 Seifert 曲面边/面属性统计。

    仅使用 0 层邻域（组件种子面片）构建局部网格，不引入额外上下文。
    """
    seed_faces = list(map(int, comp.get("face_ids", [])))
    if not seed_faces:
        print("  [WARN] 无法获取组件种子面片，跳过 Seifert 填充对比")
        return

    # 0 层邻域：只取组件种子面片
    original_faces = np.asarray(mesh.faces, dtype=np.int64)
    face_idx_sub = np.array(sorted(seed_faces), dtype=np.int64)
    sub_faces = original_faces[face_idx_sub]

    unique_verts, inverse = np.unique(sub_faces.ravel(), return_inverse=True)
    local_vertices = mesh.vertices[unique_verts]
    local_faces = inverse.reshape(-1, 3)

    local_mesh = trimesh.Trimesh(
        vertices=local_vertices,
        faces=local_faces,
        process=False,
    )

    old_to_new = {
        int(old_v): int(new_v)
        for new_v, old_v in enumerate(unique_verts)
    }

    loop_local = []
    for v in loop_original_indices:
        if int(v) not in old_to_new:
            print("  [WARN] 孔洞边界顶点不在种子面片中，跳过填充对比")
            return
        loop_local.append(old_to_new[int(v)])

    if len(loop_local) != len(seifert_boundary_indices):
        print("  [WARN] Seifert 边界映射长度不一致，跳过填充对比")
        return

    filled_mesh = _build_filled_mesh(
        local_mesh,
        seifert_mesh,
        loop_local,
        seifert_boundary_indices,
    )
    if filled_mesh is None:
        return

    # 1) 健康孔洞边界边填充前后属性
    loop_edge_tuples = []
    for i in range(len(loop_local)):
        v0 = loop_local[i]
        v1 = loop_local[(i + 1) % len(loop_local)]
        loop_edge_tuples.append(_edge_tuple(v0, v1))

    local_edge_map = _build_edge_tuple_to_faces(local_mesh)
    filled_edge_map = _build_edge_tuple_to_faces(filled_mesh)

    before = {"open": 0, "manifold": 0, "nonmanifold": 0}
    after = {"open": 0, "manifold": 0, "nonmanifold": 0}

    for key in loop_edge_tuples:
        cnt_before = len(local_edge_map.get(key, []))
        cnt_after = len(filled_edge_map.get(key, []))
        before[_classify_edge_count(cnt_before)] += 1
        after[_classify_edge_count(cnt_after)] += 1

    # 2) Seifert 曲面新增边/面属性
    new_face_start = len(local_mesh.faces)
    new_face_indices = list(range(new_face_start, len(filled_mesh.faces)))

    new_face_stats = {"open": 0, "manifold": 0, "nonmanifold": 0}
    new_edge_set = set()

    for fid in new_face_indices:
        face = filled_mesh.faces[fid]
        has_open = False
        has_nonmanifold = False

        for j in range(3):
            key = _edge_tuple(face[j], face[(j + 1) % 3])
            new_edge_set.add(key)

            cnt = len(filled_edge_map.get(key, []))
            if cnt == 1:
                has_open = True
            elif cnt >= 3:
                has_nonmanifold = True

        if has_open:
            new_face_stats["open"] += 1
        elif has_nonmanifold:
            new_face_stats["nonmanifold"] += 1
        else:
            new_face_stats["manifold"] += 1

    new_edge_stats = {"open": 0, "manifold": 0, "nonmanifold": 0}
    for key in new_edge_set:
        cnt = len(filled_edge_map.get(key, []))
        new_edge_stats[_classify_edge_count(cnt)] += 1

    print("  Seifert 曲面局部填充对比（0层邻域）:")
    print(
        f"    健康孔洞边界边: "
        f"开放={before['open']} -> {after['open']}, "
        f"流形={before['manifold']} -> {after['manifold']}, "
        f"非流形={before['nonmanifold']} -> {after['nonmanifold']}"
    )
    print(
        f"    新增 Seifert 面片: {len(new_face_indices)} 个 "
        f"(开放={new_face_stats['open']}, "
        f"流形={new_face_stats['manifold']}, "
        f"非流形={new_face_stats['nonmanifold']})"
    )
    print(
        f"    新增 Seifert 唯一边: {len(new_edge_set)} 条 "
        f"(开放={new_edge_stats['open']}, "
        f"流形={new_edge_stats['manifold']}, "
        f"非流形={new_edge_stats['nonmanifold']})"
    )

# [The remaining file content continues unchanged from the original file.]
# (...)
