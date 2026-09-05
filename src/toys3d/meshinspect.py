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


def _build_cotangent_laplacian(mesh):
    """
    构建当前网格的余切权重 Laplacian 矩阵。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n_vertices = len(vertices)
    if n_vertices == 0:
        return csr_matrix((0, 0))

    row = []
    col = []
    data = []

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

        alpha = angle_at(a, b, c)
        beta = angle_at(b, c, a)
        gamma = angle_at(c, a, b)

        edge0 = (int(face[1]), int(face[2])) if face[1] < face[2] else (int(face[2]), int(face[1]))
        edge1 = (int(face[2]), int(face[0])) if face[2] < face[0] else (int(face[0]), int(face[2]))
        edge2 = (int(face[0]), int(face[1])) if face[0] < face[1] else (int(face[1]), int(face[0]))

        cot_alpha = 1.0 / np.tan(alpha) if abs(np.tan(alpha)) > 1e-12 else 0.0
        cot_beta = 1.0 / np.tan(beta) if abs(np.tan(beta)) > 1e-12 else 0.0
        cot_gamma = 1.0 / np.tan(gamma) if abs(np.tan(gamma)) > 1e-12 else 0.0

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

    row_sums = np.asarray(L.sum(axis=1)).ravel()
    L = L - csr_matrix(
        (row_sums, (np.arange(n_vertices), np.arange(n_vertices))),
        shape=(n_vertices, n_vertices),
    )

    return L


def _laplacian_smooth_fixed_boundary(mesh, boundary_vertex_indices, iterations=200, step_size=1.0, tol=1e-7):
    """
    固定边界顶点，内部顶点按离散 Plateau 问题迭代求解。
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

        n_int = len(interior_indices)
        eps_reg = 1e-10
        Lint = Lint + csr_matrix(np.eye(n_int, dtype=np.float64) * eps_reg)

        rhs = -Lbnd @ vertices[boundary_indices]
        sol = spsolve(Lint, rhs)

        new_vertices = vertices.copy()
        new_vertices[interior_indices] = (
            vertices[interior_indices] + step_size * (sol - vertices[interior_indices])
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

    Hn = L @ vertices
    H_mag = np.linalg.norm(Hn, axis=1) / (2.0 * vertex_areas)

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
        interior_mask = np.ones(n_vertices, dtype=bool)

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


def _print_boundary_component_diagnostics(mesh, comp, boundary_type, boundary_id, neighborhood_depth):
    """
    打印指定边界组件/健康孔洞的基础诊断信息。
    """
    print(f"\n[DIAGNOSTICS] 可视化原始网格组件 (boundary_id={boundary_id}, type={boundary_type})")

    edges = comp.get("edge_vertex_pairs", [])
    vertices_set = set()
    for v0, v1 in edges:
        vertices_set.add(int(v0))
        vertices_set.add(int(v1))

    print(f"  边界边数: {len(edges)}")
    print(f"  边界顶点数: {len(vertices_set)}")

    # 顶点度数分布
    degree = Counter()
    for v0, v1 in edges:
        degree[int(v0)] += 1
        degree[int(v1)] += 1

    deg1 = sum(1 for d in degree.values() if d == 1)
    deg2 = sum(1 for d in degree.values() if d == 2)
    deg3plus = sum(1 for d in degree.values() if d >= 3)
    print(f"  度为1的顶点数: {deg1}")
    print(f"  度为2的顶点数: {deg2}")
    print(f"  度为3及以上的顶点数: {deg3plus}")

    seed_faces = comp.get("face_ids", [])
    print(f"  种子面片数: {len(seed_faces)}")

    if neighborhood_depth > 0:
        print("  邻域面片距离分布（距离0=种子面片）:")
        max_display = min(neighborhood_depth, 20)  # 最多显示到20层
        prev_set = set(seed_faces)
        print(f"    距离 0: {len(prev_set)}")
        for d in range(1, max_display + 1):
            cur_set = expand_face_neighborhood(mesh, seed_faces, d + 1)
            new_count = len(cur_set - prev_set)
            print(f"    距离 {d}: {new_count}")
            prev_set = cur_set


def print_scene_debug_info(scene, title="Scene Debug Info"):
    """
    打印场景中所有几何对象的名称、类型、尺寸与包围盒。
    """
    print_separator(title)

    geometry_items = list(scene.geometry.items())
    if not geometry_items:
        print("  scene is empty")
        return

    print(f"  geometry count: {len(geometry_items)}")

    scene_bounds = None
    for i, (name, geom) in enumerate(geometry_items):
        geom_type = type(geom).__name__

        n_vertices = len(getattr(geom, "vertices", [])) if hasattr(geom, "vertices") else 0
        n_faces = len(getattr(geom, "faces", [])) if hasattr(geom, "faces") else 0

        try:
            bounds = geom.bounds
        except Exception:
            bounds = None

        if bounds is not None:
            bmin = bounds[0]
            bmax = bounds[1]
            extents = bmax - bmin
            center = (bmin + bmax) / 2.0
            diag = float(np.linalg.norm(extents))
        else:
            bmin = np.zeros(3)
            bmax = np.zeros(3)
            extents = np.zeros(3)
            center = np.zeros(3)
            diag = 0.0

        print(f"  [{i}] name={name}")
        print(f"      type={geom_type}")
        print(f"      vertices={n_vertices}, faces={n_faces}")
        print(f"      bounds.min=[{bmin[0]:.6f}, {bmin[1]:.6f}, {bmin[2]:.6f}]")
        print(f"      bounds.max=[{bmax[0]:.6f}, {bmax[1]:.6f}, {bmax[2]:.6f}]")
        print(f"      extents=[{extents[0]:.6f}, {extents[1]:.6f}, {extents[2]:.6f}]")
        print(f"      center=[{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
        print(f"      diagonal={diag:.6f}")

        if bounds is not None:
            if scene_bounds is None:
                scene_bounds = bounds.copy()
            else:
                scene_bounds[0] = np.minimum(scene_bounds[0], bmin)
                scene_bounds[1] = np.maximum(scene_bounds[1], bmax)

    if scene_bounds is not None:
        sbmin = scene_bounds[0]
        sbmax = scene_bounds[1]
        sext = sbmax - sbmin
        scent = (sbmin + sbmax) / 2.0
        sdiag = float(np.linalg.norm(sext))
        print("  [scene]")
        print(f"      bounds.min=[{sbmin[0]:.6f}, {sbmin[1]:.6f}, {sbmin[2]:.6f}]")
        print(f"      bounds.max=[{sbmax[0]:.6f}, {sbmax[1]:.6f}, {sbmax[2]:.6f}]")
        print(f"      extents=[{sext[0]:.6f}, {sext[1]:.6f}, {sext[2]:.6f}]")
        print(f"      center=[{scent[0]:.6f}, {scent[1]:.6f}, {scent[2]:.6f}]")
        print(f"      diagonal={sdiag:.6f}")


def _filter_camera_core_points(points):
    """
    根据中位数绝对偏差过滤离群点，避免个别原点/错误点拉偏相机。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        return points

    med = np.median(points, axis=0)
    dist = np.linalg.norm(points - med, axis=1)
    med_dist = np.median(dist)

    if med_dist < 1e-12:
        return points

    ratio = dist / med_dist
    kept = ratio <= 3.0
    return points[kept]


def visualize_boundary_component(mesh, args):
    """
    可视化健康孔洞或未覆盖开放边分量及其局部三角面片。
    默认不显示整个网格，只显示目标边界和指定邻域深度内的面片。
    """
    comp = load_boundary_component_data(
        args.boundary_data_dir,
        args.boundary_id,
        args.boundary_type,
    )

    # 打印组件诊断信息
    _print_boundary_component_diagnostics(
        mesh, comp, args.boundary_type, args.boundary_id,
        args.boundary_neighborhood_depth
    )

    # 优先使用当前边集导出的顶点，避免 hole_diagnosis.json 中旧索引/异常索引
    focus_indices = comp.get("vertices")

    if not focus_indices:
        focus_indices = comp.get("healthy_hole_vertex_indices", [])

    if not focus_indices:
        focus_indices = comp.get("endpoints", [])

    # 核心取景点集：优先为健康孔洞边界顶点，其次组件顶点/端点
    camera_core_points = []
    camera_focus = None

    if focus_indices:
        focus_points = mesh.vertices[np.asarray(focus_indices, dtype=np.int64)]
        camera_core_points = focus_points.copy()
        camera_focus = focus_points.mean(axis=0)

    scene = trimesh.Scene()

    # 可选：显示半透明原始网格
    if args.boundary_show_original:
        vis_mesh = mesh.copy()
        # 赋予统一半透明颜色（确保存在 face_colors）
        alpha_uint8 = int(0.3 * 255)
        vis_mesh.visual.face_colors = np.full(
            (len(vis_mesh.faces), 4),
            [200, 200, 200, alpha_uint8],
            dtype=np.uint8,
        )
        # 双面显示原始网格背景
        if args.double_sided:
            vis_mesh = make_double_sided(vis_mesh)
        scene.add_geometry(vis_mesh)

    # 根据邻域深度显示相关三角面片
    if args.boundary_neighborhood_depth > 0:
        face_ids = comp.get("face_ids", [])
        if face_ids:
            expanded_faces = expand_face_neighborhood(
                mesh, face_ids, args.boundary_neighborhood_depth
            )
            if expanded_faces:
                sub = mesh.submesh(
                    [np.array(list(expanded_faces), dtype=np.int64)]
                )[0]

                if args.boundary_type == "uncovered":
                    sub.visual.face_colors = [255, 165, 0, 255]  # 橙色
                else:
                    sub.visual.face_colors = [144, 238, 144, 255]  # 浅绿

                # 双面显示相关三角面片
                if args.double_sided:
                    sub = make_double_sided(sub)

                scene.add_geometry(sub)

    # 计算默认圆柱半径
    radius = args.boundary_radius
    if radius is None or radius <= 0:
        bounds = mesh.bounds
        diag = np.linalg.norm(bounds[1] - bounds[0])
        radius = max(diag * 0.0005, 1e-6)

    # 绘制边界边
    if args.boundary_type == "uncovered":
        edge_color = [0, 128, 255, 255]   # 蓝色
    else:
        edge_color = [0, 255, 255, 255]   # 青色

    for v0, v1 in comp["edge_vertex_pairs"]:
        seg = trimesh.creation.cylinder(
            radius=radius,
            segment=[mesh.vertices[v0], mesh.vertices[v1]],
            sections=4,
        )
        seg.visual.face_colors = edge_color
        scene.add_geometry(seg)

    # 绘制最小包络流形边界（若存在）
    enclosing = comp.get("minimal_enclosing_boundary", {})
    if enclosing.get("success"):
        enclosing_vertices = enclosing.get("boundary_vertices", [])
        enclosing_radius = radius * 1.5   # 稍粗，更醒目

        for loop_verts in enclosing_vertices:
            for i in range(len(loop_verts) - 1):
                v0 = loop_verts[i]
                v1 = loop_verts[i + 1]
                seg = trimesh.creation.cylinder(
                    radius=enclosing_radius,
                    segment=[mesh.vertices[v0], mesh.vertices[v1]],
                    sections=6,
                )
                seg.visual.face_colors = [255, 0, 255, 255]  # 洋红色
                scene.add_geometry(seg)

    # 拟合水密包络曲面并显示交线
    if args.fit_watertight_patch:
        print("拟合水密包络曲面...")
        patch_result = fit_watertight_patch_from_component(
            mesh,
            comp,
            method=args.patch_method,
            neighborhood_depth=args.patch_neighborhood_depth,
            poisson_depth=args.patch_poisson_depth,
            density_quantile=args.patch_density_quantile,
            alpha=args.patch_alpha,
            allow_non_genus0=args.allow_non_genus0,
        )
        if patch_result["success"]:
            watertight_mesh = patch_result["watertight_mesh"]
            intersection_vertices = patch_result["intersection_vertices"]
            intersection_edges = patch_result["intersection_edges"]

            # 显示拟合曲面（半透明青色）
            # 使用用户指定的不透明度，并支持双面渲染避免背面剔除导致的结构透视
            alpha = int(np.clip(args.patch_opacity, 0.0, 1.0) * 255)
            watertight_mesh.visual.face_colors = np.full(
                (len(watertight_mesh.faces), 4),
                [0, 200, 200, alpha],
                dtype=np.uint8,
            )
            if args.double_sided:
                watertight_mesh = make_double_sided(watertight_mesh)
            scene.add_geometry(watertight_mesh)

            # 显示交线（洋红色圆柱）
            for edge in intersection_edges:
                p0 = intersection_vertices[edge[0]]
                p1 = intersection_vertices[edge[1]]
                seg = trimesh.creation.cylinder(
                    radius=radius * 1.2,
                    segment=[p0, p1],
                    sections=5,
                )
                seg.visual.face_colors = [255, 0, 255, 255]
                scene.add_geometry(seg)

            print(f"  拟合成功：交线 {len(intersection_vertices)} 个顶点，"
                  f"{len(intersection_edges)} 条边")
        else:
            print(f"  [WARN] 水密包络拟合失败: {patch_result['message']}")

    # 端点（绿色球）
    for v in comp.get("endpoints", []):
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius * 2.0)
        sphere.apply_translation(mesh.vertices[v])
        sphere.visual.face_colors = [0, 255, 0, 255]
        scene.add_geometry(sphere)

    # 分支点（红色球）
    for v in comp.get("branch_vertices", []):
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius * 2.0)
        sphere.apply_translation(mesh.vertices[v])
        sphere.visual.face_colors = [255, 0, 0, 255]
        scene.add_geometry(sphere)

    # 候选断裂点对（橙色虚线，用细圆柱表示）
    for cand in comp.get("candidate_breaks", []):
        p0 = mesh.vertices[cand["v0"]]
        p1 = mesh.vertices[cand["v1"]]
        seg = trimesh.creation.cylinder(
            radius=radius * 0.8,
            segment=[p0, p1],
            sections=4,
        )
        seg.visual.face_colors = [255, 165, 0, 255]
        scene.add_geometry(seg)

    # Seifert 曲面
    if getattr(args, 'generate_seifert_surface_strict', False):
        if args.boundary_type != "healthy":
            print("  警告: --generate-seifert-surface-strict 仅适用于 healthy 孔洞")
        else:
            loop = comp.get("healthy_hole_vertex_indices")
            if not loop:
                # 从 edge_vertex_pairs 恢复环
                edge_pairs = comp.get("edge_vertex_pairs", [])
                if edge_pairs:
                    import warnings
                    # 简化恢复：取所有边的顶点并排序？但这里直接用边构建邻接并遍历
                    # 可以省略，因为健康孔洞 JSON 中应已有 vertex_indices
                    print("  [WARN] 未找到 healthy_hole_vertex_indices")
                else:
                    print("  [WARN] 未找到任何边界信息")
                loop = []
            if loop and len(loop) >= 3:
                print("生成严格 Seifert 曲面...")
                disk_mesh, boundary_indices = _generate_initial_seifert_disk(mesh, loop)
                if disk_mesh is None:
                    print("  [WARN] 无法生成初始圆盘")
                else:
                    seifert_mesh = _laplacian_smooth_fixed_boundary(
                        disk_mesh,
                        boundary_indices,
                        iterations=args.seifert_optimize_iterations,
                        step_size=args.seifert_step_size,
                        tol=args.seifert_tolerance,
                    )
                    # 将 Seifert 曲面顶点加入相机核心取景点集
                    if len(camera_core_points) == 0:
                        camera_core_points = np.asarray(
                            seifert_mesh.vertices, dtype=np.float64
                        ).copy()
                    else:
                        camera_core_points = np.vstack([
                            camera_core_points,
                            np.asarray(seifert_mesh.vertices, dtype=np.float64),
                        ])

                    color = np.array(args.seifert_color, dtype=np.uint8)
                    seifert_mesh.visual.face_colors = np.tile(
                        color, (len(seifert_mesh.faces), 1)
                    )
                    if args.double_sided:
                        seifert_mesh = make_double_sided(seifert_mesh)
                    scene.add_geometry(seifert_mesh)
                    print(f"  Seifert 曲面已生成: {len(seifert_mesh.faces)} 个三角面片")
                    if args.seifert_curvature_report:
                        stats = _compute_curvature_statistics(
                            seifert_mesh, boundary_indices
                        )
                        print("  Seifert 曲面曲率统计:")
                        for k, v in stats.items():
                            print(f"    {k}: {v:.6f}")
            else:
                print("  [WARN] 未找到有效的健康孔洞边界环")

    if getattr(args, "debug_scene", False):
        print_scene_debug_info(scene, title="Boundary Component Scene Debug Info")

    if args.output:
        scene.export(args.output)
        print(
            f"边界组件 {args.boundary_id} 可视化已保存至: {args.output}"
        )
    if args.show:
        if len(camera_core_points) > 0:
            try:
                camera_core_points = _filter_camera_core_points(camera_core_points)
                core_pts = np.asarray(camera_core_points, dtype=np.float64)
                core_center = core_pts.mean(axis=0)
                core_radius = float(
                    np.linalg.norm(core_pts - core_center, axis=1).max()
                )
                if core_radius < 1e-8:
                    core_radius = 0.1

                distance = max(core_radius * 5.0, 1e-6)

                if getattr(args, "debug_scene", False):
                    print("  [camera] core point count:", len(core_pts))
                    print(
                        "           center=({:.6f}, {:.6f}, {:.6f})".format(
                            core_center[0], core_center[1], core_center[2]
                        )
                    )
                    print(
                        "           radius={:.6f}, distance={:.6f}".format(
                            core_radius, distance
                        )
                    )

                # 优先使用显式 center/distance；不支持时回退到旧 API
                try:
                    scene.camera.look_at(
                        core_pts,
                        center=core_center,
                        distance=distance,
                    )
                except TypeError:
                    scene.camera.look_at(core_pts)

            except Exception as e:
                print(f"[WARN] 相机自动取景失败: {e}")

        elif camera_focus is not None:
            try:
                scene.camera.look_at(
                    np.array([camera_focus], dtype=np.float64)
                )
            except Exception as e:
                print(f"[WARN] 相机自动取景失败: {e}")

        os.environ['TRIMESH_DEFAULT_VIEWER'] = 'vedo'
        scene.show()


def run_full_diagnosis_pass1(mesh, output_dir, valence_threshold=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Full Diagnosis Pass 1 ===")
    print("分析网格缺陷...")
    _, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    abnormal_mask = open_face_mask | nonmanifold_face_mask
    abnormal_indices = np.where(abnormal_mask)[0]

    print(f"异常面片总数: {len(abnormal_indices)}")

    vertex_face_counts = compute_vertex_face_counts(mesh)
    face_edge_types = compute_face_edge_types(mesh)

    # 计算所有面片的拓扑编码
    all_face_indices = np.arange(len(mesh.faces))
    codes_all, _, _ = compute_face_topology_codes(
        mesh, all_face_indices, vertex_face_counts, face_edge_types
    )
    save_codes(codes_all, output_dir / "face_codes.npy")

    if len(abnormal_indices) == 0:
        # 创建空的分类 JSON 和 checkpoint
        empty_classes = {
            "valence_threshold": valence_threshold,
            "total_abnormal_faces": 0,
            "classes": {}
        }
        with open(output_dir / "abnormal_truncated_classes.json", "w") as f:
            json.dump(empty_classes, f, indent=2)
        checkpoint = {
            "valence_threshold": valence_threshold,
            "classes": {},
            "total_classes": 0,
            "abnormal_count": 0
        }
        with open(output_dir / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
        return {}, abnormal_indices

    # 对异常面片进行截断聚类
    print("截断聚类异常面片...")
    grouped = group_faces_by_topology_codes(
        mesh, abnormal_indices, vertex_face_counts, face_edge_types,
        valence_threshold=valence_threshold
    )

    class_faces = {}
    classes_json = {}
    for key, face_list in grouped.items():
        hex_code = code_to_hex(key)
        class_faces[hex_code] = face_list
        classes_json[hex_code] = {
            "face_indices": face_list.tolist(),
            "count": int(len(face_list)),
            "status": "pending"
        }

    abnormal_data = {
        "valence_threshold": valence_threshold,
        "total_abnormal_faces": int(len(abnormal_indices)),
        "classes": classes_json
    }
    with open(output_dir / "abnormal_truncated_classes.json", "w") as f:
        json.dump(abnormal_data, f, indent=2)

    checkpoint = {
        "valence_threshold": valence_threshold,
        "classes": {hex_code: "pending" for hex_code in class_faces},
        "total_classes": len(class_faces),
        "abnormal_count": int(len(abnormal_indices))
    }
    with open(output_dir / "checkpoint.json", "w") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"发现 {len(class_faces)} 个不同拓扑类别（截断）。")
    return class_faces, abnormal_indices


def run_full_diagnosis_pass2(mesh, output_dir, class_faces,
                             open_face_mask, nonmanifold_face_mask,
                             valence_threshold=5,
                             resume=False):
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    classes_json_path = output_dir / "abnormal_truncated_classes.json"

    if resume and checkpoint_path.exists() and classes_json_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        pending_classes = [hex_code for hex_code, status in checkpoint["classes"].items()
                           if status == "pending"]
        print(f"恢复模式：已完成 {len(checkpoint['classes']) - len(pending_classes)} 类，"
              f"剩余 {len(pending_classes)} 类")
    else:
        pending_classes = list(class_faces.keys())

    if not pending_classes:
        print("没有待分析类别。")
        return {}

    # 预计算共享数据
    vertex_faces_csr = _build_vertex_face_csr(mesh)
    vertex_face_counts = compute_vertex_face_counts(mesh)
    edge_keys, edge_faces = compute_edge_to_faces(mesh)
    edge_to_faces = {int(k): v for k, v in zip(edge_keys, edge_faces)}
    face_edge_keys = compute_face_edge_keys(mesh)
    edge_valences_all = compute_face_edge_valences(mesh, edge_to_faces, face_edge_keys)

    results = {}
    for idx, hex_code in enumerate(pending_classes):
        face_indices = np.asarray(class_faces[hex_code], dtype=np.int64)
        print(f"\n=== 分析类别 {idx+1}/{len(pending_classes)} ===", flush=True)
        print(f"  编码: {hex_code}, 面片数: {len(face_indices)}", flush=True)

        areas = mesh.area_faces[face_indices]
        area_stats = {
            'count': int(len(face_indices)),
            'mean': float(np.mean(areas)),
            'min': float(np.min(areas)),
            'p1': float(np.percentile(areas, 1)),
            'p5': float(np.percentile(areas, 5)),
            'p10': float(np.percentile(areas, 10)),
            'p25': float(np.percentile(areas, 25)),
            'p50': float(np.percentile(areas, 50)),
            'p75': float(np.percentile(areas, 75)),
            'p90': float(np.percentile(areas, 90)),
            'p95': float(np.percentile(areas, 95)),
            'p99': float(np.percentile(areas, 99)),
            'max': float(np.max(areas)),
        }

        point_counts, edge_counts = compute_class_neighbor_stats(
            mesh, face_indices, open_face_mask, nonmanifold_face_mask,
            vertex_faces_csr, edge_to_faces, face_edge_keys
        )

        rep_face = int(face_indices[0])
        vertex_stats, edge_stats = compute_single_face_neighbor_stats(
            mesh, rep_face, open_face_mask, nonmanifold_face_mask,
            vertex_faces_csr, edge_to_faces, face_edge_keys
        )

        _, v_order, e_order = get_face_topology_code_and_order(
            mesh, rep_face, vertex_face_counts, edge_to_faces, face_edge_keys
        )

        aligned_vertex_stats = [vertex_stats[i] for i in v_order]
        aligned_edge_stats = [edge_stats[i] for i in e_order]

        # 截断字段的真实值分布
        verts = mesh.faces[face_indices]                # (k,3)
        v_counts = vertex_face_counts[verts]            # (k,3)
        e_counts = edge_valences_all[face_indices]      # (k,3)

        # 顶点元截断分布：只统计 >= valence_threshold 的字段
        truncated_v_mask = v_counts >= valence_threshold
        truncated_v_vals = v_counts[truncated_v_mask]
        truncated_v_dist = Counter(truncated_v_vals.tolist())

        # 边元截断分布：只统计 >= 3 的字段
        truncated_e_mask = e_counts >= 3
        truncated_e_vals = e_counts[truncated_e_mask]
        truncated_e_dist = Counter(truncated_e_vals.tolist())

        class_result = {
            'class_id': hex_code,
            'code': hex_code,
            'area_stats': area_stats,
            'point_counts': point_counts,
            'edge_counts': edge_counts,
            'representative_vertex_stats': aligned_vertex_stats,
            'representative_edge_stats': aligned_edge_stats,
            'truncated_vertex_valence_dist': dict(sorted(truncated_v_dist.items())),
            'truncated_edge_valence_dist': dict(sorted(truncated_e_dist.items())),
        }
        results[hex_code] = class_result

        # 更新 abnormal_truncated_classes.json
        with open(classes_json_path, "r") as f:
            data = json.load(f)
        if hex_code not in data["classes"]:
            # 兼容恢复时 dict 中可能没有该键
            data["classes"][hex_code] = {
                "face_indices": face_indices.tolist(),
                "count": len(face_indices)
            }
        data["classes"][hex_code]["area_stats"] = area_stats
        data["classes"][hex_code]["point_counts"] = point_counts
        data["classes"][hex_code]["edge_counts"] = edge_counts
        data["classes"][hex_code]["representative_vertex_stats"] = aligned_vertex_stats
        data["classes"][hex_code]["representative_edge_stats"] = aligned_edge_stats
        data["classes"][hex_code]["status"] = "done"

        data["classes"][hex_code]["truncated_vertex_valence_dist"] = \
            dict(sorted(truncated_v_dist.items()))
        data["classes"][hex_code]["truncated_edge_valence_dist"] = \
            dict(sorted(truncated_e_dist.items()))

        with open(classes_json_path, "w") as f:
            json.dump(data, f, indent=2)

        # 更新 checkpoint
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
        if hex_code not in ckpt["classes"]:
            ckpt["classes"][hex_code] = "pending"  # 兼容可能缺失
        ckpt["classes"][hex_code] = "done"
        with open(checkpoint_path, "w") as f:
            json.dump(ckpt, f, indent=2)

        print(f"  面积: 平均={area_stats['mean']:.6f}, p50={area_stats['p50']:.6f}", flush=True)
        print(f"  点邻: 流形={point_counts['normal']}, 开放={point_counts['open']}, "
              f"非流形={point_counts['nonmanifold']}", flush=True)
        print(f"  边邻: 流形={edge_counts['normal']}, 开放={edge_counts['open']}, "
              f"非流形={edge_counts['nonmanifold']}", flush=True)

    return results


def generate_html_report(output_dir, id_to_code, results):
    """生成 HTML 报告，包含图例、交互式 SVG 示意图和详细统计。"""
    output_dir = Path(output_dir)
    html_path = output_dir / "report.html"

    html = ["<html><head><meta charset='utf-8'><title>Full Face Diagnosis</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: center; }",
            "th { background: #f0f0f0; }",
            ".class-block { margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; }",
            "img { max-width: 300px; height: auto; }",
            ".diagram-container { display: inline-block; vertical-align: top; margin-right: 20px; }",
            ".diagram-container svg { width: 200px; height: 170px; }",
            ".legend { margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; background: #fafafa; }",
            ".legend span.legend-dot { display: inline-block; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px; }",
            ".legend span.solid { background: black; border: 1px solid black; }",
            ".legend span.hollow { background: white; border: 2px solid black; }",
            ".legend span.legend-line { display: inline-block; width: 30px; height: 0; border-top: 3px solid; margin-right: 5px; vertical-align: middle; }",
            ".legend span.blue { border-color: blue; }",
            ".legend span.green { border-color: green; }",
            ".legend span.red { border-color: red; }",
            "</style></head><body>",
            "<h1>Full Face Diagnosis Report</h1>"]

    # ---- 图例 ----
    html.append("<div class='legend'>")
    html.append("<strong>图例：</strong><br>")
    html.append("<span class='legend-dot solid'></span> 独占顶点（仅被当前面引用）<br>")
    html.append("<span class='legend-dot hollow'></span> 共享顶点（被多个面引用）<br>")
    html.append("<span class='legend-line blue'></span> 开放边（仅属于当前面）<br>")
    html.append("<span class='legend-line green'></span> 流形边（被两面共享）<br>")
    html.append("<span class='legend-line red'></span> 非流形边（被三面或更多共享）<br>")
    html.append("</div>")

    html.append("<h2>Topology Classes</h2>")

    for class_id in sorted(results.keys()):
        res = results[class_id]
        code = res['code']
        area = res['area_stats']
        pc = res['point_counts']
        ec = res['edge_counts']
        v_stats = res.get('representative_vertex_stats')
        if not v_stats:
            v_stats = [{'normal': 0, 'open': 0, 'nonmanifold': 0} for _ in range(3)]
        e_stats = res.get('representative_edge_stats')
        if not e_stats:
            e_stats = [{'normal': 0, 'open': 0, 'nonmanifold': 0} for _ in range(3)]

        # 内联 SVG 交互图
        diagram_path = output_dir / "diagrams" / f"diagram_{class_id}.svg"
        if diagram_path.exists():
            svg_content = diagram_path.read_text(encoding="utf-8")
        else:
            svg_content = "<svg xmlns='http://www.w3.org/2000/svg' width='200' height='170'><text x='10' y='80'>Diagram not generated</text></svg>"

        html.append("<div class='class-block'>")
        html.append(f"<h3>Class {class_id}: {code}</h3>")
        html.append("<div class='diagram-container'>")
        html.append(svg_content)
        html.append("</div>")
        html.append("<div style='display: inline-block; vertical-align: top;'>")

        # 面积统计表
        html.append("<table>")
        html.append("<tr><th>面积统计</th><th>值</th></tr>")
        rows = [
            ("count", "面片数"),
            ("mean", "平均"),
            ("min", "最小值"),
            ("p1", "p1"),
            ("p5", "p5"),
            ("p10", "p10"),
            ("p25", "p25"),
            ("p50", "p50"),
            ("p75", "p75"),
            ("p90", "p90"),
            ("p95", "p95"),
            ("p99", "p99"),
            ("max", "最大值"),
        ]
        for key, label in rows:
            html.append(f"<tr><td>{label}</td><td>{area[key]:.6f}</td></tr>")
        html.append("</table>")

        # 总体邻接统计（点邻 / 边邻）
        html.append("<table>")
        html.append("<tr><th></th><th>流形</th><th>开放</th><th>非流形</th></tr>")
        html.append("<tr><th>点邻（合计）</th>"
                    f"<td>{pc['normal']}</td><td>{pc['open']}</td><td>{pc['nonmanifold']}</td></tr>")
        html.append("<tr><th>边邻（合计）</th>"
                    f"<td>{ec['normal']}</td><td>{ec['open']}</td><td>{ec['nonmanifold']}</td></tr>")
        html.append("</table>")

        html.append("</div>")  # 关闭右侧容器
        html.append("</div>")  # 关闭 class-block

    html.append("</body></html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"HTML 报告已保存: {html_path}")


def generate_latex_report(output_dir, id_to_code, results):
    """生成 LaTeX 报告（包含更多面积百分位和邻接统计）。"""
    output_dir = Path(output_dir)
    tex_path = output_dir / "report.tex"

    tex = ["\\documentclass{article}",
           "\\usepackage{graphicx}",
           "\\usepackage{booktabs}",
           "\\begin{document}",
           "\\section{Full Face Diagnosis Report}"]

    for class_id in sorted(results.keys()):
        res = results[class_id]
        code = res['code']
        area = res['area_stats']
        pc = res['point_counts']
        ec = res['edge_counts']
        diagram_rel = f"diagrams/diagram_{class_id}.svg"

        tex.append(f"\\subsection{{Class {class_id}: {code}}}")
        tex.append("\\begin{figure}[h]")
        tex.append(f"\\includegraphics[width=0.25\\textwidth]{{{diagram_rel}}}")
        tex.append("\\end{figure}")

        # 面积统计表
        tex.append("\\begin{tabular}{l r}")
        tex.append("\\toprule")
        tex.append("Area Metric & Value \\\\")
        tex.append("\\midrule")
        rows = [
            ("Count", area['count']),
            ("Mean", area['mean']),
            ("Min", area['min']),
            ("p1", area['p1']),
            ("p5", area['p5']),
            ("p10", area['p10']),
            ("p25", area['p25']),
            ("p50", area['p50']),
            ("p75", area['p75']),
            ("p90", area['p90']),
            ("p95", area['p95']),
            ("p99", area['p99']),
            ("Max", area['max']),
        ]
        for label, val in rows:
            tex.append(f"{label} & {val:.6f} \\\\")
        tex.append("\\bottomrule")
        tex.append("\\end{tabular}")

        # 邻接统计表
        tex.append("\\begin{tabular}{l c c c}")
        tex.append("\\toprule")
        tex.append("Neighbor & Manifold & Open & Nonmanifold \\\\")
        tex.append("\\midrule")
        tex.append(f"Point & {pc['normal']} & {pc['open']} & {pc['nonmanifold']} \\\\")
        tex.append(f"Edge & {ec['normal']} & {ec['open']} & {ec['nonmanifold']} \\\\")
        tex.append("\\bottomrule")
        tex.append("\\end{tabular}")

    tex.append("\\end{document}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"LaTeX 报告已保存: {tex_path}")


def generate_html_report_from_json(output_dir):
    """
    从 abnormal_truncated_classes.json 生成 HTML 报告，
    包含每个类的拓扑 SVG 示意图和统计表。
    """
    output_dir = Path(output_dir)
    html_path = output_dir / "report.html"
    classes_json_path = output_dir / "abnormal_truncated_classes.json"
    diagrams_dir = output_dir / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)

    if not classes_json_path.exists():
        print(f"[ERROR] {classes_json_path} not found.")
        return

    with open(classes_json_path, "r") as f:
        data = json.load(f)

    html = ["<html><head><meta charset='utf-8'><title>Full Face Diagnosis</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: center; }",
            "th { background: #f0f0f0; }",
            ".class-block { margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; }",
            "img { max-width: 300px; height: auto; }",
            ".diagram-container { display: inline-block; vertical-align: top; margin-right: 20px; }",
            ".diagram-container svg { width: 200px; height: 170px; }",
            ".legend { margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; background: #fafafa; }",
            ".legend span.legend-dot { display: inline-block; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px; }",
            ".legend span.solid { background: black; border: 1px solid black; }",
            ".legend span.hollow { background: white; border: 2px solid black; }",
            ".legend span.legend-line { display: inline-block; width: 30px; height: 0; border-top: 3px solid; margin-right: 5px; vertical-align: middle; }",
            ".legend span.blue { border-color: blue; }",
            ".legend span.green { border-color: green; }",
            ".legend span.red { border-color: red; }",
            "</style></head><body>",
            "<h1>Full Face Diagnosis Report</h1>"]

    # 图例
    html.append("<div class='legend'>")
    html.append("<strong>图例：</strong><br>")
    html.append("<span class='legend-dot solid'></span> 独占顶点（仅被当前面引用）<br>")
    html.append("<span class='legend-dot hollow'></span> 共享顶点（被多个面引用）<br>")
    html.append("<span class='legend-line blue'></span> 开放边（仅属于当前面）<br>")
    html.append("<span class='legend-line green'></span> 流形边（被两面共享）<br>")
    html.append("<span class='legend-line red'></span> 非流形边（被三面或更多共享）<br>")
    html.append("</div>")

    html.append("<h2>Topology Classes</h2>")

    classes = data.get("classes", {})
    if not classes:
        html.append("<p>No abnormal classes found.</p>")
    else:
        for hex_code, cls_data in classes.items():
            # 生成 SVG 图
            diagram_path = diagrams_dir / f"diagram_{hex_code}.svg"
            try:
                code_arr = hex_to_code(hex_code)
                _generate_topology_diagram(code_arr, str(diagram_path))
            except Exception as e:
                print(f"  [WARN] Diagram generation for {hex_code} failed: {e}")
                diagram_path = None

            svg_content = ""
            if diagram_path and diagram_path.exists():
                svg_content = diagram_path.read_text(encoding="utf-8")
            if not svg_content:
                svg_content = "<svg xmlns='http://www.w3.org/2000/svg' width='200' height='170'><text x='10' y='80'>Diagram not generated</text></svg>"

            html.append("<div class='class-block'>")
            html.append(f"<h3>Class {hex_code}</h3>")
            html.append("<div class='diagram-container'>")
            html.append(svg_content)
            html.append("</div>")
            html.append("<div style='display: inline-block; vertical-align: top;'>")

            # 面积统计表
            area = cls_data.get("area_stats", {})
            if area:
                html.append("<table>")
                html.append("<tr><th>面积统计</th><th>值</th></tr>")
                rows = [
                    ("count", "面片数"),
                    ("mean", "平均"),
                    ("min", "最小值"),
                    ("p1", "p1"),
                    ("p5", "p5"),
                    ("p10", "p10"),
                    ("p25", "p25"),
                    ("p50", "p50"),
                    ("p75", "p75"),
                    ("p90", "p90"),
                    ("p95", "p95"),
                    ("p99", "p99"),
                    ("max", "最大值"),
                ]
                for key, label in rows:
                    if key in area:
                        html.append(f"<tr><td>{label}</td><td>{area[key]:.6f}</td></tr>")
                    else:
                        html.append(f"<tr><td>{label}</td><td>N/A</td></tr>")
                html.append("</table>")

            # 邻接统计
            pc = cls_data.get("point_counts", {})
            ec = cls_data.get("edge_counts", {})
            if pc and ec:
                html.append("<table>")
                html.append("<tr><th></th><th>流形</th><th>开放</th><th>非流形</th></tr>")
                html.append("<tr><th>点邻（合计）</th>"
                            f"<td>{pc.get('normal', 0)}</td><td>{pc.get('open', 0)}</td><td>{pc.get('nonmanifold', 0)}</td></tr>")
                html.append("<tr><th>边邻（合计）</th>"
                            f"<td>{ec.get('normal', 0)}</td><td>{ec.get('open', 0)}</td><td>{ec.get('nonmanifold', 0)}</td></tr>")
                html.append("</table>")

            # 代表面逐顶点/逐边统计（可选展示）
            v_stats = cls_data.get("representative_vertex_stats")
            e_stats = cls_data.get("representative_edge_stats")
            if v_stats:
                html.append("<table>")
                html.append("<tr><th>代表面-顶点</th><th>流形</th><th>开放</th><th>非流形</th></tr>")
                for idx, vs in enumerate(v_stats):
                    html.append(f"<tr><td>V{idx}</td>"
                                f"<td>{vs.get('normal', 0)}</td><td>{vs.get('open', 0)}</td><td>{vs.get('nonmanifold', 0)}</td></tr>")
                html.append("</table>")
            if e_stats:
                html.append("<table>")
                html.append("<tr><th>代表面-边</th><th>流形</th><th>开放</th><th>非流形</th></tr>")
                for idx, es in enumerate(e_stats):
                    html.append(f"<tr><td>E{idx}</td>"
                                f"<td>{es.get('normal', 0)}</td><td>{es.get('open', 0)}</td><td>{es.get('nonmanifold', 0)}</td></tr>")
                html.append("</table>")

            # 截断字段真实分布
            vertex_dist = cls_data.get("truncated_vertex_valence_dist", {})
            edge_dist = cls_data.get("truncated_edge_valence_dist", {})

            if vertex_dist:
                html.append("<table>")
                html.append("<tr><th>顶点元截断分布</th><th>真实 valence</th><th>计数</th></tr>")
                for val_str, cnt in vertex_dist.items():
                    html.append(f"<tr><td>valence</td><td>{val_str}</td><td>{cnt}</td></tr>")
                html.append("</table>")

            if edge_dist:
                html.append("<table>")
                html.append("<tr><th>边元截断分布</th><th>真实共享数</th><th>计数</th></tr>")
                for val_str, cnt in edge_dist.items():
                    html.append(f"<tr><td>edge</td><td>{val_str}</td><td>{cnt}</td></tr>")
                html.append("</table>")

            html.append("</div>")  # 关闭右侧容器
            html.append("</div>")  # 关闭 class-block

    html.append("</body></html>")
    html_path.write_text("\n".join(html), encoding="utf-8")
    print(f"HTML 报告已保存: {html_path}")


def perform_full_diagnosis(mesh, args):
    """
    Full Diagnosis 入口，协调 Pass 1 和 Pass 2。
    """
    output_dir = Path(args.diagnosis_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)

    resume_checkpoint = output_dir / "checkpoint.json"
    classes_json_path = output_dir / "abnormal_truncated_classes.json"

    if args.resume and resume_checkpoint.exists() and classes_json_path.exists():
        with open(classes_json_path, "r") as f:
            classes_data = json.load(f)
        class_faces = {hex_code: np.asarray(entry["face_indices"], dtype=np.int64)
                       for hex_code, entry in classes_data.get("classes", {}).items()}
        print("Resume: loading existing classifications from Pass 1.")
    else:
        class_faces, _ = run_full_diagnosis_pass1(
            mesh, output_dir,
            valence_threshold=args.valence_threshold
        )

    results = run_full_diagnosis_pass2(
        mesh, output_dir, class_faces,
        open_face_mask, nonmanifold_face_mask,
        valence_threshold=args.valence_threshold,
        resume=args.resume
    )

    if args.diagnosis_format == "html":
        generate_html_report_from_json(output_dir)
    else:
        # 新的架构中以 JSON 为单一数据源，暂未适配 LaTeX 生成，这里回退到 HTML
        print("[WARN] LaTeX report generation is not yet adapted to the new architecture.")
        print("Falling back to HTML report generation.")
        generate_html_report_from_json(output_dir)

    # 计算异常面总数（用于 meta；不再单独生成 diagnosis_results.json）
    abnormal_mask = open_face_mask | nonmanifold_face_mask
    abnormal_count = int(abnormal_mask.sum())
    print(f"\nFull Diagnosis 完成，异常面总数：{abnormal_count}，"
          f"报告已生成到 {output_dir}")


def perform_hole_diagnosis(mesh, args):
    """执行孔洞诊断，输出健康孔洞与未覆盖开放边分析。"""
    output_dir = Path(args.hole_diagnosis_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Hole Diagnosis ===")
    print("提取开放边与健康孔洞...")
    hole_data = build_hole_diagnosis_data(mesh)

    # 1. 保存二进制数据
    print("保存开放边与孔洞数据...")
    open_edge_ids = np.arange(len(hole_data['open_edge_face_ids']), dtype=np.int64)
    np.savez_compressed(
        output_dir / "hole_diagnosis_data.npz",
        open_edge_ids=open_edge_ids,
        open_edge_vertex_pairs=hole_data['open_edge_vertex_pairs'],
        open_edge_face_ids=hole_data['open_edge_face_ids'],
        open_edge_keys=hole_data['open_edge_keys'],
        hole_ids_per_edge=hole_data['hole_ids_per_edge'],
        uncovered_edge_ids=hole_data['uncovered_edge_ids'],
        uncovered_category=hole_data['uncovered_category'],
    )

    # 2. 分析未覆盖开放边连通分量（异常孔洞）
    print("分析未覆盖开放边连通分量...")
    components = analyze_uncovered_open_edge_components(mesh, hole_data)
    component_json_path = output_dir / "uncovered_component_analysis.json"
    with open(component_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_components": len(components),
            "components": components,
        }, f, indent=2, ensure_ascii=False)
    print(f"未覆盖开放边分量分析已保存: {component_json_path}")

    # 计算最小包络流形边界（可选）
    if getattr(args, 'compute_enclosing_boundaries', False):
        print("计算最小包络流形边界...")
        for comp in components:
            comp_id = comp['component_id']
            bound = find_minimal_enclosing_manifold_boundary_greedy(mesh, comp)
            comp['minimal_enclosing_boundary'] = bound
            if bound['success']:
                print(f"  Component {comp_id}: 包络边界深度 {bound['depth']}, "
                      f"内部面片 {len(bound['enclosed_faces'])}, "
                      f"边界边 {len(bound['boundary_edges'])}")
            else:
                print(f"  Component {comp_id}: 未找到包络边界 "
                      f"(最大深度 {bound['depth']})")

        # 更新 JSON 文件
        with open(component_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_components": len(components),
                "components": components,
            }, f, indent=2, ensure_ascii=False)
        print(f"已更新包含包络边界信息的: {component_json_path}")

    # 3. 构造 JSON 报告
    print("生成孔洞诊断 JSON...")
    total_open_edges = len(open_edge_ids)
    total_holes = len(hole_data['hole_vertex_lists'])
    uncovered_count = len(hole_data['uncovered_edge_ids'])
    covered_count = total_open_edges - uncovered_count

    hole_info_list = []
    for hole_id, (vert_list, edge_list) in enumerate(zip(
        hole_data['hole_vertex_lists'],
        hole_data['hole_edge_lists']
    )):
        # 计算周长和面积
        pts = mesh.vertices[np.asarray(vert_list, dtype=np.int64)]
        # 面积调用 polygon_area_from_3d_ccw
        area = polygon_area_from_3d_ccw(pts)
        perimeter = float(np.sum(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)))
        hole_info_list.append({
            "hole_id": hole_id,
            "num_edges": len(edge_list),
            "num_vertices": len(vert_list),
            "area": area,
            "perimeter": perimeter,
            "vertex_indices": vert_list,
        })

    # 未覆盖开放边分类统计
    category_counts = Counter(hole_data['uncovered_category'].tolist())
    category_names = {
        0: "孤立开放链",
        1: "悬空开放边",
        2: "分支内部开放边",
        4: "非流形关联开放边",
        5: "其他复杂开放边",
    }
    uncovered_categories_summary = {
        category_names.get(int(cat), f"cat_{cat}"): int(cnt)
        for cat, cnt in category_counts.items()
    }

    diagnosis_json = {
        "total_open_edges": total_open_edges,
        "total_healthy_holes": total_holes,
        "covered_open_edges": covered_count,
        "uncovered_open_edges": uncovered_count,
        "healthy_holes": hole_info_list,
        "uncovered_categories_summary": uncovered_categories_summary,
        "open_edge_face_ids": hole_data['open_edge_face_ids'].tolist(),  # 可用于关联面片
    }
    with open(output_dir / "hole_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(diagnosis_json, f, indent=2, ensure_ascii=False)

    # 4. 生成 HTML 报告
    print("生成孔洞诊断 HTML...")
    html_path = output_dir / "hole_report.html"
    html = ["<html><head><meta charset='utf-8'><title>Hole Diagnosis</title>",
            "<style>body{font-family:sans-serif;margin:20px}",
            "table{border-collapse:collapse;width:100%;margin-bottom:20px}",
            "th,td{border:1px solid #ccc;padding:4px 8px;text-align:center}",
            "th{background:#f0f0f0}",
            ".component-block{margin-bottom:20px;border:1px solid #ddd;padding:10px}",
            "</style></head><body>",
            "<h1>Hole Diagnosis Report</h1>",
            f"<p>总开放边: {total_open_edges}</p>",
            f"<p>健康孔洞数: {total_holes}</p>",
            f"<p>覆盖开放边: {covered_count}</p>",
            f"<p>未覆盖开放边: {uncovered_count}</p>",
            "<h2>未覆盖开放边分类</h2>",
            "<table><tr><th>分类</th><th>数量</th></tr>"]
    for cat, cnt in uncovered_categories_summary.items():
        html.append(f"<tr><td>{cat}</td><td>{cnt}</td></tr>")
    html.append("</table>")
    html.append("<h2>健康孔洞概览</h2>")
    html.append("<table><tr><th>孔洞ID</th><th>边数</th><th>面积</th><th>周长</th></tr>")
    for hole in hole_info_list:
        html.append(f"<tr><td>{hole['hole_id']}</td><td>{hole['num_edges']}</td>"
                    f"<td>{hole['area']:.6f}</td><td>{hole['perimeter']:.6f}</td></tr>")
    html.append("</table>")
    html.append("<h2>未覆盖开放边组件分析</h2>")
    html.append("<table><tr><th>组件ID</th><th>边数</th><th>端点</th><th>分支点</th>"
                "<th>断裂候选</th></tr>")
    for comp in components:
        html.append(f"<tr><td>{comp['component_id']}</td><td>{comp['num_edges']}</td>"
                    f"<td>{len(comp['endpoints'])}</td><td>{len(comp['branch_vertices'])}</td>"
                    f"<td>{len(comp['candidate_breaks'])}</td></tr>")
    html.append("</table>")

    # 生成并嵌入每个组件的 3D 图
    component_diagrams_dir = output_dir / "component_diagrams"
    component_diagrams_dir.mkdir(exist_ok=True)

    html.append("<h2>未覆盖开放边组件三维视图</h2>")
    for comp in components:
        comp_id = comp['component_id']
        diagram_path = component_diagrams_dir / f"component_{comp_id}.svg"
        try:
            _generate_component_3d_diagram(comp, mesh, str(diagram_path))
        except Exception as e:
            print(f"  [WARN] Component {comp_id} 3D diagram failed: {e}")
            diagram_path = None

        if diagram_path and diagram_path.exists():
            svg_content = diagram_path.read_text(encoding="utf-8")
        else:
            svg_content = "<svg xmlns='http://www.w3.org/2000/svg' width='200' height='170'><text x='10' y='80'>Diagram not generated</text></svg>"

        html.append("<div class='component-block'>")
        html.append(f"<h3>组件 {comp_id}（边数 {comp['num_edges']}）</h3>")
        html.append("<div class='diagram-container'>")
        html.append(svg_content)
        html.append("</div>")
        html.append("<p>")
        html.append(f"顶点数: {comp['num_vertices']} | 端点: {len(comp['endpoints'])} | "
                    f"分支点: {len(comp['branch_vertices'])} | 断裂候选: {len(comp['candidate_breaks'])}")
        html.append("</p>")
        html.append("</div>")

    html.append("</body></html>")
    html_path.write_text("\n".join(html), encoding="utf-8")
    print(f"孔洞诊断 HTML 报告已保存: {html_path}")

    print(f"\nHole Diagnosis 完成，健康孔洞：{total_holes}，未覆盖开放边：{uncovered_count}，"
          f"报告已生成到 {output_dir}")


def print_separator(title=None):
    if title:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")


def inspect_mesh(mesh, args):
    """
    主检查函数：输出统计信息并可选返回可视化场景。
    """
    if args.weld_small_holes:
        print_separator("Weld Small Holes")
        mesh = weld_small_holes(
            mesh,
            threshold=args.weld_hole_threshold,
            quantile=args.weld_hole_quantile,
            min_edges=args.weld_min_hole_edges,
            verbose=True,
        )

    stats = compute_mesh_stats(mesh)
    (
        defect_stats,
        open_face_mask,
        nonmanifold_face_mask,
        open_edge_per_face,
        manifold_edge_per_face,
        nonmanifold_edge_per_face,
    ) = analyze_mesh_defects(mesh, return_face_edge_counts=True)
    area_stats = compute_face_area_stats(mesh)
    bbox_stats = compute_bounding_box_stats(mesh)
    volume = compute_volume_if_closed(mesh)

    defect_mask = open_face_mask | nonmanifold_face_mask
    if np.any(defect_mask):
        distances = compute_face_distances(mesh, defect_mask)
        reliable_mask = (~defect_mask) & (distances >= args.reliable_distance)
    else:
        distances = np.full(len(mesh.faces), args.reliable_distance + 1, dtype=np.int32)
        reliable_mask = np.ones(len(mesh.faces), dtype=bool)

    print_separator("Mesh Topology")
    print(f"  vertices:          {stats['vertices']}")
    print(f"  faces:             {stats['faces']}")
    print(f"  edges (unique):    {stats['edges']}")
    print(f"  watertight:        {stats['is_watertight']}")
    print(f"  open edges:        {defect_stats['open_edges']}")
    print(f"  nonmanifold edges: {defect_stats['nonmanifold_edges']}")
    print(f"  open faces:        {defect_stats['open_faces']}")
    print(f"  nonmanifold faces: {defect_stats['nonmanifold_faces']}")

    hole_stats = compute_hole_area_stats(mesh)
    print(f"  open boundary loops: {hole_stats['count']}")
    print(f"  total hole area:     {hole_stats['total_area']:.6f}")
    if hole_stats['count'] > 0:
        print(f"  hole area percentiles: "
              f"p1={hole_stats['p1_area']:.6f}, "
              f"p5={hole_stats['p5_area']:.6f}, "
              f"p25={hole_stats['p25_area']:.6f}, "
              f"p50={hole_stats['p50_area']:.6f}, "
              f"p75={hole_stats['p75_area']:.6f}, "
              f"p90={hole_stats['p90_area']:.6f}, "
              f"p95={hole_stats['p95_area']:.6f}, "
              f"p99={hole_stats['p99_area']:.6f}, "
              f"max={hole_stats['max_area']:.6f}")

    print_separator("Defect Face Details")
    both_mask = open_face_mask & nonmanifold_face_mask
    open_only_mask = open_face_mask & ~nonmanifold_face_mask
    nonmanifold_only_mask = nonmanifold_face_mask & ~open_face_mask
    print(f"  open-only faces:                 {open_only_mask.sum()}")
    print(f"  nonmanifold-only faces:          {nonmanifold_only_mask.sum()}")
    print(f"  both open & nonmanifold faces:   {both_mask.sum()}")

    print_separator("Open Face Diagnostics")

    open_face_areas = mesh.area_faces[open_face_mask]
    if len(open_face_areas) == 0:
        print("  no open faces")
    else:
        print(f"  open face count: {len(open_face_areas)}")

        print("  face area percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"    p{p}: {np.percentile(open_face_areas, p):.6f}")
        print(f"    mean: {open_face_areas.mean():.6f}")
        print(f"    min:  {open_face_areas.min():.6f}")
        print(f"    max:  {open_face_areas.max():.6f}")

        # 开放边数量分布（每个开放面片有几条边是开放边）
        open_edge_hist = np.bincount(open_edge_per_face[open_face_mask])
        print("  open-edge count per open face:")
        for k, cnt in enumerate(open_edge_hist):
            if cnt > 0:
                print(f"    {k} open edge(s): {cnt} faces")

        # 流形边数量分布
        manifold_edge_hist = np.bincount(manifold_edge_per_face[open_face_mask])
        print("  manifold-edge count per open face:")
        for k, cnt in enumerate(manifold_edge_hist):
            if cnt > 0:
                print(f"    {k} manifold edge(s): {cnt} faces")

        # 非流形边数量分布
        nonmanifold_edge_hist = np.bincount(nonmanifold_edge_per_face[open_face_mask])
        if nonmanifold_edge_hist.size > 1:
            print("  nonmanifold-edge count per open face:")
            for k, cnt in enumerate(nonmanifold_edge_hist):
                if cnt > 0:
                    print(f"    {k} nonmanifold edge(s): {cnt} faces")

    print_separator("Topological Reliability Distribution")
    max_display = 5
    for d in range(max_display + 1):
        cnt = int(np.sum(distances == d))
        print(f"  distance {d}: {cnt} faces")
    rest = int(np.sum(distances > max_display))
    print(f"  distance > {max_display}: {rest} faces")

    reliable_count = int(reliable_mask.sum())
    total_faces = len(mesh.faces)
    pct = 100.0 * reliable_count / max(total_faces, 1)
    print(f"  reliable faces (distance >= {args.reliable_distance}): "
          f"{reliable_count} ({pct:.2f}%)")

    print_separator("Bounding Box")
    print(f"  min:      [{bbox_stats['min'][0]:.4f}, "
          f"{bbox_stats['min'][1]:.4f}, {bbox_stats['min'][2]:.4f}]")
    print(f"  max:      [{bbox_stats['max'][0]:.4f}, "
          f"{bbox_stats['max'][1]:.4f}, {bbox_stats['max'][2]:.4f}]")
    print(f"  extents:  [{bbox_stats['extents'][0]:.4f}, "
          f"{bbox_stats['extents'][1]:.4f}, {bbox_stats['extents'][2]:.4f}]")
    print(f"  diagonal: {bbox_stats['diagonal']:.4f}")
    print(f"  centroid: [{bbox_stats['centroid'][0]:.4f}, "
          f"{bbox_stats['centroid'][1]:.4f}, {bbox_stats['centroid'][2]:.4f}]")

    print_separator("Edge Length Statistics")
    print(f"  mean: {stats['mean_edge_length']:.6f}")
    print(f"  p1:   {stats['edge_length_p1']:.6f}")
    print(f"  p5:   {stats['edge_length_p5']:.6f}")
    print(f"  p50:  {stats['edge_length_p50']:.6f}")
    print(f"  p95:  {stats['edge_length_p95']:.6f}")
    print(f"  p99:  {stats['edge_length_p99']:.6f}")

    print_separator("Face Area Statistics")
    print(f"  count: {area_stats['count']}")
    print(f"  mean:  {area_stats['mean']:.6f}")
    print(f"  min:   {area_stats['min']:.6f}")
    print(f"  p1:    {area_stats['p1']:.6f}")
    print(f"  p5:    {area_stats['p5']:.6f}")
    print(f"  p10:   {area_stats['p10']:.6f}")
    print(f"  p25:   {area_stats['p25']:.6f}")
    print(f"  p50:   {area_stats['p50']:.6f}")
    print(f"  p75:   {area_stats['p75']:.6f}")
    print(f"  p90:   {area_stats['p90']:.6f}")
    print(f"  p95:   {area_stats['p95']:.6f}")
    print(f"  p99:   {area_stats['p99']:.6f}")
    print(f"  max:   {area_stats['max']:.6f}")

    print_separator("Surface & Volume")
    print(f"  total surface area: {mesh.area:.6f}")
    if np.isfinite(volume):
        print(f"  volume (watertight): {volume:.6f}")
    else:
        print(f"  volume: N/A (mesh is not watertight)")

    # 可视化
    scene = None

    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        if args.show or args.output:
            print("  Mesh is empty; visualization skipped.")
        return scene

    if args.show or args.output:
        # 提前加载代理网格，供叠加显示、投影和透明度控制使用
        proxy_mesh = load_proxy_mesh(args) if args.overlay_proxy else None

        if args.keep_reliable_only:
            print_separator("Reliable-Only Extracted Mesh")
            print(f"  min_distance: {args.reliable_distance}")

            # 使用之前计算好的可靠面片掩码
            reliable_count = int(reliable_mask.sum())
            print(f"  reliable faces: {reliable_count}/{len(mesh.faces)}")

            if reliable_count == 0:
                print("  No reliable faces selected; skipping extraction.")
                return None

            # 提取可靠面片子网格
            reliable_faces = np.asarray(mesh.faces, dtype=np.int64)[reliable_mask]
            flat_faces = reliable_faces.ravel()
            unique_verts, inverse = np.unique(flat_faces, return_inverse=True)
            reliable_mesh = trimesh.Trimesh(
                vertices=mesh.vertices[unique_verts],
                faces=inverse.reshape(-1, 3),
                process=False,
            )
            # 清理提取后的网格
            reliable_mesh.remove_unreferenced_vertices()
            reliable_mesh.merge_vertices()
            reliable_mesh = repair_mesh_by_removing_duplicates(reliable_mesh)

            if args.weld_small_holes:
                print_separator("Weld Small Holes (Reliable Mesh)")
                reliable_mesh = weld_small_holes(
                    reliable_mesh,
                    threshold=args.weld_hole_threshold,
                    quantile=args.weld_hole_quantile,
                    min_edges=args.weld_min_hole_edges,
                    verbose=True,
                )

            # 打印提取后的简要统计
            extracted_stats = compute_mesh_stats(reliable_mesh)
            extracted_defects, _, _ = analyze_mesh_defects(reliable_mesh)
            print(f"  extracted vertices: {extracted_stats['vertices']}")
            print(f"  extracted faces:    {extracted_stats['faces']}")
            print(f"  extracted open edges:        {extracted_defects['open_edges']}")
            print(f"  extracted nonmanifold edges: {extracted_defects['nonmanifold_edges']}")

            # 导出或显示
            if args.output:
                reliable_mesh.export(args.output)
                print(f"\nReliable-only mesh saved to: {args.output}")

            # 构造用于显示的双面网格，并设置输入网格透明度
            if args.double_sided:
                display_mesh = make_double_sided(
                    reliable_mesh,
                    backface_color=args.backface_color,
                )
            else:
                display_mesh = reliable_mesh

            if args.input_alpha < 1.0:
                set_face_alpha(display_mesh, args.input_alpha)

            scene = trimesh.Scene(display_mesh)

            # 若用户要求线框，可叠加在提取网格上
            if args.wireframe:
                add_wireframe_to_scene(
                    scene, reliable_mesh,
                    color=args.wireframe_color,
                    radius=args.wireframe_radius
                )

            # 若用户要求高亮孔洞，也可以基于提取网格显示
            if args.highlight_holes:
                add_hole_boundaries_to_scene(
                    scene, reliable_mesh,
                    radius=args.hole_radius,
                    min_edges=args.min_hole_edges,
                    min_area=args.min_hole_area,
                    verbose=True,
                )

            # 孔洞边界投影到代理网格
            if args.boundary_projection and proxy_mesh is not None:
                add_boundary_projection_to_scene(
                    scene, reliable_mesh, proxy_mesh,
                    radius=args.boundary_projection_radius,
                    verbose=True,
                    print_enclosed_vertices=args.print_shell_enclosed_vertices,
                    max_report_loops=args.max_report_boundary_loops,
                    max_report_vertices=args.max_report_shell_vertices,
                )

            # 最后叠加代理网格
            if proxy_mesh is not None:
                add_proxy_overlay_to_scene(scene, args, proxy_mesh)

            return scene

        if args.highlight_reliable:
            print_separator("Topological Reliable Visualization")
            print(f"  min_distance: {args.reliable_distance}")
            print("  green:  reliable faces (distance >= min_distance)")
            print("  yellow: intermediate faces (0 < distance < min_distance)")
            print("  red:    defect faces (distance = 0)")

            vis = build_reliable_visualization(mesh, distances, args.reliable_distance)
        else:
            print_separator("Defect Visualization")
            print("  gray:    normal faces")
            print("  yellow:  faces adjacent to open edges")
            print("  red:     faces adjacent to non-manifold edges")
            print("  orange:  faces with both defects")

            vis = build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask)

        # 默认双面渲染：薄壳从任何一侧观察都可见
        if args.double_sided:
            vis = make_double_sided(vis, args.backface_color)

        if args.input_alpha < 1.0:
            set_face_alpha(vis, args.input_alpha)

        scene = trimesh.Scene(vis)

        # 导出着色网格
        if args.output:
            vis.export(args.output)
            if args.highlight_reliable:
                print(f"\nReliable-neighborhood mesh saved to: {args.output}")
            else:
                print(f"\nColored defect mesh saved to: {args.output}")

        # 再叠加线框到场景用于可视化
        if args.wireframe:
            add_wireframe_to_scene(
                scene, vis,
                color=args.wireframe_color,
                radius=args.wireframe_radius
            )

        # 高亮显示闭合孔洞边界
        if args.highlight_holes:
            add_hole_boundaries_to_scene(
                scene, mesh,
                radius=args.hole_radius,
                min_edges=args.min_hole_edges,
                min_area=args.min_hole_area,
                verbose=True,
            )

        # 孔洞边界投影到代理网格
        if args.boundary_projection and proxy_mesh is not None:
            add_boundary_projection_to_scene(
                scene, mesh, proxy_mesh,
                radius=args.boundary_projection_radius,
                verbose=True,
                print_enclosed_vertices=args.print_shell_enclosed_vertices,
                max_report_loops=args.max_report_boundary_loops,
                max_report_vertices=args.max_report_shell_vertices,
            )

    if scene is not None and proxy_mesh is not None:
        add_proxy_overlay_to_scene(scene, args, proxy_mesh)

    if args.highlight_uncovered_edges and scene is not None:
        add_uncovered_edges_to_scene(
            scene, mesh,
            data_dir=args.uncovered_data_dir,
            radius=args.uncovered_radius,
            verbose=True,
        )

    return scene


def _parse_color_string(s):
    """
    将 'R,G,B,A' 字符串解析为整数列表。
    """
    parts = s.split(',')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Color must be 'R,G,B,A', got '{s}'"
        )
    try:
        return [int(p.strip()) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Color components must be integers, got '{s}'"
        )


def _parse_color_string_flexible(s):
    """将 'R,G,B' 或 'R,G,B,A' 字符串解析为整数列表。"""
    parts = s.split(',')
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"Color must be 'R,G,B' or 'R,G,B,A', got '{s}'"
        )
    try:
        return [int(p.strip()) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Color components must be integers, got '{s}'"
        )


def main():
    parser = argparse.ArgumentParser(
        description="检查并可视化网格模型，输出拓扑、边长、面积等统计信息。"
    )
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("-o", "--output",
                        help="输出带缺陷着色的网格文件路径 (可选)")
    parser.add_argument("--show", action="store_true",
                        help="显示可视化窗口")
    parser.add_argument("--wireframe", action="store_true",
                        help="在可视化中叠加黑色线框，观察三角剖分")
    parser.add_argument("--wireframe-radius", type=float, default=None,
                        help="线框圆柱半径（默认按包围盒自动计算）")
    parser.add_argument("--wireframe-color", type=str, default="0,0,0,255",
                        help="线框 RGBA 颜色，默认 '0,0,0,255'")
    parser.add_argument("--highlight-holes", action="store_true",
                        help="高亮显示闭合孔洞边界，相邻孔洞使用不同高饱和度颜色")
    parser.add_argument("--highlight-uncovered-edges", action="store_true",
                        help="在 --show 或 --output 时高亮 hole diagnosis 中未覆盖的开放边")
    parser.add_argument("--uncovered-data-dir", type=str,
                        default="hole_diagnosis_report",
                        help="hole diagnosis 输出目录，用于加载未覆盖开放边数据 "
                             "（默认 hole_diagnosis_report）")
    parser.add_argument("--uncovered-radius", type=float, default=None,
                        help="未覆盖开放边圆柱半径（默认自动计算，略粗于普通线框）")
    parser.add_argument("--highlight-reliable", action="store_true",
                        help="高亮显示可靠邻域（绿色=可靠，黄色=缺陷邻域，红色=不可靠）")
    parser.add_argument("--keep-reliable-only", action="store_true",
                        help="只保留可靠面片，删除其余面片。"
                             "需配合 --output 或 --show 使用。")
    parser.add_argument("--reliable-threshold", type=float, default=None,
                        help="[已废弃] 请使用 --reliable-distance。"
                             "该参数不再生效，仅保留兼容。")
    parser.add_argument("--reliable-distance", type=int, default=2,
                        help="可靠面片距离开放/非流形面的最小拓扑距离"
                             "（面邻接跳数），默认 2。")
    parser.add_argument("--hole-radius", type=float, default=None,
                        help="孔洞边界圆柱半径（默认自动计算，通常比普通线框略粗）")
    parser.add_argument("--double-sided", dest='double_sided',
                        action='store_true', default=True,
                        help="双面渲染薄壳网格（默认开启）")
    parser.add_argument("--no-double-sided", dest='double_sided',
                        action='store_false',
                        help="关闭双面渲染，恢复默认背面剔除")
    parser.add_argument("--backface-color", type=str, default=None,
                        help="双面渲染时背面子颜色，格式 'R,G,B,A'。"
                             "默认自动根据正面颜色生成同色系暗色")
    parser.add_argument("--min-hole-edges", type=int, default=3,
                        help="高亮孔洞的最小边界边数（默认 3）")
    parser.add_argument("--min-hole-area", type=float, default=0.0,
                        help="高亮孔洞的最小面积（默认 0，不过滤）")
    parser.add_argument("--weld-small-holes", action="store_true",
                        help="自动焊接面积小于阈值的小孔洞，适用于扫描去重后的伪孔洞")
    parser.add_argument("--weld-hole-threshold", type=float, default=None,
                        help="焊接孔洞的绝对面积阈值；默认使用面片面积百分位")
    parser.add_argument("--weld-hole-quantile", type=float, default=5.0,
                        help="用于计算焊接阈值的面片面积百分位，默认 5")
    parser.add_argument("--weld-min-hole-edges", type=int, default=3,
                        help="焊接孔洞的最小边数，默认 3")
    parser.add_argument("--overlay-proxy", type=str, default=None,
                        help="同时显示输入的代理网格文件（例如体素壳），路径为 STL/PLY/OBJ")
    parser.add_argument("--input-alpha", type=float, default=1.0,
                        help="输入网格（或可靠子网格）不透明度，范围 0~1，"
                             "默认 1.0。与 --overlay-proxy 配合使用时，"
                             "设置为 0.3~0.6 效果较好")
    parser.add_argument("--boundary-projection", action="store_true",
                        help="将输入网格（或可靠子网格）的孔洞边界投影到代理网格表面显示。"
                             "仅在 --overlay-proxy 且 --highlight-holes 或 --keep-reliable-only 时有效")
    parser.add_argument("--boundary-projection-radius", type=float, default=None,
                        help="边界投影线圆柱半径（默认使用与孔洞边界相同的半径）")
    parser.add_argument(
        "--print-shell-enclosed-vertices",
        action="store_true",
        help="在 --boundary-projection 模式下，打印投影孔洞多边形及被包围的代理网格顶点信息"
    )
    parser.add_argument(
        "--max-report-boundary-loops",
        type=int,
        default=None,
        help="最多打印多少个孔洞边界的投影诊断信息；默认全部"
    )
    parser.add_argument(
        "--max-report-shell-vertices",
        type=int,
        default=20,
        help="每个孔洞最多列出多少个被包围的代理网格顶点索引；默认 20"
    )
    parser.add_argument("--proxy-alpha", type=float, default=0.45,
                        help="代理网格透明度，0=全透明，1=不透明，默认 0.45")
    parser.add_argument("--proxy-color", type=str, default=None,
                        help="代理网格统一颜色，格式 'R,G,B' 或 'R,G,B,A'。"
                             "默认使用浅蓝灰色半透明 [128, 180, 255, alpha]")
    parser.add_argument("--proxy-double-sided", dest='proxy_double_sided',
                        action='store_true', default=True,
                        help="双面渲染代理网格（默认开启）")
    parser.add_argument("--no-proxy-double-sided", dest='proxy_double_sided',
                        action='store_false',
                        help="关闭代理网格双面渲染")
    parser.add_argument("--full-diagnosis", action="store_true",
                        help="执行 Full Diagnosis（2-Pass），生成异常面拓扑分类报告")
    parser.add_argument("--hole-diagnosis", action="store_true",
                        help="执行孔洞诊断（健康孔洞提取及未覆盖开放边分类）")
    parser.add_argument("--compute-enclosing-boundaries", action="store_true",
                        help="在 hole diagnosis 中计算每个未覆盖开放边组件的最小包络流形边界")
    # 新增：局部边界组件可视化参数（在 --hole-diagnosis 后插入）
    parser.add_argument("--visualize-boundary-component", action="store_true",
                        help="可视化特定孔洞/开放边分量及其三角面片")
    parser.add_argument("--boundary-type", choices=["uncovered", "healthy"],
                        default="uncovered",
                        help="要可视化的边界类型：uncovered（未覆盖开放边分量）或 healthy（健康孔洞）")
    parser.add_argument("--boundary-id", type=int, default=0,
                        help="边界组件或孔洞的 ID（默认 0）")
    parser.add_argument("--boundary-data-dir", type=str, default="hole_diagnosis_report",
                        help="hole diagnosis 输出目录（默认 hole_diagnosis_report）")
    parser.add_argument("--boundary-neighborhood-depth", type=int, default=1,
                        help="边界邻域深度：0 仅边界，1 边界所在面片，"
                             "2 边界面片+直接邻居，依此类推（默认 1）")
    parser.add_argument("--boundary-radius", type=float, default=None,
                        help="边界圆柱半径（默认自动计算）")
    parser.add_argument("--boundary-show-original", action="store_true",
                        help="同时显示原始网格（半透明背景）")
    parser.add_argument("--debug-scene", action="store_true",
                        help="在显示或导出边界组件场景前，打印场景内所有几何对象的位置与大小")
    parser.add_argument("--fit-watertight-patch", action="store_true",
                        help="在可视化边界组件时，拟合亏格0水密曲面并显示包络交线")
    parser.add_argument("--patch-method",
                        choices=["poisson", "convex_hull", "concave_hull"],
                        default="poisson",
                        help="水密包络曲面生成算法（默认 poisson）")
    parser.add_argument("--patch-neighborhood-depth", type=int, default=2,
                        help="点云提取的邻域深度（默认 2）")
    parser.add_argument("--patch-poisson-depth", type=int, default=8,
                        help="泊松重建深度（默认 8）")
    parser.add_argument("--patch-density-quantile", type=float, default=0.2,
                        help="泊松密度过滤分位（默认 0.2）")
    parser.add_argument("--patch-alpha", type=float, default=1.5,
                        help="凹包算法的 alpha 参数（默认 1.5）")
    parser.add_argument("--patch-opacity", type=float, default=0.3,
                        help="拟合水密曲面的不透明度，范围 0~1，默认 0.3")
    parser.add_argument("--allow-non-genus0", action="store_true",
                        help="允许水密但亏格非0的拟合曲面通过（用于可视化调试）")
    parser.add_argument(
        "--generate-seifert-surface-strict",
        action="store_true",
        help="在可视化健康孔洞时，生成并显示严格 Seifert 曲面（固定边界极小曲面优化）"
    )
    parser.add_argument(
        "--seifert-optimize-iterations",
        type=int,
        default=200,
        help="Seifert 曲面内部顶点优化迭代次数（默认 200）"
    )
    parser.add_argument(
        "--seifert-step-size",
        type=float,
        default=1.0,
        help="Seifert 曲面 Dirichlet 求解松弛系数（默认 1.0）"
    )
    parser.add_argument(
        "--seifert-tolerance",
        type=float,
        default=1e-7,
        help="Seifert 曲面优化收敛容差（默认 1e-7）"
    )
    parser.add_argument(
        "--seifert-color",
        type=str,
        default="255,215,0,255",
        help="Seifert 曲面显示颜色，格式 'R,G,B,A'，默认金色"
    )
    parser.add_argument(
        "--seifert-curvature-report",
        action="store_true",
        help="打印 Seifert 曲面曲率统计信息"
    )
    parser.add_argument("--hole-diagnosis-output", type=str, default="hole_diagnosis_report",
                        help="孔洞诊断输出目录（默认 hole_diagnosis_report）")
    parser.add_argument("--diagnosis-output", type=str, default="diagnosis_report",
                        help="Full Diagnosis 输出目录（默认 diagnosis_report）")
    parser.add_argument("--diagnosis-format", type=str, default="html",
                        choices=["html", "latex"],
                        help="报告格式：html 或 latex（默认 html）")
    parser.add_argument("--resume", action="store_true",
                        help="从现有检查点继续 Full Diagnosis Pass 2（跳过已完成类别）")
    parser.add_argument("--valence-threshold",
                        type=int,
                        default=5,
                        help="顶点元 valence 截断阈值，默认 5。"
                             "当顶点被引用的面片数 >= 阈值时，截断编码归并为该阈值。")

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.reliable_threshold is not None:
        print("[WARNING] --reliable-threshold is deprecated and ignored. "
              "Use --reliable-distance instead.")

    args.wireframe_color = _parse_color_string(args.wireframe_color)

    if args.backface_color is not None:
        args.backface_color = _parse_color_string(args.backface_color)

    if args.proxy_color is not None:
        args.proxy_color = _parse_color_string_flexible(args.proxy_color)

    if args.seifert_color is not None:
        args.seifert_color = _parse_color_string(args.seifert_color)

    print(f"Hey! Loading {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

    # 新增：局部边界组件可视化入口
    if args.visualize_boundary_component:
        visualize_boundary_component(mesh, args)
        return

    if args.hole_diagnosis:
        perform_hole_diagnosis(mesh, args)
        return

    if args.full_diagnosis:
        perform_full_diagnosis(mesh, args)
        return

    scene = inspect_mesh(mesh, args)

    if args.show and scene is not None:
        try:
            os.environ['TRIMESH_DEFAULT_VIEWER'] = 'vedo'
            scene.show()
        except Exception as e:
            print(f"\n[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
