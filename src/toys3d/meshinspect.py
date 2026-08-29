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
from collections import deque
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
    编码格式：A, AB, B, BC, C, CA，每个元素为 '1','2','3'。
    顶点：'1' 实心，'2' 或更大空心。
    边：'1' 蓝色，'2' 绿色，'3' 红色。
    """
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=120)  # 增大尺寸提高清晰度
    pts = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (0.5, np.sqrt(3) / 2)
    }
    vA, eAB, vB, eBC, vC, eCA = [c for c in code]

    edge_styles = {
        '1': ('blue', 'solid'),
        '2': ('green', 'solid'),
        '3': ('red', 'solid')
    }
    for (p1, p2, ecode) in [
        (pts['A'], pts['B'], eAB),
        (pts['B'], pts['C'], eBC),
        (pts['C'], pts['A'], eCA)
    ]:
        color, ls = edge_styles[ecode]
        line = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                      color=color, linewidth=2, linestyle=ls)
        ax.add_line(line)

    for (pt, vcode) in [(pts['A'], vA), (pts['B'], vB), (pts['C'], vC)]:
        fill = (vcode == '1')
        circle = Circle(pt, radius=0.05, fill=fill,
                        color='black', linewidth=2)
        ax.add_patch(circle)

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 0.7)
    ax.set_aspect('equal')
    ax.axis('off')
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


def _compute_class_neighbor_stats(mesh, face_indices,
                                  open_face_mask, nonmanifold_face_mask,
                                  vertex_faces_csr, face_adjacency):
    """
    计算指定面片集合的点邻和边邻面类型计数。
    返回 (point_counts, edge_counts) 各为 dict。
    """
    face_set = set(face_indices)

    point_counts = {'normal': 0, 'open': 0, 'nonmanifold': 0}
    edge_counts = {'normal': 0, 'open': 0, 'nonmanifold': 0}

    for fid in face_indices:
        # 边邻
        row_start = face_adjacency.indptr[fid]
        row_end = face_adjacency.indptr[fid + 1]
        edge_neighbors = face_adjacency.indices[row_start:row_end]

        edge_neighbor_set = set(edge_neighbors)

        for nb in edge_neighbors:
            if nonmanifold_face_mask[nb]:
                edge_counts['nonmanifold'] += 1
            elif open_face_mask[nb]:
                edge_counts['open'] += 1
            else:
                edge_counts['normal'] += 1

        # 点邻：三个顶点的相邻面并集，减去自身和边邻
        verts = mesh.faces[fid]
        all_nb = set()
        for v_ in verts:
            row_start = vertex_faces_csr.indptr[v_]
            row_end = vertex_faces_csr.indptr[v_ + 1]
            indices = vertex_faces_csr.indices[row_start:row_end]
            all_nb.update(indices)

        all_nb.discard(fid)
        point_neighbors = all_nb - edge_neighbor_set

        for nb in point_neighbors:
            if nonmanifold_face_mask[nb]:
                point_counts['nonmanifold'] += 1
            elif open_face_mask[nb]:
                point_counts['open'] += 1
            else:
                point_counts['normal'] += 1

    return point_counts, edge_counts


def _compute_single_face_neighbor_stats(mesh, face_id,
                                        open_face_mask, nonmanifold_face_mask,
                                        vertex_faces_csr, face_adjacency_csr):
    """
    计算指定面片的逐顶点、逐边邻居统计。

    返回：
        vertex_stats: list of 3 dicts，对应顶点 A,B,C，每个 dict 包含
                      'normal', 'open', 'nonmanifold'
        edge_stats:   list of 3 dicts，对应边 AB,BC,CA，每个 dict 包含
                      'normal', 'open', 'nonmanifold'
    """
    verts = mesh.faces[face_id]  # 当前面片的三个顶点索引

    # ---------- 获取边邻信息 ----------
    # trimesh 的 face_adjacency_edges 提供每对邻接面共享的边 (顶点对)
    if hasattr(mesh, 'face_adjacency_edges'):
        face_adj = mesh.face_adjacency
        face_adj_edges = mesh.face_adjacency_edges
        # 初始化三条边的邻居面列表
        edge_neighbors_by_edge = [[], [], []]  # 对应边 AB, BC, CA
        for (f0, f1), (ev0, ev1) in zip(face_adj, face_adj_edges):
            if f0 == face_id or f1 == face_id:
                # 确定该邻接边属于当前面的哪条边
                for i in range(3):
                    a = verts[i]
                    b = verts[(i + 1) % 3]
                    if (ev0 == a and ev1 == b) or (ev0 == b and ev1 == a):
                        neighbor = f0 if f1 == face_id else f1
                        edge_neighbors_by_edge[i].append(neighbor)
                        break
    else:
        # 后备：无法区分边时，将所有边邻面视为同一条边？这里简化处理，
        # 将边邻集合均分给三条边，但这样不精确；实际 trimesh 应有 face_adjacency_edges。
        row_start = face_adjacency_csr.indptr[face_id]
        row_end = face_adjacency_csr.indptr[face_id + 1]
        all_edge_neighbors = list(face_adjacency_csr.indices[row_start:row_end])
        edge_neighbors_by_edge = [all_edge_neighbors, all_edge_neighbors, all_edge_neighbors]

    # ---------- 逐边统计 ----------
    edge_stats = []
    for i in range(3):
        cnt = {'normal': 0, 'open': 0, 'nonmanifold': 0}
        for nb in edge_neighbors_by_edge[i]:
            if nonmanifold_face_mask[nb]:
                cnt['nonmanifold'] += 1
            elif open_face_mask[nb]:
                cnt['open'] += 1
            else:
                cnt['normal'] += 1
        edge_stats.append(cnt)

    # ---------- 逐顶点统计（点邻） ----------
    # 首先获取该面的边邻面集合
    row_start = face_adjacency_csr.indptr[face_id]
    row_end = face_adjacency_csr.indptr[face_id + 1]
    edge_neighbors_set = set(face_adjacency_csr.indices[row_start:row_end])

    vertex_stats = []
    for v in verts:
        # 该顶点的所有相邻面
        row_start = vertex_faces_csr.indptr[v]
        row_end = vertex_faces_csr.indptr[v + 1]
        all_nb = set(vertex_faces_csr.indices[row_start:row_end])
        all_nb.discard(face_id)
        # 点邻 = 顶点邻接面 - 边邻面
        point_neighbors = all_nb - edge_neighbors_set

        cnt = {'normal': 0, 'open': 0, 'nonmanifold': 0}
        for nb in point_neighbors:
            if nonmanifold_face_mask[nb]:
                cnt['nonmanifold'] += 1
            elif open_face_mask[nb]:
                cnt['open'] += 1
            else:
                cnt['normal'] += 1
        vertex_stats.append(cnt)

    return vertex_stats, edge_stats


def _generate_topology_diagram_svg(code, vertex_stats, edge_stats):
    """
    返回一个内联 SVG 字符串，表示拓扑编码示意图。
    顶点和边均带有 <title> 悬浮提示，内容为邻接统计。
    """
    vA, eAB, vB, eBC, vC, eCA = [c for c in code]
    coords = {
        'A': (20, 20),
        'B': (120, 20),
        'C': (70, 100)
    }
    # 顶点样式：独占实心，共享空心
    vertex_fill = lambda v: '#000000' if v == '1' else '#ffffff'
    # 边颜色
    edge_color = {'1': '#0000ff', '2': '#008000', '3': '#ff0000'}

    # 顶点提示文本
    vertex_tips = []
    for i, (v, stats) in enumerate(zip([vA, vB, vC], vertex_stats)):
        tip = (f"顶点{i}: 点邻面片 流形={stats['normal']}, "
               f"开放={stats['open']}, 非流形={stats['nonmanifold']}")
        vertex_tips.append(tip)

    # 边提示文本
    edge_tips = []
    for i, (e, stats) in enumerate(zip([eAB, eBC, eCA], edge_stats)):
        tip = (f"边{i}: 边邻面片 流形={stats['normal']}, "
               f"开放={stats['open']}, 非流形={stats['nonmanifold']}")
        edge_tips.append(tip)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 120" width="200" height="170">
    <line x1="{coords['A'][0]}" y1="{coords['A'][1]}" x2="{coords['B'][0]}" y2="{coords['B'][1]}"
          stroke="{edge_color[eAB]}" stroke-width="3">
        <title>{edge_tips[0]}</title>
    </line>
    <line x1="{coords['B'][0]}" y1="{coords['B'][1]}" x2="{coords['C'][0]}" y2="{coords['C'][1]}"
          stroke="{edge_color[eBC]}" stroke-width="3">
        <title>{edge_tips[1]}</title>
    </line>
    <line x1="{coords['C'][0]}" y1="{coords['C'][1]}" x2="{coords['A'][0]}" y2="{coords['A'][1]}"
          stroke="{edge_color[eCA]}" stroke-width="3">
        <title>{edge_tips[2]}</title>
    </line>
    <circle cx="{coords['A'][0]}" cy="{coords['A'][1]}" r="8"
            fill="{vertex_fill(vA)}" stroke="black" stroke-width="2">
        <title>{vertex_tips[0]}</title>
    </circle>
    <circle cx="{coords['B'][0]}" cy="{coords['B'][1]}" r="8"
            fill="{vertex_fill(vB)}" stroke="black" stroke-width="2">
        <title>{vertex_tips[1]}</title>
    </circle>
    <circle cx="{coords['C'][0]}" cy="{coords['C'][1]}" r="8"
            fill="{vertex_fill(vC)}" stroke="black" stroke-width="2">
        <title>{vertex_tips[2]}</title>
    </circle>
    </svg>'''
    return svg


def run_full_diagnosis_pass1(mesh, output_dir):
    """
    Pass 1: 计算异常面编码、绘制示意图、生成报告框架和检查点。
    返回 (class_faces, id_to_code, abnormal_indices)。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagrams_dir = output_dir / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)

    print("\n=== Full Diagnosis Pass 1 ===")
    print("计算面片拓扑编码...")

    defects, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    abnormal_mask = open_face_mask | nonmanifold_face_mask
    abnormal_indices = np.where(abnormal_mask)[0]

    print(f"异常面片总数: {len(abnormal_indices)}")

    if len(abnormal_indices) == 0:
        print("没有异常面，退出。")
        # 创建空的 classes_data 和 checkpoint
        with open(output_dir / "classes_data.json", "w") as f:
            json.dump({"class_faces": {}, "id_to_code": []}, f)
        with open(output_dir / "checkpoint.json", "w") as f:
            json.dump({"mesh_hash": None, "classes": {}, "total_classes": 0,
                       "abnormal_count": 0}, f)
        return {}, [], abnormal_indices

    vertex_face_counts = compute_vertex_face_counts(mesh)
    face_edge_types = compute_face_edge_types(mesh)

    codes, code_to_id, id_to_code = compute_face_topology_codes(
        mesh, abnormal_indices, vertex_face_counts, face_edge_types
    )

    code_ids = np.array([code_to_id[c] for c in codes], dtype=np.int64)

    unique_ids, inverse, counts = np.unique(code_ids, return_inverse=True, return_counts=True)

    print(f"发现 {len(unique_ids)} 个不同拓扑类别。")

    class_faces = {}
    for class_id in unique_ids:
        mask = inverse == np.where(unique_ids == class_id)[0][0]
        class_faces[class_id] = abnormal_indices[mask].tolist()

    # 绘制示意图
    print("生成拓扑示意图...")
    for class_id, code in enumerate(id_to_code):
        diagram_path = diagrams_dir / f"diagram_{class_id}.svg"
        _generate_topology_diagram(code, str(diagram_path))
        if class_id % 10 == 0:
            print(f"  已生成 {class_id + 1}/{len(id_to_code)} 个示意图")

    # 保存分类数据用于 resume
    classes_data = {
        "class_faces": {str(cid): faces for cid, faces in class_faces.items()},
        "id_to_code": id_to_code,
    }
    with open(output_dir / "classes_data.json", "w") as f:
        json.dump(classes_data, f, indent=2)

    # 创建检查点文件
    checkpoint = {
        "mesh_hash": str(mesh.vertices.shape) + str(mesh.faces.shape),
        "classes": {str(cid): "pending" for cid in unique_ids},
        "total_classes": len(unique_ids),
        "abnormal_count": len(abnormal_indices),
    }
    with open(output_dir / "checkpoint.json", "w") as f:
        json.dump(checkpoint, f, indent=2)

    return class_faces, id_to_code, abnormal_indices


def run_full_diagnosis_pass2(mesh, output_dir, class_faces, id_to_code,
                             open_face_mask, nonmanifold_face_mask,
                             resume=False):
    """
    Pass 2: 逐类分析并填充报告，支持断点恢复。
    """
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.json"

    if resume and checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        pending_classes = [int(cid) for cid, status in checkpoint["classes"].items()
                           if status == "pending"]
        done_classes = [int(cid) for cid, status in checkpoint["classes"].items()
                        if status == "done"]
        print(f"恢复模式：已完成 {len(done_classes)} 类，剩余 {len(pending_classes)} 类")
    else:
        pending_classes = list(class_faces.keys())

    if not pending_classes:
        print("没有待分析类别。")
        return {}

    # 预计算共享数据
    vertex_faces_csr = _build_vertex_face_csr(mesh)
    face_adjacency = mesh.face_adjacency  # (n,2) 数组
    rows = np.concatenate([face_adjacency[:, 0], face_adjacency[:, 1]])
    cols = np.concatenate([face_adjacency[:, 1], face_adjacency[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    face_adjacency_csr = csr_matrix((data, (rows, cols)),
                                    shape=(len(mesh.faces), len(mesh.faces)))

    results = {}
    for idx, class_id in enumerate(pending_classes):
        code = id_to_code[class_id]
        face_indices = class_faces[class_id]
        print(f"\n=== 分析类别 {idx+1}/{len(pending_classes)} ===", flush=True)
        print(f"  编码: {code}, 面片数: {len(face_indices)}", flush=True)

        areas = mesh.area_faces[face_indices]
        area_stats = {
            'count': len(face_indices),
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

        point_counts, edge_counts = _compute_class_neighbor_stats(
            mesh, face_indices, open_face_mask, nonmanifold_face_mask,
            vertex_faces_csr, face_adjacency_csr
        )

        # 选择代表面，计算逐顶点/逐边统计
        rep_face = int(face_indices[0])
        vertex_stats, edge_stats = _compute_single_face_neighbor_stats(
            mesh, rep_face, open_face_mask, nonmanifold_face_mask,
            vertex_faces_csr, face_adjacency_csr
        )

        class_result = {
            'class_id': int(class_id),
            'code': code,
            'area_stats': area_stats,
            'point_counts': point_counts,
            'edge_counts': edge_counts,
            'representative_vertex_stats': vertex_stats,
            'representative_edge_stats': edge_stats,
        }
        results[class_id] = class_result

        # 更新检查点
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        checkpoint["classes"][str(class_id)] = "done"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

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
        svg_content = _generate_topology_diagram_svg(code, v_stats, e_stats)

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


def perform_full_diagnosis(mesh, args):
    """
    Full Diagnosis 入口，协调 Pass 1 和 Pass 2。
    """
    output_dir = Path(args.diagnosis_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)

    resume_checkpoint = output_dir / "checkpoint.json"
    classes_data_path = output_dir / "classes_data.json"

    if args.resume and resume_checkpoint.exists() and classes_data_path.exists():
        with open(classes_data_path, "r") as f:
            data = json.load(f)
        class_faces = {int(k): v for k, v in data["class_faces"].items()}
        id_to_code = data["id_to_code"]
        print("Resume: loading existing classification from Pass 1.")
    else:
        class_faces, id_to_code, _ = run_full_diagnosis_pass1(mesh, output_dir)

    results = run_full_diagnosis_pass2(
        mesh, output_dir, class_faces, id_to_code,
        open_face_mask, nonmanifold_face_mask,
        resume=args.resume
    )

    if args.diagnosis_format == "html":
        generate_html_report(output_dir, id_to_code, results)
    else:
        generate_latex_report(output_dir, id_to_code, results)

    print(f"\nFull Diagnosis 完成，报告已生成到 {output_dir}")


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
    parser.add_argument("--diagnosis-output", type=str, default="diagnosis_report",
                        help="Full Diagnosis 输出目录（默认 diagnosis_report）")
    parser.add_argument("--diagnosis-format", type=str, default="html",
                        choices=["html", "latex"],
                        help="报告格式：html 或 latex（默认 html）")
    parser.add_argument("--resume", action="store_true",
                        help="从现有检查点继续 Full Diagnosis Pass 2（跳过已完成类别）")

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

    print(f"Hey! Loading {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

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
