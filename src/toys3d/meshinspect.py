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
import numpy as np
import trimesh

from toys3d.geometrics import (
    compute_mesh_stats,
    analyze_mesh_defects,
    compute_hole_area_stats,
    extract_boundary_loops,
    polygon_area_from_3d_ccw,
    compute_reliable_face_mask,
    repair_mesh_by_removing_duplicates,
    project_vertices_to_shell,
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


def build_reliable_visualization(mesh, weights):
    """
    根据可靠性权重生成可视化网格。
    - 绿色：可靠（权重 > 0.75）
    - 黄色：中间状态（0.25 < 权重 <= 0.75）
    - 红色：不可靠（权重 <= 0.25）
    """
    vis = mesh.copy()
    N = len(vis.faces)

    colors = np.full((N, 4), [255, 0, 0, 255], dtype=np.uint8)  # 默认红
    reliable = weights > 0.75
    intermediate = (~reliable) & (weights > 0.25)
    colors[reliable] = [0, 200, 0, 255]       # 绿
    colors[intermediate] = [255, 220, 0, 255] # 黄

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

        print(f"  [loop {loop_idx}]")
        print(f"    boundary_edges={len(ids)}")
        print(f"    input_hole_area={polygon_area_from_3d_ccw(original_pts):.6f}")
        print(f"    projected_area={polygon_area_from_3d_ccw(pts):.6f}")
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
    stats = compute_mesh_stats(mesh)
    defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    area_stats = compute_face_area_stats(mesh)
    bbox_stats = compute_bounding_box_stats(mesh)
    volume = compute_volume_if_closed(mesh)

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
            print(f"  threshold: {args.reliable_threshold}")

            # 计算可靠面片掩码
            weights = compute_reliable_face_mask(mesh)
            reliable_mask = weights > args.reliable_threshold
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
            print_separator("Reliable Neighborhood Visualization")
            print("  green:  reliable faces")
            print("  yellow: defect neighborhood (intermediate)")
            print("  red:    unreliable faces")

            weights = compute_reliable_face_mask(mesh)
            vis = build_reliable_visualization(mesh, weights)
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
    parser.add_argument("--reliable-threshold", type=float, default=0.75,
                        help="可靠面片权重阈值，默认 0.75。"
                             "仅当 --keep-reliable-only 时生效。")
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

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    args.wireframe_color = _parse_color_string(args.wireframe_color)

    if args.backface_color is not None:
        args.backface_color = _parse_color_string(args.backface_color)

    if args.proxy_color is not None:
        args.proxy_color = _parse_color_string_flexible(args.proxy_color)

    print(f"Loading: {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

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
