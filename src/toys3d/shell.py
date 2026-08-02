import sys
import os
import time

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
    build_face_adjacency,
    detect_multiscale_edges,
    segment_regions_by_edges,
    estimate_shell_thickness,
    segment_plates_by_smoothness,
    detect_thin_regions,
    compute_wall_thickness_statistics,
    extract_plate_boundary_loops,
    classify_edge_regularity,
    # 新增代理网格与局部聚类分割
    build_proxy_mesh,
    map_labels_from_proxy,
    segment_plates_by_local_clustering,
)


# ------------------------------------------------------------------
#  Timer 类用于记录耗时
# ------------------------------------------------------------------
class Timer:
    """Simple timer context manager for logging elapsed time."""
    def __init__(self, label=''):
        self.label = label
        self.start = None
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[Timer] {self.label} took {elapsed:.3f}s")


# ------------------------------------------------------------------
#  可视化辅助
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


def _jet_colormap(t):
    """简化的 jet 伪彩色映射，输入 t 在 [0,1]。"""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    r = np.clip(1.5 - 4.0 * np.abs(t - 0.75), 0.0, 1.0)
    g = np.clip(1.5 - 4.0 * np.abs(t - 0.5), 0.0, 1.0)
    b = np.clip(1.5 - 4.0 * np.abs(t - 0.25), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def visualize_thickness(mesh, thickness, reliability=None):
    """用伪彩色显示厚度场，不可靠/无效区域显示为灰色。"""
    scene = trimesh.Scene()
    vis = mesh.copy()
    N = len(vis.faces)

    colors = np.full((N, 4), 180, dtype=np.uint8)
    colors[:, 3] = 255

    if reliability is None:
        reliability = np.isfinite(thickness)
    valid = reliability & np.isfinite(thickness)

    if np.any(valid):
        t = thickness[valid]
        t_min, t_max = np.percentile(t, [2, 98])
        rng = t_max - t_min
        if rng < 1e-12:
            rng = 1.0
        norm = np.clip((t - t_min) / rng, 0.0, 1.0)
        rgb = _jet_colormap(norm)
        colors[np.where(valid)[0], :3] = rgb

    colors[~valid] = [180, 180, 180, 255]
    vis.visual.face_colors = colors

    scene.add_geometry(vis)
    origin = mesh.bounding_box.centroid
    max_ext = mesh.bounding_box.extents.max()
    add_axes_to_scene(scene, origin,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)
    return scene


def visualize_plates(mesh, labels):
    """用不同颜色显示薄板分割结果，未归类面片为灰色。"""
    scene = trimesh.Scene()
    vis = mesh.copy()
    N = len(vis.faces)

    rng = np.random.default_rng(42)
    palette = np.column_stack([
        rng.integers(60, 255, size=20),
        rng.integers(60, 255, size=20),
        rng.integers(60, 255, size=20),
        np.full(20, 255, dtype=np.uint8),
    ])

    colors = np.full((N, 4), 180, dtype=np.uint8)
    colors[:, 3] = 255

    unique = np.unique(labels)
    for i, lbl in enumerate(unique):
        if lbl < 0:
            continue
        col = palette[i % len(palette)]
        colors[labels == lbl] = col
    colors[labels < 0] = [180, 180, 180, 255]

    vis.visual.face_colors = colors
    scene.add_geometry(vis)
    origin = mesh.bounding_box.centroid
    max_ext = mesh.bounding_box.extents.max()
    add_axes_to_scene(scene, origin,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)
    return scene


def _assign_plate_colors(labels, adjacency):
    """
    为每个薄板分配调色板颜色索引，保证相邻薄板颜色不同（贪心四色）。
    所有有效标签（>=0）都会获得颜色。
    """
    from collections import defaultdict

    # 初始化所有有效标签，确保孤立薄板也有颜色
    unique_labels = np.unique(labels)
    valid_labels = [l for l in unique_labels if l >= 0]
    label_neighbors = defaultdict(set)
    for lbl in valid_labels:
        label_neighbors[lbl]  # 创建空集

    # 构建标签邻接关系
    for fi, neighbors in enumerate(adjacency):
        lbl = labels[fi]
        if lbl < 0:
            continue
        for fj in neighbors:
            lbl2 = labels[fj]
            if lbl2 < 0 or lbl2 == lbl:
                continue
            label_neighbors[lbl].add(lbl2)
            label_neighbors[lbl2].add(lbl)

    # 基础高对比度调色板
    palette = [
        [230, 60, 60],    # 红
        [60, 120, 230],   # 蓝
        [60, 200, 80],    # 绿
        [255, 200, 40],   # 黄
    ]

    color_map = {}
    for lbl in sorted(label_neighbors.keys()):
        used = {color_map[n] for n in label_neighbors[lbl] if n in color_map}
        chosen = None
        for i in range(len(palette)):
            if i not in used:
                chosen = i
                break
        if chosen is None:
            extras = [
                [180, 40, 180],   # 紫
                [0, 200, 200],    # 青
                [255, 120, 0],    # 橙
                [120, 80, 60],    # 棕
            ]
            palette.extend(extras)
            for i in range(len(palette)):
                if i not in used:
                    chosen = i
                    break
        color_map[lbl] = chosen

    return color_map, palette


def visualize_plate_centers(mesh, labels, alpha=40, plate_alpha=200):
    """
    以薄板自身面片作为中心面代理进行可视化。
    单网格混合着色：
    - 背景面片半透明灰色（alpha）
    - 薄板面片按四色贪心分配，保证相邻板颜色不同
    - 用不透明黑色勾勒每个薄板的边界环

    Parameters
    ----------
    mesh : trimesh.Trimesh
    labels : (N,) ndarray
        薄板标签。
    alpha : int
        未归类/背景面片的透明度（0-255，默认40）。
    plate_alpha : int
        薄板面片的透明度（0-255，默认200）。
    """
    scene = trimesh.Scene()
    vis = mesh.copy()
    N = len(vis.faces)

    # 构建面片邻接（用于四色分配）
    adjacency = build_face_adjacency(mesh)

    # 分配颜色
    color_map, palette = _assign_plate_colors(labels, adjacency)

    # 初始化颜色：背景半透明灰色 alpha
    colors = np.full((N, 4), 180, dtype=np.uint8)
    colors[:, 3] = alpha

    # 填充薄板颜色
    n_plates = int(labels.max()) + 1
    for lbl in range(n_plates):
        mask = labels == lbl
        if np.sum(mask) == 0:
            continue
        if lbl in color_map:
            ci = color_map[lbl]
            if ci < len(palette):
                colors[mask, :3] = palette[ci]
                colors[mask, 3] = plate_alpha

    # 未归类面片保持半透明灰色
    colors[labels < 0] = [180, 180, 180, alpha]

    # 构建 edge -> faces 映射，用于勾勒边界
    edge_face_map = {}
    for fi, (v1, v2, v3) in enumerate(mesh.faces):
        for a, b in [(int(v1), int(v2)), (int(v2), int(v3)), (int(v3), int(v1))]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(fi)

    # 对每个薄板提取边界环，并将边界邻接面片标记为不透明黑色
    for lbl in range(n_plates):
        mask = labels == lbl
        if np.sum(mask) == 0:
            continue
        loops = extract_plate_boundary_loops(mesh, mask)
        for loop in loops:
            loop_arr = np.asarray(loop, dtype=int)
            if len(loop_arr) < 3:
                continue
            for i in range(len(loop_arr)):
                a, b = int(loop_arr[i]), int(loop_arr[(i + 1) % len(loop_arr)])
                key = (a, b) if a < b else (b, a)
                for fi in edge_face_map.get(key, []):
                    colors[fi] = [10, 10, 10, 255]  # 近黑色，不透明

    vis.visual.face_colors = colors
    scene.add_geometry(vis)

    # 坐标轴
    origin = mesh.bounding_box.centroid
    max_ext = mesh.bounding_box.extents.max()
    add_axes_to_scene(scene, origin,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)
    return scene


def visualize_boundaries(mesh, labels, scale, line_tol, circle_tol, spline_tol):
    """
    按边界环规律性着色：直线=绿，圆弧=蓝，样条=黄，不规则=红。
    同时返回被判为不规则的边界环列表。
    同时输出每个薄板的边界环数量。
    """
    scene = trimesh.Scene()
    vis = mesh.copy()
    N = len(vis.faces)

    colors = np.full((N, 4), 200, dtype=np.uint8)
    colors[:, 3] = 255

    edge_map = {}
    for fi, face in enumerate(mesh.faces):
        for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            key = tuple(sorted((int(a), int(b))))
            edge_map.setdefault(key, []).append(fi)

    irregular_loops = []
    for lbl in np.unique(labels):
        if lbl < 0:
            continue
        loops = extract_plate_boundary_loops(mesh, labels == lbl)
        # 输出每个薄板的边界环数量
        print(f"  Plate {lbl}: {len(loops)} boundary loops")
        for loop in loops:
            pts = mesh.vertices[np.asarray(loop, dtype=int)]
            ctype, score = classify_edge_regularity(
                pts, scale=scale,
                line_tol=line_tol, circle_tol=circle_tol, spline_tol=spline_tol
            )
            if ctype == 'line':
                col = [0, 255, 0, 255]
            elif ctype == 'circle':
                col = [0, 0, 255, 255]
            elif ctype == 'spline':
                col = [255, 255, 0, 255]
            else:
                col = [255, 0, 0, 255]
                irregular_loops.append(loop)

            for i in range(len(loop)):
                key = tuple(sorted((int(loop[i]), int(loop[(i + 1) % len(loop)]))))
                for fi in edge_map.get(key, []):
                    colors[fi] = col

    vis.visual.face_colors = colors
    scene.add_geometry(vis)
    origin = mesh.bounding_box.centroid
    max_ext = mesh.bounding_box.extents.max()
    add_axes_to_scene(scene, origin,
                      u_x=np.array([1, 0, 0]),
                      u_y=np.array([0, 1, 0]),
                      u_z=np.array([0, 0, 1]),
                      length=max_ext * 0.5)
    return scene, irregular_loops


# ------------------------------------------------------------------
#  几何辅助
# ------------------------------------------------------------------

def _loop_geometry(mesh, loop):
    """计算边界环的周长与投影面积。"""
    pts = mesh.vertices[np.asarray(loop, dtype=int)]
    n = len(pts)

    perim = 0.0
    for i in range(n):
        perim += np.linalg.norm(pts[(i + 1) % n] - pts[i])

    # Newell 法求最佳投影平面法向
    normal = np.zeros(3, dtype=np.float64)
    for i in range(n):
        normal += np.cross(pts[i], pts[(i + 1) % n])
    nlen = np.linalg.norm(normal)
    if nlen < 1e-12:
        return perim, 0.0
    normal /= nlen

    if abs(normal[2]) < 0.9:
        u = np.cross([0.0, 0.0, 1.0], normal)
    else:
        u = np.cross([1.0, 0.0, 0.0], normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    proj = np.column_stack([pts @ u, pts @ v])
    area = 0.5 * abs(
        np.sum(proj[:, 0] * np.roll(proj[:, 1], -1)
               - np.roll(proj[:, 0], -1) * proj[:, 1])
    )
    return perim, area


def _faces_adjacent_to_loops(mesh, loops):
    """返回与任意边界环共享边的所有面片掩码。"""
    edge_map = {}
    for fi, face in enumerate(mesh.faces):
        for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            key = tuple(sorted((int(a), int(b))))
            edge_map.setdefault(key, []).append(fi)

    mask = np.zeros(len(mesh.faces), dtype=bool)
    for loop in loops:
        loop_arr = np.asarray(loop, dtype=int)
        for i in range(len(loop_arr)):
            key = tuple(sorted((int(loop_arr[i]), int(loop_arr[(i + 1) % len(loop_arr)]))))
            for fi in edge_map.get(key, []):
                mask[fi] = True
    return mask


def _repair_loop(mesh, max_iter=5):
    """通用修复循环：去重/退化面、去非流形边、补洞。"""
    for it in range(max_iter):
        mesh = repair_mesh_by_removing_duplicates(mesh)
        mesh = repair_nonmanifold_edges(mesh, verbose=(it == 0))
        mesh = fill_small_holes(mesh, max_loop_edges=50, verbose=(it == 0))

        defect_stats, _, _ = analyze_mesh_defects(mesh)
        if defect_stats['open_edges'] == 0 and defect_stats['nonmanifold_edges'] == 0:
            break
    return mesh


def segment_by_multiscale_edges(mesh, scales=(1, 2, 4, 8),
                                threshold_ratio=0.3, min_faces=30):
    """
    多尺度边缘检测 + 连通分量分割 + 边缘面片重新分配 + 小区域合并。
    """
    edge_mask, strengths = detect_multiscale_edges(
        mesh, scales=scales, threshold_ratio=threshold_ratio
    )
    labels = segment_regions_by_edges(mesh, edge_mask)

    # 将边缘面片（-1）重新分配给相邻薄板
    adjacency = build_face_adjacency(mesh)
    normals = mesh.face_normals

    for fi in range(len(labels)):
        if labels[fi] != -1:
            continue

        best_label = -1
        best_angle = np.inf
        for fj in adjacency[fi]:
            lbl = labels[fj]
            if lbl <= 0:
                continue
            dot = np.clip(np.dot(normals[fi], normals[fj]), -1.0, 1.0)
            angle = float(np.arccos(dot))
            if angle < best_angle:
                best_angle = angle
                best_label = lbl

        if best_label != -1:
            labels[fi] = best_label

    # 合并小区域
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    for lbl in unique[counts < min_faces]:
        mask = labels == lbl
        neighbor_labels = []
        for fi in np.flatnonzero(mask):
            for fj in adjacency[fi]:
                nl = labels[fj]
                if nl >= 0 and nl != lbl:
                    neighbor_labels.append(nl)
        if neighbor_labels:
            new_lbl = max(set(neighbor_labels), key=neighbor_labels.count)
            labels[mask] = new_lbl

    # 压缩标签
    valid = labels >= 0
    if np.any(valid):
        _, new_labels = np.unique(labels[valid], return_inverse=True)
        labels[valid] = new_labels

    return labels


# ------------------------------------------------------------------
#  主处理流程
# ------------------------------------------------------------------

def process_shell(mesh, num_passes=0, repair_mode=False,
                  angle_threshold_deg=30.0, min_faces=30,
                  thin_mode='adaptive', thin_threshold=0.1,
                  line_tol=0.05, circle_tol=0.05, spline_tol=0.1,
                  irregular_perimeter_ratio=0.05,
                  thickness_grid_size=128,
                  seg_mode='cluster', edge_scales=(1, 2, 4, 8),
                  edge_threshold_ratio=0.3,
                  vis_mode='plates', vis_alpha=40, plate_alpha=200,
                  proxy_faces=50000, proxy_max_edge=None,
                  cluster_depth=2, cluster_angle_deg=30.0,
                  cluster_radius=None):
    """
    薄壳扫描网格处理主函数。

    Returns
    -------
    scene : trimesh.Scene
        当前阶段主可视化场景。
    world_scene : trimesh.Scene
        厚度场可视化场景（始终基于当前处理后的网格）。
    processed_mesh : trimesh.Trimesh
        处理后的网格。
    all_stats : dict
        统计信息。
    """
    # 0. 基础统计
    with Timer("Initial stats/defect analysis"):
        stats = compute_mesh_stats(mesh)
        print("Yo, mesh stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
        print("\nMesh defect analysis:")
        print(f"  open edges: {defect_stats['open_edges']}")
        print(f"  nonmanifold edges: {defect_stats['nonmanifold_edges']}")
        print(f"  watertight (no open edges): {defect_stats['watertight_by_count']}")

    # 可选修复
    if repair_mode and (defect_stats['open_edges'] > 0 or defect_stats['nonmanifold_edges'] > 0):
        with Timer("Mesh repair"):
            print("\n[Repair mode] Attempting to fix mesh...")
            mesh = _repair_loop(mesh)
            stats = compute_mesh_stats(mesh)
            print("\nAfter repair:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
            print(f"  After repair defects: open_edges={defect_stats['open_edges']}, "
                  f"nonmanifold_edges={defect_stats['nonmanifold_edges']}")

    # 首次厚度估计与薄板分割（后续各阶段都会复用）
    def analyze_current(m, proxy_m=None, step_label=""):
        with Timer(f"{step_label} thickness estimation"):
            thickness, reliability = estimate_shell_thickness(m, grid_size=thickness_grid_size)
        with Timer(f"{step_label} thickness statistics"):
            th_stats = compute_wall_thickness_statistics(thickness, reliability)

        with Timer(f"{step_label} plate segmentation"):
            if seg_mode == 'cluster':
                # 在代理网格上分割
                p = proxy_m if proxy_m is not None else m
                proxy_labels = segment_plates_by_local_clustering(
                    p, radius=cluster_radius,
                    cluster_angle_deg=cluster_angle_deg,
                    min_faces=min_faces
                )
                labels = map_labels_from_proxy(m, p, proxy_labels)
            elif seg_mode == 'multiscale':
                labels = segment_by_multiscale_edges(
                    m,
                    scales=edge_scales,
                    threshold_ratio=edge_threshold_ratio,
                    min_faces=min_faces
                )
            else:
                labels = segment_plates_by_smoothness(
                    m, angle_threshold_deg=angle_threshold_deg, min_faces=min_faces
                )
        return thickness, reliability, th_stats, labels

    # 先估计初始厚度，用于自适应代理网格参数
    with Timer("Initial thickness estimation"):
        thickness, reliability = estimate_shell_thickness(mesh, grid_size=thickness_grid_size)
        th_stats = compute_wall_thickness_statistics(thickness, reliability)

    print(f"\n[Thickness] median={th_stats['median']:.4f}, "
          f"reliable={th_stats['reliable_ratio']:.2%}")

    # 构建代理网格（用于 cluster 模式）
    proxy_mesh = None
    if seg_mode == 'cluster':
        with Timer("Build proxy mesh"):
            auto_max_edge = proxy_max_edge
            if auto_max_edge is None and np.isfinite(th_stats.get('median', np.nan)):
                auto_max_edge = th_stats['median'] * 0.5
            proxy_mesh = build_proxy_mesh(
                mesh,
                target_faces=proxy_faces,
                max_edge_length=auto_max_edge,
                iterations=2,
                smooth=False
            )
            print(f"  Proxy mesh: {proxy_mesh.faces.shape[0]} faces, "
                  f"mean_edge={proxy_mesh.edges_unique_length.mean():.4f}")

    # 第一次完整分析（包含分割）
    with Timer("First thickness/segmentation analysis"):
        thickness, reliability, th_stats, labels = analyze_current(mesh, proxy_mesh, step_label="First")

    print("\n[Shell analysis]")
    print(f"  Median thickness: {th_stats['median']:.4f}")
    print(f"  Reliable faces: {th_stats['reliable_count']} / {len(mesh.faces)} "
          f"({th_stats['reliable_ratio']:.2%})")
    print(f"  Number of plates: {th_stats.get('num_plates', int(labels.max()) + 1)}")

    # Pass 0: 检测可视化
    if num_passes == 0:
        with Timer("pass 0 visualization"):
            if vis_mode == 'thickness':
                world_scene = visualize_thickness(mesh, thickness, reliability)
                scene = visualize_plates(mesh, labels)
            elif vis_mode == 'centers':
                world_scene = visualize_plate_centers(mesh, labels, alpha=vis_alpha, plate_alpha=plate_alpha)
                scene = world_scene
            else:  # 'plates'
                world_scene = visualize_thickness(mesh, thickness, reliability)
                scene = visualize_plates(mesh, labels)
        return scene, world_scene, mesh.copy(), {**stats, **th_stats}

    # Pass 1: 删除过薄 / 不可靠区域
    print("\n[Pass 1] Removing thin / unreliable regions...")
    thin_mask = detect_thin_regions(
        thickness, mode=thin_mode, threshold=thin_threshold,
        fallback_median=th_stats['median']
    )
    unreliable_mask = ~reliability
    remove_mask = thin_mask | unreliable_mask

    print(f"  Removing {np.sum(remove_mask)} faces "
          f"({np.sum(thin_mask)} thin + {np.sum(unreliable_mask)} unreliable)")

    if np.all(remove_mask):
        raise ValueError("All faces would be removed. Check thin-threshold or mesh quality.")

    if np.any(remove_mask):
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[~remove_mask],
            process=False
        )
        mesh.remove_unreferenced_vertices()
        mesh = fill_small_holes(mesh, max_loop_edges=100, verbose=False)

    with Timer("Second thickness/segmentation analysis (after cleanup)"):
        thickness, reliability, th_stats, labels = analyze_current(mesh, proxy_mesh, step_label="Second")
    print(f"  After cleanup: {mesh.faces.shape[0]} faces, median thickness: {th_stats['median']:.4f}")

    if num_passes == 1:
        world_scene = visualize_thickness(mesh, thickness, reliability)
        scene = visualize_plates(mesh, labels)
        return scene, world_scene, mesh.copy(), {**stats, **th_stats}

    # Pass 2: 基于边界环规律性精化
    print("\n[Pass 2] Refining by boundary regularity...")
    scale = float(np.linalg.norm(mesh.bounding_box.extents))
    if scale < 1e-12:
        scale = 1.0

    # 先可视化并收集不规则小边界环
    with Timer("visualize_boundaries (pass 2)"):
        scene, irregular_loops = visualize_boundaries(
            mesh, labels, scale=scale,
            line_tol=line_tol, circle_tol=circle_tol, spline_tol=spline_tol
        )

    print(f"  Found {len(irregular_loops)} irregular small boundary loops")

    if irregular_loops:
        remove_mask = _faces_adjacent_to_loops(mesh, irregular_loops)
        print(f"  Removing {np.sum(remove_mask)} faces adjacent to irregular loops")
        if np.any(remove_mask) and not np.all(remove_mask):
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces[~remove_mask],
                process=False
            )
            mesh.remove_unreferenced_vertices()
            mesh = fill_small_holes(mesh, max_loop_edges=100, verbose=False)

    # 重新分析用于最终可视化
    with Timer("Third thickness/segmentation analysis (after refinement)"):
        thickness, reliability, th_stats, labels = analyze_current(mesh, proxy_mesh, step_label="Third")
    print(f"  After refinement: {mesh.faces.shape[0]} faces, "
          f"median thickness: {th_stats['median']:.4f}")

    world_scene = visualize_thickness(mesh, thickness, reliability)
    scene, _ = visualize_boundaries(
        mesh, labels, scale=scale,
        line_tol=line_tol, circle_tol=circle_tol, spline_tol=spline_tol
    )

    return scene, world_scene, mesh.copy(), {**stats, **th_stats}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="薄壳扫描网格处理：厚度分析、薄板分割、去薄区、边界规律性精化。"
    )
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("--output", help="保存清理后的网格路径 (可选)")
    parser.add_argument("--num-passes", type=int, default=0, choices=[0, 1, 2],
                        help="处理阶段：0=检测，1=去薄区，2=边界规律性精化（默认0）")
    parser.add_argument("--repair", action="store_true",
                        help="尝试自动修复网格拓扑缺陷")
    parser.add_argument("--angle-threshold", type=float, default=30.0,
                        help="薄板分割二面角阈值（度，默认30）")
    parser.add_argument("--min-faces", type=int, default=30,
                        help="薄板最小面片数（默认30）")
    parser.add_argument("--thin-mode", type=str, default='adaptive',
                        choices=['adaptive', 'absolute'],
                        help="薄区阈值模式：adaptive=厚度中位数比例，absolute=绝对值")
    parser.add_argument("--thin-threshold", type=float, default=0.1,
                        help="薄区阈值（adaptive 默认 0.1*median；absolute 为绝对厚度）")
    parser.add_argument("--line-tol", type=float, default=0.05,
                        help="边界环直线拟合归一化误差阈值")
    parser.add_argument("--circle-tol", type=float, default=0.05,
                        help="边界环圆弧拟合归一化误差阈值")
    parser.add_argument("--spline-tol", type=float, default=0.1,
                        help="边界环样条拟合归一化误差阈值")
    parser.add_argument("--irregular-perimeter-ratio", type=float, default=0.05,
                        help="不规则边界环周长小于该比例*包围盒对角线时会被删除")
    parser.add_argument("--thickness-grid-size", type=int, default=128,
                        help="体素分辨率用于厚度估计（默认128）")
    parser.add_argument("--seg-mode", type=str, default='cluster',
                        choices=['smoothness', 'multiscale', 'cluster'],
                        help="薄板分割模式：smoothness=法向连通性，multiscale=多尺度边缘，cluster=局部法向聚类")
    parser.add_argument("--edge-scales", type=int, nargs='+', default=(1, 2, 4, 8),
                        help="多尺度边缘检测的邻域尺度，默认 1 2 4 8")
    parser.add_argument("--edge-threshold", type=float, default=0.3,
                        help="多尺度边缘检测相对阈值（默认0.3）")
    parser.add_argument("--vis-mode", type=str, default='plates',
                        choices=['plates', 'centers', 'thickness'],
                        help="Pass 0 可视化模式：plates=薄板着色, centers=中心面高亮, thickness=厚度场")
    parser.add_argument("--vis-alpha", type=int, default=40,
                        help="原始网格透明度（仅 centers 模式，0-255，默认40）")
    parser.add_argument("--plate-alpha", type=int, default=200,
                        help="薄板面片透明度（仅 centers 模式，0-255，默认200）")
    parser.add_argument("--proxy-faces", type=int, default=50000,
                        help="代理网格目标面片数（仅 cluster 模式，默认50000）")
    parser.add_argument("--proxy-max-edge", type=float, default=None,
                        help="代理网格最大边长（默认自动=0.5*厚度中位数）")
    parser.add_argument("--cluster-depth", type=int, default=2,
                        help="局部法向聚类邻域树深度（默认2）")
    parser.add_argument("--cluster-angle", type=float, default=30.0,
                        help="同一法向簇最大夹角（度，默认30）")
    parser.add_argument("--cluster-radius", type=float, default=None,
                        help="球状邻域半径（默认自动=4*厚度中位数）")
    parser.add_argument("--show", action="store_true", help="显示可视化窗口")
    args = parser.parse_args()

    mesh = trimesh.load(args.input_file)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")
    print(f"Yo, loading model: {args.input_file}")

    scene, world_scene, processed_mesh, all_stats = process_shell(
        mesh,
        num_passes=args.num_passes,
        repair_mode=args.repair,
        angle_threshold_deg=args.angle_threshold,
        min_faces=args.min_faces,
        thin_mode=args.thin_mode,
        thin_threshold=args.thin_threshold,
        line_tol=args.line_tol,
        circle_tol=args.circle_tol,
        spline_tol=args.spline_tol,
        irregular_perimeter_ratio=args.irregular_perimeter_ratio,
        thickness_grid_size=args.thickness_grid_size,
        seg_mode=args.seg_mode,
        edge_scales=tuple(args.edge_scales),
        edge_threshold_ratio=args.edge_threshold,
        vis_mode=args.vis_mode,
        vis_alpha=args.vis_alpha,
        plate_alpha=args.plate_alpha,
        proxy_faces=args.proxy_faces,
        proxy_max_edge=args.proxy_max_edge,
        cluster_depth=args.cluster_depth,
        cluster_angle_deg=args.cluster_angle,
        cluster_radius=args.cluster_radius,
    )

    if args.output:
        processed_mesh.export(args.output)
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
