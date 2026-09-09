# src/toys3d/visualization.py
"""
基于 trimesh 的可视化与渲染工具。
"""
import argparse
import colorsys
import json
import os
from pathlib import Path

import numpy as np
import trimesh

from .geometrics import (
    extract_boundary_loops,
    polygon_area_from_3d_ccw,
    project_vertices_to_shell,
    normalize,
)
from .reporting import load_uncovered_edge_data


def add_axes_to_scene(scene, origin, u_x, u_y, u_z, length=0.3, radius=0.01):
    """在场景中添加红、绿、蓝三根坐标轴箭头。"""
    def add_arrow(o, d, color):
        cyl = trimesh.creation.cylinder(
            radius=radius,
            segment=[o, o + d * length],
        )
        cyl.visual.face_colors = color
        scene.add_geometry(cyl)

        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius * 3)
        sphere.apply_translation(o + d * length)
        sphere.visual.face_colors = color
        scene.add_geometry(sphere)

    add_arrow(origin, u_x, [255, 0, 0, 255])
    add_arrow(origin, u_y, [0, 255, 0, 255])
    add_arrow(origin, u_z, [0, 0, 255, 255])


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

    colors[open_only] = [255, 220, 0, 255]
    colors[nonmanifold_only] = [255, 0, 0, 255]
    colors[both] = [255, 128, 0, 255]

    vis.visual.face_colors = colors
    return vis


def build_reliable_visualization(mesh, distances, min_distance):
    """
    根据拓扑距离生成可视化网格。
    - 绿色：可靠
    - 黄色：中间状态
    - 红色：缺陷面
    """
    vis = mesh.copy()
    N = len(vis.faces)

    colors = np.full((N, 4), [255, 0, 0, 255], dtype=np.uint8)
    intermediate = (distances > 0) & (distances < min_distance)
    reliable = distances >= min_distance
    colors[intermediate] = [255, 220, 0, 255]
    colors[reliable] = [0, 200, 0, 255]

    vis.visual.face_colors = colors
    return vis


def make_double_sided(mesh, backface_color=None):
    """
    将网格渲染为双面几何，避免薄壳背面被背面剔除而显示为透明。
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
    """将面片颜色 alpha 通道设置为指定透明度。"""
    if hasattr(mesh.visual, 'face_colors') and \
            mesh.visual.face_colors.shape[0] == len(mesh.faces):
        mesh.visual.face_colors[:, 3] = int(np.clip(alpha, 0.0, 1.0) * 255)
    return mesh


def add_wireframe_to_scene(scene, mesh, color=None, radius=None):
    """将网格边以圆柱线段形式加入场景。"""
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
    palette = [
        (255,   0,   0, 255),
        (  0, 255,   0, 255),
        (  0, 128, 255, 255),
        (255, 255,   0, 255),
        (255,   0, 255, 255),
        (  0, 255, 255, 255),
        (255, 128,   0, 255),
        (128,   0, 255, 255),
        (  0, 255, 128, 255),
        (255,   0, 128, 255),
    ]
    return [np.array(c, dtype=np.uint8) for c in palette]


def _greedy_color_hole_loops(loops):
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
    """检测闭合孔洞边界环并高亮绘制。"""
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
    """打印投影孔洞边界包围的代理网格顶点信息。"""
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
                point_in_polygon_2d(tuple(p), poly2d)
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


def point_in_polygon_2d(pt, poly):
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


def add_boundary_projection_to_scene(scene, boundary_mesh, proxy_mesh,
                                     radius=None, verbose=False,
                                     print_enclosed_vertices=False,
                                     max_report_loops=None,
                                     max_report_vertices=20):
    """将边界环投影到代理网格表面并绘制。"""
    loops = extract_boundary_loops(boundary_mesh)
    if not loops:
        if verbose:
            print("  No boundary loops to project.")
        return

    try:
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

    color = np.array([0, 255, 255, 255], dtype=np.uint8)

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

        pts.append(pts[0])
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


def load_proxy_mesh(proxy_path):
    """加载代理网格，返回代理网格或 None。"""
    if not proxy_path:
        return None

    proxy = trimesh.load(proxy_path, force="mesh")
    if isinstance(proxy, trimesh.Scene):
        proxy = proxy.dump(concatenate=True)

    if len(proxy.faces) == 0:
        print("[WARNING] Proxy mesh is empty; skipping overlay")
        return None

    return proxy


def add_proxy_overlay_to_scene(scene, proxy, color=None, alpha=0.45,
                               double_sided=True):
    """将代理网格以半透明方式叠加到场景。"""
    if proxy is None:
        return

    if color is not None:
        color_components = color
        if len(color_components) == 3:
            alpha_uint8 = int(np.clip(alpha, 0.0, 1.0) * 255)
            color_arr = [*color_components, alpha_uint8]
        else:
            color_arr = color_components
    else:
        base = [128, 180, 255]
        alpha_uint8 = int(np.clip(alpha, 0.0, 1.0) * 255)
        color_arr = [*base, alpha_uint8]

    color_arr = np.array(color_arr, dtype=np.uint8)
    proxy.visual.face_colors = np.tile(color_arr, (len(proxy.faces), 1))

    if double_sided:
        proxy = make_double_sided(proxy, backface_color=None)

    scene.add_geometry(proxy)


def add_uncovered_edges_to_scene(scene, mesh, data_dir,
                                 radius=None, verbose=False):
    """将 hole diagnosis 中未覆盖的开放边高亮添加到场景。"""
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
        0: (0, 0, 255, 255),
        1: (255, 255, 0, 255),
        2: (255, 128, 0, 255),
        4: (255, 0, 0, 255),
        5: (128, 128, 128, 255),
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


def print_scene_debug_info(scene, title="Scene Debug Info"):
    """打印场景中几何对象信息。"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")

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


def _print_camera_info(info):
    """命令行打印摄像机信息。"""
    print("\n[Camera Info]")
    ws = info.get("window_size")
    if ws:
        print(f"  window_size:        {ws[0]} x {ws[1]}")

    if "camera_position_display" in info:
        c = info["camera_position_display"]
        print(f"  camera_position (display): [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
    if "focal_point_display" in info:
        c = info["focal_point_display"]
        print(f"  focal_point (display):     [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
    if "view_up" in info:
        c = info["view_up"]
        print(f"  view_up:            [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
    if "view_direction" in info:
        c = info["view_direction"]
        print(f"  view_direction:     [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
    if "camera_distance" in info:
        print(f"  camera_distance:    {info['camera_distance']:.6f}")
    if "clipping_range" in info:
        c = info["clipping_range"]
        print(f"  clipping_range:     [{c[0]:.6f}, {c[1]:.6f}]")

    if "scene_translation" in info:
        c = info["scene_translation"]
        print(f"  scene_translation:  [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")

    if "camera_position_world" in info:
        c = info["camera_position_world"]
        print(f"  camera_position (world):   [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
    if "focal_point_world" in info:
        c = info["focal_point_world"]
        print(f"  focal_point (world):       [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")

    err = info.get("error")
    if err:
        print(f"  [ERROR] {err}")


def _capture_vedo_camera_info(plotter_or_viewer):
    """从 vedo Plotter 或 trimesh vedo viewer 捕获相机/窗口信息。"""
    info = {}
    try:
        plt = getattr(plotter_or_viewer, "plotter", plotter_or_viewer)

        win = getattr(plt, "window", None)
        if win is not None:
            sz = win.GetSize()
            info["window_size"] = [int(sz[0]), int(sz[1])]

        cam = getattr(plt, "camera", None)
        if cam is None:
            return {"error": "plotter has no camera"}

        pos = np.asarray(cam.GetPosition(), dtype=np.float64)
        focal = np.asarray(cam.GetFocalPoint(), dtype=np.float64)
        if hasattr(cam, "GetViewUp"):
            up = np.asarray(cam.GetViewUp(), dtype=np.float64)
        else:
            up = np.asarray(cam.GetUp(), dtype=np.float64)

        view_dir = focal - pos
        dist = float(np.linalg.norm(view_dir))

        info["camera_position_display"] = pos.tolist()
        info["focal_point_display"] = focal.tolist()
        info["view_up"] = up.tolist()
        info["view_direction"] = normalize(view_dir).tolist()
        info["camera_distance"] = dist

        cr = cam.GetClippingRange()
        info["clipping_range"] = [float(cr[0]), float(cr[1])]

    except Exception as e:
        info["error"] = str(e)

    return info


def _try_get_trimesh_vedo_viewer():
    """尝试导入 trimesh 内置的 VedoViewer（旧版本才有）。"""
    for module_path in ("trimesh.viewers.vedo_viewer", "trimesh.viewer.vedo_viewer"):
        try:
            mod = __import__(module_path, fromlist=["VedoViewer"])
            return getattr(mod, "VedoViewer", None)
        except Exception:
            continue
    return None


def _show_scene_with_vedo(scene):
    """使用 vedo 直接显示 trimesh.Scene，并返回相机信息。"""
    import vedo

    merged = scene.to_geometry()
    actor = vedo.Mesh(merged)

    if (hasattr(merged.visual, "face_colors") and
            merged.visual.face_colors.shape[0] == len(merged.faces)):
        colors = np.asarray(merged.visual.face_colors)
        actor.cellcolors = colors[:, :3]

    plt = vedo.Plotter()
    plt.show(actor, interactive=True)

    return _capture_vedo_camera_info(plt)


def _show_scene_with_camera_info(scene, args, scene_translation=None):
    """统一封装 scene.show()，使用 vedo 显示，并按需捕获/打印相机信息。"""
    info = None

    VedoViewer = _try_get_trimesh_vedo_viewer()
    if VedoViewer is not None:
        class _CameraInfoViewer(VedoViewer):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.camera_info = None

            def show(self, **kw):
                result = super().show(**kw)
                self.camera_info = _capture_vedo_camera_info(self)
                return result

        viewer = _CameraInfoViewer(scene)
        viewer.show()
        info = viewer.camera_info or {}
    else:
        try:
            info = _show_scene_with_vedo(scene)
        except ImportError as e:
            print(f"[WARN] vedo 未安装，回退到默认显示: {e}")
            scene.show()
            return
        except Exception as e:
            print(f"[WARN] vedo 直接显示失败，回退到默认显示: {e}")
            scene.show()
            return

    if info and scene_translation is not None and "error" not in info:
        t = np.asarray(scene_translation, dtype=np.float64)
        if "camera_position_display" in info:
            info["camera_position_world"] = (
                np.asarray(info["camera_position_display"]) - t
            ).tolist()
        if "focal_point_display" in info:
            info["focal_point_world"] = (
                np.asarray(info["focal_point_display"]) - t
            ).tolist()
        info["scene_translation"] = t.tolist()

    if info and (args.print_camera_info or args.camera_info_output):
        _print_camera_info(info)
        if args.camera_info_output:
            out = Path(args.camera_info_output)
            out.write_text(
                json.dumps(info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"相机信息已保存: {out}")


def _filter_camera_core_points(points):
    """根据中位数绝对偏差过滤离群点，避免个别原点/错误点拉偏相机。"""
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


def _parse_color_string(s):
    """将 'R,G,B,A' 解析为整数列表。"""
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
    """将 'R,G,B' 或 'R,G,B,A' 解析为整数列表。"""
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
