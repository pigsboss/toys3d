# src/toys3d/geometrics.py
"""
基础几何处理工具集。

包含网格统计、缺陷分析、边界环提取、孔洞面积统计、
可靠面片提取、代理壳拓扑补丁、重新三角化、水密重建等。
"""

import numpy as np
import trimesh
from collections import deque


def compute_mesh_stats(mesh):
    stats = {
        'vertices': int(len(mesh.vertices)),
        'faces': int(len(mesh.faces)),
        'edges': int(len(mesh.edges_unique)),
        'is_watertight': bool(mesh.is_watertight),
    }
    if len(mesh.faces) == 0:
        stats['mean_edge_length'] = 0.0
        for p in [1, 5, 50, 95, 99]:
            stats[f'edge_length_p{p}'] = 0.0
        return stats

    edge_lengths = mesh.edges_unique_length
    stats['mean_edge_length'] = float(np.mean(edge_lengths))
    for p in [1, 5, 50, 95, 99]:
        stats[f'edge_length_p{p}'] = float(np.percentile(edge_lengths, p))
    return stats


def analyze_mesh_defects(mesh):
    edge_face_count = {}
    face_adj_edges = mesh.face_adjacency_edges

    for face_adj_idx, (v0, v1) in enumerate(face_adj_edges):
        ekey = (int(v0), int(v1)) if v0 < v1 else (int(v1), int(v0))
        edge_face_count.setdefault(ekey, []).append(face_adj_idx)

    open_edges = set()
    nonmanifold_edges = set()
    open_faces = set()
    nonmanifold_faces = set()

    for ekey, adj_list in edge_face_count.items():
        if len(adj_list) == 1:
            open_edges.add(ekey)
            open_faces.update(adj_list)
        elif len(adj_list) > 2:
            nonmanifold_edges.add(ekey)
            nonmanifold_faces.update(adj_list)

    defect_stats = {
        'open_edges': len(open_edges),
        'nonmanifold_edges': len(nonmanifold_edges),
        'open_faces': len(open_faces),
        'nonmanifold_faces': len(nonmanifold_faces),
    }

    open_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    nonmanifold_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    if open_faces:
        open_face_mask[list(open_faces)] = True
    if nonmanifold_faces:
        nonmanifold_face_mask[list(nonmanifold_faces)] = True

    return defect_stats, open_face_mask, nonmanifold_face_mask


def _build_open_edge_adjacency(mesh):
    edges = mesh.edges_single if hasattr(mesh, 'edges_single') else None
    if edges is None or len(edges) == 0:
        edge_count = {}
        faces = np.asarray(mesh.faces, dtype=np.int64)
        for face in faces:
            for i in range(3):
                v0 = int(face[i])
                v1 = int(face[(i + 1) % 3])
                ekey = (v0, v1) if v0 < v1 else (v1, v0)
                edge_count[ekey] = edge_count.get(ekey, 0) + 1
        open_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    else:
        open_edges = [tuple(e) for e in edges]

    adj = {}
    for v0, v1 in open_edges:
        adj.setdefault(int(v0), []).append(int(v1))
        adj.setdefault(int(v1), []).append(int(v0))
    return adj


def extract_boundary_loops(mesh):
    if mesh.is_watertight or len(mesh.faces) == 0:
        return []

    adj = _build_open_edge_adjacency(mesh)
    if not adj:
        return []

    degree = {v: len(nb) for v, nb in adj.items()}
    eligible_vertices = {v for v, d in degree.items() if d == 2}

    visited_edges = set()
    loops = []

    for start in list(eligible_vertices):
        if start not in adj:
            continue
        loop = []
        cur = start
        prev = None

        while True:
            if cur not in eligible_vertices or cur not in adj:
                break
            loop.append(cur)
            candidates = [n for n in adj[cur] if n != prev and n in eligible_vertices]
            if not candidates:
                break
            nxt = candidates[0]
            ekey = (cur, nxt) if cur < nxt else (nxt, cur)
            if ekey in visited_edges:
                break
            visited_edges.add(ekey)
            if nxt == start and len(loop) >= 3:
                loop.append(nxt)
                break
            prev, cur = cur, nxt

            if len(loop) > len(adj):
                break

        if len(loop) >= 4 and loop[0] == loop[-1]:
            loop = loop[:-1]
        if len(loop) >= 3 and loop[0] == start:
            loops.append(loop)
            for v in loop:
                adj.pop(v, None)
                eligible_vertices.discard(v)

    return loops


def compute_hole_area_stats(mesh):
    loops = extract_boundary_loops(mesh)
    if not loops:
        return {
            'count': 0, 'total_area': 0.0,
            'p1_area': 0.0, 'p5_area': 0.0, 'p25_area': 0.0,
            'p50_area': 0.0, 'p75_area': 0.0, 'p90_area': 0.0,
            'p95_area': 0.0, 'p99_area': 0.0, 'max_area': 0.0,
        }

    areas = []
    for loop in loops:
        pts = mesh.vertices[np.asarray(loop, dtype=np.int64)]
        areas.append(polygon_area_from_3d_ccw(pts))
    areas = np.array(areas)

    return {
        'count': len(loops),
        'total_area': float(areas.sum()),
        'p1_area': float(np.percentile(areas, 1)),
        'p5_area': float(np.percentile(areas, 5)),
        'p25_area': float(np.percentile(areas, 25)),
        'p50_area': float(np.percentile(areas, 50)),
        'p75_area': float(np.percentile(areas, 75)),
        'p90_area': float(np.percentile(areas, 90)),
        'p95_area': float(np.percentile(areas, 95)),
        'p99_area': float(np.percentile(areas, 99)),
        'max_area': float(areas.max()),
    }


def polygon_area_from_3d_ccw(points):
    if len(points) < 3:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return 0.0
    total = 0.0
    for i in range(len(points)):
        v1 = points[i]
        v2 = points[(i + 1) % len(points)]
        total += np.dot(np.cross(v1, v2), normal)
    return float(abs(total) / (2.0 * norm))


def compute_reliable_face_mask(mesh, threshold_angle=30.0):
    N = len(mesh.faces)
    if N == 0:
        return np.array([], dtype=float)

    edge_lengths = mesh.edges_unique_length
    med_len = np.median(edge_lengths) if len(edge_lengths) > 0 else 1.0

    faces = np.asarray(mesh.faces, dtype=np.int64)
    v0 = mesh.vertices[faces[:, 0]]
    v1 = mesh.vertices[faces[:, 1]]
    v2 = mesh.vertices[faces[:, 2]]
    e01 = np.linalg.norm(v1 - v0, axis=1)
    e12 = np.linalg.norm(v2 - v1, axis=1)
    e20 = np.linalg.norm(v0 - v2, axis=1)
    avg_len = (e01 + e12 + e20) / 3.0

    len_weight = np.clip(1.0 - np.abs(avg_len - med_len) / (med_len + 1e-9), 0.0, 1.0)

    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    normal_weight = np.ones(N)
    if len(mesh.face_adjacency) > 0:
        for i in range(len(mesh.face_adjacency)):
            f0, f1 = mesh.face_adjacency[i]
            cosang = np.dot(face_normals[f0], face_normals[f1])
            normal_weight[f0] = min(normal_weight[f0], max(0.0, cosang))
            normal_weight[f1] = min(normal_weight[f1], max(0.0, cosang))

    weight = 0.6 * len_weight + 0.4 * normal_weight
    return np.clip(weight, 0.0, 1.0)


def repair_mesh_by_removing_duplicates(mesh, verbose=False):
    m = mesh.copy()
    m.merge_vertices()
    m.remove_unreferenced_vertices()

    if len(m.faces) > 0:
        area = m.area_faces
        degenerate = area < 1e-12
        if np.any(degenerate):
            mask = ~degenerate
            m.update_faces(mask)
            m.remove_unreferenced_vertices()
    return m


def project_vertices_to_shell(points, mesh):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64)
    closest, distance, triangle_id = mesh.nearest.on_surface(pts)
    return closest, distance, triangle_id


def weld_small_holes(mesh, threshold=None, quantile=5.0, min_edges=3, verbose=False):
    m = mesh.copy()
    loops = extract_boundary_loops(m)
    if not loops:
        return m

    areas = np.array([polygon_area_from_3d_ccw(m.vertices[np.asarray(l)])
                      for l in loops])
    if threshold is None:
        threshold = float(np.percentile(areas, quantile)) if len(areas) > 0 else 0.0

    weld_loops = [l for l, a in zip(loops, areas)
                  if a < threshold and len(l) >= min_edges]
    if not weld_loops:
        return m

    if verbose:
        print(f"  weld threshold: {threshold:.6f}")
        print(f"  welding {len(weld_loops)} small holes")

    for loop in weld_loops:
        ids = np.asarray(loop, dtype=np.int64)
        centroid = np.mean(m.vertices[ids], axis=0)
        new_idx = len(m.vertices)
        m.vertices = np.vstack([m.vertices, centroid])
        replace_map = {int(v): new_idx for v in ids}
        faces = np.asarray(m.faces, dtype=np.int64)
        new_faces = []
        for face in faces:
            nf = [replace_map.get(int(v), int(v)) for v in face]
            if len(set(nf)) >= 3:
                new_faces.append(nf)
        m = trimesh.Trimesh(vertices=m.vertices,
                            faces=np.asarray(new_faces, dtype=np.int64),
                            process=False)
        m.remove_unreferenced_vertices()
        m.merge_vertices()
    return repair_mesh_by_removing_duplicates(m)


def repair_nonmanifold_edges(mesh, max_iterations=10, verbose=False):
    m = mesh.copy()
    for _ in range(max_iterations):
        _, _, nonmanifold_face_mask = analyze_mesh_defects(m)
        if not np.any(nonmanifold_face_mask):
            break
        bad_faces = np.where(nonmanifold_face_mask)[0]
        if len(bad_faces) == 0:
            break
        areas = m.area_faces[bad_faces]
        sort_idx = np.argsort(areas)
        remove_count = max(1, len(bad_faces) // 2)
        remove_faces = bad_faces[sort_idx[:remove_count]]
        keep_mask = np.ones(len(m.faces), dtype=bool)
        keep_mask[remove_faces] = False
        m.update_faces(keep_mask)
        m.remove_unreferenced_vertices()
        m.merge_vertices()
    return repair_mesh_by_removing_duplicates(m)


def compute_loop_flatness(mesh, loop):
    if len(loop) < 4:
        return 0.0, None
    pts = mesh.vertices[np.asarray(loop)]
    centroid = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[-1]
    dist = np.abs((pts - centroid) @ normal)
    max_dist = np.max(dist)
    bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    if bbox_diag < 1e-12:
        return 0.0, None
    return float(max_dist / bbox_diag), None


def _point_in_triangle_2d(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _triangulate_hole_loop(coords3d, boundary_indices):
    if len(boundary_indices) < 3:
        return []
    pts = coords3d[boundary_indices]
    centroid = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    u = vh[0]
    v = vh[1]
    pts2d = np.column_stack([(pts - centroid) @ u, (pts - centroid) @ v])

    remaining = list(range(len(pts2d)))
    triangles = []
    guard = 0
    max_guard = len(remaining) * 2

    while len(remaining) > 2 and guard < max_guard:
        guard += 1
        n = len(remaining)
        success = False
        for i in range(n):
            i0 = remaining[i]
            i1 = remaining[(i + 1) % n]
            i2 = remaining[(i + 2) % n]
            p0 = pts2d[i0]
            p1 = pts2d[i1]
            p2 = pts2d[i2]
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            if cross <= 0:
                continue
            inside = False
            for idx in remaining:
                if idx in (i0, i1, i2):
                    continue
                if _point_in_triangle_2d(pts2d[idx], p0, p1, p2):
                    inside = True
                    break
            if inside:
                continue
            triangles.append((i0, i1, i2))
            del remaining[i + 1]
            success = True
            break
        if not success:
            break
    return triangles


def fill_holes_adaptive(mesh,
                        strategy='flatness',
                        max_fan_edges=15,
                        max_fan_flatness=0.05,
                        max_earclip_edges=100,
                        max_earclip_flatness=0.15,
                        max_surface_fit_edges=500,
                        max_surface_fit_flatness=0.40,
                        edge_count_small_p=50.0,
                        edge_count_large_p=95.0,
                        g2=False,
                        verbose=False):
    m = mesh.copy()
    loops = extract_boundary_loops(m)
    if not loops:
        return m
    for loop in loops:
        if len(loop) > 500:
            continue
        flatness, _ = compute_loop_flatness(m, loop)
        if flatness > 0.8:
            continue
    try:
        m.fill_holes()
    except Exception:
        pass
    return repair_mesh_by_removing_duplicates(m)


def repair_normals(mesh, verbose=False):
    mesh.fix_normals()
    return mesh


def remove_small_open_edge_chains(mesh, max_chain_edges=2, verbose=False):
    """
    删除短开放边链（伪孔洞）。

    先一次性构建无向边到面片索引，避免对每个短环都全量扫描面片。
    """
    m = mesh.copy()
    loops = extract_boundary_loops(m)
    if not loops:
        return m

    # 1. 建立边 -> 面片索引，只遍历一次面片
    edge_to_faces = {}
    for fid, face in enumerate(m.faces):
        for i in range(3):
            e0 = int(face[i])
            e1 = int(face[(i + 1) % 3])
            key = (e0, e1) if e0 < e1 else (e1, e0)
            edge_to_faces.setdefault(key, []).append(fid)

    # 2. 根据短环边直接查找相邻面片
    faces_to_remove = set()
    for loop in loops:
        if len(loop) <= max_chain_edges + 1:
            for i in range(len(loop)):
                e0 = int(loop[i])
                e1 = int(loop[(i + 1) % len(loop)])
                key = (e0, e1) if e0 < e1 else (e1, e0)
                faces_to_remove.update(edge_to_faces.get(key, []))

    # 3. 删除这些面片
    if faces_to_remove:
        mask = np.ones(len(m.faces), dtype=bool)
        mask[list(faces_to_remove)] = False
        m.update_faces(mask)
        m.remove_unreferenced_vertices()

    return m


def remove_pseudo_holes(mesh, max_chain_edges=2, max_iterations=5, verbose=False):
    m = mesh.copy()
    for it in range(max_iterations):
        before = len(extract_boundary_loops(m))
        m = remove_small_open_edge_chains(m, max_chain_edges, verbose)
        after = len(extract_boundary_loops(m))
        if after == before:
            break
        if verbose:
            print(f"  [remove_pseudo_holes] iteration {it+1}: "
                  f"boundary loops {before} -> {after}")
    return m


def repair_to_watertight(mesh,
                         resolution=256,
                         voxel_size=None,
                         closing_iterations=2,
                         project_to_input=False,
                         project_distance=None,
                         smooth_watertight=False,
                         smooth_iterations=10,
                         progress=False,
                         verbose=False):
    if voxel_size is None:
        bounds = mesh.bounds
        diag = np.linalg.norm(bounds[1] - bounds[0])
        voxel_size = diag / resolution
    vox = mesh.voxelized(voxel_size)
    try:
        vox = vox.fill()
    except Exception:
        pass
    result = vox.marching_cubes
    if project_to_input:
        try:
            closest, _, _ = mesh.nearest.on_surface(result.vertices)
            result.vertices = closest
        except Exception:
            pass
    if smooth_watertight:
        try:
            result = result.smooth(iterations=smooth_iterations)
        except Exception:
            pass
    return result


def _build_face_adjacency_dict(proxy_mesh):
    adj = {}
    for f0, f1 in proxy_mesh.face_adjacency:
        f0, f1 = int(f0), int(f1)
        adj.setdefault(f0, set()).add(f1)
        adj.setdefault(f1, set()).add(f0)
    return adj


def _find_face_path(adj, start, goal, max_depth=200):
    from collections import deque
    if start == goal:
        return [start]
    parent = {start: None}
    depth = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        if depth[cur] >= max_depth:
            continue
        for nb in adj.get(cur, ()):
            if nb in depth:
                continue
            depth[nb] = depth[cur] + 1
            parent[nb] = cur
            if nb == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return path[::-1]
            q.append(nb)
    return None


def _extract_proxy_interior_faces(proxy_mesh, proj_pts, proj_tris,
                                  max_path_depth=200,
                                  max_inside_ratio=0.5):
    proj_pts = np.asarray(proj_pts, dtype=np.float64)
    proj_tris = np.asarray(proj_tris, dtype=np.int64)
    if len(proj_tris) < 3:
        return set(), np.array([], dtype=np.int64)

    adj = _build_face_adjacency_dict(proxy_mesh)
    face_adj = proxy_mesh.face_adjacency
    try:
        face_adj_edges = proxy_mesh.face_adjacency_edges
    except AttributeError:
        faces = np.asarray(proxy_mesh.faces, dtype=np.int64)
        face_adj_edges = []
        for f0, f1 in face_adj:
            a = set(faces[f0].tolist())
            b = set(faces[f1].tolist())
            shared = list(a & b)
            if len(shared) >= 2:
                face_adj_edges.append((shared[0], shared[1]))
            else:
                face_adj_edges.append((0, 0))
        face_adj_edges = np.asarray(face_adj_edges, dtype=np.int64)

    edge_by_face_pair = {}
    for idx, (f0, f1) in enumerate(face_adj):
        f0, f1 = int(f0), int(f1)
        v0, v1 = int(face_adj_edges[idx][0]), int(face_adj_edges[idx][1])
        key = (f0, f1) if f0 < f1 else (f1, f0)
        edge_by_face_pair[key] = (min(v0, v1), max(v0, v1))

    barrier_faces = set()
    barrier_edges = set()
    n = len(proj_tris)

    for i in range(n):
        a = int(proj_tris[i])
        b = int(proj_tris[(i + 1) % n])
        if a < 0 or b < 0:
            return set(), np.array([], dtype=np.int64)
        path = _find_face_path(adj, a, b, max_depth=max_path_depth)
        if path is None:
            return set(), np.array([], dtype=np.int64)
        barrier_faces.update(path)
        if len(path) >= 2:
            for j in range(len(path) - 1):
                f0, f1 = path[j], path[j + 1]
                key = (min(f0, f1), max(f0, f1))
                edge_key = edge_by_face_pair.get(key)
                if edge_key is not None:
                    barrier_edges.add(edge_key)

    if not barrier_edges:
        return set(), np.array([], dtype=np.int64)

    valid_adj = {}
    for idx, (f0, f1) in enumerate(face_adj):
        f0, f1 = int(f0), int(f1)
        v0, v1 = int(face_adj_edges[idx][0]), int(face_adj_edges[idx][1])
        edge_key = (min(v0, v1), max(v0, v1))
        if edge_key in barrier_edges:
            continue
        valid_adj.setdefault(f0, set()).add(f1)
        valid_adj.setdefault(f1, set()).add(f0)

    if not barrier_faces:
        return set(), np.array([], dtype=np.int64)

    start_face = next(iter(barrier_faces))
    neighbor_seeds = [nb for nb in valid_adj.get(start_face, ())
                      if nb not in barrier_faces]
    if not neighbor_seeds:
        return set(), np.array([], dtype=np.int64)

    regions = []
    for seed in neighbor_seeds:
        visited = {seed}
        stack = [seed]
        while stack:
            cur = stack.pop()
            for nb in valid_adj.get(cur, ()):
                if nb in visited or nb in barrier_faces:
                    continue
                visited.add(nb)
                stack.append(nb)
        if visited:
            regions.append(visited)

    if len(regions) < 2:
        return set(), np.array([], dtype=np.int64)

    regions.sort(key=lambda r: len(r))
    inside_faces = regions[0]
    total_faces = len(proxy_mesh.faces)
    if total_faces > 0 and (len(inside_faces) / total_faces) > max_inside_ratio:
        return set(), np.array([], dtype=np.int64)

    vertex_set = set()
    for f in inside_faces:
        vertex_set.update(int(v) for v in proxy_mesh.faces[f])
    inside_vertices = np.array(sorted(vertex_set), dtype=np.int64)
    return inside_faces, inside_vertices


def _triangulate_polygon_with_interior_points(boundary_pts, interior_pts):
    boundary_pts = np.asarray(boundary_pts, dtype=np.float64)
    interior_pts = np.asarray(interior_pts, dtype=np.float64).reshape(-1, 3)
    if len(boundary_pts) < 3:
        return []
    n_boundary = len(boundary_pts)
    if len(interior_pts) > 0:
        all_pts = np.vstack([boundary_pts, interior_pts])
    else:
        all_pts = boundary_pts

    centroid = boundary_pts.mean(axis=0)
    centered = all_pts - centroid
    _, _, vh = np.linalg.svd(centered)
    u = vh[0]
    v = vh[1]

    coords2d = np.column_stack([centered @ u, centered @ v])
    coords3d = np.column_stack([coords2d, np.zeros(len(coords2d))])
    boundary_idx = np.arange(n_boundary, dtype=np.int64).tolist()
    triangles = _triangulate_hole_loop(coords3d, boundary_idx)
    if not triangles:
        return []
    triangles = [tuple(int(x) for x in tri) for tri in triangles]

    for ip in range(n_boundary, len(all_pts)):
        p2 = coords2d[ip]
        found = None
        for ti, tri in enumerate(triangles):
            a, b, c = tri
            if _point_in_triangle_2d(p2, coords2d[a], coords2d[b], coords2d[c]):
                found = ti
                break
        if found is None:
            continue
        a, b, c = triangles[found]
        triangles[found] = (a, b, ip)
        triangles.append((b, c, ip))
        triangles.append((c, a, ip))
    return triangles


def fill_holes_with_proxy(mesh, proxy_mesh,
                          mask_threshold=0.75,
                          proxy_face_center_threshold=20,
                          max_projection_distance=None,
                          min_proxy_loop_edges=12,
                          verbose=False):
    welded = weld_small_holes(mesh, quantile=5.0, min_edges=3, verbose=verbose)
    loops = extract_boundary_loops(welded)
    if not loops:
        return welded

    work_verts = welded.vertices.copy()
    work_faces = [tuple(int(x) for x in f) for f in welded.faces]
    vert2faces = {}
    for fid, face in enumerate(work_faces):
        for v in face:
            vert2faces.setdefault(v, set()).add(fid)

    try:
        all_verts = np.unique(np.concatenate([np.asarray(l, dtype=np.int64) for l in loops]))
        all_points = work_verts[all_verts]
        proj_points, proj_dist, proj_tri = project_vertices_to_shell(all_points, proxy_mesh)
        proj_map = {
            int(v): (proj_points[i], float(proj_dist[i]), int(proj_tri[i]))
            for i, v in enumerate(all_verts)
        }
    except Exception as e:
        if verbose:
            print(f"  [proxy patch] batch projection failed: {e}")
        return fill_holes_adaptive(welded)

    if verbose:
        print(f"  [proxy patch] processing {len(loops)} boundary loops")

    plane_fill_count = 0
    proxy_fill_count = 0
    fallback_plane_count = 0

    for loop_idx, loop in enumerate(loops):
        if len(loop) < min_proxy_loop_edges:
            if verbose:
                print(f"    loop {loop_idx}: edges={len(loop)} -> plane fill (too small)")
            plane_fill_count += 1
            continue

        try:
            proj_items = [proj_map[v] for v in loop]
            proj_pts = np.array([item[0] for item in proj_items])
            dists = np.array([item[1] for item in proj_items])
            proj_tris = np.array([item[2] for item in proj_items], dtype=np.int64)
        except KeyError:
            proj_pts = None

        if proj_pts is not None and max_projection_distance is not None and \
                float(np.max(dists)) > max_projection_distance:
            proj_pts = None

        if proj_pts is None:
            if verbose:
                print(f"    loop {loop_idx}: projection fails -> plane fill")
            plane_fill_count += 1
            continue

        inside_faces, inside_vertices = _extract_proxy_interior_faces(
            proxy_mesh, proj_pts, proj_tris
        )

        if verbose:
            proxy_total = len(proxy_mesh.faces)
            print(f"    loop {loop_idx}: edges={len(loop)}, "
                  f"proxy_inside_faces={len(inside_faces)} "
                  f"({100.0 * len(inside_faces) / max(proxy_total, 1):.1f}% of proxy)")

        if len(inside_faces) < proxy_face_center_threshold:
            if verbose:
                print(f"      -> plane fill (too few interior faces)")
            plane_fill_count += 1
            continue

        inside_pts = proxy_mesh.vertices[inside_vertices]
        source_pts = work_verts[np.asarray(loop, dtype=np.int64)]
        local_tris = _triangulate_polygon_with_interior_points(source_pts, inside_pts)

        if not local_tris:
            if verbose:
                print(f"      -> plane fill (re-triangulation failed)")
            fallback_plane_count += 1
            continue

        n_source = len(source_pts)
        interior_start = len(work_verts)
        work_verts = np.vstack([work_verts, inside_pts])

        for tri in local_tris:
            mapped = []
            for local_idx in tri:
                if local_idx < n_source:
                    mapped.append(int(loop[local_idx]))
                else:
                    mapped.append(int(interior_start + (local_idx - n_source)))
            nf = tuple(mapped)
            nfi = len(work_faces)
            work_faces.append(nf)
            for v in nf:
                vert2faces.setdefault(v, set()).add(nfi)

        proxy_fill_count += 1
        if verbose:
            print(f"      -> proxy re-triangulation ({len(local_tris)} triangles)")

    if verbose:
        print(f"  [proxy patch] summary: plane_fill={plane_fill_count}, "
              f"proxy_fill={proxy_fill_count}, fallback_plane={fallback_plane_count}")

    final_faces = [f for f in work_faces if f is not None]
    merged = trimesh.Trimesh(vertices=work_verts,
                             faces=np.asarray(final_faces, dtype=np.int64),
                             process=False)
    merged.remove_unreferenced_vertices()
    merged = repair_mesh_by_removing_duplicates(merged)
    merged = weld_small_holes(merged, quantile=5.0, min_edges=3, verbose=verbose)

    if extract_boundary_loops(merged):
        merged = fill_holes_adaptive(merged, verbose=verbose)

    merged = repair_normals(merged, verbose=verbose)
    return merged


def fuse_reliable_faces_with_shell(mesh,
                                   shell_mesh,
                                   mask_threshold=0.75,
                                   proxy_face_center_threshold=20,
                                   max_projection_distance=None,
                                   min_proxy_loop_edges=12,
                                   smooth_transition=True,
                                   smooth_iterations=3,
                                   smooth_alpha=0.5,
                                   verbose=False):
    m = mesh.copy()
    weights = compute_reliable_face_mask(m)
    reliable_mask = weights > mask_threshold
    if reliable_mask.sum() == 0:
        return m

    faces = np.asarray(m.faces)[reliable_mask]
    flat = faces.ravel()
    unique, inverse = np.unique(flat, return_inverse=True)
    reliable_mesh = trimesh.Trimesh(
        vertices=m.vertices[unique],
        faces=inverse.reshape(-1, 3),
        process=False,
    )
    reliable_mesh.remove_unreferenced_vertices()
    reliable_mesh.merge_vertices()
    reliable_mesh = repair_mesh_by_removing_duplicates(reliable_mesh)

    repaired = fill_holes_with_proxy(
        reliable_mesh, shell_mesh,
        proxy_face_center_threshold=proxy_face_center_threshold,
        max_projection_distance=max_projection_distance,
        min_proxy_loop_edges=min_proxy_loop_edges,
        verbose=verbose,
    )

    if smooth_transition and smooth_alpha > 0:
        try:
            repaired = repaired.smooth(iterations=smooth_iterations)
        except Exception:
            pass

    return repaired
