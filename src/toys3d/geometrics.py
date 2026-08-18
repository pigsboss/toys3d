"""Geometry utilities for mesh inspection and repair."""

import numpy as np
import trimesh


def compute_mesh_stats(mesh):
    """Compute basic statistics about a mesh."""
    stats = {}
    stats['vertices'] = int(len(mesh.vertices))
    stats['faces'] = int(len(mesh.faces))
    stats['edges'] = int(len(mesh.edges_unique))
    stats['is_watertight'] = bool(mesh.is_watertight)

    if len(mesh.edges_unique) > 0:
        v1 = mesh.vertices[mesh.edges_unique[:, 0]]
        v2 = mesh.vertices[mesh.edges_unique[:, 1]]
        edge_len = np.linalg.norm(v1 - v2, axis=1)
        stats['mean_edge_length'] = float(np.mean(edge_len))
        for p in [1, 5, 50, 95, 99]:
            stats[f'edge_length_p{p}'] = float(np.percentile(edge_len, p))
    else:
        stats['mean_edge_length'] = 0.0
        for p in [1, 5, 50, 95, 99]:
            stats[f'edge_length_p{p}'] = 0.0

    stats['area'] = float(mesh.area)
    return stats


def analyze_mesh_defects(mesh):
    """
    Analyze open and non‑manifold edges.

    Returns
    -------
    defect_stats : dict
    open_face_mask : np.ndarray, bool
    nonmanifold_face_mask : np.ndarray, bool
    """
    n_faces = len(mesh.faces)
    edges = np.asarray(mesh.edges, dtype=np.int64)
    edges_sorted = np.sort(edges, axis=1)

    edge_to_faces = {}
    for i, e in enumerate(edges_sorted):
        key = (int(e[0]), int(e[1]))
        face_idx = i // 3
        edge_to_faces.setdefault(key, []).append(face_idx)

    open_edges = 0
    nonmanifold_edges = 0
    open_face_mask = np.zeros(n_faces, dtype=bool)
    nonmanifold_face_mask = np.zeros(n_faces, dtype=bool)

    for faces in edge_to_faces.values():
        count = len(faces)
        if count == 1:
            open_edges += 1
            open_face_mask[faces] = True
        elif count > 2:
            nonmanifold_edges += 1
            nonmanifold_face_mask[faces] = True

    defect_stats = {
        'open_edges': open_edges,
        'nonmanifold_edges': nonmanifold_edges,
        'open_faces': int(open_face_mask.sum()),
        'nonmanifold_faces': int(nonmanifold_face_mask.sum()),
    }
    return defect_stats, open_face_mask, nonmanifold_face_mask


def repair_mesh_by_removing_duplicates(mesh):
    """Remove duplicate and degenerate faces."""
    if len(mesh.faces) == 0:
        return mesh.copy()

    faces = np.asarray(mesh.faces, dtype=np.int64)
    sorted_faces = np.sort(faces, axis=1)

    # First occurrence of each unique sorted face
    _, inverse = np.unique(sorted_faces, axis=0, return_inverse=True)
    first_occurrence = np.zeros(len(faces), dtype=bool)
    first_occurrence[np.unique(inverse, return_index=True)[1]] = True

    # Mark degenerate faces (zero area)
    verts = mesh.vertices
    tri_verts = verts[faces]
    cross = np.cross(tri_verts[:, 1] - tri_verts[:, 0],
                     tri_verts[:, 2] - tri_verts[:, 0], axis=1)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid_area = areas > 1e-12

    keep = first_occurrence & valid_area
    new_mesh = mesh.copy()
    new_mesh.update_faces(keep)
    new_mesh.remove_unreferenced_vertices()
    return new_mesh


def repair_nonmanifold_edges(mesh, max_iterations=10, verbose=False):
    """Remove faces that are adjacent to non‑manifold edges."""
    repaired = mesh.copy()
    for _ in range(max_iterations):
        defects, _, nonmanifold_faces = analyze_mesh_defects(repaired)
        if defects['nonmanifold_edges'] == 0:
            break
        keep = ~nonmanifold_faces
        if not keep.any():
            break
        repaired.update_faces(keep)
        repaired.remove_unreferenced_vertices()
    return repaired


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
                        verbose=True):
    """
    自适应孔洞修复。

    该实现先用边界环大小/平坦度对孔洞分类并打印统计，然后：
    - 对超过阈值的孔洞临时用扇形面片封住，避免被 trimesh fill_holes 填充；
    - 调用 trimesh.repair.fill_holes 填充剩余的小、中、大孔洞；
    - 最后移除临时扇形面片，保留被跳过的孔洞。
    """
    import time

    loops = extract_boundary_loops(mesh)
    if not loops:
        if verbose:
            print("  Boundary loops: 0")
        return mesh.copy()

    counts = {'fan': 0, 'earclip': 0, 'surface_fit': 0, 'skipped': 0}
    skip_loops = []
    selected_loops = []

    for loop in loops:
        n = len(loop)
        if n < 3:
            counts['skipped'] += 1
            continue

        if strategy == 'edge-count':
            if n <= max_fan_edges:
                key = 'fan'
            elif n <= max_earclip_edges:
                key = 'earclip'
            elif n <= max_surface_fit_edges:
                key = 'surface_fit'
            else:
                key = 'skipped'
        else:
            flat = compute_loop_flatness(mesh, loop)[0]
            if n <= max_fan_edges and flat <= max_fan_flatness:
                key = 'fan'
            elif n <= max_earclip_edges and flat <= max_earclip_flatness:
                key = 'earclip'
            elif n <= max_surface_fit_edges and flat <= max_surface_fit_flatness:
                key = 'surface_fit'
            else:
                key = 'skipped'

        counts[key] += 1
        if key == 'skipped':
            skip_loops.append(loop)
        else:
            selected_loops.append(loop)

    if verbose:
        print(f"  Strategy: {strategy}")
        print(f"  Thresholds: "
              f"small<={max_fan_edges}, "
              f"medium<={max_earclip_edges}, "
              f"large<={max_surface_fit_edges}")
        print(f"  Boundary loops: {len(loops)}")
        print(f"    fan fill: {counts['fan']}")
        print(f"    ear clip: {counts['earclip']}")
        print(f"    surface fit: {counts['surface_fit']}")
        print(f"    skipped: {counts['skipped']}")

    if not selected_loops:
        if verbose:
            print("  No holes within thresholds to fill.")
        return mesh.copy()

    # 临时封住 skipped 孔洞，避免 trimesh.fill_holes 填充它们
    temp_mesh = mesh.copy()
    temp_face_keys = set()

    if skip_loops:
        added_faces = []
        for loop in skip_loops:
            n = len(loop)
            for k in range(1, n - 1):
                tri = (int(loop[0]), int(loop[k]), int(loop[k + 1]))
                added_faces.append(tri)
                temp_face_keys.add(tuple(sorted(tri)))

        if added_faces:
            temp_mesh.faces = np.vstack([
                temp_mesh.faces,
                np.asarray(added_faces, dtype=np.int64)
            ])

    from trimesh.repair import fill_holes as trimesh_fill_holes

    t0 = time.time()
    trimesh_fill_holes(temp_mesh)
    if verbose:
        print(f"  Trimesh fill_holes completed in {time.time() - t0:.2f}s")

    # 移除临时 skipped 孔洞扇形面片
    if temp_face_keys:
        def is_keep(face):
            return tuple(sorted(face)) not in temp_face_keys

        keep = np.array([is_keep(f) for f in temp_mesh.faces], dtype=bool)
        temp_mesh.update_faces(keep)
        temp_mesh.remove_unreferenced_vertices()

    return temp_mesh


def extract_boundary_loops(mesh):
    """Extract closed boundary loops as lists of vertex indices."""
    edges = np.asarray(mesh.edges, dtype=np.int64)
    if len(edges) == 0:
        return []
    edges_sorted = np.sort(edges, axis=1)

    edge_counts = {}
    for e in edges_sorted:
        key = (int(e[0]), int(e[1]))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    boundary = [key for key, cnt in edge_counts.items() if cnt == 1]
    if not boundary:
        return []

    adj = {}
    for a, b in boundary:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    loops = []
    visited = set()
    for start in list(adj.keys()):
        if start in visited:
            continue
        loop = []
        current = start
        prev = None
        while True:
            if current in visited:
                break
            loop.append(current)
            visited.add(current)
            nbrs = adj.get(current, [])
            nxt = None
            for nb in nbrs:
                if nb != prev:
                    nxt = nb
                    break
            if nxt is None:
                break
            prev = current
            current = nxt

        # Remove duplicate end if the loop is closed
        if loop and loop[0] == loop[-1]:
            loop = loop[:-1]

        if len(loop) > 1:
            loops.append(loop)

    return loops


def compute_loop_flatness(mesh, loop):
    """Return a flatness measure for a boundary loop (0 = perfect plane)."""
    pts = mesh.vertices[np.asarray(loop, dtype=np.int64)]
    if len(pts) < 3:
        return [0.0]

    centroid = pts.mean(axis=0)
    cov = (pts - centroid).T @ (pts - centroid)
    eigvals = np.linalg.eigvalsh(cov)
    max_eig = eigvals[-1]
    if max_eig <= 1e-12:
        return [0.0]
    flatness = eigvals[0] / max_eig
    return [float(flatness)]


def polygon_area_from_3d_ccw(pts):
    """Compute area of a 3D polygon using Newell's method."""
    if len(pts) < 3:
        return 0.0
    normal = np.zeros(3)
    n = len(pts)
    for i in range(n):
        v1 = pts[i]
        v2 = pts[(i + 1) % n]
        normal += np.cross(v1, v2)
    return 0.5 * np.linalg.norm(normal)


def repair_normals(mesh, reference_face_mask=None, verbose=False):
    """
    修复网格面片绕序，使相邻流形面片法线方向一致。

    仅处理流形连通分量内部：对每条流形边，要求两个面片的有向半边
    互为反向。若不一致，则翻转邻居面片。非流形边不参与传播；未访问
    到的面片保持原方向。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    reference_face_mask : (N,) bool ndarray or None
        可选，标记载信/原始面片。若提供，每个连通分量优先选择这些面片
        中面积最大的作为传播种子。
    verbose : bool
        是否打印进度和各阶段耗时。
    """
    import time

    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    N = len(faces)
    if N == 0:
        return mesh.copy()

    t0 = time.time()

    # --------------------------------------------------------------
    # Phase 0: 构建边信息与流形邻接
    # --------------------------------------------------------------
    edge_data = {}
    for fi, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_data.setdefault(key, []).append((fi, (a, b)))

    manifold_adjacency = [[] for _ in range(N)]
    for key, face_list in edge_data.items():
        if len(face_list) == 2:
            fi, _fi_dir = face_list[0]
            fj, _fj_dir = face_list[1]
            manifold_adjacency[fi].append((fj, key))
            manifold_adjacency[fj].append((fi, key))

    # 记录每个面片在每条边上的原始有向半边
    halfedge_lookup = {}
    for key, face_list in edge_data.items():
        for fi, directed in face_list:
            halfedge_lookup[(fi, key)] = directed

    if verbose:
        print(f"  [repair_normals] edge data built in "
              f"{time.time() - t0:.2f}s", flush=True)

    # --------------------------------------------------------------
    # Phase 1: 计算流形连通分量
    # --------------------------------------------------------------
    t0 = time.time()

    visited = np.zeros(N, dtype=bool)
    labels = -np.ones(N, dtype=np.int64)
    comp_id = 0

    for start in range(N):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        labels[start] = comp_id
        while queue:
            i = queue.pop(0)
            for j, _key in manifold_adjacency[i]:
                if not visited[j]:
                    visited[j] = True
                    labels[j] = comp_id
                    queue.append(j)
        comp_id += 1

    n_components = comp_id
    area_faces = mesh.area_faces

    if verbose:
        print(f"  [repair_normals] components computed in "
              f"{time.time() - t0:.2f}s, n_components={n_components}",
              flush=True)

    # --------------------------------------------------------------
    # Phase 2: 对每个分量选择种子并 BFS 传播
    # --------------------------------------------------------------
    t0 = time.time()

    flipped = np.zeros(N, dtype=bool)
    visited = np.zeros(N, dtype=bool)

    conflict_count = 0

    components = {}
    for fi, lbl in enumerate(labels):
        components.setdefault(int(lbl), []).append(fi)

    if verbose:
        print("  [repair_normals] propagating orientations...",
              flush=True)

    report_interval = max(1, n_components // 10)
    processed_components = 0

    for comp_faces in components.values():
        if verbose and processed_components % report_interval == 0:
            print(f"    component {processed_components}/{n_components} "
                  f"({100 * processed_components / n_components:.0f}%) "
                  f"+{time.time() - t0:.2f}s", flush=True)
        processed_components += 1

        comp_arr = np.asarray(comp_faces, dtype=int)
        candidates = comp_arr

        if reference_face_mask is not None:
            ref_mask = np.asarray(reference_face_mask, dtype=bool)
            ref_candidates = comp_arr[ref_mask[comp_arr]]
            if len(ref_candidates) > 0:
                candidates = ref_candidates

        seed = int(candidates[np.argmax(area_faces[candidates])])

        visited[seed] = True
        flipped[seed] = False
        queue = [seed]

        while queue:
            i = queue.pop(0)

            for j, key in manifold_adjacency[i]:
                if visited[j]:
                    continue

                di = halfedge_lookup[(i, key)]
                dj = halfedge_lookup[(j, key)]

                actual_di = (di[1], di[0]) if flipped[i] else di

                # 正确流形要求 j 的有向半边等于 actual_di 的反向
                required_dj = (actual_di[1], actual_di[0])

                if dj == required_dj:
                    flipped[j] = False
                else:
                    if (dj[1], dj[0]) == required_dj:
                        flipped[j] = True
                    else:
                        # 理论上退化/冲突，保守保持原方向
                        flipped[j] = False
                        conflict_count += 1

                visited[j] = True
                queue.append(j)

    unvisited_count = int(np.sum(~visited))

    if verbose:
        print(f"  [repair_normals] orientation propagation done in "
              f"{time.time() - t0:.2f}s", flush=True)

    # --------------------------------------------------------------
    # Phase 3: 应用翻转并重建网格
    # --------------------------------------------------------------
    t0 = time.time()

    flip_count = int(np.sum(flipped))
    faces[flipped] = faces[flipped][:, [1, 0, 2]]

    repaired = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=faces,
        process=False,
    )

    if verbose:
        print(f"  [repair_normals] flip applied in "
              f"{time.time() - t0:.2f}s", flush=True)
        print(f"  Normal repair: flipped {flip_count} faces")
        print(f"  Normal repair: conflicts {conflict_count}")
        print(f"  Normal repair: unvisited faces {unvisited_count}")

    return repaired


def remove_isolated_components(mesh,
                               min_faces=20,
                               min_area=None,
                               min_ratio=0.001,
                               verbose=False):
    """Remove small isolated components."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh.copy()

    total_faces = len(mesh.faces)
    total_area = mesh.area
    keep = []
    for comp in components:
        nf = len(comp.faces)
        if nf < min_faces:
            continue
        if min_area is not None and comp.area < min_area:
            continue
        if nf / max(total_faces, 1) < min_ratio:
            continue
        keep.append(comp)

    if not keep:
        return mesh.copy()
    return trimesh.util.concatenate(keep)


def remove_quasi_isolated_components(mesh,
                                     radius=None,
                                     n_sample=2000,
                                     min_faces=30,
                                     max_ratio=0.05,
                                     remove_bridge=False,
                                     rng=None,
                                     verbose=False):
    """Remove components that are small and weakly connected (simplified)."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh.copy()

    total_faces = len(mesh.faces)
    keep = []
    for comp in components:
        frac = len(comp.faces) / max(total_faces, 1)
        if frac > max_ratio:
            keep.append(comp)
        elif len(comp.faces) >= min_faces:
            keep.append(comp)
        # Otherwise discard

    if not keep:
        return mesh.copy()
    return trimesh.util.concatenate(keep)


def compute_hole_area_stats(mesh):
    """Compute statistics for boundary loops of a mesh."""
    loops = extract_boundary_loops(mesh)
    areas = []
    for loop in loops:
        pts = mesh.vertices[np.asarray(loop, dtype=np.int64)]
        areas.append(polygon_area_from_3d_ccw(pts))

    stats = {'count': len(loops)}
    if areas:
        areas = np.array(areas)
        stats['total_area'] = float(areas.sum())
        for p in [1, 5, 25, 50, 75, 90, 95, 99]:
            stats[f'p{p}_area'] = float(np.percentile(areas, p))
        stats['max_area'] = float(areas.max())
    else:
        stats['total_area'] = 0.0
        for p in [1, 5, 25, 50, 75, 90, 95, 99]:
            stats[f'p{p}_area'] = 0.0
        stats['max_area'] = 0.0
    return stats
