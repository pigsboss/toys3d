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


def build_face_adjacency(mesh):
    """
    构建每个面片的邻接面片索引列表。

    返回的列表长度等于面片数量，每个元素是 np.ndarray，
    表示与当前面片共享至少一条边的其它面片索引。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    n_faces = len(faces)

    if n_faces == 0:
        return [np.array([], dtype=np.int64) for _ in range(n_faces)]

    adjacency = [set() for _ in range(n_faces)]
    edge_to_faces = {}

    for fi, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in ((v1, v2), (v2, v3), (v3, v1)):
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(fi)

    for face_indices in edge_to_faces.values():
        if len(face_indices) < 2:
            continue
        for i in face_indices:
            for j in face_indices:
                if i != j:
                    adjacency[i].add(j)

    return [np.array(sorted(s), dtype=np.int64) for s in adjacency]


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


def _signed_area_2d(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(
        np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    )


def _point_in_triangle_2d(pt, a, b, c, eps=1e-12):
    """Barycentric test: pt is inside/on triangle a,b,c in 2D."""
    v0x, v0y = c[0] - a[0], c[1] - a[1]
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = pt[0] - a[0], pt[1] - a[1]

    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-30:
        return False

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv

    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def _triangulate_hole_loop(vertices, loop):
    """
    将单个 3D 边界环三角化。

    此函数将环投影到最佳拟合平面，使用 ear clipping 生成三角形。
    若 ear clipping 失败，则退化为 fan triangulation。
    """
    loop = np.asarray(loop, dtype=np.int64)
    n = len(loop)

    if n < 3:
        return []
    if n == 3:
        return [tuple(int(v) for v in loop)]

    pts = vertices[loop]
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # 用 SVD 找到最佳拟合平面，将 3D 环投影到 2D
    _, _, vh = np.linalg.svd(centered)
    u = vh[0]
    v = vh[1]
    coords = np.column_stack([centered @ u, centered @ v])

    # 保持 2D 多边形为逆时针
    if _signed_area_2d(coords) < 0:
        loop = loop[::-1]
        coords = coords[::-1]

    idxs = list(range(n))
    triangles = []

    eps = 1e-12

    while len(idxs) > 3:
        m = len(idxs)
        ear_found = False

        for i in range(m):
            p = idxs[i - 1]
            c = idxs[i]
            q = idxs[(i + 1) % m]

            # 凸顶点判断
            cross = (
                (coords[c, 0] - coords[p, 0]) *
                (coords[q, 1] - coords[p, 1]) -
                (coords[c, 1] - coords[p, 1]) *
                (coords[q, 0] - coords[p, 0])
            )
            if cross <= eps:
                continue

            inside = False
            for other in idxs:
                if other in (p, c, q):
                    continue
                if _point_in_triangle_2d(
                    coords[other],
                    coords[p],
                    coords[c],
                    coords[q],
                ):
                    inside = True
                    break

            if not inside:
                triangles.append((
                    int(loop[p]),
                    int(loop[c]),
                    int(loop[q]),
                ))
                del idxs[i]
                ear_found = True
                break

        if not ear_found:
            # 退化情况，退化为 fan
            break

    if len(idxs) == 3:
        p, c, q = idxs
        triangles.append((
            int(loop[p]),
            int(loop[c]),
            int(loop[q]),
        ))
    elif len(idxs) > 3:
        # fallback fan from first vertex
        for k in range(1, len(idxs) - 1):
            triangles.append((
                int(loop[idxs[0]]),
                int(loop[idxs[k]]),
                int(loop[idxs[k + 1]]),
            ))

    return triangles


def _extract_boundary_loops_oriented(mesh):
    """
    提取开放边界环，并调整为适合补孔的方向。

    新三角面片需要与相邻面片保持相反的共享有向半边。此函数返回
    经过方向校正后的边界环。
    """
    loops = extract_boundary_loops(mesh)
    if not loops:
        return []

    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

    edge_counts = {}
    for face in faces:
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    boundary_face_dir = {}
    for face in faces:
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            if edge_counts.get(key, 0) == 1:
                boundary_face_dir[key] = (a, b)

    oriented = []
    for loop in loops:
        loop = list(loop)
        if len(loop) < 3:
            oriented.append(loop)
            continue

        for i in range(len(loop)):
            a = int(loop[i])
            b = int(loop[(i + 1) % len(loop)])
            key = (a, b) if a < b else (b, a)
            face_dir = boundary_face_dir.get(key)
            if face_dir is not None:
                # 若当前 loop 方向与相邻面片一致，需要反向
                if face_dir == (a, b):
                    loop = loop[::-1]
                break

        oriented.append(loop)

    return oriented


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

    对阈值内的开放边界环直接三角化：
    - 小环：fan fill
    - 中环：ear clip
    - 大环：surface fit（此处使用同一 ear clipping 三角化）
    - 超过最大阈值的环保持开放，不填充
    """
    import time

    loops = _extract_boundary_loops_oriented(mesh)
    if not loops:
        if verbose:
            print("  Boundary loops: 0")
        return mesh.copy()

    counts = {'fan': 0, 'earclip': 0, 'surface_fit': 0, 'skipped': 0}
    selected_by_type = {'fan': [], 'earclip': [], 'surface_fit': []}

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
        if key != 'skipped':
            selected_by_type[key].append(loop)

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

    total_selected = sum(len(v) for v in selected_by_type.values())
    if total_selected == 0:
        if verbose:
            print("  No holes within thresholds to fill.")
        return mesh.copy()

    t0 = time.time()
    added_faces = []

    for fill_type in ['fan', 'earclip', 'surface_fit']:
        loops_of_type = selected_by_type[fill_type]
        if not loops_of_type:
            continue

        if verbose:
            print(f"  Filling {fill_type} loops ({len(loops_of_type)}):",
                  flush=True)

        report_interval = max(1, len(loops_of_type) // 10)

        for idx, loop in enumerate(loops_of_type, 1):
            tris = _triangulate_hole_loop(mesh.vertices, loop)
            added_faces.extend(tris)

            if verbose and idx % report_interval == 0:
                print(f"    [{idx}/{len(loops_of_type)}] "
                      f"{fill_type} fill "
                      f"({100 * idx / len(loops_of_type):.0f}%) "
                      f"+{time.time() - t0:.2f}s", flush=True)

    if not added_faces:
        if verbose:
            print("  Hole filling produced no triangles.")
        return mesh.copy()

    new_faces = np.vstack([
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(added_faces, dtype=np.int64).reshape(-1, 3),
    ])

    repaired = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=new_faces,
        process=False,
    )

    repaired.remove_unreferenced_vertices()

    if verbose:
        print(f"  Hole filling completed in {time.time() - t0:.2f}s, "
              f"added {len(added_faces)} faces")

    return repaired


def extract_boundary_loops(mesh):
    """
    只提取真正闭合的开放边界环。

    孤立的开放边或断开的开放边链不会作为闭合环返回，
    避免后续补孔逻辑把它们误认为可以三角化的孔洞。
    """
    edges = np.asarray(mesh.edges, dtype=np.int64)
    if len(edges) == 0:
        return []

    edges_sorted = np.sort(edges, axis=1)
    edge_counts = {}
    for e in edges_sorted:
        key = (int(e[0]), int(e[1]))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    boundary_edges = [key for key, cnt in edge_counts.items() if cnt == 1]
    if not boundary_edges:
        return []

    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    visited_edges = set()
    loops = []

    for a, b in boundary_edges:
        key = (a, b) if a < b else (b, a)
        if key in visited_edges:
            continue

        # 提取该边界边所在的完整连通分量
        component_edges = []
        stack = [key]

        while stack:
            u, v = stack.pop()
            k = (u, v) if u < v else (v, u)
            if k in visited_edges:
                continue
            visited_edges.add(k)
            component_edges.append((u, v))

            for w in adj.get(u, ()):
                k2 = (u, w) if u < w else (w, u)
                if k2 not in visited_edges:
                    stack.append(k2)

            for w in adj.get(v, ()):
                k2 = (v, w) if v < w else (w, v)
                if k2 not in visited_edges:
                    stack.append(k2)

        vertices = set()
        for u, v in component_edges:
            vertices.add(u)
            vertices.add(v)

        deg = {v: len(adj.get(v, ())) for v in vertices}

        # 是否为单一简单闭合环：
        # 顶点数 == 边数，每个顶点度数为 2，且至少 3 个顶点
        is_closed_loop = (
            len(component_edges) == len(vertices) and
            len(vertices) >= 3 and
            all(d == 2 for d in deg.values())
        )

        if not is_closed_loop:
            continue

        component_adj = {v: list(adj[v]) for v in vertices}
        start = next(iter(vertices))
        prev = None
        cur = start
        loop = []

        while True:
            loop.append(cur)

            nbrs = [nb for nb in component_adj.get(cur, []) if nb != prev]
            if not nbrs:
                loop = []
                break

            nxt = nbrs[0]
            prev, cur = cur, nxt

            if cur == start:
                break

        if len(loop) >= 3:
            loops.append(loop)

    return loops


def extract_open_edge_chains(mesh):
    """
    提取不闭合的开放边链。

    返回列表，每个元素为 dict：
        edges:      list of (u, v)
        vertices:   set of vertices
        is_cycle:   bool
        endpoints:  度数为 1 的端点列表
    """
    edges = np.asarray(mesh.edges, dtype=np.int64)
    if len(edges) == 0:
        return []

    edges_sorted = np.sort(edges, axis=1)
    edge_counts = {}
    for e in edges_sorted:
        key = (int(e[0]), int(e[1]))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    boundary_edges = [key for key, cnt in edge_counts.items() if cnt == 1]
    if not boundary_edges:
        return []

    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    visited_edges = set()
    chains = []

    for a, b in boundary_edges:
        key = (a, b) if a < b else (b, a)
        if key in visited_edges:
            continue

        component_edges = []
        stack = [key]

        while stack:
            u, v = stack.pop()
            k = (u, v) if u < v else (v, u)
            if k in visited_edges:
                continue
            visited_edges.add(k)
            component_edges.append((u, v))

            for w in adj.get(u, ()):
                k2 = (u, w) if u < w else (w, u)
                if k2 not in visited_edges:
                    stack.append(k2)

            for w in adj.get(v, ()):
                k2 = (v, w) if v < w else (w, v)
                if k2 not in visited_edges:
                    stack.append(k2)

        vertices = set()
        for u, v in component_edges:
            vertices.add(u)
            vertices.add(v)

        deg = {v: len(adj.get(v, ())) for v in vertices}

        is_cycle = (
            len(component_edges) == len(vertices) and
            len(vertices) >= 3 and
            all(d == 2 for d in deg.values())
        )

        endpoints = [v for v in vertices if deg[v] == 1]

        chains.append({
            'edges': component_edges,
            'vertices': vertices,
            'is_cycle': is_cycle,
            'endpoints': endpoints,
        })

    return chains


def remove_small_open_edge_chains(mesh,
                                  max_chain_edges=2,
                                  verbose=False):
    """
    删除短小且不闭合的开放边链关联面片。

    这类结构通常是由非流形修复或 STL 拼接产生的细长尖刺/孤边，
    无法通过孔洞三角化补成水密。删除后可让下一轮填孔更稳定。
    """
    chains = extract_open_edge_chains(mesh)
    if not chains:
        return mesh.copy()

    edge_to_faces = {}
    for fi, face in enumerate(mesh.faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in ((v1, v2), (v2, v3), (v3, v1)):
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(fi)

    face_remove = set()
    removed_chain_count = 0

    for chain in chains:
        if chain['is_cycle']:
            continue

        if len(chain['edges']) > max_chain_edges:
            continue

        # 只处理简单开放链，不处理分叉结构
        if len(chain['endpoints']) != 2:
            continue

        # 边数 == 顶点数 - 1，确保是没有分支的简单路径
        if len(chain['vertices']) != len(chain['edges']) + 1:
            continue

        for a, b in chain['edges']:
            key = (a, b) if a < b else (b, a)
            face_remove.update(edge_to_faces.get(key, []))

        removed_chain_count += 1

    if not face_remove:
        return mesh.copy()

    keep = np.ones(len(mesh.faces), dtype=bool)
    for fi in face_remove:
        keep[fi] = False

    repaired = mesh.copy()
    repaired.update_faces(keep)
    repaired.remove_unreferenced_vertices()

    if verbose:
        print(f"  Removed {removed_chain_count} small open edge chains, "
              f"{len(face_remove)} faces")

    return repaired


def remove_pseudo_holes(mesh,
                        max_chain_edges=2,
                        max_iterations=5,
                        verbose=False):
    """
    迭代删除短小且不闭合的开放边链（伪孔洞）。

    这类结构通常不是真正的拓扑孔洞，而是由拼接、错位或非流形修复
    产生的细长尖刺/裂缝。直接补孔容易引入新的非流形边，因此先删除
    它们可以显著减少后续修复和融合时的边界环数量。

    Parameters
    ----------
    mesh : trimesh.Trimesh
        输入网格。
    max_chain_edges : int
        伪孔洞开放边链的最大边数，默认 2。
    max_iterations : int
        最大迭代清理次数。每次删除会暴露新的短链，因此需要迭代。
    verbose : bool
        是否打印每轮删除统计。

    Returns
    -------
    cleaned : trimesh.Trimesh
    """
    cleaned = mesh.copy()
    merged_vertices = False

    for it in range(1, max_iterations + 1):
        before_faces = len(cleaned.faces)
        cleaned = remove_small_open_edge_chains(
            cleaned,
            max_chain_edges=max_chain_edges,
            verbose=verbose,
        )
        after_faces = len(cleaned.faces)

        if verbose:
            print(f"  [remove_pseudo_holes] iteration {it}: "
                  f"removed {before_faces - after_faces} faces")

        if after_faces == before_faces:
            break

    # 清理可能因删除产生的孤立顶点和重复面片
    cleaned.merge_vertices()
    cleaned.remove_unreferenced_vertices()
    cleaned = repair_mesh_by_removing_duplicates(cleaned)

    return cleaned


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


def repair_to_watertight(mesh,
                         resolution=256,
                         voxel_size=None,
                         closing_iterations=2,
                         min_component_faces=100,
                         project_to_input=False,
                         project_distance=None,
                         smooth_watertight=False,
                         smooth_iterations=10,
                         progress=False,
                         verbose=False):
    """
    通过体素化 + 形态学封闭 + Marching Cubes 重建水密网格。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    resolution : int
        默认体素分辨率，按包围盒对角线均分。
    voxel_size : float or None
        显式指定体素边长；若给出，则 resolution 被忽略。
    closing_iterations : int
        3D 形态学闭运算迭代次数，用于关闭细小孔洞和缝隙。
    min_component_faces : int
        删除小于该面片数的孤立连通分量。
    project_to_input : bool
        是否将重建网格顶点投影回原始输入网格表面，减少体素锯齿感。
    project_distance : float or None
        投影最大搜索/移动距离。默认取 0.5 * pitch。
    smooth_watertight : bool
        是否在投影后执行 Taubin 平滑。
    smooth_iterations : int
        Taubin 平滑迭代次数，默认 10。
    progress : bool
        是否显示形态学闭运算等阶段进度条。需要 tqdm；如未安装则打印百分比。
    verbose : bool
        是否输出进度与耗时。
    """
    import time

    try:
        import scipy.ndimage as ndi
    except Exception as e:
        raise RuntimeError("scipy is required for --watertight mode") from e

    t0 = time.time()

    m = mesh.copy()
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    m = repair_mesh_by_removing_duplicates(m)

    if len(m.faces) == 0:
        return m

    extents = m.bounding_box.extents
    diag = float(np.linalg.norm(extents))
    pitch = voxel_size if voxel_size is not None else (diag / max(resolution, 1))
    pitch = float(pitch)

    if verbose:
        print(
            f"  [watertight] resolution={resolution}, "
            f"voxel_size={voxel_size}, "
            f"closing_iterations={closing_iterations}, "
            f"diag={diag:.4f}, pitch={pitch:.6f}"
        )

    from trimesh.voxel.creation import voxelize

    if verbose:
        print("  [watertight] voxelizing surface...", flush=True)

    t_vox = time.time()
    vox = voxelize(m, pitch=pitch)
    grid = np.asarray(vox.matrix, dtype=bool)

    if verbose:
        print(f"  [watertight] voxelization done in "
              f"{time.time() - t_vox:.2f}s, "
              f"grid shape={grid.shape}",
              flush=True)

    pad = closing_iterations + 1
    big = np.pad(grid, pad, mode='constant', constant_values=False)

    if closing_iterations > 0:
        struct = ndi.generate_binary_structure(3, 1)

        has_tqdm = False
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        total_steps = closing_iterations * 2

        if progress and has_tqdm:
            pbar = tqdm(total=total_steps,
                        desc="Morphological closing",
                        unit="step")
        elif progress:
            pbar = None
            print(f"  Morphological closing: 0/{total_steps} steps",
                  flush=True)
        else:
            pbar = None

        t_close = time.time()

        # Dilation
        for i in range(closing_iterations):
            big = ndi.binary_dilation(big, structure=struct)
            if pbar is not None:
                pbar.update(1)
            elif progress:
                print(f"    closing dilation {i+1}/{closing_iterations}",
                      flush=True)

        # Erosion
        for i in range(closing_iterations):
            big = ndi.binary_erosion(big, structure=struct)
            if pbar is not None:
                pbar.update(1)
            elif progress:
                print(f"    closing erosion {i+1}/{closing_iterations}",
                      flush=True)

        if pbar is not None:
            pbar.close()

        if verbose:
            print(f"  [watertight] morphological closing done in "
                  f"{time.time() - t_close:.2f}s",
                  flush=True)

    # 填充内部空腔；对于已经封闭的表面，这会把内部变成实心。
    big = ndi.binary_fill_holes(big)

    # 使用完整 padded 体素栅格提取表面，不再裁剪。
    # 否则裁剪会在体素边界处制造人工开放边。
    grid_filled = big

    if verbose:
        print("  [watertight] marching cubes...", flush=True)
    t_mc = time.time()

    try:
        from skimage.measure import marching_cubes

        verts, faces, normals, _ = marching_cubes(
            grid_filled.astype(np.float32),
            level=0.5,
        )

        # big 比原始 grid 多出 pad 层，因此世界坐标原点需要回退 pad * pitch
        origin = np.asarray(vox.transform[:3, 3]) - pitch * pad
        verts_world = origin + verts * pitch

        result = trimesh.Trimesh(
            vertices=verts_world,
            faces=faces,
            process=False,
        )
    except Exception as e:
        # 如果 scikit-image 不可用，尝试用调整过 transform 的 VoxelGrid
        try:
            from trimesh.voxel import VoxelGrid

            new_transform = np.array(vox.transform)
            new_transform[:3, 3] -= pitch * pad

            big_vox = VoxelGrid(big, transform=new_transform)
            result = big_vox.marching_cubes
            if callable(result):
                result = result()
        except Exception as e2:
            raise RuntimeError(
                "Unable to extract watertight surface. "
                "Install scikit-image or adjust trimesh voxel grid handling."
            ) from e2

    if verbose:
        print(f"  [watertight] marching cubes done in "
              f"{time.time() - t_mc:.2f}s",
              flush=True)

    result.merge_vertices()
    result.remove_unreferenced_vertices()
    result = repair_mesh_by_removing_duplicates(result)

    # -------------------------------------------------------------
    # 可选：把体素化代理壳顶点投影回原始网格表面
    # 可显著减少轴对齐体素造成的阶梯状锯齿。
    # -------------------------------------------------------------
    if project_to_input:
        if project_distance is None:
            project_distance = 0.5 * pitch

        # 精确表面投影依赖 rtree；没有 rtree 时放弃投影
        try:
            import rtree  # noqa
            have_rtree = True
        except ImportError:
            have_rtree = False

        if not have_rtree:
            if verbose:
                print("  [watertight] rtree not available; "
                      "skipping projection to input surface.",
                      flush=True)
        else:
            if verbose:
                print(f"  [watertight] projecting to input surface "
                      f"(distance={project_distance:.4f})...",
                      flush=True)

            verts = result.vertices.astype(np.float64)

            try:
                from trimesh.proximity import closest_point

                closest, distances, _ = closest_point(m, verts)
            except Exception as e:
                if verbose:
                    print(f"  [watertight] projection failed: {e}; "
                          f"skipping projection.",
                          flush=True)
                closest = None

            if closest is not None:
                distances = np.asarray(distances, dtype=np.float64)
                moved_mask = distances < project_distance

                if verbose:
                    print(f"  [watertight] projected "
                          f"{int(moved_mask.sum())}/{len(verts)} vertices",
                          flush=True)

                verts[moved_mask] = closest[moved_mask]

                result = trimesh.Trimesh(
                    vertices=verts,
                    faces=result.faces,
                    process=False,
                )

                result.merge_vertices()
                result.remove_unreferenced_vertices()
                result = repair_mesh_by_removing_duplicates(result)

    # -------------------------------------------------------------
    # 可选：Taubin 平滑，进一步降低残留锯齿。
    # -------------------------------------------------------------
    if smooth_watertight:
        if verbose:
            print(f"  [watertight] applying Taubin smoothing "
                  f"(iterations={smooth_iterations})...",
                  flush=True)

        try:
            from trimesh.smoothing import filter_taubin

            result = filter_taubin(
                result,
                lamb=0.5,
                nu=-0.53,
                iterations=smooth_iterations,
            )
        except Exception as e:
            if verbose:
                print(f"  [watertight] smoothing failed: {e}",
                      flush=True)

        result.merge_vertices()
        result.remove_unreferenced_vertices()
        result = repair_mesh_by_removing_duplicates(result)

    # 投影/平滑可能重新产生少量开放边和非流形边。
    # 这里进行多轮局部修复，尽量在不整体回退的前提下恢复水密性。
    max_local_repair_iterations = 5

    for repair_iter in range(1, max_local_repair_iterations + 1):
        defects, _, _ = analyze_mesh_defects(result)
        if defects['open_edges'] == 0 and defects['nonmanifold_edges'] == 0:
            if verbose:
                print(f"  [watertight] enhancement repair converged "
                      f"after {repair_iter - 1} iteration(s).",
                      flush=True)
            break

        if verbose:
            print(f"  [watertight] enhancement repair iteration {repair_iter}: "
                  f"open_edges={defects['open_edges']}, "
                  f"nonmanifold_edges={defects['nonmanifold_edges']}",
                  flush=True)

        # 1. 删除非流形边关联面片
        if defects['nonmanifold_edges'] > 0:
            result = repair_nonmanifold_edges(
                result,
                max_iterations=5,
                verbose=False,
            )

        # 2. 删除投影产生的细碎开放边链（伪孔洞）
        #    不闭合的短链无法补孔，先清理掉。
        result = remove_small_open_edge_chains(
            result,
            max_chain_edges=3,
            verbose=False,
        )

        # 3. 清理重复/退化面片
        result.merge_vertices()
        result.remove_unreferenced_vertices()
        result = repair_mesh_by_removing_duplicates(result)

        # 4. 填补剩余闭合孔洞
        defects, _, _ = analyze_mesh_defects(result)
        if defects['open_edges'] > 0:
            result = fill_holes_adaptive(
                result,
                strategy='edge-count',
                max_fan_edges=20,
                max_earclip_edges=200,
                max_surface_fit_edges=1000,
                verbose=False,
            )

        result.merge_vertices()
        result.remove_unreferenced_vertices()
        result = repair_mesh_by_removing_duplicates(result)

    # 经过局部修复后，即使仍有少量缺陷，也保留增强网格，不再整体回退。
    defects_final, _, _ = analyze_mesh_defects(result)
    if defects_final['open_edges'] > 0 or defects_final['nonmanifold_edges'] > 0:
        if verbose:
            print("  [watertight] warning: enhancement repair did not fully "
                  "resolve all defects; keeping enhanced mesh anyway.",
                  flush=True)
    else:
        if verbose:
            print("  [watertight] enhancement repair completed successfully.",
                  flush=True)

    if len(result.faces) > 0:
        result = repair_normals(result, verbose=verbose)

    if min_component_faces and min_component_faces > 0:
        result = remove_isolated_components(
            result,
            min_faces=min_component_faces,
            verbose=verbose,
        )

    # 诊断最终仍有多少开放边 / 非流形边
    defects, _, _ = analyze_mesh_defects(result)
    if verbose:
        print(f"  [watertight] remaining defects: "
              f"open_edges={defects['open_edges']}, "
              f"nonmanifold_edges={defects['nonmanifold_edges']}")

    if verbose:
        print(f"  [watertight] reconstruction completed in "
              f"{time.time() - t0:.2f}s")
        print(f"  watertight result: is_watertight={result.is_watertight}, "
              f"vertices={len(result.vertices)}, faces={len(result.faces)}")

    return result


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


def weld_small_holes(mesh,
                     threshold=None,
                     quantile=5.0,
                     min_edges=3,
                     verbose=False):
    """
    焊接面积小于阈值的小孔洞。

    对于面积很小的开放边界环，将环上所有顶点合并到环重心，
    再合并重复顶点、删除退化面片。适合扫描模型去重后残留的微裂缝。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    threshold : float or None
        孔洞面积阈值。若为 None，则使用 quantile 根据面片面积计算。
    quantile : float
        面片面积百分位，默认 5，即 p5。
    min_edges : int
        小于此边数的环直接忽略。
    verbose : bool
    """
    if len(mesh.faces) == 0:
        return mesh.copy()

    loops = extract_boundary_loops(mesh)
    if not loops:
        return mesh.copy()

    if threshold is None:
        face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
        if len(face_areas) == 0:
            return mesh.copy()
        threshold = float(np.percentile(face_areas, quantile))

    if verbose:
        print(f"  weld threshold: {threshold:.6f}")

    small_loops = []
    for loop in loops:
        if len(loop) < min_edges:
            continue
        pts = mesh.vertices[np.asarray(loop, dtype=np.int64)]
        area = polygon_area_from_3d_ccw(pts)
        if area <= threshold:
            small_loops.append((area, loop))

    if not small_loops:
        if verbose:
            print("  no small holes to weld")
        return mesh.copy()

    if verbose:
        print(f"  welding {len(small_loops)} small holes")

    new_vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    for _area, loop in small_loops:
        loop = np.asarray(loop, dtype=np.int64)
        centroid = new_vertices[loop].mean(axis=0)
        new_vertices[loop] = centroid

    welded = trimesh.Trimesh(
        vertices=new_vertices,
        faces=faces,
        process=False,
    )
    welded.merge_vertices()
    welded.remove_unreferenced_vertices()
    welded = repair_mesh_by_removing_duplicates(welded)

    if verbose:
        defects, _, _ = analyze_mesh_defects(welded)
        print(f"  after welding: open_edges={defects['open_edges']}, "
              f"nonmanifold_edges={defects['nonmanifold_edges']}")

    return welded


# =====================================================================
# 新增：代理网格支撑的孔洞修补辅助函数
# =====================================================================

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


def _loop_arc_params(pts):
    """闭环顶点的归一化弧长参数（[0,1)，不含终点）。"""
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    tot = float(seg.sum())
    if tot <= 1e-12:
        return None
    return np.concatenate([[0.0], np.cumsum(seg[:-1])]) / tot


def _sample_loop_pts(pts, cum, tt):
    """按归一化弧长 tt 在闭环折线上采样一个点。"""
    n = len(pts)
    k = int(np.searchsorted(cum, tt, side='right') - 1)
    if k >= n - 1:
        span = 1.0 - cum[-1]
        lam = (tt - cum[-1]) / span if span > 1e-30 else 0.0
        return (1 - lam) * pts[-1] + lam * pts[0]
    span = cum[k + 1] - cum[k]
    lam = (tt - cum[k]) / span if span > 1e-30 else 0.0
    return (1 - lam) * pts[k] + lam * pts[k + 1]


def _split_loop_edges_in_work_mesh(work_verts, work_faces, vert2faces,
                                   loop, t_samples, tol=1e-9):
    """
    在演化中的工作网格上，按归一化弧长 t_samples（∈[0,1)，升序）对闭环
    loop 重采样：与原顶点重合则复用索引，否则在边界边上插入新顶点，
    并扇形分裂共享该边的面片（支持同一面片多条边被切）。

    work_faces 为 list（允许元素被置 None 表示已删除），vert2faces 原地
    更新；work_verts 只追加顶点（原索引保持有效）。

    Returns
    -------
    work_verts : (N', 3) float   # 追加新顶点后的数组
    idx_map : (len(t_samples),) int
    """
    loop = [int(v) for v in loop]
    n = len(loop)
    pts = work_verts[np.asarray(loop)]
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    total = float(seg.sum())
    cum = np.concatenate([[0.0], np.cumsum(seg[:-1])])

    idx_map = np.empty(len(t_samples), dtype=np.int64)
    split_map = {}  # undirected key -> {'dir': (u, w), 'pts': [(lam, new_idx)]}

    for si, t in enumerate(t_samples):
        tt = t - np.floor(t)
        d = tt * total
        edge_end = cum + seg
        k = int(np.searchsorted(edge_end, d, side='right') - 1)
        k = min(max(k, 0), n - 1)
        lam = (d - cum[k]) / seg[k] if seg[k] > 0 else 0.0
        u = loop[k]
        w = loop[(k + 1) % n]
        if lam <= tol:
            idx_map[si] = u
            continue
        if lam >= 1.0 - tol:
            idx_map[si] = w
            continue
        p = (1.0 - lam) * work_verts[u] + lam * work_verts[w]
        new_idx = len(work_verts)
        work_verts = np.vstack([work_verts, p[None, :]])
        idx_map[si] = new_idx
        key = (u, w) if u < w else (w, u)
        split_map.setdefault(key, {'dir': (u, w), 'pts': []})['pts'] \
            .append((lam, new_idx))

    if not split_map:
        return work_verts, idx_map

    # 只需检查 loop 顶点关联的候选面片
    cand = set()
    for v in loop:
        cand |= vert2faces.get(v, set())

    for fi in cand:
        f = work_faces[fi]
        if f is None:
            continue
        v0, v1, v2 = f
        fedges = [(v0, v1), (v1, v2), (v2, v0)]
        poly = []
        touched = False
        for (a, b) in fedges:
            poly.append(a)
            key = (a, b) if a < b else (b, a)
            entry = split_map.get(key)
            if entry is None:
                continue
            touched = True
            u, w = entry['dir']
            pts_e = sorted(entry['pts'], key=lambda x: x[0])
            if (a, b) != (u, w):
                pts_e = [(1.0 - lam, idx) for lam, idx in pts_e][::-1]
            poly.extend(idx for _, idx in pts_e)
        if not touched:
            continue
        # 移除旧面片
        work_faces[fi] = None
        for v in (v0, v1, v2):
            vert2faces.get(v, set()).discard(fi)
        # 从 poly[0] 扇形分裂，保持原面片方向
        for i in range(1, len(poly) - 1):
            nf = (poly[0], poly[i], poly[i + 1])
            nfi = len(work_faces)
            work_faces.append(nf)
            for v in nf:
                vert2faces.setdefault(v, set()).add(nfi)

    return work_verts, idx_map


def _stitch_patch_into_mesh(work_verts, work_faces, vert2faces, loop, patch):
    """
    将拓扑圆盘补丁 patch 拉链缝合到工作网格的孔洞边界环 loop 上。

    源边界环与补丁外边界环分别按弧长参数化，取双方归一化断点的并集
    重采样两条环（重合原顶点复用，边上插点并分裂面片），得到等长且
    一一对应的 S'、B'，再逐边生成四边形条带（每边两个三角形）。
    条带边在源/补丁/条带中各出现一次，桥接边恰好两次，接缝处拓扑
    水密（流形）。方向一致性由后续 repair_normals 统一。

    Returns
    -------
    (work_verts, work_faces, vert2faces) 或 None（无法缝合时）
    """
    p_loops = extract_boundary_loops(patch)
    if len(p_loops) != 1:
        return None
    B = p_loops[0]
    S = [int(v) for v in loop]
    if len(S) < 3 or len(B) < 3:
        return None

    s_pts = work_verts[np.asarray(S)]
    b_pts = patch.vertices[np.asarray(B)]

    # 1. B 起点取离 S[0] 最近者；方向用均匀采样总距离判定
    j0 = int(np.argmin(np.linalg.norm(b_pts - s_pts[0], axis=1)))
    t_S = _loop_arc_params(s_pts)
    if t_S is None:
        return None
    K = 32
    ts = np.linspace(0.0, 1.0, K, endpoint=False)
    best_cost = None
    best_dir = 0
    for direction in (0, 1):
        if direction == 0:
            idx = [(j0 + k) % len(B) for k in range(len(B))]
        else:
            idx = [(j0 - k) % len(B) for k in range(len(B))]
        pts_j = b_pts[idx]
        t_j = _loop_arc_params(pts_j)
        if t_j is None:
            continue
        cost = 0.0
        for tt in ts:
            cost += float(np.linalg.norm(
                _sample_loop_pts(s_pts, t_S, tt)
                - _sample_loop_pts(pts_j, t_j, tt)))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_dir = direction
    if best_cost is None:
        return None
    if best_dir == 0:
        idx = [(j0 + k) % len(B) for k in range(len(B))]
    else:
        idx = [(j0 - k) % len(B) for k in range(len(B))]
    B = [B[i] for i in idx]

    # 2. 断点并集
    t_B = _loop_arc_params(patch.vertices[np.asarray(B)])
    if t_B is None:
        return None
    T = np.sort(np.concatenate([t_S, t_B]))
    T = T[np.concatenate([[True], np.diff(T) > 1e-9])]
    if len(T) < 3:
        return None

    # 3. 源侧重采样与面片分裂
    work_verts, S_idx = _split_loop_edges_in_work_mesh(
        work_verts, work_faces, vert2faces, S, T)

    # 4. 补丁侧重采样与面片分裂
    p_faces = [tuple(int(x) for x in f) for f in patch.faces]
    p_v2f = {}
    for fi, f in enumerate(p_faces):
        for v in f:
            p_v2f.setdefault(v, set()).add(fi)
    p_verts, B_idx = _split_loop_edges_in_work_mesh(
        patch.vertices, p_faces, p_v2f, B, T)
    p_faces = [f for f in p_faces if f is not None]

    # 5. 追加补丁顶点/面片与拉链条带
    off = len(work_verts)
    work_verts = np.vstack([work_verts, p_verts])
    for f in p_faces:
        nf = tuple(off + int(v) for v in f)
        nfi = len(work_faces)
        work_faces.append(nf)
        for v in nf:
            vert2faces.setdefault(v, set()).add(nfi)

    m = len(T)
    for i in range(m):
        i2 = (i + 1) % m
        s0 = int(S_idx[i]); s1 = int(S_idx[i2])
        t0 = off + int(B_idx[i]); t1 = off + int(B_idx[i2])
        # 四边形 s0-s1-t1-t0 拆成两个三角形
        for nf in ((s0, s1, t1), (s0, t1, t0)):
            nfi = len(work_faces)
            work_faces.append(nf)
            for v in nf:
                vert2faces.setdefault(v, set()).add(nfi)

    return work_verts, work_faces, vert2faces


def _count_proxy_face_centers_inside_projected_polygon(
        projected_points, proxy_mesh,
        center_tree=None, centers=None, verbose=False):
    """
    统计代理网格中面片中心落在 projected_points 构成的投影多边形内的数量。

    使用 cKDTree 球查询预筛选候选点，避免遍历全部面片中心。

    Parameters
    ----------
    projected_points : (N, 3) float
        源孔洞边界环顶点在代理网格表面的投影点。
    proxy_mesh : trimesh.Trimesh
        代理网格。
    center_tree : scipy.spatial.cKDTree or None
        预先构建的代理面片中心 cKDTree。若为 None，将在此函数内构建。
    centers : (M, 3) float or None
        代理面片中心坐标数组。若为 None，将在此函数内计算。
    verbose : bool
        是否打印耗时信息。

    Returns
    -------
    count : int
        位于投影多边形内部的代理面片中心数量。
    inside_indices : np.ndarray
        代理网格中落在投影多边形内部的面片索引数组。
    """
    import time

    t0 = time.time()

    if len(projected_points) < 3 or len(proxy_mesh.faces) == 0:
        return 0, np.empty(0, dtype=np.int64)

    # 拟合平面，将投影点变换到 2D
    pts = np.asarray(projected_points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    u = vh[0]
    v = vh[1]
    poly2d = np.column_stack([
        (pts - centroid) @ u,
        (pts - centroid) @ v,
    ])

    # 代理面片中心（若未传入则计算）
    if centers is None:
        centers = np.asarray(proxy_mesh.triangles_center, dtype=np.float64)

    if verbose:
        print(f"      [count] SVD + center extraction: {time.time() - t0:.4f}s")

    # 使用球查询快速筛选候选点
    radius = float(np.linalg.norm(poly2d - poly2d.mean(axis=0), axis=1).max()) + 1e-12
    # 注意：这里使用 3D 球查询，质心为原始投影点质心（3D）
    if center_tree is None:
        from scipy.spatial import cKDTree
        center_tree = cKDTree(centers)

    t1 = time.time()
    candidate_indices = center_tree.query_ball_point(centroid, r=radius)
    if verbose:
        print(f"      [count] KDTree query: {time.time() - t1:.4f}s "
              f"({len(candidate_indices)} candidates)")

    if not candidate_indices:
        return 0, np.empty(0, dtype=np.int64)

    # 只对候选点投影到 2D 并测试
    cand_pts = centers[candidate_indices]
    cand2d = np.column_stack([
        (cand_pts - centroid) @ u,
        (cand_pts - centroid) @ v,
    ])

    t2 = time.time()
    inside_mask = [
        _point_in_polygon_2d(c2d, poly2d)
        for c2d in cand2d
    ]
    inside_indices = np.asarray(candidate_indices, dtype=np.int64)[inside_mask]
    count = len(inside_indices)

    if verbose:
        print(f"      [count] polygon test for candidates: {time.time() - t2:.4f}s")
        print(f"      [count] total: {time.time() - t0:.4f}s")

    return count, inside_indices


def _build_face_adjacency_dict(proxy_mesh):
    """构建代理网格的面片邻接字典。"""
    adj = {}
    for f0, f1 in proxy_mesh.face_adjacency:
        f0, f1 = int(f0), int(f1)
        adj.setdefault(f0, set()).add(f1)
        adj.setdefault(f1, set()).add(f0)
    return adj


def _find_face_path(adj, start, goal, max_depth=200):
    """在面片邻接图中寻找 start 到 goal 的局部最短路径。"""
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
                                  max_path_depth=200):
    """
    返回投影边界内部的代理网格面片集合与相关顶点索引。

    Parameters
    ----------
    proxy_mesh : trimesh.Trimesh
    proj_pts : (N, 3) float
        源孔洞边界在代理网格表面上的投影点。
    proj_tris : (N,) int64
        每个投影点所在的代理网格三角形索引。

    Returns
    -------
    inside_faces : set[int]
        投影边界内部的代理网格面片索引集合。
    inside_vertices : np.ndarray
        这些内部面片使用到的代理网格顶点索引。
    """
    proj_pts = np.asarray(proj_pts, dtype=np.float64)
    proj_tris = np.asarray(proj_tris, dtype=np.int64)

    if len(proj_tris) < 3:
        return set(), np.array([], dtype=np.int64)

    adj = _build_face_adjacency_dict(proxy_mesh)

    # 1. 以投影点所在三角形为种子，沿面片邻接图连接相邻投影点，
    #    得到闭合的拓扑分界线。
    barrier = {int(t) for t in proj_tris if t >= 0}
    n = len(proj_tris)

    for i in range(n):
        a = int(proj_tris[i])
        b = int(proj_tris[(i + 1) % n])
        if a < 0 or b < 0:
            continue
        path = _find_face_path(adj, a, b, max_depth=max_path_depth)
        if path is not None:
            barrier.update(path)

    if not barrier:
        return set(), np.array([], dtype=np.int64)

    # 2. 用投影点质心在代理网格上的最近三角形作为内部种子。
    centroid = np.mean(proj_pts, axis=0)
    seed_points, _, seed_tri = project_vertices_to_shell(
        np.array([centroid]), proxy_mesh
    )
    seed = int(seed_tri[0]) if len(seed_tri) > 0 and seed_tri[0] >= 0 else None

    if seed is None:
        return set(), np.array([], dtype=np.int64)

    # 如果种子刚好落在分界线上，尝试用相邻非分界面片。
    if seed in barrier:
        replaced = False
        for nb in adj.get(seed, ()):
            if nb not in barrier:
                seed = nb
                replaced = True
                break
        if not replaced:
            return set(), np.array([], dtype=np.int64)

    # 3. 避开分界线，从种子泛洪。
    visited = {seed}
    stack = [seed]
    while stack:
        cur = stack.pop()
        for nb in adj.get(cur, ()):
            if nb in visited or nb in barrier:
                continue
            visited.add(nb)
            stack.append(nb)

    # 4. 收集内部面片和内部顶点。
    vertex_set = set()
    for f in visited:
        vertex_set.update(int(v) for v in proxy_mesh.faces[f])

    inside_vertices = np.array(sorted(vertex_set), dtype=np.int64)
    return visited, inside_vertices


def _triangulate_polygon_with_interior_points(boundary_pts, interior_pts):
    """
    将源孔洞边界和内部代理顶点一起重新三角化。

    boundary_pts 为源孔洞边界 3D 顶点序列，interior_pts 为代理网格内部
    3D 顶点。函数将全部点投影到局部最佳拟合平面，在 2D 中对边界多边形
    做初始 ear clipping，然后逐个插入内部点，从而保持源边界不变。

    Returns
    -------
    faces : list[tuple[int, int, int]]
        局部索引，0..len(boundary_pts)-1 对应源边界顶点，
        其后的索引对应 interior_pts。
    """
    boundary_pts = np.asarray(boundary_pts, dtype=np.float64)
    interior_pts = np.asarray(interior_pts, dtype=np.float64).reshape(-1, 3)

    if len(boundary_pts) < 3:
        return []

    n_boundary = len(boundary_pts)
    if len(interior_pts) > 0:
        all_pts = np.vstack([boundary_pts, interior_pts])
    else:
        all_pts = boundary_pts

    # 局部参数平面：使用所有点拟合，降低弯曲表面下的退化风险。
    centroid = boundary_pts.mean(axis=0)
    centered = all_pts - centroid
    _, _, vh = np.linalg.svd(centered)
    u = vh[0]
    v = vh[1]

    coords2d = np.column_stack([
        centered @ u,
        centered @ v,
    ])
    coords3d = np.column_stack([coords2d, np.zeros(len(coords2d))])

    # 先只对源边界做 ear clipping，保证外边界正确。
    boundary_idx = np.arange(n_boundary, dtype=np.int64).tolist()
    triangles = _triangulate_hole_loop(coords3d, boundary_idx)
    if not triangles:
        return []

    triangles = [tuple(int(x) for x in tri) for tri in triangles]

    # 逐个插入内部点：定位当前包含它的三角形并一分为三。
    for ip in range(n_boundary, len(all_pts)):
        p2 = coords2d[ip]
        found = None
        for ti, tri in enumerate(triangles):
            a, b, c = tri
            if _point_in_triangle_2d(
                p2,
                coords2d[a],
                coords2d[b],
                coords2d[c],
            ):
                found = ti
                break

        if found is None:
            continue

        a, b, c = triangles[found]
        triangles[found] = (a, b, ip)
        triangles.append((b, c, ip))
        triangles.append((c, a, ip))

    return triangles


def fill_holes_with_proxy(mesh,
                          proxy_mesh,
                          proxy_face_center_threshold=20,
                          max_projection_distance=None,
                          min_proxy_loop_edges=12,
                          patch_area_ratio_range=(0.25, 8.0),
                          verbose=False):
    """
    使用代理网格支撑填补源网格上的孔洞（拓扑分界 + 重新三角化版）。

    策略：
    1. 异常小孔洞先通过 weld_small_holes 焊接。
    2. 所有边界环顶点一次性批量投影到代理网格，得到投影点与三角形索引。
    3. 根据投影三角形构造代理网格上的拓扑分界线，从内部种子泛洪
       得到内部面片/内部顶点。
    4. 对“源孔洞边界顶点 + 内部代理顶点”进行带约束的重新三角化。
    5. 收尾：残存小孔再次焊接 + fill_holes_adaptive，法线修复。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    proxy_mesh : trimesh.Trimesh
    proxy_face_center_threshold : int
        默认 20。
    max_projection_distance : float or None
    min_proxy_loop_edges : int
        使用代理支撑的最小边界边数，默认 12。
    patch_area_ratio_range : (float, float)
        保留兼容参数，不再用于裁剪补丁。
    verbose : bool
    """
    import time

    t_start = time.time()

    # 1. 焊接异常小孔洞
    welded = weld_small_holes(mesh, quantile=5.0, min_edges=3,
                              verbose=verbose)

    loops = extract_boundary_loops(welded)
    if not loops:
        return welded

    have_proxy = proxy_mesh is not None and len(proxy_mesh.faces) > 0

    proj_map = None
    if have_proxy:
        # 批量投影：一次得到代理表面最近点、距离以及所在三角形索引。
        try:
            all_verts = np.unique(np.concatenate(
                [np.asarray(l, dtype=np.int64) for l in loops]))
            all_points = welded.vertices[all_verts]
            proj_points, proj_dist, proj_tri = project_vertices_to_shell(
                all_points, proxy_mesh
            )
            proj_map = {
                int(v): (
                    proj_points[i],
                    float(proj_dist[i]),
                    int(proj_tri[i]),
                )
                for i, v in enumerate(all_verts)
            }
        except Exception as e:
            if verbose:
                print(f"  [proxy patch] batch projection failed: {e}")
            proj_map = None

    if verbose:
        print(f"  [proxy patch] processing {len(loops)} boundary loops")

    # 演化中的工作网格（顶点只追加，原索引保持有效）
    work_verts = welded.vertices.copy()
    work_faces = [tuple(int(x) for x in f) for f in welded.faces]
    vert2faces = {}
    for fi, f in enumerate(work_faces):
        for v in f:
            vert2faces.setdefault(v, set()).add(fi)

    plane_fill_count = 0
    proxy_fill_count = 0
    fallback_plane_count = 0

    for loop_idx, loop in enumerate(loops):
        loop_t0 = time.time()
        if len(loop) < 3:
            continue

        loop = [int(v) for v in loop]
        loop_verts = welded.vertices[np.asarray(loop)]
        hole_area = polygon_area_from_3d_ccw(loop_verts)

        # 判断是否可用代理支撑
        use_proxy = False
        inside_faces = None
        inside_vertices = None

        if have_proxy and proj_map is not None and \
                len(loop) >= min_proxy_loop_edges:
            try:
                proj_items = [proj_map[v] for v in loop]
                proj_pts = np.array([item[0] for item in proj_items])
                dists = np.array([item[1] for item in proj_items])
                proj_tris = np.array([item[2] for item in proj_items],
                                     dtype=np.int64)
            except KeyError:
                proj_pts = None

            if proj_pts is not None and \
                    max_projection_distance is not None and \
                    float(np.max(dists)) > max_projection_distance:
                proj_pts = None

            if proj_pts is not None:
                inside_faces, inside_vertices = \
                    _extract_proxy_interior_faces(
                        proxy_mesh, proj_pts, proj_tris
                    )
                proxy_inside_count = len(inside_faces)

                if verbose:
                    print(f"    loop {loop_idx}: edges={len(loop)}, "
                          f"proxy_inside_faces={proxy_inside_count}")

                if proxy_inside_count >= proxy_face_center_threshold:
                    use_proxy = True

        stitched = False
        if use_proxy:
            if verbose:
                print("      -> proxy re-triangulation (topological interior)")

            inside_pts = proxy_mesh.vertices[inside_vertices]
            source_pts = work_verts[np.asarray(loop, dtype=np.int64)]

            local_tris = _triangulate_polygon_with_interior_points(
                source_pts, inside_pts
            )

            if local_tris:
                n_source = len(source_pts)
                interior_start = len(work_verts)
                work_verts = np.vstack([work_verts, inside_pts])

                for tri in local_tris:
                    mapped = []
                    for local_idx in tri:
                        if local_idx < n_source:
                            mapped.append(int(loop[local_idx]))
                        else:
                            mapped.append(
                                int(interior_start + (local_idx - n_source))
                            )
                    nf = tuple(mapped)
                    nfi = len(work_faces)
                    work_faces.append(nf)
                    for v in nf:
                        vert2faces.setdefault(v, set()).add(nfi)

                proxy_fill_count += 1
                stitched = True
            else:
                fallback_plane_count += 1

        if not stitched:
            if verbose:
                print("      -> plane fill")
            for t in _triangulate_hole_loop(work_verts, loop):
                t = tuple(int(x) for x in t)
                nfi = len(work_faces)
                work_faces.append(t)
                for v in t:
                    vert2faces.setdefault(v, set()).add(nfi)
            plane_fill_count += 1

        if verbose:
            print(f"    [loop {loop_idx}] total time: "
                  f"{time.time() - loop_t0:.4f}s")

    # 2. 组装最终网格（不 merge_vertices：重新三角化后的面片共享边界）
    final_faces = [f for f in work_faces if f is not None]
    merged = trimesh.Trimesh(
        vertices=work_verts,
        faces=np.asarray(final_faces, dtype=np.int64),
        process=False,
    )
    merged.remove_unreferenced_vertices()
    merged = repair_mesh_by_removing_duplicates(merged)

    # 3. 收尾：残存小孔焊接 + 平面填补（不再调用代理补丁）
    merged = weld_small_holes(merged, quantile=5.0, min_edges=3,
                              verbose=verbose)
    if extract_boundary_loops(merged):
        merged = fill_holes_adaptive(merged, strategy='flatness',
                                     verbose=verbose)

    merged = repair_normals(merged, verbose=verbose)

    if verbose:
        print(f"  [proxy patch] summary: plane_fill={plane_fill_count}, "
              f"proxy_fill={proxy_fill_count}, "
              f"proxy_fallback_plane={fallback_plane_count}")
        defects, _, _ = analyze_mesh_defects(merged)
        print(f"  [proxy patch] final defects: "
              f"open_edges={defects['open_edges']}, "
              f"nonmanifold_edges={defects['nonmanifold_edges']}")
        print(f"  [proxy patch] total time: {time.time() - t_start:.4f}s")

    return merged


def project_vertices_to_shell(vertices, shell_mesh):
    """
    将输入点集投影到 shell_mesh 的三角形表面。

    返回的是表面上距离输入点最近的点（可能位于三角形内部或边上），
    以及该点所在的三角形索引。这样可以避免仅仅查询顶点最近点而误选
    背面或错误一侧的问题。

    Parameters
    ----------
    vertices : (N, 3) float
        待投影点集。
    shell_mesh : trimesh.Trimesh
        代理网格。

    Returns
    -------
    closest_points : (N, 3) float
        表面上距离输入点最近的点。
    distances : (N,) float
        最近点与输入点之间的距离。
    triangle_indices : (N,) int
        最近点所在的三角形索引。
    """
    if not isinstance(shell_mesh, trimesh.Trimesh):
        raise TypeError("shell_mesh must be trimesh.Trimesh")

    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    try:
        from trimesh.proximity import ProximityQuery
        prox = ProximityQuery(shell_mesh)
        closest_points, distances, triangle_indices = prox.on_surface(vertices)
        return (
            np.asarray(closest_points, dtype=np.float64),
            np.asarray(distances, dtype=np.float64),
            np.asarray(triangle_indices, dtype=np.int64),
        )
    except Exception:
        # 回退到 closest_point，它同样返回表面最近点和三角形索引
        from trimesh.proximity import closest_point
        closest_points, distances, triangle_indices = closest_point(
            shell_mesh, vertices
        )
        return (
            np.asarray(closest_points, dtype=np.float64),
            np.asarray(distances, dtype=np.float64),
            np.asarray(triangle_indices, dtype=np.int64),
        )


def compute_reliable_face_mask(mesh,
                               k_defect=3,
                               min_area_ratio=0.01,
                               normal_thresh_deg=60,
                               min_component_faces=20,
                               use_soft_weights=True):
    """
    计算每个三角面片的可靠性权重。

    综合以下因素：
    - 开放边/非流形边的邻域扩展
    - 退化面片（面积过小）
    - 法线一致性（与邻域平均法线夹角过大）
    - 孤立小连通分量

    Parameters
    ----------
    mesh : trimesh.Trimesh
    k_defect : int
        缺陷邻域扩展步数
    min_area_ratio : float
        面积过滤阈值比例（相对于平均面积）
    normal_thresh_deg : float
        法线一致性角度阈值（度）
    min_component_faces : int
        连通分量最小面片数
    use_soft_weights : bool
        若 True，返回连续权重（1.0可靠，0.5缺陷邻域，0.0不可靠）；
        若 False，返回二值权重（>0.5视为可靠）

    Returns
    -------
    weights : np.ndarray, shape (N,), dtype float32
        每个面片的可靠性权重，取值范围 0~1
    """
    n_faces = len(mesh.faces)
    weights = np.ones(n_faces, dtype=np.float32)  # 初始全部可靠

    # 1. 拓扑缺陷邻域扩展
    _, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    defect_mask = open_face_mask | nonmanifold_face_mask
    if k_defect > 0 and defect_mask.any():
        current = defect_mask.copy()
        visited = current.copy()
        for _ in range(k_defect):
            if not current.any():
                break
            idx_current = np.where(current)[0]
            # 找到与当前缺陷面相邻的边
            mask_edges = (
                np.isin(mesh.face_adjacency[:, 0], idx_current) |
                np.isin(mesh.face_adjacency[:, 1], idx_current)
            )
            neighbor_faces = np.unique(mesh.face_adjacency[mask_edges].ravel())
            new_faces = neighbor_faces[~visited[neighbor_faces]]
            if len(new_faces) == 0:
                break
            visited[new_faces] = True
            current.fill(False)
            current[new_faces] = True
        defect_mask = visited
    # 缺陷邻域赋予中间权重 0.5（后续可能被其他硬条件覆盖为 0）
    weights[defect_mask] = 0.5

    # 2. 退化面片
    areas = mesh.area_faces
    mean_area = areas.mean() if len(areas) > 0 else 0.0
    area_thresh = max(1e-12, min_area_ratio * mean_area)
    degenerate_mask = areas < area_thresh
    weights[degenerate_mask] = 0.0

    # 3. 法线一致性（邻域平均法线夹角）
    normal_bad_mask = np.zeros(n_faces, dtype=bool)
    if normal_thresh_deg is not None:
        face_normals = mesh.face_normals
        neighbor_sum = np.zeros_like(face_normals, dtype=np.float64)
        neighbor_count = np.zeros(n_faces, dtype=np.int32)
        # 累加每个面的邻域法线
        for f0, f1 in mesh.face_adjacency:
            neighbor_sum[f0] += face_normals[f1]
            neighbor_count[f0] += 1
            neighbor_sum[f1] += face_normals[f0]
            neighbor_count[f1] += 1

        valid = neighbor_count > 0
        avg_normals = np.zeros_like(face_normals)
        avg_normals[valid] = neighbor_sum[valid] / neighbor_count[valid, None]
        avg_normals[~valid] = face_normals[~valid]  # 无邻居的面暂时认为自己一致，后续可能因孤立分量被剔除

        dot = np.einsum('ij,ij->i', face_normals, avg_normals)
        norms = np.linalg.norm(face_normals, axis=1) * np.linalg.norm(avg_normals, axis=1)
        cos_angle = np.clip(dot / (norms + 1e-12), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        normal_bad_mask = angle_deg > normal_thresh_deg
        weights[normal_bad_mask] = 0.0

    # 4. 孤立小连通分量（只对当前权重>0.5的面片进行）
    if min_component_faces and min_component_faces > 0:
        # 候选可靠面片：权重 > 0.5，即未因缺陷邻域、退化或法线问题被置零
        candidate_mask = weights > 0.5
        if candidate_mask.any():
            # 只保留两端都是候选面的邻接边
            candidate_edges = mesh.face_adjacency[
                candidate_mask[mesh.face_adjacency[:, 0]] &
                candidate_mask[mesh.face_adjacency[:, 1]]
            ]
            if len(candidate_edges) > 0:
                try:
                    from scipy.sparse import csr_matrix
                    from scipy.sparse.csgraph import connected_components
                    orig_indices = np.where(candidate_mask)[0]
                    index_map = -np.ones(n_faces, dtype=np.int64)
                    index_map[orig_indices] = np.arange(len(orig_indices))
                    rows = index_map[candidate_edges[:, 0]]
                    cols = index_map[candidate_edges[:, 1]]
                    data = np.ones(len(rows))
                    # 对称矩阵
                    rows_all = np.concatenate([rows, cols])
                    cols_all = np.concatenate([cols, rows])
                    data_all = np.concatenate([data, data])
                    graph = csr_matrix(
                        (data_all, (rows_all, cols_all)),
                        shape=(len(orig_indices), len(orig_indices))
                    )
                    n_components, labels = connected_components(graph, directed=False)
                    comp_sizes = np.bincount(labels)
                    small_components = np.where(comp_sizes < min_component_faces)[0]
                    small_orig_indices = orig_indices[np.isin(labels, small_components)]
                    weights[small_orig_indices] = 0.0
                except ImportError:
                    # 无 scipy 时跳过连通分量过滤
                    pass

    if not use_soft_weights:
        weights = (weights > 0.5).astype(np.float32)

    return weights


def fuse_reliable_faces_with_shell(
    source_mesh,
    shell_mesh,
    reliable_face_mask=None,
    mask_threshold=0.75,
    proxy_face_center_threshold=20,
    max_projection_distance=None,
    min_proxy_loop_edges=12,
    smooth_transition=True,
    smooth_iterations=3,
    smooth_alpha=0.5,
    verbose=False,
):
    """
    将 source_mesh 的可靠面片与 watertight shell_mesh 融合。

    流程：
    1. 提取可靠面片子网格；
    2. 破坏性清理前置（非流形边删除、短开放链删除），先于填补执行；
    3. 单次代理支撑填补（拓扑分界 + 重新三角化），不再二次调用；
    4. 可选接缝平滑（向量化拉普拉斯，固定可靠子网格顶点）；
    5. 法线修复与孤立小组件移除；最终只报告缺陷，不再删面。
    """
    if not isinstance(source_mesh, trimesh.Trimesh) or \
            not isinstance(shell_mesh, trimesh.Trimesh):
        raise TypeError("Both meshes must be trimesh.Trimesh instances")

    # 1. 可靠面片掩码
    if reliable_face_mask is None:
        weights = compute_reliable_face_mask(source_mesh)
        reliable_face_mask = np.asarray(weights > mask_threshold, dtype=bool)
    else:
        reliable_face_mask = np.asarray(reliable_face_mask, dtype=bool)
        if len(reliable_face_mask) != len(source_mesh.faces):
            raise ValueError(
                "reliable_face_mask length does not match source_mesh faces")

    if not reliable_face_mask.any():
        raise ValueError("No reliable faces selected")

    # 2. 提取可靠子网格
    src_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    rel_faces = src_faces[np.where(reliable_face_mask)[0]]
    unique_verts, inverse = np.unique(rel_faces, return_inverse=True)
    reliable_mesh = trimesh.Trimesh(
        vertices=source_mesh.vertices[unique_verts],
        faces=inverse.reshape(-1, 3),
        process=False,
    )
    reliable_mesh.remove_unreferenced_vertices()
    reliable_mesh.merge_vertices()
    reliable_mesh = repair_mesh_by_removing_duplicates(reliable_mesh)

    if len(reliable_mesh.faces) == 0:
        raise RuntimeError("Reliable submesh became empty after cleaning")

    # 3. 破坏性清理前置：先删非流形面片与短开放链，再填补
    reliable_mesh = repair_nonmanifold_edges(reliable_mesh, max_iterations=5)
    reliable_mesh = remove_small_open_edge_chains(
        reliable_mesh, max_chain_edges=3, verbose=verbose)
    reliable_mesh.merge_vertices()
    reliable_mesh.remove_unreferenced_vertices()
    reliable_mesh = repair_mesh_by_removing_duplicates(reliable_mesh)

    # 4. 单次代理支撑填补（拓扑分界 + 重新三角化）
    fused_mesh = fill_holes_with_proxy(
        reliable_mesh,
        proxy_mesh=shell_mesh,
        proxy_face_center_threshold=proxy_face_center_threshold,
        max_projection_distance=max_projection_distance,
        min_proxy_loop_edges=min_proxy_loop_edges,
        verbose=verbose,
    )

    # 5. 可选：向量化拉普拉斯平滑（固定可靠子网格顶点）
    if smooth_transition and len(fused_mesh.vertices) > 0:
        fixed_mask = np.zeros(len(fused_mesh.vertices), dtype=bool)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.asarray(fused_mesh.vertices, dtype=np.float64))
            dist, idx = tree.query(
                np.asarray(reliable_mesh.vertices, dtype=np.float64), k=1)
            fixed_mask[idx] = dist < 1e-7
        except ImportError:
            fixed_mask = None
            if verbose:
                print("  [fuse] scipy not available; skip fixed mask")

        if fixed_mask is not None:
            if verbose:
                print(f"  [fuse] smoothing transition with "
                      f"{int(fixed_mask.sum())} fixed vertices")
            try:
                from scipy.sparse import csr_matrix
                edges = np.asarray(fused_mesh.edges_unique, dtype=np.int64)
                n_v = len(fused_mesh.vertices)
                rows = np.concatenate([edges[:, 0], edges[:, 1]])
                cols = np.concatenate([edges[:, 1], edges[:, 0]])
                A = csr_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(n_v, n_v))
                deg = np.asarray(A.sum(axis=1)).ravel()
                deg[deg == 0] = 1.0
                verts = np.asarray(fused_mesh.vertices,
                                   dtype=np.float64).copy()
                move = ~fixed_mask
                for _ in range(smooth_iterations):
                    avg = (A @ verts) / deg[:, None]
                    verts[move] = (1.0 - smooth_alpha) * verts[move] \
                        + smooth_alpha * avg[move]
                fused_mesh = fused_mesh.copy()
                fused_mesh.vertices = verts
            except ImportError:
                if verbose:
                    print("  [fuse] scipy.sparse not available; "
                          "skip smoothing")

        fused_mesh.merge_vertices()
        fused_mesh.remove_unreferenced_vertices()
        fused_mesh = repair_mesh_by_removing_duplicates(fused_mesh)

    # 6. 最终修复（不删面、不再次填补）
    fused_mesh = repair_normals(fused_mesh, verbose=verbose)
    fused_mesh = remove_isolated_components(
        fused_mesh, min_faces=20, verbose=verbose)

    if verbose:
        defects, _, _ = analyze_mesh_defects(fused_mesh)
        print(f"  [fuse] final defects: "
              f"open_edges={defects['open_edges']}, "
              f"nonmanifold_edges={defects['nonmanifold_edges']}")

    return fused_mesh
