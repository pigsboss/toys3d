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


def analyze_mesh_defects(mesh, return_face_edge_counts=False):
    """
    分析网格缺陷，返回统计信息以及开放面片和非流形面片的布尔掩码。

    如果 return_face_edge_counts=True，额外返回：
        open_edge_per_face : 每个面片的开放边数
        manifold_edge_per_face : 每个面片的流形边数
        nonmanifold_edge_per_face : 每个面片的非流形边数
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces)

    if n_faces == 0:
        if return_face_edge_counts:
            empty = np.zeros(0, dtype=np.uint8)
            return (
                {'open_edges': 0, 'nonmanifold_edges': 0,
                 'open_faces': 0, 'nonmanifold_faces': 0},
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=bool),
                empty.copy(), empty.copy(), empty.copy()
            )
        else:
            return (
                {'open_edges': 0, 'nonmanifold_edges': 0,
                 'open_faces': 0, 'nonmanifold_faces': 0},
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=bool)
            )

    # 1. 构建所有边的两个端点（共 3F 条边）
    # 先构建形状 (n_faces, 3, 2)，每行对应一个面的三条边
    edge_pairs = np.stack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], axis=1).reshape(-1, 2)

    ea = edge_pairs[:, 0]
    eb = edge_pairs[:, 1]
    face_ids = np.repeat(np.arange(n_faces, dtype=np.int64), 3)

    # 将边端点排序，使 (v0,v1) 和 (v1,v0) 被视为同一条边
    min_e = np.minimum(ea, eb)
    max_e = np.maximum(ea, eb)

    # 编码键（注意使用足够大的基数）
    n_vertices = int(mesh.vertices.shape[0])
    if n_vertices == 0:
        max_vertex = int(max_e.max()) + 1
    else:
        max_vertex = n_vertices
    keys = min_e.astype(np.int64) * (max_vertex + 1) + max_e

    # 2. 对键排序，同时重排面片索引
    order = np.argsort(keys, kind='stable')
    keys_sorted = keys[order]
    face_ids_sorted = face_ids[order]

    # 3. 找出每个唯一键的连续段
    diff = np.empty(keys_sorted.shape[0], dtype=bool)
    diff[0] = True
    diff[1:] = keys_sorted[1:] != keys_sorted[:-1]
    start_idx = np.flatnonzero(diff)
    end_idx = np.append(start_idx[1:], keys_sorted.shape[0])
    counts = end_idx - start_idx

    # 4. 统计缺陷
    open_edges = int(np.sum(counts == 1))
    nonmanifold_edges = int(np.sum(counts >= 3))

    open_face_mask = np.zeros(n_faces, dtype=bool)
    nonmanifold_face_mask = np.zeros(n_faces, dtype=bool)

    if return_face_edge_counts:
        open_edge_per_face = np.zeros(n_faces, dtype=np.uint8)
        manifold_edge_per_face = np.zeros(n_faces, dtype=np.uint8)
        nonmanifold_edge_per_face = np.zeros(n_faces, dtype=np.uint8)

    # 处理开放边（counts == 1）
    open_pos = start_idx[counts == 1]
    if len(open_pos) > 0:
        open_f = face_ids_sorted[open_pos]
        open_face_mask[open_f] = True
        if return_face_edge_counts:
            np.add.at(open_edge_per_face, open_f, 1)

    # 处理流形边（counts == 2）
    manifold_pos = start_idx[counts == 2]
    if len(manifold_pos) > 0:
        f0 = face_ids_sorted[manifold_pos]
        f1 = face_ids_sorted[manifold_pos + 1]  # 段内第二个
        if return_face_edge_counts:
            np.add.at(manifold_edge_per_face, f0, 1)
            np.add.at(manifold_edge_per_face, f1, 1)

    # 处理非流形边（counts >= 3）
    nonmanifold_start = start_idx[counts >= 3]
    nonmanifold_vals = counts[counts >= 3]
    for start, cnt in zip(nonmanifold_start, nonmanifold_vals):
        seg_face_ids = face_ids_sorted[start:start + cnt]
        nonmanifold_face_mask[seg_face_ids] = True
        if return_face_edge_counts:
            np.add.at(nonmanifold_edge_per_face, seg_face_ids, 1)

    defect_stats = {
        'open_edges': open_edges,
        'nonmanifold_edges': nonmanifold_edges,
        'open_faces': int(open_face_mask.sum()),
        'nonmanifold_faces': int(nonmanifold_face_mask.sum()),
    }

    if return_face_edge_counts:
        return (
            defect_stats,
            open_face_mask,
            nonmanifold_face_mask,
            open_edge_per_face,
            manifold_edge_per_face,
            nonmanifold_edge_per_face,
        )
    else:
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


def compute_topological_reliable_face_mask(mesh, min_distance=2):
    """
    计算基于拓扑距离的可靠面片掩码。

    可靠面片 = 面片本身是流形面，且沿面邻接图到最近缺陷面
    （开放面或非流形面）的距离 >= min_distance。

    返回:
        reliable_mask : bool array
        distances     : int array, 每个面片到最近缺陷面的拓扑距离
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

    _, open_mask, nonmanifold_mask = analyze_mesh_defects(mesh)
    defect_mask = open_mask | nonmanifold_mask

    if not np.any(defect_mask):
        return np.ones(n_faces, dtype=bool), np.full(n_faces, min_distance + 1, dtype=int)

    # 构建面邻接表
    adjacency = [[] for _ in range(n_faces)]
    face_adj = getattr(mesh, 'face_adjacency', None)
    if face_adj is not None and len(face_adj) > 0:
        for f0, f1 in face_adj:
            f0, f1 = int(f0), int(f1)
            if f0 < 0 or f1 < 0:
                continue
            adjacency[f0].append(f1)
            adjacency[f1].append(f0)

    # 多源 BFS
    dist = np.full(n_faces, -1, dtype=int)
    queue = deque()
    for fid in np.where(defect_mask)[0]:
        dist[fid] = 0
        queue.append(int(fid))

    while queue:
        cur = queue.popleft()
        for nb in adjacency[cur]:
            if dist[nb] == -1:
                dist[nb] = dist[cur] + 1
                queue.append(nb)

    # 未到达的孤立面（极少见）视为远离缺陷
    dist[dist == -1] = min_distance + 1

    reliable_mask = (~defect_mask) & (dist >= min_distance)
    return reliable_mask, dist


def repair_mesh_by_removing_duplicates(mesh):
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
    """
    焊接面积小于阈值的小孔洞。

    通过一次性构建“旧顶点 -> 新顶点”查表，并用 NumPy 向量化替换面片索引，
    避免对每个小孔洞都全量扫描百万级面片。
    """
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

    # 1. 为每个待焊接环创建一个质心新顶点，并记录旧顶点映射
    old_to_new = {}
    extra_vertices = list(m.vertices)

    for loop in weld_loops:
        ids = np.asarray(loop, dtype=np.int64)
        centroid = np.mean(m.vertices[ids], axis=0)

        new_idx = len(extra_vertices)
        extra_vertices.append(centroid)

        for v in ids:
            old_to_new[int(v)] = new_idx

    # 2. 构建旧顶点索引到新顶点索引的查询表
    n_old = len(m.vertices)
    lookup = np.arange(n_old, dtype=np.int64)
    for old_v, new_v in old_to_new.items():
        lookup[old_v] = new_v

    # 3. 新顶点数组
    new_vertices = np.asarray(extra_vertices, dtype=np.float64)

    # 4. 向量化替换所有面片的顶点索引
    faces = np.asarray(m.faces, dtype=np.int64)
    flat_faces = faces.ravel()
    new_flat = lookup[flat_faces]
    new_faces = new_flat.reshape(-1, 3)

    # 5. 删除替换后产生的退化面片
    valid = ~((new_faces[:, 0] == new_faces[:, 1]) |
              (new_faces[:, 1] == new_faces[:, 2]) |
              (new_faces[:, 2] == new_faces[:, 0]))
    new_faces = new_faces[valid]

    m2 = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    m2.remove_unreferenced_vertices()
    m2.merge_vertices()   # 合并重复质心顶点
    return repair_mesh_by_removing_duplicates(m2)


def trim_isolated_faces(mesh, verbose=False):
    """
    删除完全孤立的面片（开放边数 == 3）。

    调试模式下，在删除前打印 3 开放边面片的统计信息：
      - 数量
      - 面积分布
      - 顶点是否被其它面片共享
    """
    m = mesh.copy()

    (
        defect_stats,
        open_face_mask,
        nonmanifold_face_mask,
        open_edge_per_face,
        manifold_edge_per_face,
        nonmanifold_edge_per_face,
    ) = analyze_mesh_defects(m, return_face_edge_counts=True)

    bad_mask = open_edge_per_face == 3
    count_removed = int(np.sum(bad_mask))

    if verbose:
        print(
            f"  [trim_isolated_faces] found {count_removed} isolated faces "
            f"(3 open edges)"
        )

        if count_removed > 0:
            bad_face_indices = np.where(bad_mask)[0]
            bad_faces = m.faces[bad_face_indices]
            bad_areas = m.area_faces[bad_face_indices]

            # 面积统计
            print("    area stats:")
            print(f"      min:  {bad_areas.min():.6f}")
            print(f"      max:  {bad_areas.max():.6f}")
            print(f"      mean: {bad_areas.mean():.6f}")
            print(f"      p50:  {np.percentile(bad_areas, 50):.6f}")
            print(f"      p90:  {np.percentile(bad_areas, 90):.6f}")
            print(f"      p99:  {np.percentile(bad_areas, 99):.6f}")

            # 顶点共享统计
            all_face_vertices = m.faces.ravel()
            vertex_face_counts = np.bincount(
                all_face_vertices, minlength=len(m.vertices)
            )

            counts_per_face = vertex_face_counts[bad_faces]  # shape (N, 3)
            is_solo = counts_per_face == 1
            solo_counts = is_solo.sum(axis=1)

            n_all_shared = int(np.sum(solo_counts == 0))
            n_one_solo   = int(np.sum(solo_counts == 1))
            n_two_solo   = int(np.sum(solo_counts == 2))
            n_all_solo   = int(np.sum(solo_counts == 3))

            print("    vertex sharing (among all faces):")
            print(f"      0 solo vertices (all shared):            {n_all_shared}")
            print(f"      1 solo vertex:                          {n_one_solo}")
            print(f"      2 solo vertices:                        {n_two_solo}")
            print(f"      3 solo vertices (fully isolated):       {n_all_solo}")

            mean_ref_count = counts_per_face.mean()
            print(f"    mean vertex reference count (in these faces): {mean_ref_count:.2f}")

    if count_removed > 0:
        m.update_faces(~bad_mask)
        m.remove_unreferenced_vertices()

    return m


def repair_to_watertight(mesh,
                         resolution=256,
                         voxel_size=None,
                         project_to_input=False,
                         smooth_watertight=False,
                         smooth_iterations=10):
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


def compute_vertex_face_counts(mesh):
    """返回每个顶点被多少个面片引用（np.int64 array）。"""
    return np.bincount(mesh.faces.ravel(), minlength=len(mesh.vertices))


def compute_face_edge_types(mesh):
    """
    返回每个面片的三条边的类型矩阵，形状 (n_faces, 3)，值为：
      1: 开放边（只被1个面使用）
      2: 流形边（被2个面使用）
      3: 非流形边（被>=3个面使用）
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces)
    if n_faces == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    edge_pairs = np.stack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], axis=1).reshape(-1, 2)

    ea = edge_pairs[:, 0]
    eb = edge_pairs[:, 1]
    face_ids = np.repeat(np.arange(n_faces, dtype=np.int64), 3)

    min_e = np.minimum(ea, eb)
    max_e = np.maximum(ea, eb)
    n_vertices = int(mesh.vertices.shape[0])
    if n_vertices == 0:
        max_vertex = int(max_e.max()) + 1
    else:
        max_vertex = n_vertices
    keys = min_e.astype(np.int64) * (max_vertex + 1) + max_e

    order = np.argsort(keys, kind='stable')
    keys_sorted = keys[order]
    face_ids_sorted = face_ids[order]

    diff = np.empty(keys_sorted.shape[0], dtype=bool)
    diff[0] = True
    diff[1:] = keys_sorted[1:] != keys_sorted[:-1]
    start_idx = np.flatnonzero(diff)
    end_idx = np.append(start_idx[1:], keys_sorted.shape[0])
    counts = end_idx - start_idx

    # 每个唯一边的类型：1/2/3
    edge_type_by_unique = np.ones(len(start_idx), dtype=np.uint8)
    edge_type_by_unique[counts == 2] = 2
    edge_type_by_unique[counts >= 3] = 3

    # 将原始边键映射回类型
    unique_keys = keys_sorted[start_idx]
    pos = np.searchsorted(unique_keys, keys)
    face_edge_types = edge_type_by_unique[pos].reshape(-1, 3)

    return face_edge_types


def compute_edge_to_faces(mesh):
    """
    返回两个数组：
        edge_keys: 每条非开放边的唯一键（int64）
        edge_faces: 列表的列表，每个列表包含共享该边的所有面索引
    仅包含流形边（2个面）和非流形边（≥3个面）。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces)
    if n_faces == 0:
        return np.array([], dtype=np.int64), []

    edge_pairs = np.stack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], axis=1).reshape(-1, 2)

    ea = edge_pairs[:, 0]
    eb = edge_pairs[:, 1]
    face_ids = np.repeat(np.arange(n_faces, dtype=np.int64), 3)

    min_e = np.minimum(ea, eb)
    max_e = np.maximum(ea, eb)
    n_vertices = int(mesh.vertices.shape[0])
    if n_vertices == 0:
        max_vertex = int(max_e.max()) + 1
    else:
        max_vertex = n_vertices
    keys = min_e.astype(np.int64) * (max_vertex + 1) + max_e

    order = np.argsort(keys, kind='stable')
    keys_sorted = keys[order]
    face_ids_sorted = face_ids[order]

    diff = np.empty(keys_sorted.shape[0], dtype=bool)
    diff[0] = True
    diff[1:] = keys_sorted[1:] != keys_sorted[:-1]
    start_idx = np.flatnonzero(diff)
    end_idx = np.append(start_idx[1:], keys_sorted.shape[0])
    counts = end_idx - start_idx

    # 筛选出非开放边
    valid = counts >= 2
    valid_start = start_idx[valid]
    valid_counts = counts[valid]
    valid_keys = keys_sorted[valid_start]

    edge_faces = []
    for s, c in zip(valid_start, valid_counts):
        edge_faces.append(face_ids_sorted[s:s+c].tolist())

    return valid_keys, edge_faces


def compute_face_edge_keys(mesh):
    """
    返回形状 (n_faces, 3) 的 int64 数组，每个元素为该面对应边的唯一键，
    顺序与 (v0,v1), (v1,v2), (v2,v0) 一致。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces)
    if n_faces == 0:
        return np.zeros((0, 3), dtype=np.int64)

    edge_pairs = np.stack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], axis=1).reshape(-1, 2)

    ea = edge_pairs[:, 0]
    eb = edge_pairs[:, 1]

    min_e = np.minimum(ea, eb)
    max_e = np.maximum(ea, eb)
    n_vertices = int(mesh.vertices.shape[0])
    if n_vertices == 0:
        max_vertex = int(max_e.max()) + 1
    else:
        max_vertex = n_vertices
    keys = min_e.astype(np.int64) * (max_vertex + 1) + max_e

    return keys.reshape(-1, 3)


def compute_class_neighbor_stats(mesh, face_indices,
                                 open_face_mask, nonmanifold_face_mask,
                                 vertex_faces_csr, edge_to_faces, face_edge_keys):
    """
    计算指定面片集合的点邻和边邻面类型计数。
    边邻基于精确的 edge_to_faces 映射，点邻 = 顶点邻接面 - 边邻面。
    """
    point_counts = {'normal': 0, 'open': 0, 'nonmanifold': 0}
    edge_counts = {'normal': 0, 'open': 0, 'nonmanifold': 0}

    for fid in face_indices:
        # 边邻：通过三条边键查询所有共享面
        edge_neighbors = set()
        for key in face_edge_keys[fid]:
            for nb in edge_to_faces.get(int(key), []):
                if nb != fid:
                    edge_neighbors.add(nb)
                    if nonmanifold_face_mask[nb]:
                        edge_counts['nonmanifold'] += 1
                    elif open_face_mask[nb]:
                        edge_counts['open'] += 1
                    else:
                        edge_counts['normal'] += 1

        # 点邻：顶点邻接面并集减去边邻面
        verts = mesh.faces[fid]
        all_nb = set()
        for v in verts:
            row_start = vertex_faces_csr.indptr[v]
            row_end = vertex_faces_csr.indptr[v + 1]
            indices = vertex_faces_csr.indices[row_start:row_end]
            all_nb.update(indices)
        all_nb.discard(fid)
        point_neighbors = all_nb - edge_neighbors

        for nb in point_neighbors:
            if nonmanifold_face_mask[nb]:
                point_counts['nonmanifold'] += 1
            elif open_face_mask[nb]:
                point_counts['open'] += 1
            else:
                point_counts['normal'] += 1

    return point_counts, edge_counts


def compute_single_face_neighbor_stats(mesh, face_id,
                                       open_face_mask, nonmanifold_face_mask,
                                       vertex_faces_csr, edge_to_faces, face_edge_keys):
    """
    计算指定面片的逐顶点、逐边邻居统计。
    返回 vertex_stats (list of 3 dicts) 和 edge_stats (list of 3 dicts)。
    """
    verts = mesh.faces[face_id]

    # ---- 逐边统计 ----
    edge_stats = []
    for i in range(3):
        key = int(face_edge_keys[face_id, i])
        neighbor_faces = [nb for nb in edge_to_faces.get(key, []) if nb != face_id]

        cnt = {'normal': 0, 'open': 0, 'nonmanifold': 0}
        for nb in neighbor_faces:
            if nonmanifold_face_mask[nb]:
                cnt['nonmanifold'] += 1
            elif open_face_mask[nb]:
                cnt['open'] += 1
            else:
                cnt['normal'] += 1
        edge_stats.append(cnt)

    # ---- 收集该面的所有边邻面，用于点邻计算 ----
    edge_neighbors_set = set()
    for key in face_edge_keys[face_id]:
        for nb in edge_to_faces.get(int(key), []):
            if nb != face_id:
                edge_neighbors_set.add(nb)

    # ---- 逐顶点统计 ----
    vertex_stats = []
    for v in verts:
        row_start = vertex_faces_csr.indptr[v]
        row_end = vertex_faces_csr.indptr[v + 1]
        all_nb = set(vertex_faces_csr.indices[row_start:row_end])
        all_nb.discard(face_id)
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


def compute_face_topology_codes(mesh, face_indices, vertex_face_counts, face_edge_types):
    """
    计算指定面片的标准化拓扑编码。

    返回:
        codes : np.ndarray, 形状 (n,6), dtype uint8
                每行 6 个字段依次为: vA, eAB, vB, eBC, vC, eCA
                顶点元为真实 valence 截断到 255；边元为 1/2/3。
        code_to_id : dict, 6 字节 bytes 编码 -> 整数 id
        id_to_code : list, 整数 id -> 6 字节 bytes 编码
    """
    face_indices = np.asarray(face_indices, dtype=np.int64)
    n = len(face_indices)
    if n == 0:
        return np.zeros((0, 6), dtype=np.uint8), {}, []

    faces = np.asarray(mesh.faces)[face_indices]
    # 顶点元：真实 valence，截断到 uint8 上限
    v_counts_raw = vertex_face_counts[faces]
    v_counts = np.clip(v_counts_raw, 0, 255).astype(np.uint8)

    # 边元：face_edge_types 已为 1/2/3
    e_types = face_edge_types[face_indices].astype(np.uint8)

    all_codes = []

    for i in range(n):
        va, vb, vc = v_counts[i]
        eab, ebc, eca = e_types[i]

        # 原始 6 字段序列
        raw = np.array([va, eab, vb, ebc, vc, eca], dtype=np.uint8)

        # 生成 6 个候选：3 旋转 + 3 镜像
        candidates = []
        # 旋转
        for shift in [0, 2, 4]:
            candidates.append(np.concatenate([raw[shift:], raw[:shift]]))
        # 镜像：vA, eCA, vC, eBC, vB, eAB
        mirror = np.concatenate([raw[[0]], raw[[5]], raw[[4]], raw[[3]], raw[[2]], raw[[1]]])
        candidates.append(mirror)
        for shift in [2, 4]:
            candidates.append(np.concatenate([mirror[shift:], mirror[:shift]]))

        # 取字典序最小的 uint8 数组
        min_code = min(candidates, key=lambda x: tuple(x))
        all_codes.append(min_code)

    codes = np.array(all_codes, dtype=np.uint8)

    # 建立编码到 id 的映射（按字典序排序）
    unique_bytes = sorted(set(bytes(c) for c in all_codes))
    code_to_id = {b: i for i, b in enumerate(unique_bytes)}
    id_to_code = unique_bytes

    return codes, code_to_id, id_to_code


def get_face_topology_code_and_order(mesh, face_id, vertex_face_counts, edge_to_faces, face_edge_keys):
    """
    返回指定面片的标准化拓扑编码（6 字节 bytes）以及对应的顶点/边顺序映射。

    顶点元为真实 valence 截断到 255；边元为真实共享数（开放边为 1，流形边为 2，非流形边为 >=3）。
    移除拓扑一致性修正。
    """
    verts = mesh.faces[face_id]

    # 真实顶点共享数，截断到 255
    v_counts = np.clip(vertex_face_counts[verts], 0, 255).astype(np.uint8)

    # 边共享数
    e_counts_list = []
    for j in range(3):
        key = int(face_edge_keys[face_id, j])
        shared_faces = edge_to_faces.get(key, [])
        e_counts_list.append(len(shared_faces) if shared_faces else 1)
    e_counts = np.clip(e_counts_list, 0, 255).astype(np.uint8)

    raw = np.array([
        v_counts[0], e_counts[0],
        v_counts[1], e_counts[1],
        v_counts[2], e_counts[2]
    ], dtype=np.uint8)

    # 六种对称变换对应的顶点/边顺序映射
    transforms = [
        {"vertex_order": [0, 1, 2], "edge_order": [0, 1, 2]},
        {"vertex_order": [1, 2, 0], "edge_order": [1, 2, 0]},
        {"vertex_order": [2, 0, 1], "edge_order": [2, 0, 1]},
        {"vertex_order": [0, 2, 1], "edge_order": [2, 1, 0]},
        {"vertex_order": [2, 1, 0], "edge_order": [1, 0, 2]},
        {"vertex_order": [1, 0, 2], "edge_order": [0, 2, 1]},
    ]

    candidates = []
    # 旋转候选
    for shift, idx in [(0, 0), (2, 1), (4, 2)]:
        cand = np.concatenate([raw[shift:], raw[:shift]])
        candidates.append((cand, idx))
    # 镜像候选
    mirror = np.concatenate([raw[[0]], raw[[5]], raw[[4]], raw[[3]], raw[[2]], raw[[1]]])
    candidates.append((mirror, 3))
    for shift, offset in [(2, 1), (4, 2)]:
        cand = np.concatenate([mirror[shift:], mirror[:shift]])
        candidates.append((cand, 3 + offset))

    # 字典序最小
    min_code, min_idx = min(candidates, key=lambda x: tuple(x[0]))
    transform = transforms[min_idx]

    standard_code = bytes(min_code)
    return standard_code, transform["vertex_order"], transform["edge_order"]


def group_faces_by_topology_codes(mesh, face_indices, vertex_face_counts, face_edge_types, valence_threshold=5):
    """
    按截断后的标准拓扑编码对面片进行聚类。

    参数:
        mesh : trimesh.Trimesh
        face_indices : 待聚类的面片索引数组
        vertex_face_counts : np.ndarray, 每个顶点被面片引用的真实次数
        face_edge_types : np.ndarray, 形状 (n_faces, 3)，值为 1/2/3
        valence_threshold : int, 顶点元截断阈值，默认 5。
                            当顶点 valence >= valence_threshold 时，顶点元归并为阈值。

    返回:
        dict : {6 字节 bytes 编码 : np.ndarray of face indices}
    """
    face_indices = np.asarray(face_indices, dtype=np.int64)
    n = len(face_indices)
    if n == 0:
        return {}

    faces = np.asarray(mesh.faces)[face_indices]

    # 顶点元：真实 valence 截断到阈值后转为 uint8
    v_counts_raw = vertex_face_counts[faces]
    v_counts_clipped = np.minimum(v_counts_raw, valence_threshold).astype(np.uint8)
    # 边元：1/2/3
    e_types = face_edge_types[face_indices].astype(np.uint8)

    grouped = {}

    for i, fid in enumerate(face_indices):
        va, vb, vc = v_counts_clipped[i]
        eab, ebc, eca = e_types[i]

        raw = np.array([va, eab, vb, ebc, vc, eca], dtype=np.uint8)

        candidates = []
        for shift in [0, 2, 4]:
            candidates.append(np.concatenate([raw[shift:], raw[:shift]]))
        mirror = np.concatenate([raw[[0]], raw[[5]], raw[[4]], raw[[3]], raw[[2]], raw[[1]]])
        candidates.append(mirror)
        for shift in [2, 4]:
            candidates.append(np.concatenate([mirror[shift:], mirror[:shift]]))

        min_code = min(candidates, key=lambda x: tuple(x))
        key = bytes(min_code)

        grouped.setdefault(key, []).append(fid)

    # 将列表转为 np.ndarray
    for key in grouped:
        grouped[key] = np.array(grouped[key], dtype=np.int64)

    return grouped


class FaceTopologyCode6:
    """
    表示三角面片拓扑编码的六元数，六个字段均为 uint8：
        [vA, eAB, vB, eBC, vC, eCA]
    """
    __slots__ = ("data",)

    def __init__(self, *values):
        if len(values) == 1 and isinstance(values[0], np.ndarray):
            arr = values[0]
        else:
            arr = np.array(values, dtype=np.uint8)
        if arr.shape != (6,) or arr.dtype != np.uint8:
            # 尝试转换并检查长度
            arr = np.asarray(arr, dtype=np.uint8).reshape(-1)
            if arr.shape != (6,):
                raise ValueError("code must have exactly 6 uint8 fields")
        self.data = arr

    def __bytes__(self):
        return self.data.tobytes()

    def __repr__(self):
        return f"FaceTopologyCode6({self.data.tolist()})"

    def __eq__(self, other):
        if not isinstance(other, FaceTopologyCode6):
            return NotImplemented
        return np.array_equal(self.data, other.data)

    def __hash__(self):
        return hash(bytes(self.data))

    def to_hex(self):
        return self.data.tobytes().hex()

    @classmethod
    def from_hex(cls, s):
        return cls(np.frombuffer(bytes.fromhex(s), dtype=np.uint8))


def canonicalize_single_code(code):
    """
    输入长度 6 的 uint8 序列（或 FaceTopologyCode6），
    返回规范化后的长度 6 的 uint8 数组（旋转/镜像无关，字典序最小）。
    """
    if isinstance(code, FaceTopologyCode6):
        raw = code.data.copy()
    else:
        raw = np.asarray(code, dtype=np.uint8)
        if raw.shape != (6,):
            raise ValueError("code must have exactly 6 uint8 fields")

    candidates = []
    # 旋转
    for shift in [0, 2, 4]:
        candidates.append(np.concatenate([raw[shift:], raw[:shift]]))
    # 镜像：vA, eCA, vC, eBC, vB, eAB
    mirror = np.concatenate([raw[[0]], raw[[5]], raw[[4]], raw[[3]], raw[[2]], raw[[1]]])
    candidates.append(mirror)
    for shift in [2, 4]:
        candidates.append(np.concatenate([mirror[shift:], mirror[:shift]]))

    return min(candidates, key=lambda x: tuple(x))


def canonicalize_faces(codes):
    """
    输入形状 (n,6) 的 uint8 数组，返回规范化后的 (n,6) 数组。
    与 compute_face_topology_codes 中的归一化逻辑一致。
    """
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != 6:
        raise ValueError("codes must be of shape (n,6)")
    n = len(codes)
    result = np.empty((n, 6), dtype=np.uint8)
    for i in range(n):
        result[i] = canonicalize_single_code(codes[i])
    return result


def truncate_codes(codes, valence_threshold=5):
    """
    对顶点元（索引 0,2,4）按阈值截断，返回新的 (n,6) uint8 数组。
    """
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != 6:
        raise ValueError("codes must be of shape (n,6)")
    out = codes.copy()
    # 顶点元索引
    v_idx = [0, 2, 4]
    clipped = np.minimum(out[:, v_idx], valence_threshold).astype(np.uint8)
    out[:, v_idx] = clipped
    return out


def code_to_bytes(code):
    """
    将单个长度 6 的 uint8 序列或 FaceTopologyCode6 转为 6 字节 bytes。
    """
    if isinstance(code, FaceTopologyCode6):
        return bytes(code)
    arr = np.asarray(code, dtype=np.uint8)
    if arr.shape != (6,):
        raise ValueError("code must have exactly 6 uint8 fields")
    return arr.tobytes()


def bytes_to_code(b):
    """
    从 6 字节 bytes 恢复长度 6 的 uint8 数组。
    """
    if not isinstance(b, (bytes, bytearray)):
        raise TypeError("input must be bytes-like")
    if len(b) != 6:
        raise ValueError("bytes must have length 6")
    return np.frombuffer(b, dtype=np.uint8).copy()


def code_to_hex(code):
    """
    将单个长度 6 的 uint8 序列转为十六进制字符串（不含前缀）。
    例如 [1,2,3,4,5,6] -> '010203040506'
    """
    return code_to_bytes(code).hex()


def hex_to_code(s):
    """
    从十六进制字符串恢复长度 6 的 uint8 数组。
    """
    # 若字符串以 0x 开头，移除前缀
    if s.lower().startswith("0x"):
        s = s[2:]
    if len(s) != 12:
        raise ValueError("hex string must represent exactly 6 bytes (12 hex digits)")
    return bytes_to_code(bytes.fromhex(s))


def save_codes(codes, path):
    """
    将形状 (n,6) 的 uint8 数组保存为 .npy 文件。
    """
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != 6:
        raise ValueError("codes must be of shape (n,6)")
    np.save(path, codes)


def load_codes(path):
    """
    从 .npy 文件加载形状 (n,6) 的 uint8 数组。
    """
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("loaded array must be of shape (n,6)")
    return arr


def validate_code(code):
    """
    校验一个编码是否是合法的 6 字段 uint8 序列。
    返回 bool。
    """
    try:
        arr = np.asarray(code, dtype=np.uint8)
    except (TypeError, ValueError):
        return False
    if arr.shape != (6,):
        return False
    return True
