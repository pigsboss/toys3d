# src/toys3d/geometrics.py
"""
基础几何处理工具集。

包含网格统计、缺陷分析、边界环提取、孔洞面积统计、
可靠面片提取、代理壳拓扑补丁、重新三角化、水密重建等。
"""

import numpy as np
import trimesh
from collections import deque
from scipy.sparse import csr_matrix


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

    # 直接处理 bytes / bytearray 输入
    if isinstance(code, (bytes, bytearray)):
        if len(code) != 6:
            raise ValueError("code must have exactly 6 uint8 fields")
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


def compute_face_edge_valences(mesh, edge_to_faces, face_edge_keys):
    """
    返回形状 (n_faces, 3) 的 int32 数组，每个元素为对应边的真实共享面数：
        1：开放边
        2：流形边
        n>=3：非流形边（保留真实 n）
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros((0, 3), dtype=np.int32)

    edge_valences = np.ones((n_faces, 3), dtype=np.int32)
    for fid in range(n_faces):
        for j in range(3):
            key = int(face_edge_keys[fid, j])
            shared = edge_to_faces.get(key, [])
            edge_valences[fid, j] = len(shared) if shared else 1
    return edge_valences


def compute_open_edge_data(mesh):
    """
    提取网格中所有开放边的核心数据。

    返回字典：
        open_edge_vertex_pairs : (E,2) int64, 每条开放边的两个顶点索引
        open_edge_face_ids     : (E,)   int64, 每条开放边唯一所属的面片 id
        open_edge_keys         : (E,)   int64, 边的唯一键（min_v * max_vertex + max_v）
        open_edge_key_to_id    : dict,  键 -> 开放边 id
        vertex_open_edges_csr  : scipy.sparse.csr_matrix, 形状 (n_vertices, E),
                                 每行存储关联该顶点的开放边 id
        vertex_degree          : (n_vertices,) int32, 每个顶点在开放边图上的度数
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces)
    n_vertices = len(mesh.vertices)

    if n_faces == 0 or n_vertices == 0:
        return {
            'open_edge_vertex_pairs': np.zeros((0, 2), dtype=np.int64),
            'open_edge_face_ids': np.zeros(0, dtype=np.int64),
            'open_edge_keys': np.zeros(0, dtype=np.int64),
            'open_edge_key_to_id': {},
            'vertex_open_edges_csr': csr_matrix((n_vertices, 0), dtype=np.int64),
            'vertex_degree': np.zeros(n_vertices, dtype=np.int32),
        }

    # 构造所有有向边
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
    max_vertex = n_vertices
    keys = min_e.astype(np.int64) * (max_vertex + 1) + max_e

    order = np.argsort(keys, kind='stable')
    keys_sorted = keys[order]
    face_ids_sorted = face_ids[order]
    ea_sorted = ea[order]
    eb_sorted = eb[order]

    diff = np.empty(keys_sorted.shape[0], dtype=bool)
    diff[0] = True
    diff[1:] = keys_sorted[1:] != keys_sorted[:-1]
    start_idx = np.flatnonzero(diff)
    end_idx = np.append(start_idx[1:], keys_sorted.shape[0])
    counts = end_idx - start_idx

    # 筛选开放边（counts == 1）
    open_mask = counts == 1
    open_start = start_idx[open_mask]
    open_face_ids = face_ids_sorted[open_start]
    open_vertex_pairs = np.column_stack([
        ea_sorted[open_start],
        eb_sorted[open_start]
    ])
    open_keys = keys_sorted[open_start]

    E = len(open_face_ids)
    open_edge_key_to_id = {int(k): i for i, k in enumerate(open_keys)}

    # 构建顶点 -> 开放边 CSR
    rows = open_vertex_pairs.ravel()
    cols = np.repeat(np.arange(E, dtype=np.int64), 2)
    data = np.ones(2 * E, dtype=np.int8)
    vertex_open_edges_csr = csr_matrix(
        (data, (rows, cols)),
        shape=(n_vertices, E),
        dtype=np.int64,
    )

    vertex_degree = np.asarray(vertex_open_edges_csr.getnnz(axis=1)).ravel().astype(np.int32)

    return {
        'open_edge_vertex_pairs': open_vertex_pairs.astype(np.int64),
        'open_edge_face_ids': open_face_ids.astype(np.int64),
        'open_edge_keys': open_keys.astype(np.int64),
        'open_edge_key_to_id': open_edge_key_to_id,
        'vertex_open_edges_csr': vertex_open_edges_csr,
        'vertex_degree': vertex_degree,
    }


def build_hole_diagnosis_data(mesh):
    """
    构建孔洞诊断数据结构。

    返回字典包含：
        open_edge_vertex_pairs, open_edge_face_ids, open_edge_keys,
        open_edge_key_to_id, vertex_open_edges_csr, vertex_degree,
        hole_vertex_lists : list of list[int]，健康孔洞的顶点序列
        hole_edge_lists   : list of list[int]，健康孔洞的开放边 id 序列
        hole_ids_per_edge : (E,) int32，每条开放边所属健康孔洞 id，-1 表示未覆盖
        uncovered_edge_ids : (U,) int64，未被任何健康孔洞覆盖的开放边 id
        uncovered_category : (U,) int8，未覆盖开放边分类（见规则）
    """
    open_data = compute_open_edge_data(mesh)
    vertex_pairs = open_data['open_edge_vertex_pairs']
    edge_keys = open_data['open_edge_keys']
    key_to_id = open_data['open_edge_key_to_id']

    E = len(vertex_pairs)
    hole_ids_per_edge = np.full(E, -1, dtype=np.int32)
    hole_vertex_lists = []
    hole_edge_lists = []

    # 健康孔洞提取：使用现有 extract_boundary_loops 获取闭合、度数为2的环
    loops = extract_boundary_loops(mesh)

    for loop in loops:
        # 环的顶点列表
        loop_vertices = [int(v) for v in loop]
        edge_ids = []
        valid = True
        for i in range(len(loop_vertices)):
            v0 = loop_vertices[i]
            v1 = loop_vertices[(i + 1) % len(loop_vertices)]
            min_v = min(v0, v1)
            max_v = max(v0, v1)
            key = min_v * (len(mesh.vertices) + 1) + max_v
            if key not in key_to_id:
                valid = False
                break
            edge_ids.append(key_to_id[key])
        if not valid:
            continue

        hole_id = len(hole_vertex_lists)
        hole_vertex_lists.append(loop_vertices)
        hole_edge_lists.append(edge_ids)
        for eid in edge_ids:
            hole_ids_per_edge[eid] = hole_id

    # 未覆盖开放边 id
    uncovered = np.where(hole_ids_per_edge == -1)[0].astype(np.int64)

    # 分类未覆盖开放边
    degree = open_data['vertex_degree']
    nonmanifold_mask = None
    # 获取非流形面片掩码
    _, _, nonmanifold_face_mask = analyze_mesh_defects(mesh)

    categories = np.zeros(len(uncovered), dtype=np.int8)
    for idx, edge_id in enumerate(uncovered):
        v0, v1 = vertex_pairs[edge_id]
        d0 = degree[v0]
        d1 = degree[v1]

        # 非流形关联优先
        face_id = open_data['open_edge_face_ids'][edge_id]
        if nonmanifold_face_mask[face_id]:
            categories[idx] = 4
            continue

        # 两端度数都为1：孤立开放链
        if d0 == 1 and d1 == 1:
            categories[idx] = 0
        # 一端度数1，另一端>=2：悬空开放边
        elif (d0 == 1 and d1 >= 2) or (d1 == 1 and d0 >= 2):
            categories[idx] = 1
        # 两端度数都>=2：分支内部开放边
        elif d0 >= 2 and d1 >= 2:
            categories[idx] = 2
        else:
            categories[idx] = 5

    return {
        **open_data,
        'hole_vertex_lists': hole_vertex_lists,
        'hole_edge_lists': hole_edge_lists,
        'hole_ids_per_edge': hole_ids_per_edge,
        'uncovered_edge_ids': uncovered,
        'uncovered_category': categories,
        'nonmanifold_face_mask': nonmanifold_face_mask,
    }


def analyze_uncovered_open_edge_components(mesh, hole_data, spatial_threshold=None):
    """
    进一步分析“异常孔洞”：对未覆盖开放边做连通分量分析，找出导致
    健康孔洞无法闭合的异常点/断裂点。

    返回:
        list of dict，每个元素描述一个未覆盖开放边连通分量，包含：
            component_id         : int，分量编号
            num_edges            : int，分量内边数
            num_vertices         : int，分量内顶点数
            vertices             : list[int]，分量顶点索引
            endpoints            : list[int]，度数为 1 的顶点（端点）
            branch_vertices      : list[int]，度数 >= 3 的顶点（分支点）
            is_cycle             : bool，是否所有顶点度数均为 2（闭合环）
            face_ids             : list[int]，相邻面片索引（去重）
            open_face_count      : int，开放面数量（所有相邻面均为开放面）
            nonmanifold_face_count : int，非流形面数量
            candidate_breaks     : list[dict]，空间上可能断裂的端点对
                                   [{'v0':int,'v1':int,'distance':float}]
    """
    from scipy.spatial import cKDTree

    uncovered_ids = hole_data['uncovered_edge_ids']
    all_vertex_pairs = hole_data['open_edge_vertex_pairs']
    all_face_ids = hole_data['open_edge_face_ids']
    vertex_csr = hole_data['vertex_open_edges_csr']
    hole_ids_per_edge = hole_data['hole_ids_per_edge']
    nonmanifold_face_mask = hole_data['nonmanifold_face_mask']

    U = len(uncovered_ids)
    if U == 0:
        return []

    # 建立全局开放边 ID 到未覆盖索引的映射
    uncovered_index = np.full(len(hole_ids_per_edge), -1, dtype=np.int64)
    uncovered_index[uncovered_ids] = np.arange(U, dtype=np.int64)

    visited = np.zeros(U, dtype=bool)
    components = []

    # BFS 搜索连通分量
    for start in range(U):
        if visited[start]:
            continue
        comp_indices = []
        queue = deque([start])
        visited[start] = True

        while queue:
            ue_idx = queue.popleft()
            comp_indices.append(ue_idx)
            global_eid = uncovered_ids[ue_idx]
            v0, v1 = all_vertex_pairs[global_eid]

            for v in (v0, v1):
                row_start = vertex_csr.indptr[v]
                row_end = vertex_csr.indptr[v + 1]
                for j in range(row_start, row_end):
                    gid = vertex_csr.indices[j]
                    if hole_ids_per_edge[gid] != -1:
                        continue
                    nu = uncovered_index[gid]
                    if nu == -1 or visited[nu]:
                        continue
                    visited[nu] = True
                    queue.append(int(nu))

        # 本分量数据
        comp_local = np.array(comp_indices, dtype=np.int64)
        comp_global = uncovered_ids[comp_local]
        comp_edges = all_vertex_pairs[comp_global]
        comp_face_ids = np.unique(all_face_ids[comp_global])

        # 顶点度数（未覆盖子图）
        verts = comp_edges.ravel()
        unique_verts, counts = np.unique(verts, return_counts=True)
        endpoints = unique_verts[counts == 1]
        branch_vertices = unique_verts[counts >= 3]
        is_cycle = bool(np.all(counts == 2) and len(unique_verts) == len(comp_global))

        # 几何断裂候选
        candidate_breaks = []
        if len(endpoints) >= 2:
            endpoint_pts = mesh.vertices[endpoints]
            if spatial_threshold is None:
                edge_lengths = np.linalg.norm(
                    mesh.vertices[comp_edges[:, 1]] - mesh.vertices[comp_edges[:, 0]],
                    axis=1
                )
                if len(edge_lengths) == 0:
                    spatial_threshold = 0.0
                else:
                    spatial_threshold = float(np.median(edge_lengths)) * 3.0

            if spatial_threshold > 0:
                tree = cKDTree(endpoint_pts)
                pairs = tree.query_pairs(spatial_threshold, output_type='ndarray')

                # 排除直接相连的边（同一条边的两个端点）
                direct_edges = set()
                for a, b in comp_edges:
                    direct_edges.add((int(a), int(b)))
                    direct_edges.add((int(b), int(a)))

                for i, j in pairs:
                    u = int(endpoints[i])
                    v = int(endpoints[j])
                    if (u, v) in direct_edges or (v, u) in direct_edges:
                        continue
                    dist = float(np.linalg.norm(mesh.vertices[u] - mesh.vertices[v]))
                    candidate_breaks.append({'v0': u, 'v1': v, 'distance': dist})

        # 非流形面统计
        nonmanifold_face_count = int(np.sum(nonmanifold_face_mask[comp_face_ids]))
        open_face_count = len(comp_face_ids)  # 所有相邻面都含开放边

        component_info = {
            'component_id': len(components),
            'num_edges': int(len(comp_global)),
            'num_vertices': int(len(unique_verts)),
            'vertices': unique_verts.tolist(),
            'edge_vertex_pairs': comp_edges.tolist(),   # 新增
            'endpoints': endpoints.tolist(),
            'branch_vertices': branch_vertices.tolist(),
            'is_cycle': bool(is_cycle),
            'face_ids': comp_face_ids.tolist(),
            'open_face_count': open_face_count,
            'nonmanifold_face_count': nonmanifold_face_count,
            'candidate_breaks': candidate_breaks,
        }
        components.append(component_info)

    return components


def build_manifold_face_adjacency(mesh):
    """
    基于流形边构建面片邻接表。

    只保留被恰好两个面共享的边（流形边）；
    开放边和非流形边不参与面片邻接。

    返回:
        list[list[int]]，长度为 n_faces，每个元素为该面片的流形邻接面片索引列表。
    """
    n_faces = len(mesh.faces)
    adj = [[] for _ in range(n_faces)]

    _, edge_faces = compute_edge_to_faces(mesh)
    for faces in edge_faces:
        if len(faces) == 2:
            f0, f1 = int(faces[0]), int(faces[1])
            adj[f0].append(f1)
            adj[f1].append(f0)

    return adj


def is_manifold_closed_boundary(mesh, face_set):
    """
    检查给定面片集合的边界中，是否存在由流形边构成的闭合环。
    内部开放边（孔洞）和非流形边不影响判定。
    """
    face_set = set(map(int, face_set))
    if not face_set:
        return False

    face_edge_keys = compute_face_edge_keys(mesh)

    # 构建边 key -> 顶点对
    edge_key_to_vertex_pair = {}
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    for fid in range(len(faces_all)):
        verts = faces_all[fid]
        for j in range(3):
            key = int(face_edge_keys[fid, j])
            if key not in edge_key_to_vertex_pair:
                edge_key_to_vertex_pair[key] = (
                    int(verts[j]), int(verts[(j + 1) % 3])
                )

    edge_keys, edge_faces = compute_edge_to_faces(mesh)
    edge_to_faces = {}
    for key, faces_list in zip(edge_keys, edge_faces):
        edge_to_faces[int(key)] = faces_list

    # 收集面集边界中所有流形边（全局共享 ==2）
    manifold_boundary_vertices = {}
    manifold_boundary_edges = []
    for fid in face_set:
        for j in range(3):
            key = int(face_edge_keys[fid, j])
            shared = edge_to_faces.get(key, [])
            inner_count = sum(1 for f in shared if int(f) in face_set)
            if inner_count == 1 and len(shared) == 2:
                # 面集边界上的流形边
                manifold_boundary_edges.append(key)
                v0, v1 = edge_key_to_vertex_pair[key]
                manifold_boundary_vertices.setdefault(v0, []).append(v1)
                manifold_boundary_vertices.setdefault(v1, []).append(v0)

    if not manifold_boundary_edges:
        return False

    # 检查流形边界子图是否所有顶点度数为2，且边数等于顶点数（一个或多个环）
    for v, nbrs in manifold_boundary_vertices.items():
        if len(nbrs) != 2:
            return False

    return len(manifold_boundary_vertices) == len(manifold_boundary_edges)


def find_minimal_enclosing_manifold_boundary_greedy(
    mesh, component, max_depth=12
):
    """
    从组件面片出发，逐层沿流形边扩展，直到找到满足流形闭合边界的最小区域。

    返回:
        dict:
            success          : bool  是否找到包络边界
            depth            : int   使用的扩展深度
            enclosed_faces    : list[int] 包络内部面片索引
            boundary_vertices : list[list[int]] 边界顶点环列表
            boundary_edges    : list[int] 包络边界边 key
    """
    seed_faces = set(map(int, component.get('face_ids', [])))
    if not seed_faces:
        return {
            'success': False,
            'depth': 0,
            'enclosed_faces': [],
            'boundary_vertices': [],
            'boundary_edges': [],
        }

    adj = build_manifold_face_adjacency(mesh)
    current = set(seed_faces)

    for depth in range(1, max_depth + 1):
        if is_manifold_closed_boundary(mesh, current):
            # 提取边界信息
            face_edge_keys = compute_face_edge_keys(mesh)
            edge_keys, edge_faces = compute_edge_to_faces(mesh)
            edge_to_faces = {}
            for key, faces_list in zip(edge_keys, edge_faces):
                edge_to_faces[int(key)] = faces_list

            # 收集边界边 key
            boundary_edge_keys = []
            for fid in current:
                for j in range(3):
                    key = int(face_edge_keys[fid, j])
                    shared = edge_to_faces.get(key, [])
                    inner_count = sum(
                        1 for f in shared if int(f) in current
                    )
                    if inner_count == 1:
                        boundary_edge_keys.append(key)

            # 构建边界顶点环
            boundary_vertices = _extract_boundary_loops_from_edge_keys(
                mesh, boundary_edge_keys
            )

            return {
                'success': True,
                'depth': depth,
                'enclosed_faces': sorted(current),
                'boundary_vertices': boundary_vertices,
                'boundary_edges': boundary_edge_keys,
            }

        # 扩展到下一层（仅沿流形边）
        next_faces = set(current)
        for f in current:
            for nb in adj[f]:
                if nb not in next_faces:
                    next_faces.add(nb)

        if len(next_faces) == len(current):
            break   # 没有新的面片，无法继续扩展
        current = next_faces

    return {
        'success': False,
        'depth': max_depth,
        'enclosed_faces': [],
        'boundary_vertices': [],
        'boundary_edges': [],
    }


def _extract_boundary_loops_from_edge_keys(mesh, edge_keys):
    """
    从边界边 key 列表（流形边界边）提取闭合顶点环列表。

    返回:
        list of list[int]，每个内层列表是一个闭合环的顶点索引序列。
    """
    # 构建边 key -> 顶点对
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    face_edge_keys = compute_face_edge_keys(mesh)
    edge_key_to_vertex_pair = {}
    for fid in range(len(faces_all)):
        verts = faces_all[fid]
        for j in range(3):
            key = int(face_edge_keys[fid, j])
            if key not in edge_key_to_vertex_pair:
                edge_key_to_vertex_pair[key] = (
                    int(verts[j]), int(verts[(j + 1) % 3])
                )

    # 构建顶点邻接（无向图）
    adj = {}
    used_edges = set()
    for key in edge_keys:
        key = int(key)
        if key in used_edges:
            continue
        used_edges.add(key)
        v0, v1 = edge_key_to_vertex_pair[key]
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    # 使用 BFS 提取所有简单环
    visited_edges = set()
    loops = []

    for start in list(adj.keys()):
        if start not in adj:
            continue
        loop = []
        cur = start
        prev = None

        while True:
            loop.append(cur)
            candidates = [n for n in adj[cur] if n != prev and n in adj]
            nxt = None
            for cand in candidates:
                ekey = (cur, cand) if cur < cand else (cand, cur)
                if ekey not in visited_edges:
                    nxt = cand
                    break

            if nxt is None:
                break

            ekey = (cur, nxt) if cur < nxt else (nxt, cur)
            visited_edges.add(ekey)

            if nxt == start and len(loop) >= 3:
                loop.append(nxt)
                loops.append(loop)
                break

            prev, cur = cur, nxt

            if len(loop) > len(adj):
                break

    return loops
