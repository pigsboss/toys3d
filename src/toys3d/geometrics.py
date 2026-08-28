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
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]

    # 三条边分别为 (v0,v1), (v1,v2), (v2,v0)
    ea = np.concatenate([v0, v1, v2])
    eb = np.concatenate([v1, v2, v0])

    # 对应每条边的面片索引
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
