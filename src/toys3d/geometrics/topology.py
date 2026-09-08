# src/toys3d/geometrics/topology.py
"""
拓扑几何工具：边/面关系、连通性、边界环、拓扑编码等。
"""
import numpy as np
from scipy.sparse import csr_matrix
from collections import deque


def analyze_mesh_defects(mesh, return_face_edge_counts=False):
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces)

    if n_faces == 0:
        if return_face_edge_counts:
            empty = np.zeros(0, dtype=np.uint8)
            return (
                {
                    'total_faces': 0,
                    'raw_edges_count': 0,
                    'unique_edges_count': 0,
                    'open_edges': 0,
                    'manifold_edges': 0,
                    'nonmanifold_edges': 0,
                    'open_faces': 0,
                    'nonmanifold_faces': 0,
                    'both_defect_faces': 0,
                    'watertight_by_count': True,
                },
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=bool),
                empty.copy(), empty.copy(), empty.copy()
            )
        else:
            return (
                {
                    'total_faces': 0,
                    'raw_edges_count': 0,
                    'unique_edges_count': 0,
                    'open_edges': 0,
                    'manifold_edges': 0,
                    'nonmanifold_edges': 0,
                    'open_faces': 0,
                    'nonmanifold_faces': 0,
                    'both_defect_faces': 0,
                    'watertight_by_count': True,
                },
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=bool)
            )

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

    open_edges = int(np.sum(counts == 1))
    nonmanifold_edges = int(np.sum(counts >= 3))

    open_face_mask = np.zeros(n_faces, dtype=bool)
    nonmanifold_face_mask = np.zeros(n_faces, dtype=bool)

    if return_face_edge_counts:
        open_edge_per_face = np.zeros(n_faces, dtype=np.uint8)
        manifold_edge_per_face = np.zeros(n_faces, dtype=np.uint8)
        nonmanifold_edge_per_face = np.zeros(n_faces, dtype=np.uint8)

    open_pos = start_idx[counts == 1]
    if len(open_pos) > 0:
        open_f = face_ids_sorted[open_pos]
        open_face_mask[open_f] = True
        if return_face_edge_counts:
            np.add.at(open_edge_per_face, open_f, 1)

    manifold_pos = start_idx[counts == 2]
    if len(manifold_pos) > 0:
        f0 = face_ids_sorted[manifold_pos]
        f1 = face_ids_sorted[manifold_pos + 1]
        if return_face_edge_counts:
            np.add.at(manifold_edge_per_face, f0, 1)
            np.add.at(manifold_edge_per_face, f1, 1)

    nonmanifold_start = start_idx[counts >= 3]
    nonmanifold_vals = counts[counts >= 3]
    for start, cnt in zip(nonmanifold_start, nonmanifold_vals):
        seg_face_ids = face_ids_sorted[start:start + cnt]
        nonmanifold_face_mask[seg_face_ids] = True
        if return_face_edge_counts:
            np.add.at(nonmanifold_edge_per_face, seg_face_ids, 1)

    defect_stats = {
        'total_faces': n_faces,
        'raw_edges_count': n_faces * 3,
        'unique_edges_count': int(len(start_idx)),
        'open_edges': open_edges,
        'manifold_edges': int(np.sum(counts == 2)),
        'nonmanifold_edges': nonmanifold_edges,
        'open_faces': int(open_face_mask.sum()),
        'nonmanifold_faces': int(nonmanifold_face_mask.sum()),
        'both_defect_faces': int(np.sum(open_face_mask & nonmanifold_face_mask)),
        'watertight_by_count': bool(open_edges == 0),
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


def compute_topological_reliable_face_mask(mesh, min_distance=2):
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

    _, open_mask, nonmanifold_mask = analyze_mesh_defects(mesh)
    defect_mask = open_mask | nonmanifold_mask

    if not np.any(defect_mask):
        return np.ones(n_faces, dtype=bool), np.full(n_faces, min_distance + 1, dtype=int)

    adjacency = [[] for _ in range(n_faces)]
    face_adj = getattr(mesh, 'face_adjacency', None)
    if face_adj is not None and len(face_adj) > 0:
        for f0, f1 in face_adj:
            f0, f1 = int(f0), int(f1)
            if f0 < 0 or f1 < 0:
                continue
            adjacency[f0].append(f1)
            adjacency[f1].append(f0)

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

    dist[dist == -1] = min_distance + 1
    reliable_mask = (~defect_mask) & (dist >= min_distance)
    return reliable_mask, dist


def compute_vertex_face_counts(mesh):
    return np.bincount(mesh.faces.ravel(), minlength=len(mesh.vertices))


def compute_face_edge_types(mesh):
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

    edge_type_by_unique = np.ones(len(start_idx), dtype=np.uint8)
    edge_type_by_unique[counts == 2] = 2
    edge_type_by_unique[counts >= 3] = 3

    unique_keys = keys_sorted[start_idx]
    pos = np.searchsorted(unique_keys, keys)
    face_edge_types = edge_type_by_unique[pos].reshape(-1, 3)

    return face_edge_types


def compute_edge_to_faces(mesh):
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

    valid = counts >= 2
    valid_start = start_idx[valid]
    valid_counts = counts[valid]
    valid_keys = keys_sorted[valid_start]

    edge_faces = []
    for s, c in zip(valid_start, valid_counts):
        edge_faces.append(face_ids_sorted[s:s+c].tolist())

    return valid_keys, edge_faces


def compute_face_edge_keys(mesh):
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
    point_counts = {'normal': 0, 'open': 0, 'nonmanifold': 0}
    edge_counts = {'normal': 0, 'open': 0, 'nonmanifold': 0}

    for fid in face_indices:
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
    verts = mesh.faces[face_id]

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

    edge_neighbors_set = set()
    for key in face_edge_keys[face_id]:
        for nb in edge_to_faces.get(int(key), []):
            if nb != face_id:
                edge_neighbors_set.add(nb)

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
    face_indices = np.asarray(face_indices, dtype=np.int64)
    n = len(face_indices)
    if n == 0:
        return np.zeros((0, 6), dtype=np.uint8), {}, []

    faces = np.asarray(mesh.faces)[face_indices]
    v_counts_raw = vertex_face_counts[faces]
    v_counts = np.clip(v_counts_raw, 0, 255).astype(np.uint8)

    e_types = face_edge_types[face_indices].astype(np.uint8)

    all_codes = []

    for i in range(n):
        va, vb, vc = v_counts[i]
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
        all_codes.append(min_code)

    codes = np.array(all_codes, dtype=np.uint8)

    unique_bytes = sorted(set(bytes(c) for c in all_codes))
    code_to_id = {b: i for i, b in enumerate(unique_bytes)}
    id_to_code = unique_bytes

    return codes, code_to_id, id_to_code


def get_face_topology_code_and_order(mesh, face_id, vertex_face_counts, edge_to_faces, face_edge_keys):
    verts = mesh.faces[face_id]

    v_counts = np.clip(vertex_face_counts[verts], 0, 255).astype(np.uint8)

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

    transforms = [
        {"vertex_order": [0, 1, 2], "edge_order": [0, 1, 2]},
        {"vertex_order": [1, 2, 0], "edge_order": [1, 2, 0]},
        {"vertex_order": [2, 0, 1], "edge_order": [2, 0, 1]},
        {"vertex_order": [0, 2, 1], "edge_order": [2, 1, 0]},
        {"vertex_order": [2, 1, 0], "edge_order": [1, 0, 2]},
        {"vertex_order": [1, 0, 2], "edge_order": [0, 2, 1]},
    ]

    candidates = []
    for shift, idx in [(0, 0), (2, 1), (4, 2)]:
        cand = np.concatenate([raw[shift:], raw[:shift]])
        candidates.append((cand, idx))
    mirror = np.concatenate([raw[[0]], raw[[5]], raw[[4]], raw[[3]], raw[[2]], raw[[1]]])
    candidates.append((mirror, 3))
    for shift, offset in [(2, 1), (4, 2)]:
        cand = np.concatenate([mirror[shift:], mirror[:shift]])
        candidates.append((cand, 3 + offset))

    min_code, min_idx = min(candidates, key=lambda x: tuple(x[0]))
    transform = transforms[min_idx]

    standard_code = bytes(min_code)
    return standard_code, transform["vertex_order"], transform["edge_order"]


def group_faces_by_topology_codes(mesh, face_indices, vertex_face_counts, face_edge_types, valence_threshold=5):
    face_indices = np.asarray(face_indices, dtype=np.int64)
    n = len(face_indices)
    if n == 0:
        return {}

    faces = np.asarray(mesh.faces)[face_indices]

    v_counts_raw = vertex_face_counts[faces]
    v_counts_clipped = np.minimum(v_counts_raw, valence_threshold).astype(np.uint8)
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

    for key in grouped:
        grouped[key] = np.array(grouped[key], dtype=np.int64)

    return grouped


class FaceTopologyCode6:
    __slots__ = ("data",)

    def __init__(self, *values):
        if len(values) == 1 and isinstance(values[0], np.ndarray):
            arr = values[0]
        else:
            arr = np.array(values, dtype=np.uint8)
        if arr.shape != (6,) or arr.dtype != np.uint8:
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
    if isinstance(code, FaceTopologyCode6):
        raw = code.data.copy()
    else:
        raw = np.asarray(code, dtype=np.uint8)
        if raw.shape != (6,):
            raise ValueError("code must have exactly 6 uint8 fields")

    candidates = []
    for shift in [0, 2, 4]:
        candidates.append(np.concatenate([raw[shift:], raw[:shift]]))
    mirror = np.concatenate([raw[[0]], raw[[5]], raw[[4]], raw[[3]], raw[[2]], raw[[1]]])
    candidates.append(mirror)
    for shift in [2, 4]:
        candidates.append(np.concatenate([mirror[shift:], mirror[:shift]]))

    return min(candidates, key=lambda x: tuple(x))


def canonicalize_faces(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != 6:
        raise ValueError("codes must be of shape (n,6)")
    n = len(codes)
    result = np.empty((n, 6), dtype=np.uint8)
    for i in range(n):
        result[i] = canonicalize_single_code(codes[i])
    return result


def truncate_codes(codes, valence_threshold=5):
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != 6:
        raise ValueError("codes must be of shape (n,6)")
    out = codes.copy()
    v_idx = [0, 2, 4]
    clipped = np.minimum(out[:, v_idx], valence_threshold).astype(np.uint8)
    out[:, v_idx] = clipped
    return out


def code_to_bytes(code):
    if isinstance(code, FaceTopologyCode6):
        return bytes(code)

    if isinstance(code, (bytes, bytearray)):
        if len(code) != 6:
            raise ValueError("code must have exactly 6 uint8 fields")
        return bytes(code)

    arr = np.asarray(code, dtype=np.uint8)
    if arr.shape != (6,):
        raise ValueError("code must have exactly 6 uint8 fields")
    return arr.tobytes()


def bytes_to_code(b):
    if not isinstance(b, (bytes, bytearray)):
        raise TypeError("input must be bytes-like")
    if len(b) != 6:
        raise ValueError("bytes must have length 6")
    return np.frombuffer(b, dtype=np.uint8).copy()


def code_to_hex(code):
    return code_to_bytes(code).hex()


def hex_to_code(s):
    if s.lower().startswith("0x"):
        s = s[2:]
    if len(s) != 12:
        raise ValueError("hex string must represent exactly 6 bytes (12 hex digits)")
    return bytes_to_code(bytes.fromhex(s))


def save_codes(codes, path):
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != 6:
        raise ValueError("codes must be of shape (n,6)")
    np.save(path, codes)


def load_codes(path):
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("loaded array must be of shape (n,6)")
    return arr


def validate_code(code):
    try:
        arr = np.asarray(code, dtype=np.uint8)
    except (TypeError, ValueError):
        return False
    if arr.shape != (6,):
        return False
    return True


def compute_face_edge_valences(mesh, edge_to_faces, face_edge_keys):
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


def build_manifold_face_adjacency(mesh):
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
    face_set = set(map(int, face_set))
    if not face_set:
        return False

    face_edge_keys = compute_face_edge_keys(mesh)
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

    manifold_boundary_vertices = {}
    manifold_boundary_edges = []
    for fid in face_set:
        for j in range(3):
            key = int(face_edge_keys[fid, j])
            shared = edge_to_faces.get(key, [])
            inner_count = sum(1 for f in shared if int(f) in face_set)
            if inner_count == 1 and len(shared) == 2:
                manifold_boundary_edges.append(key)
                v0, v1 = edge_key_to_vertex_pair[key]
                manifold_boundary_vertices.setdefault(v0, []).append(v1)
                manifold_boundary_vertices.setdefault(v1, []).append(v0)

    if not manifold_boundary_edges:
        return False

    for v, nbrs in manifold_boundary_vertices.items():
        if len(nbrs) != 2:
            return False

    return len(manifold_boundary_vertices) == len(manifold_boundary_edges)


def _extract_boundary_loops_from_edge_keys(mesh, edge_keys):
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


def _expand_face_neighborhood_geometrics(mesh, seed_faces, depth):
    if depth <= 0:
        return set()
    seed_faces = set(map(int, seed_faces))
    if depth == 1:
        return seed_faces.copy()

    n_faces = len(mesh.faces)
    adjacency = [[] for _ in range(n_faces)]
    for f0, f1 in mesh.face_adjacency:
        adjacency[int(f0)].append(int(f1))
        adjacency[int(f1)].append(int(f0))

    current = list(seed_faces)
    visited = set(seed_faces)

    for _ in range(depth - 1):
        next_layer = []
        for f in current:
            for nb in adjacency[f]:
                if nb not in visited:
                    visited.add(nb)
                    next_layer.append(nb)
        current = next_layer
        if not current:
            break
    return visited


def expand_face_neighborhood(mesh, seed_faces, depth):
    """
    公开的邻域扩展接口，内部调用已有私有实现。
    """
    return _expand_face_neighborhood_geometrics(mesh, seed_faces, depth)


def compute_face_distances(mesh, source_mask):
    """
    计算每个面片到源面片集的最短拓扑距离。
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

    dist[dist == -1] = np.iinfo(np.int32).max
    return dist


def reconstruct_loop_from_edges(edge_vertex_pairs):
    """
    从无序边集恢复闭合顶点环。
    """
    if not edge_vertex_pairs:
        return []

    adj = {}
    for a, b in edge_vertex_pairs:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    start = next(iter(adj))
    loop = [start]
    prev = None
    cur = start

    while True:
        nxts = [v for v in adj[cur] if v != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt

        if len(loop) > len(adj):
            break

    return loop
