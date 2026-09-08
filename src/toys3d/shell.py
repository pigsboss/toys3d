import sys
import os
import time

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

import numpy as np
import trimesh

from toys3d.geometrics import (
    compute_mesh_stats,
    analyze_mesh_defects,
    repair_mesh_by_removing_duplicates,
    repair_nonmanifold_edges,
    fill_small_holes,
)

# ------------------------------------------------------------------
#  Shell self-contained algorithms
#  These functions were moved from geometrics_older.py so that shell.py
#  no longer has any dependency on that legacy module.
# ------------------------------------------------------------------

def build_face_adjacency(mesh):
    """
    返回面片邻接表，adj[fi] 为与 fi 共享一条边的面片索引列表。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    N = len(faces)
    adjacency = [[] for _ in range(N)]
    edge_map = {}
    for fi, (v1, v2, v3) in enumerate(faces):
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            if key in edge_map:
                fj = edge_map[key]
                adjacency[fi].append(fj)
                adjacency[fj].append(fi)
            else:
                edge_map[key] = fi
    return adjacency


def get_k_ring_neighbors(adjacency, seed, k=1):
    """
    获取面片的 k-ring 邻域（包括 seed 自身）。
    """
    visited = {seed}
    frontier = {seed}
    for _ in range(k):
        new_frontier = set()
        for f in frontier:
            new_frontier.update(adjacency[f])
        frontier = new_frontier - visited
        visited.update(frontier)
        if not frontier:
            break
    return np.array(list(visited), dtype=int)


def compute_multiscale_face_normals(mesh, scales=(1, 2, 4, 8)):
    """
    在不同邻域尺度下计算面片法向（面积加权平均）。
    """
    adjacency = build_face_adjacency(mesh)
    N = len(mesh.faces)
    base_normals = mesh.face_normals.copy()
    areas = mesh.area_faces

    scale_normals = []
    for k in scales:
        smoothed = np.zeros_like(base_normals)
        for i in range(N):
            neighbors = get_k_ring_neighbors(adjacency, i, k=k)
            w = areas[neighbors]
            w_sum = np.sum(w)
            if w_sum < 1e-12:
                avg = np.mean(base_normals[neighbors], axis=0)
            else:
                avg = np.average(base_normals[neighbors], axis=0, weights=w)
            norm = np.linalg.norm(avg)
            smoothed[i] = avg / norm if norm > 1e-12 else base_normals[i]
        scale_normals.append(smoothed)

    return scale_normals


def compute_edge_strength_multiscale(mesh, scale_normals):
    """
    对每个尺度，计算每个面片与其 1-ring 邻域的法向平均差异（弧度）。
    """
    adjacency = build_face_adjacency(mesh)
    N = len(mesh.faces)
    n_scales = len(scale_normals)

    strengths = np.zeros((n_scales, N), dtype=np.float64)
    for s, normals in enumerate(scale_normals):
        for i in range(N):
            neighbors = adjacency[i]
            if len(neighbors) == 0:
                continue
            dots = np.clip(np.dot(normals[neighbors], normals[i]), -1.0, 1.0)
            angles = np.arccos(dots)
            strengths[s, i] = float(np.mean(angles))

    return strengths


def detect_multiscale_edges(mesh, scales=(1, 2, 4, 8),
                            threshold_ratio=0.3,
                            min_consistent_scales=None):
    """
    多尺度边缘检测。
    """
    scale_normals = compute_multiscale_face_normals(mesh, scales=scales)
    strengths = compute_edge_strength_multiscale(mesh, scale_normals)

    n_scales, N = strengths.shape
    if min_consistent_scales is None:
        min_consistent_scales = max(2, (n_scales + 1) // 2)

    max_val = np.max(strengths, axis=1, keepdims=True) + 1e-12
    normalized = strengths / max_val

    consistent = np.sum(normalized > threshold_ratio, axis=0) >= min_consistent_scales

    if n_scales >= 2:
        monotonic = np.all(np.diff(strengths, axis=0) <= 0, axis=0)
    else:
        monotonic = np.ones(N, dtype=bool)

    edge_face_mask = consistent & monotonic

    return edge_face_mask, strengths


def segment_regions_by_edges(mesh, edge_face_mask):
    """
    移除边缘面片后，对剩余面片做连通分量分割。
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    N = len(mesh.faces)
    adjacency = build_face_adjacency(mesh)

    valid_mask = ~np.asarray(edge_face_mask, dtype=bool)
    valid_indices = np.flatnonzero(valid_mask)
    n_valid = len(valid_indices)

    remap = np.full(N, -1, dtype=np.int64)
    remap[valid_indices] = np.arange(n_valid)

    rows, cols = [], []
    for i in valid_indices:
        for j in adjacency[i]:
            if j > i and valid_mask[j]:
                ii, jj = remap[i], remap[j]
                rows.extend([ii, jj])
                cols.extend([jj, ii])

    labels = np.full(N, -1, dtype=int)
    if n_valid == 0:
        return labels

    if len(rows) > 0:
        data = np.ones(len(rows), dtype=np.int8)
        graph = csr_matrix((data, (rows, cols)), shape=(n_valid, n_valid))
        _, comps = connected_components(graph, directed=False)
        labels[valid_indices] = comps
    else:
        labels[valid_indices] = np.arange(n_valid)

    return labels


def estimate_shell_thickness(mesh, grid_size=128, margin=1.05, k_neighbors=20):
    """
    基于 k-d 树最近邻搜索估计每个面片的局部厚度。
    """
    from scipy.spatial import cKDTree

    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    N = len(centers)

    thickness = np.full(N, np.nan, dtype=np.float64)
    reliability = np.zeros(N, dtype=bool)

    if N == 0:
        return thickness, reliability

    k = min(k_neighbors, N - 1) if N > 1 else 1

    tree = cKDTree(centers)
    dists, inds = tree.query(centers, k=k)

    for i in range(N):
        for j in range(k):
            idx = inds[i, j]
            if idx == i:
                continue
            d = float(dists[i, j])
            if d < 1e-12:
                continue
            dot = float(np.dot(normals[i], normals[idx]))
            if dot < -0.3:  # 法向反平行
                thickness[i] = d
                reliability[i] = True
                break

    return thickness, reliability


def detect_thin_regions(thickness, mode='adaptive', threshold=0.1,
                        fallback_median=None):
    """
    标记局部厚度过小的面片。
    """
    thickness = np.asarray(thickness, dtype=np.float64)
    valid = np.isfinite(thickness)
    if not np.any(valid):
        return np.zeros(len(thickness), dtype=bool)

    if mode == 'adaptive':
        med = fallback_median if fallback_median is not None else np.median(thickness[valid])
        abs_thr = threshold * med
    elif mode == 'absolute':
        abs_thr = threshold
    else:
        raise ValueError("mode must be 'adaptive' or 'absolute'")

    return valid & (thickness < abs_thr)


def compute_wall_thickness_statistics(thickness, reliability=None):
    """
    计算厚度场的统计信息，用于自适应阈值与诊断输出。
    """
    thickness = np.asarray(thickness, dtype=np.float64)
    if reliability is None:
        reliability = np.isfinite(thickness)

    stats = {
        'reliable_count': int(np.sum(reliability)),
        'reliable_ratio': float(np.sum(reliability) / max(len(thickness), 1)),
    }

    if not np.any(reliability):
        for key in ['median', 'mean', 'std', 'min', 'max',
                    'p25', 'p75', 'iqr']:
            stats[key] = np.nan
        return stats

    vals = thickness[reliability]
    p25, p75 = np.percentile(vals, [25, 75])

    stats['median'] = float(np.median(vals))
    stats['mean'] = float(np.mean(vals))
    stats['std'] = float(np.std(vals))
    stats['min'] = float(np.min(vals))
    stats['max'] = float(np.max(vals))
    stats['p25'] = float(p25)
    stats['p75'] = float(p75)
    stats['iqr'] = float(p75 - p25)
    return stats


def extract_plate_boundary_loops(mesh, plate_mask):
    """
    提取指定薄板面片集合的所有边界环。
    """
    faces = np.asarray(mesh.faces).reshape(-1, 3)
    mask = np.asarray(plate_mask, dtype=bool)
    plate_indices = np.flatnonzero(mask)

    if len(plate_indices) == 0:
        return []

    edge_count = {}
    for fi in plate_indices:
        v = faces[fi]
        for j in range(3):
            a, b = int(v[j]), int(v[(j + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        return []

    adjacency = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    visited = set()
    loops = []
    for start in adjacency:
        if start in visited:
            continue

        loop = [start]
        visited.add(start)
        prev, curr = None, start

        while True:
            neighbors = [v for v in adjacency.get(curr, []) if v != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            if nxt == start and len(loop) > 2:
                break
            if nxt in visited:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt

        if len(loop) >= 3:
            loops.append(loop)

    return loops


def fit_line_3d(points):
    """
    三维点最小二乘直线拟合。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return np.inf, None, pts.mean(axis=0) if n > 0 else None

    center = pts.mean(axis=0)
    centered = pts - center
    if n == 2:
        direction = centered[1] - centered[0]
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return np.inf, None, center
        direction = direction / norm
        return 0.0, direction, center

    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    if s[0] < 1e-12:
        return np.inf, None, center

    direction = vh[0]
    projections = centered @ direction
    residuals = centered - projections[:, None] * direction
    rmse = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))
    return rmse, direction, center


def fit_circle_3d(points):
    """
    三维点最小二乘圆拟合。先投影到最佳拟合平面，再在平面内做圆拟合。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return np.inf, None, None, 0.0

    center = pts.mean(axis=0)
    centered = pts - center
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    if s[1] < 1e-12:
        return np.inf, None, None, 0.0

    normal = vh[2]
    basis_u = vh[0]
    basis_v = vh[1]

    coords = np.column_stack([centered @ basis_u, centered @ basis_v])

    A = np.column_stack([coords[:, 0], coords[:, 1], np.ones(n)])
    b = coords[:, 0] ** 2 + coords[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    radius = np.sqrt(cx * cx + cy * cy + sol[2])
    center3d = center + cx * basis_u + cy * basis_v

    radii = np.linalg.norm(coords - np.array([cx, cy]), axis=1)
    rmse = np.sqrt(np.mean((radii - radius) ** 2))
    return rmse, center3d, normal, float(radius)


def fit_spline_3d(points, degree=3, num_samples=100):
    """
    三维 B 样条拟合。若 scipy 不可用或点数不足，返回 inf。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < degree + 1:
        return np.inf, None

    try:
        from scipy.interpolate import splprep, splev
    except ImportError:
        return np.inf, None

    try:
        tck, _ = splprep(pts.T, k=degree, s=n * 0.01)
        u_fine = np.linspace(0, 1, num_samples)
        fitted = np.array(splev(u_fine, tck)).T

        dists = np.sqrt(np.min(
            np.sum((pts[:, None, :] - fitted[None, :, :]) ** 2, axis=2),
            axis=1
        ))
        rmse = float(np.sqrt(np.mean(dists ** 2)))
        return rmse, fitted
    except Exception:
        return np.inf, None


def classify_edge_regularity(loop_pts, scale=1.0, line_tol=0.1,
                             circle_tol=0.1, spline_tol=0.1):
    """
    判断三维边界环属于直线、圆弧、样条还是不规则。
    """
    pts = np.asarray(loop_pts, dtype=np.float64)
    if len(pts) < 3:
        return 'irregular', np.inf

    line_err, _, _ = fit_line_3d(pts)
    circle_err, _, _, _ = fit_circle_3d(pts)
    spline_err, _ = fit_spline_3d(pts)

    line_score = line_err / scale
    circle_score = circle_err / scale
    spline_score = spline_err / scale

    if line_score < line_tol:
        return 'line', line_score
    if circle_score < circle_tol:
        return 'circle', circle_score
    if spline_score < spline_tol and np.isfinite(spline_score):
        return 'spline', spline_score
    return 'irregular', min(line_score, circle_score, spline_score)


def build_proxy_mesh(mesh, target_faces=50000, max_edge_length=None,
                     iterations=2, smooth=False):
    """
    从原始扫描网格构建均匀、低分辨率、近似水密的代理网格。
    """
    proxy = mesh.copy()

    defect_stats, _, _ = analyze_mesh_defects(proxy)
    if defect_stats['open_edges'] > 0 or defect_stats['nonmanifold_edges'] > 0:
        for _ in range(3):
            proxy = repair_mesh_by_removing_duplicates(proxy)
            proxy = repair_nonmanifold_edges(proxy, verbose=False)
            proxy = fill_small_holes(proxy, max_loop_edges=50, verbose=False)
            defect_stats, _, _ = analyze_mesh_defects(proxy)
            if defect_stats['open_edges'] == 0 and defect_stats['nonmanifold_edges'] == 0:
                break

    if len(proxy.faces) > target_faces:
        proxy = proxy.simplify_quadric_decimation(face_count=target_faces)

    if max_edge_length is None:
        diag = float(np.linalg.norm(proxy.bounding_box.extents))
        max_edge_length = diag * 0.02
    if max_edge_length > 0 and len(proxy.edges_unique) > 0:
        mean_edge = float(np.mean(proxy.edges_unique_length))
        if mean_edge > max_edge_length * 1.5:
            est_factor = (mean_edge / max_edge_length) ** 2
            if len(proxy.faces) * est_factor < max(target_faces * 4, 200000):
                proxy = proxy.subdivide_to_size(max_edge_length)

    proxy.merge_vertices()
    faces = np.asarray(proxy.faces, dtype=np.int64)
    unique_faces = np.unique(faces, axis=0)
    if unique_faces.shape[0] < faces.shape[0]:
        proxy = trimesh.Trimesh(vertices=proxy.vertices,
                                faces=unique_faces,
                                process=False)
    nd_mask = proxy.nondegenerate_faces
    if not np.all(nd_mask):
        proxy = trimesh.Trimesh(vertices=proxy.vertices,
                                faces=proxy.faces[nd_mask],
                                process=False)
    proxy.remove_unreferenced_vertices()

    if smooth and hasattr(proxy, 'smoothed'):
        proxy = proxy.smoothed(iterations=1)

    proxy.fix_normals()
    return proxy


def map_labels_from_proxy(original_mesh, proxy_mesh, proxy_labels):
    """
    将代理网格上的薄板标签映射回原始网格。
    """
    from scipy.spatial import cKDTree

    proxy_centers = np.asarray(proxy_mesh.triangles_center, dtype=np.float64)
    original_centers = np.asarray(original_mesh.triangles_center, dtype=np.float64)

    if len(proxy_centers) == 0 or len(original_centers) == 0:
        return np.zeros(len(original_centers), dtype=int)

    tree = cKDTree(proxy_centers)
    _, indices = tree.query(original_centers, k=1)
    return np.asarray(proxy_labels, dtype=int)[indices]


class _NormalCluster:
    """用于无符号法向聚类的简单辅助类，缓存平均方向。"""
    def __init__(self, normal):
        self.normals = [normal]
        self._mean_dirty = True
        self._mean_cache = normal.copy()

    def add(self, normal):
        self.normals.append(normal)
        self._mean_dirty = True

    @property
    def mean_axis(self):
        if self._mean_dirty:
            avg = np.mean(self.normals, axis=0)
            n = np.linalg.norm(avg)
            self._mean_cache = avg / n if n > 1e-12 else self.normals[0]
            self._mean_dirty = False
        return self._mean_cache


def segment_plates_by_local_clustering(mesh, radius=None,
                                       cluster_angle_deg=30.0,
                                       min_faces=30):
    """
    基于球状欧氏邻域的局部法向聚类薄板分割。
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    import time

    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    N = len(centers)

    if N == 0:
        return np.zeros(0, dtype=int)

    if radius is None:
        print("  Estimating thickness for default cluster radius...")
        t0 = time.time()
        thickness, _ = estimate_shell_thickness(mesh, k_neighbors=10)
        med = float(np.nanmedian(thickness))
        if not np.isfinite(med) or med < 1e-6:
            med = float(np.mean(mesh.edges_unique_length))
        radius = max(1.5 * med, 1e-3)
        print(f"    thickness median={med:.4f}, radius={radius:.4f} "
              f"({time.time() - t0:.2f}s)")

    print("  Building k-d tree for face centers...")
    t0 = time.time()
    tree = cKDTree(centers)
    neighbors_list = tree.query_ball_point(centers, radius)

    cos_thr = np.cos(np.deg2rad(cluster_angle_deg))
    boundary_mask = np.zeros(N, dtype=bool)

    print("  Clustering normals per face...")
    t0 = time.time()
    for fi in range(N):
        neighbors = neighbors_list[fi]
        if len(neighbors) < 3:
            continue

        clusters = []
        for n in normals[neighbors]:
            n_unit = n / (np.linalg.norm(n) + 1e-12)
            added = False
            for cl in clusters:
                if abs(np.dot(n_unit, cl.mean_axis)) >= cos_thr:
                    cl.add(n_unit)
                    added = True
                    break
            if not added:
                clusters.append(_NormalCluster(n_unit))

        if len(clusters) >= 2:
            boundary_mask[fi] = True

    print(f"    clustering done {time.time() - t0:.2f}s, "
          f"boundary={np.sum(boundary_mask)}")

    valid_mask = ~boundary_mask
    valid_indices = np.flatnonzero(valid_mask)
    n_valid = len(valid_indices)

    labels = np.full(N, -1, dtype=int)
    if n_valid > 0:
        adjacency = build_face_adjacency(mesh)
        remap = np.full(N, -1, dtype=np.int64)
        remap[valid_indices] = np.arange(n_valid)

        rows, cols = [], []
        for i in valid_indices:
            for j in adjacency[i]:
                if j > i and valid_mask[j]:
                    ii, jj = remap[i], remap[j]
                    rows.extend([ii, jj])
                    cols.extend([jj, ii])

        if len(rows) > 0:
            data = np.ones(len(rows), dtype=np.int8)
            graph = csr_matrix((data, (rows, cols)), shape=(n_valid, n_valid))
            _, comps = connected_components(graph, directed=False)
            labels[valid_indices] = comps
        else:
            labels[valid_indices] = np.arange(n_valid)

    # Reassign boundary faces to neighbours
    adjacency = build_face_adjacency(mesh)
    for fi in range(N):
        if labels[fi] != -1:
            continue
        best_sim = -np.inf
        best_label = -1
        for fj in adjacency[fi]:
            lbl = labels[fj]
            if lbl < 0:
                continue
            sim = abs(np.dot(normals[fi], normals[fj]))
            if sim > best_sim:
                best_sim = sim
                best_label = lbl
        if best_label != -1:
            labels[fi] = best_label

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

    valid = labels >= 0
    if np.any(valid):
        _, new_labels = np.unique(labels[valid], return_inverse=True)
        labels[valid] = new_labels

    return labels


def ransac_plane_fitting(points, max_iter=500, inlier_threshold=0.1,
                         rng=None):
    """
    Fit a single plane to 3D points using RANSAC.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return None, np.zeros(pts.shape[0], dtype=bool)

    if rng is None:
        rng = np.random.default_rng()

    best_inliers = None
    best_n = None
    best_p = None
    best_score = -1

    n = pts.shape[0]
    for _ in range(max_iter):
        idxs = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idxs]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            continue
        normal = normal / norm_len
        dists = np.abs(np.dot(pts - p0, normal))
        inliers = dists <= inlier_threshold
        score = int(np.sum(inliers))
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_n = normal
            best_p = p0

    if best_inliers is None:
        return None, np.zeros(n, dtype=bool)

    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    cov = (inlier_pts - centroid).T @ (inlier_pts - centroid)
    eigvals, eigvecs = np.linalg.eigh(cov)
    refined_normal = eigvecs[:, np.argmin(eigvals)]
    if np.dot(refined_normal, best_n) < 0:
        refined_normal = -refined_normal
    return (refined_normal, centroid), best_inliers


def multi_ransac_planes(points, max_planes=3, inlier_threshold=0.1,
                        min_points_per_plane=5, max_iter=500, rng=None):
    """
    Sequentially extract up to max_planes dominant planes from points.
    """
    pts = np.asarray(points, dtype=np.float64)
    remaining = np.ones(len(pts), dtype=bool)
    planes = []

    for _ in range(max_planes):
        if np.sum(remaining) < min_points_per_plane:
            break
        sub_pts = pts[remaining]
        plane_params, inliers_sub = ransac_plane_fitting(
            sub_pts, max_iter=max_iter,
            inlier_threshold=inlier_threshold, rng=rng
        )
        if plane_params is None:
            break
        global_inliers = np.zeros(len(pts), dtype=bool)
        global_inliers[remaining] = inliers_sub
        if np.sum(global_inliers) < min_points_per_plane:
            break
        planes.append((plane_params[0], plane_params[1], global_inliers))
        remaining &= ~global_inliers

    return planes


def _spatial_split_inliers(centers, inlier_mask, adjacency, ball_face_set):
    """Split inlier faces into spatial connected components using face adjacency list."""
    idx = np.where(inlier_mask)[0]
    if len(idx) <= 1:
        return [inlier_mask]

    local_map = {global_id: i for i, global_id in enumerate(idx)}
    n_in = len(idx)

    rows, cols = [], []
    for i, global_id in enumerate(idx):
        neighbors = adjacency[global_id] if global_id < len(adjacency) else []
        for neighbor in neighbors:
            if neighbor not in ball_face_set:
                continue
            if neighbor in local_map:
                j = local_map[neighbor]
                rows.append(i); cols.append(j)
                rows.append(j); cols.append(i)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if len(rows) == 0:
        comps = np.arange(n_in)
    else:
        graph = csr_matrix((np.ones(len(rows), dtype=np.int8),
                            (rows, cols)), shape=(n_in, n_in))
        _, comps = connected_components(graph, directed=False)

    unique_comps = np.unique(comps)
    components = []
    for comp in unique_comps:
        comp_mask = np.zeros(len(centers), dtype=bool)
        comp_mask[idx[comps == comp]] = True
        components.append(comp_mask)
    return components


def detect_boundary_ball(centers, ball_face_indices, mesh_adjacency,
                         max_planes=3,
                         inlier_threshold=0.1, max_iter=300, rng=None):
    """
    Returns True if there are >=2 spatially separated plane patches.
    """
    if len(ball_face_indices) < 6:
        return False

    ball_pts = centers[ball_face_indices]
    planes = multi_ransac_planes(
        ball_pts, max_planes=max_planes,
        inlier_threshold=inlier_threshold,
        min_points_per_plane=3, max_iter=max_iter, rng=rng
    )

    total_components = 0
    for plane_params in planes:
        normal, point, inlier_mask = plane_params
        sub_mask = np.zeros(len(centers), dtype=bool)
        sub_mask[ball_face_indices] = inlier_mask
        pieces = _spatial_split_inliers(
            centers, sub_mask, mesh_adjacency, set(ball_face_indices)
        )
        total_components += len(pieces)
        if total_components >= 2:
            return True

    return False


def segment_plates_by_plane_fitting(mesh, radius=5.0,
                                     inlier_threshold=0.1,
                                     max_planes=3, min_faces=30,
                                     rng=None):
    """
    Segment mesh into plates using plane fitting in local ball neighborhoods.
    """
    from scipy.spatial import cKDTree
    import time

    centers = mesh.triangles_center
    N = len(centers)
    if N == 0:
        return np.zeros(0, dtype=int)

    adjacency = build_face_adjacency(mesh)

    tree = cKDTree(centers)
    neighbors_list = tree.query_ball_point(centers, radius)

    boundary_mask = np.zeros(N, dtype=bool)
    if rng is None:
        rng = np.random.default_rng()

    report_interval = max(1, N // 10)
    for fi in range(N):
        if fi % report_interval == 0:
            print(f"      process {fi}/{N} ({100 * fi / N:.1f}%)")
        ball_indices = neighbors_list[fi]
        if len(ball_indices) < 6:
            continue
        boundary_mask[fi] = detect_boundary_ball(
            centers, ball_indices, adjacency,
            max_planes=max_planes,
            inlier_threshold=inlier_threshold, max_iter=300, rng=rng
        )

    valid_mask = ~boundary_mask
    valid_indices = np.flatnonzero(valid_mask)
    n_valid = len(valid_indices)
    labels = np.full(N, -1, dtype=int)

    if n_valid > 0:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        remap = np.full(N, -1, dtype=np.int64)
        remap[valid_indices] = np.arange(n_valid)

        rows, cols = [], []
        for i in valid_indices:
            for j in adjacency[i]:
                if j > i and valid_mask[j]:
                    ii, jj = remap[i], remap[j]
                    rows.extend([ii, jj])
                    cols.extend([jj, ii])

        if len(rows) > 0:
            graph = csr_matrix((np.ones(len(rows), dtype=np.int8),
                                (rows, cols)), shape=(n_valid, n_valid))
            _, comps = connected_components(graph, directed=False)
            labels[valid_indices] = comps
        else:
            labels[valid_indices] = np.arange(n_valid)

    adjacency = build_face_adjacency(mesh)
    normals = mesh.face_normals
    for fi in range(N):
        if labels[fi] != -1:
            continue
        best_dot = -1.0
        best_label = -1
        for nj in adjacency[fi]:
            lbl = labels[nj]
            if lbl < 0:
                continue
            dot = np.dot(normals[fi], normals[nj])
            if dot > best_dot:
                best_dot = dot
                best_label = lbl
        if best_label != -1:
            labels[fi] = best_label

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    for lbl in unique[counts < min_faces]:
        mask = labels == lbl
        neighbor_labels = []
        for fi in np.flatnonzero(mask):
            for nj in adjacency[fi]:
                nl = labels[nj]
                if nl >= 0 and nl != lbl:
                    neighbor_labels.append(nl)
        if neighbor_labels:
            new_lbl = max(set(neighbor_labels), key=neighbor_labels.count)
            labels[mask] = new_lbl

    valid = labels >= 0
    if np.any(valid):
        _, new_labels = np.unique(labels[valid], return_inverse=True)
        labels[valid] = new_labels

    return labels


def segment_plates_by_smoothness(mesh, angle_threshold_deg=30.0, min_faces=10):
    """
    基于相邻面片二面角进行区域增长，分割出光滑薄板区域。
    使用向量化图连通分量算法，避免 Python DFS 开销。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    angle_threshold_deg : float   二面角阈值（度）
    min_faces : int               最小面片数

    Returns
    -------
    labels : (N,) ndarray, int   面片区域标签（-1 为被合并/舍弃）
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    N = len(faces)
    angle_thr = np.deg2rad(angle_threshold_deg)
    cos_thr = np.cos(angle_thr)

    # 构建相邻面片对（仅保留恰好被 2 个面片共享的边）
    edge_map = {}
    for fi, (v1, v2, v3) in enumerate(faces):
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_map.setdefault(key, []).append(fi)

    pairs = []
    for fl in edge_map.values():
        if len(fl) == 2:
            pairs.append((fl[0], fl[1]))

    pair_count = len(pairs)

    if pair_count == 0:
        labels = np.arange(N, dtype=int)
    else:
        pairs = np.array(pairs, dtype=np.int64)
        i = pairs[:, 0]
        j = pairs[:, 1]
        normals = mesh.face_normals
        dots = normals[i, 0] * normals[j, 0] + \
               normals[i, 1] * normals[j, 1] + \
               normals[i, 2] * normals[j, 2]

        # 保留二面角小于阈值的边（点积 > cos(theta)）
        mask = dots >= cos_thr

        n_keep = int(np.sum(mask))
        if n_keep == 0:
            labels = np.arange(N, dtype=int)
        else:
            rows = np.empty(2 * n_keep, dtype=np.int64)
            cols = np.empty(2 * n_keep, dtype=np.int64)
            rows[:n_keep] = i[mask]
            cols[:n_keep] = j[mask]
            rows[n_keep:] = j[mask]
            cols[n_keep:] = i[mask]
            data = np.ones(2 * n_keep, dtype=np.int8)
            graph = csr_matrix((data, (rows, cols)), shape=(N, N))
            _, labels = connected_components(graph, directed=False)

    # 移除过小的区域
    unique, counts = np.unique(labels, return_counts=True)
    small_mask = counts < min_faces
    if np.any(small_mask):
        small_labels = unique[small_mask]
        for lbl in small_labels:
            labels[labels == lbl] = -1

    # 重新编号（紧凑的从 0 开始）
    valid = labels >= 0
    if np.any(valid):
        _, new_labels = np.unique(labels[valid], return_inverse=True)
        labels[valid] = new_labels

    return labels


# ------------------------------------------------------------------
#  Timer class
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
    """
    from collections import defaultdict

    unique_labels = np.unique(labels)
    valid_labels = [l for l in unique_labels if l >= 0]
    label_neighbors = defaultdict(set)
    for lbl in valid_labels:
        label_neighbors[lbl]

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
    """
    scene = trimesh.Scene()
    vis = mesh.copy()
    N = len(vis.faces)

    adjacency = build_face_adjacency(mesh)
    color_map, palette = _assign_plate_colors(labels, adjacency)

    colors = np.full((N, 4), 180, dtype=np.uint8)
    colors[:, 3] = alpha

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

    colors[labels < 0] = [180, 180, 180, alpha]

    edge_face_map = {}
    for fi, (v1, v2, v3) in enumerate(mesh.faces):
        for a, b in [(int(v1), int(v2)), (int(v2), int(v3)), (int(v3), int(v1))]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(fi)

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
                    colors[fi] = [10, 10, 10, 255]

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


def visualize_boundaries(mesh, labels, scale, line_tol, circle_tol, spline_tol):
    """
    按边界环规律性着色：直线=绿，圆弧=蓝，样条=黄，不规则=红。
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
    pts = mesh.vertices[np.asarray(loop, dtype=int)]
    n = len(pts)

    perim = 0.0
    for i in range(n):
        perim += np.linalg.norm(pts[(i + 1) % n] - pts[i])

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

    valid = labels >= 0
    if np.any(valid):
        _, new_labels = np.unique(labels[valid], return_inverse=True)
        labels[valid] = new_labels

    return labels


# ------------------------------------------------------------------
#  主处理流程
# ------------------------------------------------------------------

def process_shell(mesh, num_passes=0, repair_mode=False,
                  angle_threshold_deg=30.0, min_faces=500,
                  thin_mode='adaptive', thin_threshold=0.1,
                  line_tol=0.05, circle_tol=0.05, spline_tol=0.1,
                  irregular_perimeter_ratio=0.05,
                  thickness_grid_size=128,
                  seg_mode='cluster', edge_scales=(1, 2, 4, 8),
                  edge_threshold_ratio=0.3,
                  vis_mode='plates', vis_alpha=40, plate_alpha=200,
                  proxy_faces=50000, proxy_max_edge=None,
                  cluster_depth=2, cluster_angle_deg=45.0,
                  cluster_radius=None,
                  ransac_inlier_thr=0.1, max_planes=3):
    """薄壳处理主函数（完整实现）。"""
    # 0. 基础统计
    with Timer("Initial stats/defect analysis"):
        stats = compute_mesh_stats(mesh)
        print("Hey, mesh stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
        print("\nMesh defect analysis:")
        print(f"  open edges: {defect_stats['open_edges']}")
        print(f"  nonmanifold edges: {defect_stats['nonmanifold_edges']}")
        print(f"  watertight (no open edges): {defect_stats['watertight_by_count']}")

    if repair_mode and (defect_stats['open_edges'] > 0 or defect_stats['nonmanifold_edges'] > 0):
        with Timer("Mesh repair"):
            print("\n[Repair mode] Attempting to fix mesh...")
            mesh = _repair_loop(mesh)
            stats = compute_mesh_stats(mesh)
            print("\nAfter repair:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)

    # 先按需构造代理网格
    proxy_mesh = None
    if seg_mode in ('cluster', 'planar'):
        with Timer("Build proxy mesh"):
            print("[Proxy] Building proxy mesh...")
            proxy_mesh = build_proxy_mesh(
                mesh,
                target_faces=proxy_faces,
                max_edge_length=proxy_max_edge,
                iterations=2,
                smooth=False
            )
            print(f"  Proxy faces: {proxy_mesh.faces.shape[0]}")

    # 分析并分割
    with Timer("Thickness estimation"):
        thickness, reliability = estimate_shell_thickness(mesh, grid_size=thickness_grid_size)
        th_stats = compute_wall_thickness_statistics(thickness, reliability)
    print(f"\n[Thickness] median={th_stats['median']:.4f}, reliable={th_stats['reliable_ratio']:.2%}")

    with Timer("Plate segmentation"):
        # 共享参数
        if seg_mode == 'cluster':
            labels = segment_plates_by_local_clustering(
                mesh,
                radius=cluster_radius,
                cluster_angle_deg=cluster_angle_deg,
                min_faces=min_faces
            )
        elif seg_mode == 'multiscale':
            labels = segment_by_multiscale_edges(
                mesh,
                scales=edge_scales,
                threshold_ratio=edge_threshold_ratio,
                min_faces=min_faces
            )
        elif seg_mode == 'planar':
            labels = segment_plates_by_plane_fitting(
                mesh,
                radius=cluster_radius if cluster_radius is not None else 1.5*max(1e-3, th_stats['median']),
                inlier_threshold=ransac_inlier_thr,
                max_planes=max_planes,
                min_faces=min_faces,
                rng=None
            )
        else:
            labels = segment_plates_by_smoothness(
                mesh,
                angle_threshold_deg=angle_threshold_deg,
                min_faces=min_faces
            )

    if num_passes == 0:
        with Timer("pass 0 visualization"):
            if vis_mode == 'thickness':
                world_scene = visualize_thickness(mesh, thickness, reliability)
                scene = visualize_plates(mesh, labels)
            elif vis_mode == 'centers':
                world_scene = visualize_plate_centers(mesh, labels, alpha=vis_alpha, plate_alpha=plate_alpha)
                scene = world_scene
            else:
                world_scene = visualize_thickness(mesh, thickness, reliability)
                scene = visualize_plates(mesh, labels)
        return scene, world_scene, mesh.copy(), {**stats, **th_stats}

    print("\n[Pass 1] Removing thin and unreliable regions...")
    thin_mask = detect_thin_regions(
        thickness, mode=thin_mode, threshold=thin_threshold,
        fallback_median=th_stats['median']
    )
    remove_mask = thin_mask | (~reliability)
    if np.all(remove_mask):
        raise ValueError("All faces would be removed.")
    if np.any(remove_mask):
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[~remove_mask],
            process=False
        )
        mesh.remove_unreferenced_vertices()
        mesh = fill_small_holes(mesh, max_loop_edges=100, verbose=False)

    # 重新分析
    with Timer("Second analysis"):
        thickness, reliability = estimate_shell_thickness(mesh, grid_size=thickness_grid_size)
        th_stats = compute_wall_thickness_statistics(thickness, reliability)
        if seg_mode == 'cluster':
            labels = segment_plates_by_local_clustering(mesh, radius=cluster_radius, cluster_angle_deg=cluster_angle_deg, min_faces=min_faces)
        elif seg_mode == 'multiscale':
            labels = segment_by_multiscale_edges(mesh, scales=edge_scales, threshold_ratio=edge_threshold_ratio, min_faces=min_faces)
        elif seg_mode == 'planar':
            labels = segment_plates_by_plane_fitting(mesh, radius=cluster_radius if cluster_radius is not None else 1.5*max(1e-3, th_stats['median']), inlier_threshold=ransac_inlier_thr, max_planes=max_planes, min_faces=min_faces)
        else:
            labels = segment_plates_by_smoothness(mesh, angle_threshold_deg=angle_threshold_deg, min_faces=min_faces)

    if num_passes == 1:
        world_scene = visualize_thickness(mesh, thickness, reliability)
        scene = visualize_plates(mesh, labels)
        return scene, world_scene, mesh.copy(), {**stats, **th_stats}

    # Pass 2: boundary refinement
    print("\n[Pass 2] Refining by boundary regularity...")
    scale = float(np.linalg.norm(mesh.bounding_box.extents))
    if scale < 1e-12:
        scale = 1.0

    scene, irregular_loops = visualize_boundaries(
        mesh, labels, scale=scale,
        line_tol=line_tol, circle_tol=circle_tol, spline_tol=spline_tol
    )
    print(f"  Found {len(irregular_loops)} irregular small boundary loops")

    if irregular_loops:
        remove_mask = _faces_adjacent_to_loops(mesh, irregular_loops)
        if np.any(remove_mask) and not np.all(remove_mask):
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces[~remove_mask],
                process=False
            )
            mesh.remove_unreferenced_vertices()
            mesh = fill_small_holes(mesh, max_loop_edges=100, verbose=False)

    # Final analysis
    with Timer("Third analysis"):
        thickness, reliability = estimate_shell_thickness(mesh, grid_size=thickness_grid_size)
        th_stats = compute_wall_thickness_statistics(thickness, reliability)
        if seg_mode == 'cluster':
            labels = segment_plates_by_local_clustering(mesh, radius=cluster_radius, cluster_angle_deg=cluster_angle_deg, min_faces=min_faces)
        elif seg_mode == 'multiscale':
            labels = segment_by_multiscale_edges(mesh, scales=edge_scales, threshold_ratio=edge_threshold_ratio, min_faces=min_faces)
        elif seg_mode == 'planar':
            labels = segment_plates_by_plane_fitting(mesh, radius=cluster_radius if cluster_radius is not None else 1.5*max(1e-3, th_stats['median']), inlier_threshold=ransac_inlier_thr, max_planes=max_planes, min_faces=min_faces)
        else:
            labels = segment_plates_by_smoothness(mesh, angle_threshold_deg=angle_threshold_deg, min_faces=min_faces)

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
    parser.add_argument("--min-faces", type=int, default=500,
                        help="薄板最小面片数（默认500）")
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
    parser.add_argument("--thickness-grid-size", type=int, default=128,
                        help="体素分辨率用于厚度估计（默认128）")
    parser.add_argument("--seg-mode", type=str, default='cluster',
                        choices=['smoothness', 'multiscale', 'cluster', 'planar'],
                        help="薄板分割模式")
    parser.add_argument("--edge-scales", type=int, nargs='+', default=(1, 2, 4, 8),
                        help="多尺度边缘检测的邻域尺度")
    parser.add_argument("--edge-threshold", type=float, default=0.3,
                        help="多尺度边缘检测相对阈值")
    parser.add_argument("--vis-mode", type=str, default='plates',
                        choices=['plates', 'centers', 'thickness'],
                        help="Pass 0 可视化模式")
    parser.add_argument("--vis-alpha", type=int, default=40)
    parser.add_argument("--plate-alpha", type=int, default=200)
    parser.add_argument("--proxy-faces", type=int, default=50000)
    parser.add_argument("--proxy-max-edge", type=float, default=None)
    parser.add_argument("--cluster-depth", type=int, default=2)
    parser.add_argument("--cluster-angle", type=float, default=45.0)
    parser.add_argument("--cluster-radius", type=float, default=None)
    parser.add_argument("--ransac-threshold", type=float, default=0.1)
    parser.add_argument("--max-planes", type=int, default=3)
    parser.add_argument("--show", action="store_true", help="显示可视化窗口")
    args = parser.parse_args()

    mesh = trimesh.load(args.input_file)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")
    print(f"Hey, loading model: {args.input_file}")

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
        ransac_inlier_thr=args.ransac_threshold,
        max_planes=args.max_planes,
    )

    if args.output:
        processed_mesh.export(args.output)
        print(f"Processed mesh saved to {args.output}")

    if args.show:
        try:
            os.environ['TRIMESH_DEFAULT_VIEWER'] = 'vedo'
            world_scene.show()
        except IndexError as e:
            print(f"\n[ERROR] macOS/pygllet display error: {e}")
        except Exception as e:
            print(f"\n[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
