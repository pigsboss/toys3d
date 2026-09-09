# src/toys3d/geometrics/discrete.py
"""
离散几何工具：统计、体素化、Marching Cubes。
"""
import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


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


def _repair_to_watertight_mesh(mesh, voxel_size=None):
    if mesh.is_watertight:
        return mesh
    if voxel_size is None:
        bounds = mesh.bounds
        diag = np.linalg.norm(bounds[1] - bounds[0])
        voxel_size = diag / 128

    vox = mesh.voxelized(voxel_size)
    try:
        vox = vox.fill()
    except Exception:
        pass

    repaired = vox.marching_cubes

    if hasattr(vox, 'transform') and vox.transform is not None:
        repaired.apply_transform(vox.transform)
    else:
        print("  [DEBUG] VoxelGrid 没有 transform 属性，尝试使用 origin/pitch 修正")
        if hasattr(vox, 'origin') and hasattr(vox, 'pitch'):
            repaired.vertices = repaired.vertices * vox.pitch + vox.origin

    return repaired


def compute_face_area_stats(mesh):
    """
    计算三角面片面积的统计量。
    """
    areas = mesh.area_faces
    stats = {}
    if len(areas) == 0:
        stats['count'] = 0
        for key in ['mean', 'min', 'max', 'p1', 'p5', 'p10',
                    'p25', 'p50', 'p75', 'p90', 'p95', 'p99']:
            stats[key] = 0.0
        return stats

    stats['count'] = int(len(areas))
    stats['mean'] = float(np.mean(areas))
    stats['min'] = float(np.min(areas))
    stats['max'] = float(np.max(areas))
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        stats[f'p{p}'] = float(np.percentile(areas, p))
    return stats


def compute_bounding_box_stats(mesh):
    """
    计算包围盒相关统计。
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return {
            'min': np.zeros(3),
            'max': np.zeros(3),
            'extents': np.zeros(3),
            'diagonal': 0.0,
            'centroid': np.zeros(3),
        }

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extents = vmax - vmin

    return {
        'min': vmin,
        'max': vmax,
        'extents': extents,
        'diagonal': float(np.linalg.norm(extents)),
        'centroid': (vmin + vmax) / 2.0,
    }


def compute_volume_if_closed(mesh):
    """
    若网格水密，返回体积；否则返回 NaN。
    """
    if mesh.is_watertight:
        return float(mesh.volume)
    return float(np.nan)


def build_cotangent_laplacian(mesh):
    """
    构建当前网格的余切权重 Laplacian 矩阵。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n_vertices = len(vertices)
    if n_vertices == 0:
        return csr_matrix((0, 0))

    edge_to_faces = {}
    for fid, face in enumerate(faces):
        for i in range(3):
            v0 = int(face[i])
            v1 = int(face[(i + 1) % 3])
            key = (v0, v1) if v0 < v1 else (v1, v0)
            edge_to_faces.setdefault(key, []).append(fid)

    row = []
    col = []
    data = []

    for fid, face in enumerate(faces):
        tri = vertices[face]
        a, b, c = tri[0], tri[1], tri[2]

        def angle_at(p, q, r):
            v1 = q - p
            v2 = r - p
            dot = np.dot(v1, v2)
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < 1e-12:
                return 0.0
            cos_angle = np.clip(dot / denom, -1.0, 1.0)
            return float(np.arccos(cos_angle))

        alpha = angle_at(a, b, c)
        beta = angle_at(b, c, a)
        gamma = angle_at(c, a, b)

        edge0 = (int(face[1]), int(face[2])) if face[1] < face[2] else (int(face[2]), int(face[1]))
        edge1 = (int(face[2]), int(face[0])) if face[2] < face[0] else (int(face[0]), int(face[2]))
        edge2 = (int(face[0]), int(face[1])) if face[0] < face[1] else (int(face[1]), int(face[0]))

        cot_alpha = 1.0 / np.tan(alpha) if abs(np.tan(alpha)) > 1e-12 else 0.0
        cot_beta = 1.0 / np.tan(beta) if abs(np.tan(beta)) > 1e-12 else 0.0
        cot_gamma = 1.0 / np.tan(gamma) if abs(np.tan(gamma)) > 1e-12 else 0.0

        def add_weight(e0, e1, w):
            if w == 0:
                return
            row.append(e0)
            col.append(e1)
            data.append(w)
            row.append(e1)
            col.append(e0)
            data.append(w)

        add_weight(edge0[0], edge0[1], cot_alpha)
        add_weight(edge1[0], edge1[1], cot_beta)
        add_weight(edge2[0], edge2[1], cot_gamma)

    if not row:
        return csr_matrix((n_vertices, n_vertices))

    L = csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(row, dtype=np.int64), np.array(col, dtype=np.int64))),
        shape=(n_vertices, n_vertices),
    )

    row_sums = np.asarray(L.sum(axis=1)).ravel()
    L = L - csr_matrix(
        (row_sums, (np.arange(n_vertices), np.arange(n_vertices))),
        shape=(n_vertices, n_vertices),
    )

    return L


def _dirichlet_energy_from_laplacian(L, vertices):
    vec = L @ vertices
    return float(np.sum(vec * vertices))


def _mesh_has_valid_positive_areas(vertices, faces, min_area_ratio=1e-8):
    if len(faces) == 0:
        return False, None

    tri = np.asarray(vertices, dtype=np.float64)[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    if np.any(~np.isfinite(areas)) or np.any(areas <= 0):
        return False, areas

    median_area = float(np.median(areas))
    if median_area < 1e-12:
        return False, areas

    if np.any(areas < min_area_ratio * median_area):
        return False, areas

    return True, areas


def laplacian_smooth_fixed_boundary(
    mesh,
    boundary_vertex_indices,
    iterations=200,
    step_size=1.0,
    tol=1e-7,
    verbose=False,
    return_info=False,
    max_backtracking=8,
    min_area_ratio=1e-8,
):
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = mesh.vertices.copy()
    n_vertices = len(vertices)
    boundary_set = set(int(v) for v in boundary_vertex_indices)

    info = {
        "success": False,
        "iterations": 0,
        "converged": False,
        "max_move": 0.0,
        "energy": None,
        "message": "",
    }

    if n_vertices == 0 or len(faces) == 0:
        info["message"] = "空网格，无法优化"
        if return_info:
            return mesh, info
        return mesh

    all_indices = np.arange(n_vertices, dtype=np.int64)
    interior_indices = np.array(
        [i for i in all_indices if int(i) not in boundary_set],
        dtype=np.int64,
    )
    boundary_indices = np.array(sorted(boundary_set), dtype=np.int64)

    if len(interior_indices) == 0:
        info.update({"success": True, "message": "没有内部顶点"})
        if return_info:
            return mesh, info
        return mesh

    step_size = float(np.clip(step_size, 0.0, 1.0))

    for it in range(iterations):
        current = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        L = build_cotangent_laplacian(current)

        Lint = L[interior_indices, :][:, interior_indices].tocsr()
        Lbnd = L[interior_indices, :][:, boundary_indices].tocsr()

        n_int = len(interior_indices)
        eps_reg = 1e-10
        Lint = Lint + csr_matrix(np.eye(n_int, dtype=np.float64) * eps_reg)

        rhs = -Lbnd @ vertices[boundary_indices]

        try:
            sol = spsolve(Lint, rhs)
        except Exception as e:
            info["message"] = f"线性求解失败: {e}"
            info["iterations"] = it
            if verbose:
                print(f"    Seifert 优化中止: {info['message']}")
            if return_info:
                return mesh, info
            return mesh

        if not np.all(np.isfinite(sol)):
            info["message"] = "线性求解出现 NaN/Inf，已中止"
            info["iterations"] = it
            if verbose:
                print(f"    Seifert 优化中止: {info['message']}")
            if return_info:
                return mesh, info
            return mesh

        current_int = vertices[interior_indices]
        old_energy = _dirichlet_energy_from_laplacian(L, vertices)

        accepted = False
        alpha = step_size
        new_vertices = vertices.copy()

        for _ in range(max_backtracking + 1):
            trial_int = current_int + alpha * (sol - current_int)
            trial_vertices = vertices.copy()
            trial_vertices[interior_indices] = trial_int

            valid_areas, _areas = _mesh_has_valid_positive_areas(
                trial_vertices, faces, min_area_ratio
            )
            if not valid_areas:
                alpha *= 0.5
                continue

            trial_energy = _dirichlet_energy_from_laplacian(L, trial_vertices)

            if trial_energy <= old_energy:
                new_vertices = trial_vertices
                energy = trial_energy
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            info["message"] = "优化失败：能量未下降或网格退化，已快速中止"
            info["iterations"] = it
            info["energy"] = old_energy
            if verbose:
                print(f"    {info['message']}")
            if return_info:
                return mesh, info
            return mesh

        moves = np.linalg.norm(
            new_vertices[interior_indices] - vertices[interior_indices],
            axis=1,
        )
        max_move = float(np.max(moves)) if len(moves) > 0 else 0.0
        vertices = new_vertices

        info["iterations"] = it + 1
        info["energy"] = energy
        info["max_move"] = max_move

        if verbose:
            print(
                f"    Seifert 迭代 {it+1}: energy={energy:.10e}, "
                f"max_move={max_move:.6e}"
            )

        if max_move < tol:
            info["success"] = True
            info["converged"] = True
            info["message"] = "已收敛"
            break

    else:
        info["success"] = True
        info["converged"] = False
        info["message"] = "达到最大迭代次数"

    result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if return_info:
        return result_mesh, info
    return result_mesh


def compute_curvature_statistics(mesh, boundary_vertex_indices):
    """
    计算网格内部顶点的离散曲率统计。
    与旧 geometrics.py 中 compute_curvature_statistics 行为一致。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n_vertices = len(vertices)
    boundary_set = set(int(v) for v in boundary_vertex_indices)

    L = build_cotangent_laplacian(mesh)

    area_faces = mesh.area_faces
    vertex_areas = np.bincount(
        faces.ravel(),
        weights=np.repeat(area_faces, 3),
        minlength=n_vertices,
    ) / 3.0
    vertex_areas[vertex_areas < 1e-12] = 1.0

    Hn = L @ vertices
    H_mag = np.linalg.norm(Hn, axis=1) / (2.0 * vertex_areas)

    angle_sum = np.zeros(n_vertices, dtype=np.float64)
    for face in faces:
        tri = vertices[face]
        for j in range(3):
            v_idx = int(face[j])
            p = tri[j]
            q = tri[(j + 1) % 3]
            r = tri[(j + 2) % 3]
            v1 = q - p
            v2 = r - p
            dot = np.dot(v1, v2)
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < 1e-12:
                angle = 0.0
            else:
                cos_angle = np.clip(dot / denom, -1.0, 1.0)
                angle = float(np.arccos(cos_angle))
            angle_sum[v_idx] += angle
    K = (2.0 * np.pi - angle_sum) / vertex_areas

    interior_mask = np.array(
        [i not in boundary_set for i in range(n_vertices)],
        dtype=bool,
    )
    if not np.any(interior_mask):
        interior_mask = np.ones(n_vertices, dtype=bool)

    H_int = H_mag[interior_mask]
    K_int = K[interior_mask]

    return {
        "mean_abs_mean_curvature": float(np.mean(H_int)),
        "median_abs_mean_curvature": float(np.median(H_int)),
        "max_abs_mean_curvature": float(np.max(H_int)),
        "p95_abs_mean_curvature": float(np.percentile(H_int, 95)),
        "std_abs_mean_curvature": float(np.std(H_int)),
        "mean_gaussian_curvature": float(np.mean(K_int)),
        "median_gaussian_curvature": float(np.median(K_int)),
        "max_gaussian_curvature": float(np.max(K_int)),
        "min_gaussian_curvature": float(np.min(K_int)),
        "area": float(mesh.area),
        "perimeter": float(np.sum(np.linalg.norm(
            vertices[boundary_vertex_indices] -
            np.roll(vertices[boundary_vertex_indices], -1, axis=0), axis=1))),
    }


def voxelize_mesh_watertight(mesh, grid_size=96, margin=1.1):
    """
    将水密网格体素化为三值数据立方：0=内部，1=表面，2=外部。

    Returns
    -------
    grid : (G, G, G) ndarray uint8
    origin : (3,)  世界坐标系中网格原点（体素 [0,0,0] 的位置）
    pitch : float  体素边长
    """
    from scipy import ndimage

    bbox = mesh.bounding_box
    extents = bbox.extents
    pitch = extents.max() / grid_size
    voxel = mesh.voxelized(pitch)
    surface = voxel.matrix.astype(bool)

    grid = np.zeros_like(surface, dtype=np.uint8)
    grid[surface] = 1

    outside = ~surface
    labeled, num = ndimage.label(outside)
    if num == 0:
        grid.fill(2)
        grid[surface] = 1
        return grid, voxel.translation, voxel.pitch

    corner_label = labeled[0, 0, 0]
    grid[labeled == corner_label] = 2

    return grid, voxel.translation, voxel.pitch


def voxel_symmetry_score_watertight(mesh, plane_normal, plane_offset=None,
                                    grid_size=96, metric='gradient'):
    """
    基于三值体素立方的对称性评分。

    在体素索引空间完成关于平面 n·x = offset 的镜像，然后比较原始网格与
    镜像网格的 occupancy 差异。

    返回负的加权均方误差，越大表示越对称。
    """
    from scipy import ndimage

    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    if plane_offset is None:
        plane_offset = float(np.median(mesh.vertices @ n))

    grid, translation, pitch = voxelize_mesh_watertight(mesh, grid_size)

    # 镜像矩阵：关于法向 n 的反射
    A = np.eye(3) - 2.0 * np.outer(n, n)

    # 平移：使镜像平面过 plane_offset
    # 推导：
    #   p  = translation + pitch * q
    #   p' = p - 2*(n·p - offset)*n
    #   q' = (p' - translation) / pitch
    #      = q - 2*(n·translation - offset)/pitch * n - 2*(n·q)*n
    #      = A @ q + b
    b = -2.0 * (np.dot(translation, n) - plane_offset) / pitch * n

    mirrored = ndimage.affine_transform(
        grid.astype(np.float32),
        A,
        offset=b,
        order=1,
        mode='constant',
        cval=2.0
    )

    a = grid.astype(np.float32)
    b_grid = mirrored

    mask = (a > 0) | (b_grid > 0)
    if not np.any(mask):
        return 0.0

    a0 = a[mask] - np.mean(a[mask])
    b0 = b_grid[mask] - np.mean(b_grid[mask])
    diff = a0 - b0

    if metric == 'identity':
        W = np.ones_like(diff)
    elif metric == 'gradient':
        gx, gy, gz = np.gradient(a)
        Gmag = np.sqrt(gx * gx + gy * gy + gz * gz)
        W = Gmag[mask]
    elif metric == 'structure':
        gx, gy, gz = np.gradient(a)
        Gmag = gx * gx + gy * gy + gz * gz
        W = Gmag[mask]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    W = np.maximum(W, 1e-12)
    W = W / W.sum()

    score = -np.sum(W * diff * diff)
    return float(score)


def estimate_symmetry_plane_voxel(mesh, candidate_normals=None,
                                  grid_size=96, metric='gradient'):
    """
    在候选法向中搜索最佳对称平面（基于水密网格体素化）。

    Returns
    -------
    best_normal : (3,)
    best_offset : float
    best_score : float
    """
    pts = mesh.vertices
    if candidate_normals is None:
        cov = np.cov(pts.T)
        _, evec = np.linalg.eigh(cov)
        candidate_normals = [evec[:, i] for i in range(3)]

    best_score = -np.inf
    best_n = None
    best_offset = 0.0

    for n in candidate_normals:
        signed = pts @ np.asarray(n, dtype=np.float64)
        offset = float(np.median(signed))
        score = voxel_symmetry_score_watertight(
            mesh, n, offset, grid_size, metric
        )
        if score > best_score:
            best_score = score
            best_n = np.asarray(n, dtype=np.float64)
            best_offset = offset

    if best_n is None:
        best_n = np.array([0.0, 1.0, 0.0])
        best_offset = 0.0
        best_score = 0.0

    return best_n, best_offset, best_score
