# src/toys3d/geometrics/analysis.py
"""分析工具：OBB / 长方体坐标框架、平面检测。"""
import numpy as np

from .euclidean import (
    average_antiparallel_directions,
    signed_distance_to_plane,
    normalize,
)


def voxelize_mesh(mesh, grid_size=128, method='surface'):
    """
    将网格体素化为 trimesh VoxelGrid 对象。
    method: 'surface' 或 'filled'（若不支持 filled 则回退到 surface）
    """
    bbox = mesh.bounding_box
    pitch = bbox.extents.max() / grid_size
    if pitch <= 0:
        pitch = 1.0
    voxel = mesh.voxelized(pitch)
    if method == 'filled' and hasattr(voxel, 'fill'):
        voxel = voxel.fill()
    return voxel


def get_occupied_voxels(voxel, method='surface'):
    """
    从 VoxelGrid 中提取被占据体素的世界坐标。
    """
    if method == 'filled' and hasattr(voxel, 'matrix_filled'):
        coords = np.argwhere(voxel.matrix_filled)
    else:
        coords = np.argwhere(voxel.matrix)
    return coords * voxel.pitch + voxel.translation


def compute_obb_volume(points, axes):
    """
    计算点集在给定正交轴下的轴对齐包围盒体积。

    axes: (3, 3)，每行是一个单位轴方向
    """
    proj = points @ axes.T
    extents = proj.max(axis=0) - proj.min(axis=0)
    return float(np.prod(extents))


def points_bounding_box(points, axes):
    """
    计算点集在给定正交轴下的包围盒中心和尺寸。

    Returns
    -------
    origin : (3,)  世界坐标系中的中心点
    extents : (3,)  三个轴方向的尺寸
    """
    proj = points @ axes.T
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    center_local = (mins + maxs) / 2.0
    extents = maxs - mins
    origin = center_local @ axes
    return origin, extents


def initial_obb_axes_pca(points, weights=None):
    """
    用 PCA 估计初始 OBB 三轴。

    Returns
    -------
    axes : (3, 3)，每行一个轴方向，右手系
    """
    pts = np.asarray(points, dtype=np.float64)
    if weights is None:
        cov = np.cov(pts.T)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / (w.sum() + 1e-12)
        mean = np.sum(w[:, None] * pts, axis=0)
        centered = pts - mean
        cov = (w[:, None] * centered).T @ centered

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    axes = eigvecs[:, order].T.copy()

    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]
    return axes


def rotation_matrix_from_euler(angles, order='xyz'):
    """
    由欧拉角构造旋转矩阵。

    Parameters
    ----------
    angles : (3,)  rx, ry, rz（弧度）
    order : 'xyz' 或 'zyx'

    Returns
    -------
    R : (3, 3)
    """
    ax, ay, az = angles
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]], dtype=np.float64)

    if order == 'xyz':
        return Rx @ Ry @ Rz
    elif order == 'zyx':
        return Rz @ Ry @ Rx
    else:
        raise ValueError(f"Unknown euler order: {order}")


def optimize_obb_axes(points, initial_axes, method='Powell', max_iter=200):
    """
    数值优化最小化包围盒体积，从 initial_axes 出发搜索最佳旋转。
    """
    from scipy.optimize import minimize

    def objective(angles):
        R = rotation_matrix_from_euler(angles)
        axes = initial_axes @ R.T
        for i in range(3):
            axes[i] = normalize(axes[i])
        return compute_obb_volume(points, axes)

    res = minimize(objective, x0=np.zeros(3), method=method,
                   options={'maxiter': max_iter})
    R_opt = rotation_matrix_from_euler(res.x)
    axes = initial_axes @ R_opt.T
    for i in range(3):
        axes[i] = normalize(axes[i])
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]
    return axes


def build_frame_from_obb(points, axes):
    """
    由 OBB 轴和点集构造世界↔局部变换矩阵。
    """
    origin, extents = points_bounding_box(points, axes)
    R = axes.T  # 局部 -> 世界的旋转

    T_world_to_local = np.eye(4)
    T_world_to_local[:3, :3] = axes
    T_world_to_local[:3, 3] = -axes @ origin

    T_local_to_world = np.eye(4)
    T_local_to_world[:3, :3] = R
    T_local_to_world[:3, 3] = origin

    return T_world_to_local, T_local_to_world, axes[0], axes[1], axes[2], origin, extents


def evaluate_obb_fit(mesh, origin, axes, extents):
    """
    评估网格与拟合 OBB 的吻合程度。
    """
    centers = mesh.triangles_center
    local = (centers - origin) @ axes.T
    half = extents / 2.0
    inside = np.all(np.abs(local) <= half, axis=1)
    areas = mesh.area_faces
    total_area = areas.sum()
    inside_ratio = float(np.sum(areas[inside]) / total_area) if total_area > 0 else 0.0

    outside_dist = np.maximum(0, np.max(np.abs(local) - half, axis=1))
    outside_mask = outside_dist > 0

    return {
        'inside_ratio': inside_ratio,
        'max_outside_distance': float(np.max(outside_dist)),
        'mean_outside_distance': float(np.mean(outside_dist[outside_mask])) if np.any(outside_mask) else 0.0,
        'rms_outside_distance': float(np.sqrt(np.mean(outside_dist[outside_mask]**2))) if np.any(outside_mask) else 0.0,
    }


def build_box_aligned_frame_voxel(mesh, grid_size=128, optimize=True,
                                  voxel_method='surface'):
    """
    针对音箱、手机等长方体扫描网格，基于体素 OBB 建立局部正交坐标系。

    坐标系约定：
      u_x : PCA 第一主成分（最长方向）
      u_y : PCA 第二主成分
      u_z : u_x × u_y
      原点：OBB 几何中心
    """
    voxel = voxelize_mesh(mesh, grid_size=grid_size, method=voxel_method)
    points = get_occupied_voxels(voxel, method=voxel_method)

    if len(points) < 3:
        raise ValueError("Too few occupied voxels to estimate OBB.")

    axes = initial_obb_axes_pca(points)
    if optimize:
        axes = optimize_obb_axes(points, axes)

    T_w2l, T_l2w, u_x, u_y, u_z, origin, extents = build_frame_from_obb(points, axes)
    fit_info = evaluate_obb_fit(mesh, origin, axes, extents)

    return T_w2l, T_l2w, u_x, u_y, u_z, origin, extents, fit_info


def detect_side_planes(centers, normals, areas, u_z, n_side_planes=4,
                       distance_thr_ratio=0.02, normal_thr_deg=30.0,
                       max_iter=5000, rng=None):
    """
    在侧壁带内用 RANSAC 检测侧面平面。
    侧面法向必须接近水平（垂直于 u_z）。
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(centers)
    bbox_diag = np.linalg.norm(centers.max(axis=0) - centers.min(axis=0))
    d_thr = distance_thr_ratio * bbox_diag
    cos_thr = np.cos(np.deg2rad(normal_thr_deg))
    z_cos_thr = np.sin(np.deg2rad(30.0))  # 法向与水平面夹角不超过 30 度

    remaining = np.ones(N, dtype=bool)
    planes = []

    for _ in range(n_side_planes):
        idx_rem = np.flatnonzero(remaining)
        if len(idx_rem) < 10:
            break

        c_rem = centers[idx_rem]
        n_rem = normals[idx_rem]
        a_rem = areas[idx_rem]

        best_score = 0.0
        best_plane = None
        best_inliers = None

        for _ in range(max_iter):
            sample = rng.choice(len(idx_rem), size=3, replace=False)
            p = c_rem[sample]
            pn = np.cross(p[1] - p[0], p[2] - p[0])
            pn_norm = np.linalg.norm(pn)
            if pn_norm < 1e-12:
                continue
            pn = pn / pn_norm

            # 侧面法向应接近水平
            if abs(np.dot(pn, u_z)) > z_cos_thr:
                continue

            dists = signed_distance_to_plane(c_rem, p.mean(axis=0), pn)
            normal_dots = np.abs(n_rem @ pn)
            inliers = (np.abs(dists) <= d_thr) & (normal_dots >= cos_thr)

            score = float(np.sum(a_rem[inliers]))
            if score > best_score:
                best_score = score
                best_plane = (pn, p.mean(axis=0))
                best_inliers = inliers

        if best_plane is None:
            break

        global_inliers = np.zeros(N, dtype=bool)
        global_inliers[idx_rem] = best_inliers

        # 精化平面
        pts_in = centers[global_inliers]
        weights = areas[global_inliers]
        mean = np.sum(weights[:, None] * pts_in, axis=0) / np.sum(weights)
        centered = pts_in - mean
        cov = (weights[:, None] * centered).T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        refined_normal = eigvecs[:, np.argmin(eigvals)]
        if np.dot(refined_normal, best_plane[0]) < 0:
            refined_normal = -refined_normal

        # 重新选择内点，保持水平约束
        dists = signed_distance_to_plane(centers, mean, refined_normal)
        normal_dots = np.abs(normals @ refined_normal)
        global_inliers = (np.abs(dists) <= d_thr) & \
                         (normal_dots >= cos_thr) & \
                         (np.abs(normals @ u_z) <= z_cos_thr)

        planes.append({'normal': refined_normal, 'origin': mean})
        remaining &= ~global_inliers

    return planes


def merge_side_planes_to_xy(planes, u_z):
    """
    把检测到的侧面法向合并为 x、y 两个正交水平方向。
    """
    # 投影到水平面
    horizontals = []
    for p in planes:
        n = p['normal']
        h = n - np.dot(n, u_z) * u_z
        hn = np.linalg.norm(h)
        if hn > 1e-12:
            horizontals.append(h / hn)

    if len(horizontals) < 2:
        raise ValueError(f"Only {len(horizontals)} horizontal normals, need 2.")

    # 用迭代聚类把法向分成两组（应近似正交）
    g0 = np.array(horizontals[0])
    g1 = np.cross(u_z, g0)
    for _ in range(10):
        cluster0, cluster1 = [], []
        for h in horizontals:
            if abs(np.dot(h, g0)) >= abs(np.dot(h, g1)):
                cluster0.append(h)
            else:
                cluster1.append(h)
        if len(cluster0) == 0 or len(cluster1) == 0:
            break
        new_g0 = np.mean(cluster0, axis=0)
        new_g1 = np.mean(cluster1, axis=0)
        new_g0 = new_g0 - np.dot(new_g0, u_z) * u_z
        new_g1 = new_g1 - np.dot(new_g1, u_z) * u_z
        if np.linalg.norm(new_g0) < 1e-12 or np.linalg.norm(new_g1) < 1e-12:
            break
        g0, g1 = new_g0, new_g1

    u_x = normalize(g0)
    u_y = normalize(g1 - np.dot(g1, u_x) * u_x)

    # 右手系
    if np.dot(np.cross(u_x, u_y), u_z) < 0:
        u_y = -u_y

    return u_x, u_y


def detect_dominant_planes(mesh, n_planes=2, distance_thr_ratio=0.02,
                           normal_thr_deg=30.0, max_iter=5000, rng=None):
    """
    用 RANSAC 检测网格中面积最大的 n_planes 个主导平面。

    Returns
    -------
    planes : list of dict
        每个元素包含：
        - 'normal' : (3,) 平面单位法向
        - 'origin' : (3,) 平面上一点
        - 'mask'   : (N,) bool 内点面片掩码
    """
    if rng is None:
        rng = np.random.default_rng()

    centers = mesh.triangles_center
    normals = mesh.face_normals
    areas = mesh.area_faces
    N = len(mesh.faces)

    bbox_diag = np.linalg.norm(mesh.bounding_box.extents)
    d_thr = distance_thr_ratio * bbox_diag
    cos_thr = np.cos(np.deg2rad(normal_thr_deg))

    remaining = np.ones(N, dtype=bool)
    planes = []

    for _ in range(n_planes):
        idx_remaining = np.flatnonzero(remaining)
        if len(idx_remaining) < 10:
            break

        c_rem = centers[idx_remaining]
        n_rem = normals[idx_remaining]
        a_rem = areas[idx_remaining]

        best_score = 0.0
        best_plane = None
        best_inliers = None

        for _ in range(max_iter):
            sample = rng.choice(len(idx_remaining), size=3, replace=False)
            p = c_rem[sample]
            v1 = p[1] - p[0]
            v2 = p[2] - p[0]
            pn = np.cross(v1, v2)
            pn_norm = np.linalg.norm(pn)
            if pn_norm < 1e-12:
                continue
            pn = pn / pn_norm

            dists = signed_distance_to_plane(c_rem, p.mean(axis=0), pn)
            normal_dots = np.abs(n_rem @ pn)
            inliers = (np.abs(dists) <= d_thr) & (normal_dots >= cos_thr)

            score = float(np.sum(a_rem[inliers]))
            if score > best_score:
                best_score = score
                best_plane = (pn, p.mean(axis=0))
                best_inliers = inliers

        if best_plane is None:
            break

        # 用所有内点精化平面
        global_inliers = np.zeros(N, dtype=bool)
        global_inliers[idx_remaining] = best_inliers

        pts_in = centers[global_inliers]
        weights = areas[global_inliers]
        if np.sum(weights) < 1e-12:
            break

        # 加权最小二乘拟合平面
        mean = np.sum(weights[:, None] * pts_in, axis=0) / np.sum(weights)
        centered = pts_in - mean
        cov = (weights[:, None] * centered).T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        refined_normal = eigvecs[:, np.argmin(eigvals)]

        # 确保法向与原始采样法向一致
        if np.dot(refined_normal, best_plane[0]) < 0:
            refined_normal = -refined_normal

        dists = signed_distance_to_plane(centers, mean, refined_normal)
        normal_dots = np.abs(normals @ refined_normal)
        global_inliers = (np.abs(dists) <= d_thr) & (normal_dots >= cos_thr)

        planes.append({
            'normal': refined_normal,
            'origin': mean,
            'mask': global_inliers,
        })
        remaining &= ~global_inliers

    return planes


def fit_obb_2d(points_2d):
    """
    对二维点集拟合最小面积包围矩形，返回矩形两条边的单位方向。
    """
    if len(points_2d) < 3:
        raise ValueError("Too few points for 2D OBB")

    angles = np.linspace(0, np.pi / 2, 180, endpoint=False)
    best_area = np.inf
    best_axes = None

    for theta in angles:
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]])
        rot = points_2d @ R.T
        w = rot[:, 0].max() - rot[:, 0].min()
        h = rot[:, 1].max() - rot[:, 1].min()
        area = w * h
        if area < best_area:
            best_area = area
            best_axes = np.array([[c, -s], [s, c]])

    return best_axes, best_area


def build_box_aligned_frame_mesh(mesh, distance_thr_ratio=0.02,
                                 normal_thr_deg=30.0, max_iter=5000,
                                 shell_depths=None, refine=True, rng=None):
    """
    基于网格几何的平面检测建立长方体坐标系。

    shell_depths: ((x_neg, x_pos), (y_neg, y_pos), (z_neg, z_pos))
                  当前实现仅使用 z 方向分量排除顶/底面纹理区域。
                  x/y 分量保留为未来扩展通用壳厚度。

    refine=True  (Pass 2): 检测顶/底面后，用侧壁带 RANSAC 精化侧面方向。
    refine=False (Pass 1): 检测顶/底面后，用整体顶点投影的 2D OBB 快速估计侧面方向。
    """
    if rng is None:
        rng = np.random.default_rng()
    if shell_depths is None:
        shell_depths = ((0.1, 0.1), (0.1, 0.1), (0.2, 0.2))

    # 1. 检测顶/底两个主导平面，确定 z 轴
    planes = detect_dominant_planes(
        mesh, n_planes=2, distance_thr_ratio=distance_thr_ratio,
        normal_thr_deg=normal_thr_deg, max_iter=max_iter, rng=rng
    )
    if len(planes) < 2:
        raise ValueError(f"Only {len(planes)} dominant plane found, need 2.")

    n0, n1 = planes[0]['normal'], planes[1]['normal']
    if np.dot(n0, n1) > 0:
        n1 = -n1
    u_z = average_antiparallel_directions(n0, n1)

    # 构造 xy 平面临时正交基
    if abs(u_z[2]) < 0.9:
        temp_x = np.cross(u_z, [0, 0, 1])
    else:
        temp_x = np.cross(u_z, [1, 0, 0])
    temp_x = temp_x / np.linalg.norm(temp_x)
    temp_y = np.cross(u_z, temp_x)

    if refine:
        # Pass 2: 在侧壁带内精化侧面方向
        proj_z = mesh.vertices @ u_z
        z_min, z_max = proj_z.min(), proj_z.max()
        z_extent = z_max - z_min
        z_neg, z_pos = shell_depths[2]
        z_lo = z_min + z_neg * z_extent
        z_hi = z_max - z_pos * z_extent

        centers_z = mesh.triangles_center @ u_z
        side_mask = (centers_z > z_lo) & (centers_z < z_hi)

        print(f"  Side band: z in [{z_lo:.3f}, {z_hi:.3f}], "
              f"faces={np.sum(side_mask)}")

        if np.sum(side_mask) < 10:
            raise ValueError("Too few faces in side band; check shell depths.")

        side_centers = mesh.triangles_center[side_mask]
        side_normals = mesh.face_normals[side_mask]
        side_areas = mesh.area_faces[side_mask]

        side_planes = detect_side_planes(
            side_centers, side_normals, side_areas, u_z,
            n_side_planes=4, distance_thr_ratio=distance_thr_ratio,
            normal_thr_deg=normal_thr_deg, max_iter=max_iter, rng=rng
        )

        if len(side_planes) < 2:
            raise ValueError(f"Only {len(side_planes)} side plane found, need 2.")

        u_x, u_y = merge_side_planes_to_xy(side_planes, u_z)
    else:
        # Pass 1: 用所有顶点投影的 2D OBB 快速估计 x, y
        pts_3d = mesh.vertices
        proj_x = pts_3d @ temp_x
        proj_y = pts_3d @ temp_y
        proj_2d = np.column_stack([proj_x, proj_y])

        axes_2d, _ = fit_obb_2d(proj_2d)
        u_x = axes_2d[0, 0] * temp_x + axes_2d[0, 1] * temp_y
        u_y = axes_2d[1, 0] * temp_x + axes_2d[1, 1] * temp_y

        if np.dot(np.cross(u_x, u_y), u_z) < 0:
            u_y = -u_y

    # 计算包围盒中心和尺寸
    axes = np.vstack([u_x, u_y, u_z])
    origin, extents = points_bounding_box(mesh.vertices, axes)

    # 变换矩阵
    T_w2l = np.eye(4)
    T_w2l[:3, :3] = axes
    T_w2l[:3, 3] = -axes @ origin

    T_l2w = np.eye(4)
    T_l2w[:3, :3] = axes.T
    T_l2w[:3, 3] = origin

    fit_info = evaluate_obb_fit(mesh, origin, axes, extents)

    return T_w2l, T_l2w, u_x, u_y, u_z, origin, extents, fit_info


def fit_line_3d(points):
    """
    三维点最小二乘直线拟合。

    Returns
    -------
    rmse : float
    direction : (3,) ndarray or None
    center : (3,) ndarray
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

    Returns
    -------
    rmse : float
    center : (3,) ndarray or None
    normal : (3,) ndarray or None
    radius : float
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


def ransac_plane_fitting(points, max_iter=500, inlier_threshold=0.1,
                         rng=None):
    """
    Fit a single plane to 3D points using RANSAC.
    Returns (normal, point_on_plane), inlier_mask.
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

    # refine plane using all inliers
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
    Returns list of (normal, point, inlier_mask).
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


def map_labels_from_proxy(original_mesh, proxy_mesh, proxy_labels):
    """
    将代理网格上的薄板标签映射回原始网格。

    Returns
    -------
    labels : (N,) ndarray
    """
    from scipy.spatial import cKDTree

    proxy_centers = np.asarray(proxy_mesh.triangles_center, dtype=np.float64)
    original_centers = np.asarray(original_mesh.triangles_center, dtype=np.float64)

    if len(proxy_centers) == 0 or len(original_centers) == 0:
        return np.zeros(len(original_centers), dtype=int)

    tree = cKDTree(proxy_centers)
    _, indices = tree.query(original_centers, k=1)
    return np.asarray(proxy_labels, dtype=int)[indices]
