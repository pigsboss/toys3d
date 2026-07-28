import numpy as np
import trimesh

def plucker_design_matrix(centers, normals, areas, ref_point=None):
    """
    计算加权普吕克设计矩阵，用于从三角面片拟合空间直线。

    Parameters
    ----------
    centers : (N, 3) np.ndarray
        三角面片的几何中心坐标。
    normals : (N, 3) np.ndarray
        单位法向量。
    areas : (N,) np.ndarray
        面片面积。
    ref_point : (3,) np.ndarray or None
        参考基准点。若为 None，则使用面积加权的面片中心作为基准点。

    Returns
    -------
    C : (6, 6) np.ndarray
        加权普吕克设计矩阵，对其最小特征值对应的特征向量即为最优
        普吕克直线坐标 [m; d]（力矩；方向）。
    moments : (N, 3) np.ndarray
        各面片法向射线相对于基准点的力矩向量 m_i = (c_i - p) × n_i。
    ref_point : (3,) np.ndarray
        实际使用的参考基准点。
    """
    if ref_point is None:
        ref_point = np.average(centers, axis=0, weights=areas)

    r = centers - ref_point                  # (N, 3)
    moments = np.cross(r, normals)           # (N, 3)

    # 构建加权矩阵 A，其行 i 为 sqrt(w_i) * [m_i, d_i]
    w = areas
    A = np.hstack([moments, normals]) * np.sqrt(w[:, None])  # (N, 6)

    C = A.T @ A  # (6, 6)

    return C, moments, ref_point

def axis_from_plucker(C, ref_point=None):
    """
    从加权普吕克设计矩阵求解最优空间直线（轴线）。

    Parameters
    ----------
    C : (6, 6) np.ndarray
        由 `plucker_design_matrix` 返回的加权普吕克矩阵。
    ref_point : (3,) np.ndarray or None
        计算设计矩阵时所用的参考基准点。提供后，输出的基点将转换回
        绝对坐标系（即基准点加上相对偏移）；若为 None，则基点相对于
        原参考点（原点）的坐标。

    Returns
    -------
    direction : (3,) np.ndarray
        轴线的单位方向向量。
    base_point : (3,) np.ndarray
        轴线上的一点，该点与 ref_point 的连线垂直于轴线。
        若 ref_point 为 None，该点坐标相对于原始参考点（原点）。
    """
    # 对称矩阵特征分解，取最小特征值对应的特征向量
    eigvals, eigvecs = np.linalg.eigh(C)
    min_idx = np.argmin(eigvals)
    v = eigvecs[:, min_idx]       # 普吕克坐标 [m; d]

    m, d = v[:3], v[3:]

    # 方向归一化
    norm_d = np.linalg.norm(d)
    if norm_d < 1e-12:
        raise ValueError("方向向量范数接近零，无法确定有效轴线。")
    d = d / norm_d

    # 消除数值误差，确保力矩与方向正交
    m = m - np.dot(m, d) * d

    # 相对参考点的最近点偏移量 (d × m)
    p_rel = np.cross(d, m)

    if ref_point is None:
        base_point = p_rel
    else:
        base_point = np.asarray(ref_point) + p_rel

    return d, base_point

def line_line_distance_and_midpoint(dir1, point1, dir2, point2):
    """
    计算空间两条直线的公垂线长度及公垂线段中点。

    Parameters
    ----------
    dir1 : (3,) np.ndarray
        直线 1 的方向向量（无需归一化）。
    point1 : (3,) np.ndarray
        直线 1 上的一点。
    dir2 : (3,) np.ndarray
        直线 2 的方向向量（无需归一化）。
    point2 : (3,) np.ndarray
        直线 2 上的一点。

    Returns
    -------
    distance : float
        两条直线之间的最短距离（公垂线长度）。
    midpoint : (3,) np.ndarray
        公垂线段的中点坐标（即两垂足连线的中点）。
    """
    d1 = np.asarray(dir1, dtype=np.float64)
    p1 = np.asarray(point1, dtype=np.float64)
    d2 = np.asarray(dir2, dtype=np.float64)
    p2 = np.asarray(point2, dtype=np.float64)

    # 叉积判断是否平行
    cross = np.cross(d1, d2)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-12:                     # 平行或重合
        v = p2 - p1
        d1_sq = np.dot(d1, d1)
        t1 = np.dot(v, d1) / d1_sq             # P2 在 L1 上的投影参数
        foot = p1 + t1 * d1                    # 垂足
        distance = np.linalg.norm(foot - p2)
        midpoint = (foot + p2) / 2.0
        return distance, midpoint

    # 非平行情况，解线性方程组
    d1_d1 = np.dot(d1, d1)
    d2_d2 = np.dot(d2, d2)
    d1_d2 = np.dot(d1, d2)
    v = p2 - p1
    rhs1 = np.dot(v, d1)
    rhs2 = np.dot(v, d2)

    det = d1_d1 * d2_d2 - d1_d2 * d1_d2
    if abs(det) < 1e-12:                       # 数值退化，退化为平行处理
        v = p2 - p1
        t1 = np.dot(v, d1) / d1_d1
        foot = p1 + t1 * d1
        distance = np.linalg.norm(foot - p2)
        midpoint = (foot + p2) / 2.0
        return distance, midpoint

    t1 = ( rhs1 * d2_d2 - rhs2 * d1_d2) / det
    t2 = (-rhs1 * d1_d2 + rhs2 * d1_d1) / det

    q1 = p1 + t1 * d1
    q2 = p2 + t2 * d2

    distance = np.linalg.norm(q2 - q1)
    midpoint = (q1 + q2) / 2.0

    return distance, midpoint


def orthogonalize_axes(dir_x, point_x, weight_x,
                       dir_y, point_y, weight_y):
    """
    从两条近似垂直的直线及其置信度构造正交坐标系，并输出坐标变换矩阵。

    算法：
    1. 归一化初始方向。
    2. 权重较高的轴保持不动，另一轴投影到其正交补，确保正交。
    3. 用正交化后的方向及原始点定义两条空间直线，计算公垂线垂足。
    4. 以置信度为权重求垂足的加权中点作为坐标系原点。
    5. 平移坐标轴过原点，构造右手系基向量。
    6. 生成世界→新坐标系  及  新坐标系→世界  的 4×4 刚体变换矩阵。

    Parameters
    ----------
    dir_x : (3,) array_like
        x 轴初始方向向量（无需归一化）。
    point_x : (3,) array_like
        x 轴上的一点。
    weight_x : float
        x 轴的置信度（>0 表明更可信；可为 0）。
    dir_y : (3,) array_like
        y 轴初始方向向量（无需归一化）。
    point_y : (3,) array_like
        y 轴上的一点。
    weight_y : float
        y 轴的置信度。

    Returns
    -------
    T_world_to_local : ndarray (4,4)
        原世界坐标系 → 新正交坐标系的刚体变换矩阵。
        用法：``mesh.apply_transform(T_world_to_local)`` 可将 mesh 变换到新坐标系。
    T_local_to_world : ndarray (4,4)
        新正交坐标系 → 原世界坐标系的刚体变换矩阵。
    u_x : ndarray (3,)
        新坐标系 x 轴在原世界系中的单位方向向量。
    u_y : ndarray (3,)
        新坐标系 y 轴在原世界系中的单位方向向量。
    u_z : ndarray (3,)
        新坐标系 z 轴在原世界系中的单位方向向量（通过右手定则计算）。
    origin : ndarray (3,)
        新坐标系原点在原世界系中的坐标。
    """
    # 转 numpy 数组
    dx = np.asarray(dir_x, dtype=np.float64)
    px = np.asarray(point_x, dtype=np.float64)
    dy = np.asarray(dir_y, dtype=np.float64)
    py = np.asarray(point_y, dtype=np.float64)

    # 归一化初始方向
    dx = dx / np.linalg.norm(dx)
    dy = dy / np.linalg.norm(dy)

    # 正交化：保留权重大的轴，修正另一轴
    if weight_x >= weight_y:
        u_x = dx
        # dy 投影到与 u_x 正交的方向
        u_y = dy - np.dot(dy, u_x) * u_x
        norm_uy = np.linalg.norm(u_y)
        if norm_uy < 1e-12:
            # y 轴与 x 轴几乎共线，无法构造正交系
            raise ValueError("y 轴方向与 x 轴平行，无法构造正交坐标系")
        u_y /= norm_uy
    else:
        u_y = dy
        u_x = dx - np.dot(dx, u_y) * u_y
        norm_ux = np.linalg.norm(u_x)
        if norm_ux < 1e-12:
            raise ValueError("x 轴方向与 y 轴平行，无法构造正交坐标系")
        u_x /= norm_ux

    # 右手系 z 轴
    u_z = np.cross(u_x, u_y)

    # 计算两条正交但不一定共面的直线的公垂线垂足
    # Lx: px + t * u_x; Ly: py + s * u_y
    # 由于 u_x · u_y = 0，解 t, s 使得 (px + t*u_x) - (py + s*u_y) 垂直于两者
    v = py - px
    t = np.dot(v, u_x)                    # 投影到 u_x
    s = -np.dot(v, u_y)                   # 投影到 u_y（注意符号）
    qx = px + t * u_x
    qy = py + s * u_y

    # 加权中点作为原点
    w_sum = weight_x + weight_y
    if w_sum < 1e-12:
        # 无有效权重，取普通中点
        origin = (qx + qy) / 2.0
    else:
        origin = (weight_x * qx + weight_y * qy) / w_sum

    # 构造基矩阵 R = [u_x, u_y, u_z] (3x3)，注意这是列向量排列
    R = np.column_stack((u_x, u_y, u_z))

    # 世界 → 新坐标系：p_local = R.T @ (p_world - origin)
    T_world_to_local = np.eye(4)
    T_world_to_local[:3, :3] = R.T
    T_world_to_local[:3, 3] = -R.T @ origin

    # 新坐标系 → 世界：p_world = R @ p_local + origin
    T_local_to_world = np.eye(4)
    T_local_to_world[:3, :3] = R
    T_local_to_world[:3, 3] = origin

    return T_world_to_local, T_local_to_world, u_x, u_y, u_z, origin

def segment_tubular_regions(normals, areas=None, threshold=0.1, min_faces=100,
                            max_regions=5, max_iterations=1000, rng=None):
    """
    使用 RANSAC 根据面片法线将三角网格分割为不同的管状区域。

    管状区域的特征是所有面片法线大致垂直于该区域的轴线。随机抽取两个面片，
    其法线叉乘作为候选轴线，将法线与轴线点积绝对值小于阈值的面片作为内点。
    重复此过程以取出多个区域。

    Parameters
    ----------
    normals : (N, 3) np.ndarray
        面片单位法向量（必须已归一化）。
    areas : (N,) np.ndarray or None
        面片面积，用于加权评分。若为 None，则使用等权（每个面片权重为 1）。
    threshold : float
        内点判定阈值：若 |dot(normal, axis)| <= threshold，该面片属于当前区域。
        数值越小越严格。
    min_faces : int
        一个区域必须包含的最少面片数。
    max_regions : int
        最多提取的区域数量。
    max_iterations : int
        每个区域 RANSAC 的最大迭代次数。
    rng : numpy.random.Generator or None
        随机数生成器，用于可重复性。若为 None 则使用默认随机状态。

    Returns
    -------
    labels : (N,) np.ndarray (int)
        面片区域标签。0 表示未归类，1..k 表示第 k 个区域。
    axes : list of ndarray (3,)
        每个区域对应的单位轴线方向（顺序与标签编号一致）。
    """
    N = normals.shape[0]
    if N == 0:
        return np.zeros(0, dtype=int), []

    # 面积权重
    if areas is None:
        weights = np.ones(N, dtype=np.float64)
    else:
        weights = np.asarray(areas, dtype=np.float64)

    # 检查并过滤零/无效法向量，避免除零
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_mask = norms[:, 0] > 1e-12
    n = np.zeros_like(normals, dtype=np.float64)
    n[valid_mask] = normals[valid_mask] / norms[valid_mask]

    # 随机数生成器
    if rng is None:
        rng = np.random.default_rng()

    labels = np.zeros(N, dtype=int)
    axes = []

    # 当前可用的面片索引（未归类的且法向有效）
    active = np.where((labels == 0) & valid_mask)[0]

    for region_id in range(1, max_regions + 1):
        if len(active) < 2 or len(active) < min_faces:
            break

        # 当前活动子集
        active_n = n[active]
        active_w = weights[active]

        best_score = -1.0
        best_axis = None
        best_inliers = None  # 相对于 active 的布尔掩码

        # 自适应迭代次数：考虑期望内点比例至少 min_faces/len(active)
        desired_inlier_ratio = max(min_faces / len(active), 0.01)
        # 概率采样模型所需最少迭代次数（保证至少一次全内点抽样概率 0.99）
        iters = min(
            max_iterations,
            int(np.log(0.01) / np.log(1 - desired_inlier_ratio**2))
        )
        iters = max(1, iters)

        for _ in range(iters):
            # 随机抽取两个不同的活动面片
            i, j = rng.choice(len(active), size=2, replace=False)
            n1, n2 = active_n[i], active_n[j]

            cross = np.cross(n1, n2)
            norm_cross = np.linalg.norm(cross)
            if norm_cross < 1e-9:   # 法线几乎平行，无法定义轴线
                continue
            axis = cross / norm_cross

            # 计算所有活动面片与该轴线的点积绝对值
            dots = np.abs(active_n @ axis)
            inliers = dots <= threshold

            # 评分 = 内点权重之和
            score = np.sum(active_w[inliers])

            # 内点数必须满足最小要求
            if np.sum(inliers) < min_faces:
                continue

            if score > best_score:
                best_score = score
                best_axis = axis.copy()
                best_inliers = inliers

        # 如果没有找到合格模型，停止
        if best_axis is None:
            break

        # 将内点标记到全局 labels
        active_indices = active[best_inliers]
        labels[active_indices] = region_id
        axes.append(best_axis)

        # 更新 active 列表
        active = np.where(labels == 0)[0]

    return labels, axes

def project_points_to_plane(points, plane_origin, plane_normal):
    """
    将三维点投影到切平面，返回二维坐标及平面基向量。

    Parameters
    ----------
    points : (N, 3) np.ndarray
    plane_origin : (3,) array_like  平面上一点
    plane_normal : (3,) array_like  平面法向量

    Returns
    -------
    pts_2d : (N, 2) np.ndarray
    basis_u, basis_v : (3,) ndarray  平面内相互正交的两个基向量
    """
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # 构造两个与 normal 正交的基向量
    if abs(normal[0]) > 1e-6 or abs(normal[1]) > 1e-6:
        basis_u = np.cross(normal, [0, 0, 1])
    else:
        basis_u = np.cross(normal, [1, 0, 0])
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)

    vec = points - np.asarray(plane_origin)
    pts_2d = np.column_stack([np.dot(vec, basis_u), np.dot(vec, basis_v)])
    return pts_2d, basis_u, basis_v


def polygon_area_from_3d_ccw(vertices, normal=None):
    """
    计算有序三维闭合多边形的面积（使用叉积求和，与顺序有关）。

    Parameters
    ----------
    vertices : (M, 3) ndarray, 按顺序排列的顶点坐标
    normal : (3,) ndarray or None  可选法向量，用于符号修正（暂未使用）

    Returns
    -------
    area : float
    """
    if len(vertices) < 3:
        return 0.0
    total = np.zeros(3)
    n = len(vertices)
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i+1) % n]
        total += np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(total)
    return area


def path3d_to_polygons(path3d, plane_origin, plane_normal):
    """
    将 trimesh.path.Path3D 投影到平面并转换为多边形列表。

    Parameters
    ----------
    path3d : trimesh.path.Path3D  来自 mesh.section()
    plane_origin, plane_normal : (3,)  平面定义

    Returns
    -------
    polygons : list of trimesh.path.polygons.Polygon (每个都有 .area)
    """
    if path3d is None or len(path3d.entities) == 0:
        return []

    pts_2d, _, _ = project_points_to_plane(path3d.vertices, plane_origin, plane_normal)
    path2d = trimesh.path.Path2D(entities=path3d.entities, vertices=pts_2d)
    # polygons_full 生成所有闭合多边形（外环与内环，如有）
    return list(path2d.polygons_full)


def compute_cross_section_area(mesh, plane_origin, plane_normal, face_mask=None):
    """
    计算网格在指定平面处的截面总面积。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    plane_origin : (3,)  平面上一点
    plane_normal : (3,)  平面法向量
    face_mask : (N,) bool or None  仅使用指定面片，None 表示全部

    Returns
    -------
    area : float  所有闭合环的面积之和（不区分内外）
    """
    if face_mask is not None:
        indices = np.where(face_mask)[0]
        if len(indices) == 0:
            return 0.0
        submesh = mesh.submesh([indices])[0]
    else:
        submesh = mesh

    section = submesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        return 0.0

    polygons = path3d_to_polygons(section, plane_origin, plane_normal)
    total_area = sum(poly.area for poly in polygons)
    return total_area


def suggest_slice_spacing(mesh, axis_dir, region_mask=None,
                          factor=1.5, min_count=10, max_count=200):
    """
    根据网格边长和轴线长度推荐切片间距。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    axis_dir : (3,) ndarray  轴线方向（单位化）
    region_mask : (N,) bool or None
    factor : float  间距 = factor * 平均边长
    min_count, max_count : int  切片数量的允许范围

    Returns
    -------
    spacing : float
    """
    if region_mask is not None:
        indices = np.where(region_mask)[0]
        submesh = mesh.submesh(indices)[0]
    else:
        submesh = mesh

    mean_edge = np.mean(submesh.edges_unique_length)
    dir_u = axis_dir / np.linalg.norm(axis_dir)

    verts = submesh.vertices
    proj = np.dot(verts, dir_u)
    axis_length = proj.max() - proj.min()

    spacing = factor * mean_edge
    if axis_length > 0:
        count = max(1, int(axis_length / spacing))
        if count < min_count:
            spacing = axis_length / min_count
        elif count > max_count:
            spacing = axis_length / max_count
    return spacing


def sample_axial_section_areas(mesh, axis_dir, axis_point, distances,
                               face_mask=None):
    """
    沿轴线在给定距离处采样横截面积。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    axis_dir : (3,) ndarray  轴线单位方向
    axis_point : (3,) ndarray  轴线上参考点
    distances : array_like  相对参考点的轴向距离（可为负）
    face_mask : (N,) bool or None

    Returns
    -------
    dists : ndarray  复制的 distances
    areas : ndarray  各位置的截面总面积
    """
    dir_u = axis_dir / np.linalg.norm(axis_dir)
    areas = []
    for d in distances:
        origin = axis_point + d * dir_u
        area = compute_cross_section_area(mesh, origin, dir_u, face_mask)
        areas.append(area)
    return np.asarray(distances), np.asarray(areas)


def kmeans_1d(data, k, max_iter=100, tol=1e-4, rng=None):
    """
    对一维数据执行 K-Means 聚类，返回聚类标签和中心。

    Parameters
    ----------
    data : (N,) array_like  一维观测值
    k : int  聚类数目
    max_iter : int
    tol : float
    rng : numpy.random.Generator or None

    Returns
    -------
    labels : (N,) ndarray  从 0 开始的簇编号（按中心升序）
    centers : (k,) ndarray  升序排列的簇中心
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    if rng is None:
        rng = np.random.default_rng()

    if k < 1:
        raise ValueError("k 必须 >= 1")
    if k == 1:
        return np.zeros(len(data), dtype=int), np.array([np.mean(data)])

    init_indices = rng.choice(len(data), size=k, replace=False)
    centers = data[init_indices].copy()
    centers.sort()

    for _ in range(max_iter):
        dists = np.abs(data[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)

        new_centers = np.array([data[labels == j].mean() for j in range(k)])
        for j in range(k):
            if np.sum(labels == j) == 0:
                new_centers[j] = centers[j]

        shift = np.max(np.abs(new_centers - centers))
        centers = new_centers
        if shift < tol:
            break

    order = np.argsort(centers)
    centers_sorted = centers[order]
    label_map = {old: new for new, old in enumerate(order)}
    labels = np.array([label_map[l] for l in labels], dtype=int)
    return labels, centers_sorted


def detect_core_segment_by_area_profile(distances, areas,
                                        spike_factor=3.0,
                                        min_segment_ratio=0.15):
    """
    根据沿轴线的截面积分布，识别最可能是核心管段的连续区间。

    策略：
    1. 轻微平滑截面积序列，抑制噪声。
    2. 计算相邻截面的相对变化率。
    3. 使用类 Tukey 方法标记面积突然增大的突变位置。
    4. 在突变位置处将轴线分割为若干候选区间。
    5. 在满足最小长度比例的区间中，选择平均截面积最小的作为核心。

    Parameters
    ----------
    distances : (N,) ndarray
        各截面沿轴线的距离。
    areas : (N,) ndarray
        对应截面的总面积。
    spike_factor : float
        突变阈值倍数，越大越宽容（基于 IQR）。
    min_segment_ratio : float
        候选区间长度占总轴长的最小比例。

    Returns
    -------
    core_distances : (M,) ndarray
        核心区间包含的截面距离。
    """
    n = len(areas)
    if n < 5:
        return distances

    # 三点移动平均平滑
    areas_smooth = areas.copy()
    if n >= 3:
        areas_smooth[1:-1] = (areas[:-2] + areas[1:-1] + areas[2:]) / 3.0

    # 相邻截面相对变化率
    d_area = np.abs(np.diff(areas_smooth))
    denom = np.maximum(areas_smooth[:-1], 1e-12)
    rates = d_area / denom

    # Tukey 风格阈值：Q3 + factor * IQR
    q1, q3 = np.percentile(rates, [25, 75])
    iqr = q3 - q1
    threshold = q3 + spike_factor * iqr
    if threshold <= 0 or not np.isfinite(threshold):
        threshold = np.percentile(rates, 90)

    # 突变位置索引
    spike_indices = np.flatnonzero(rates > threshold)

    # 在突变位置后分割
    split_indices = [0] + list(spike_indices + 1) + [n]
    split_indices = sorted(set(split_indices))

    total_length = distances[-1] - distances[0]
    candidates = []
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        if end - start < 3:
            continue
        seg_areas = areas[start:end]
        seg_dists = distances[start:end]
        mean_area = np.mean(seg_areas)
        length = seg_dists[-1] - seg_dists[0]
        length_ratio = length / total_length if total_length > 0 else 0
        if length_ratio < min_segment_ratio:
            continue
        candidates.append((mean_area, start, end, seg_dists))

    if not candidates:
        return distances

    candidates.sort(key=lambda x: x[0])
    return candidates[0][3]


def point_line_distance(points, line_point, line_dir):
    """
    计算三维点到无限直线的垂直距离。

    Parameters
    ----------
    points : (N, 3) ndarray
    line_point : (3,) ndarray  直线上一点
    line_dir : (3,) ndarray    直线方向向量（无需单位化）

    Returns
    -------
    distances : (N,) ndarray
    """
    line_dir = np.asarray(line_dir, dtype=np.float64)
    line_dir = line_dir / np.linalg.norm(line_dir)
    vec = np.asarray(points, dtype=np.float64) - np.asarray(line_point, dtype=np.float64)
    proj_len = np.dot(vec, line_dir)
    perp = vec - proj_len[:, None] * line_dir
    return np.linalg.norm(perp, axis=1)


# ------------------------------------------------------------------
#  新增通用几何工具函数（气动把三分区需要）
# ------------------------------------------------------------------
def reflect_vector_across_plane(v, plane_normal):
    """
    将向量关于法向量为 plane_normal 的平面做镜像反射。
    平面过原点；若需关于任意平面镜像，先平移。
    """
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    v = np.asarray(v, dtype=np.float64)
    return v - 2.0 * np.dot(v, n) * n


def signed_distance_to_plane(points, plane_origin, plane_normal):
    """计算三维点到平面的有向距离（点在法向同侧为正）。"""
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    pts = np.asarray(points, dtype=np.float64)
    return np.dot(pts - np.asarray(plane_origin, dtype=np.float64), n)


def intersect_line_plane(line_dir, line_point, plane_origin, plane_normal):
    """
    计算无限直线与平面的交点。
    若直线与平面平行，返回 None。
    """
    d = np.asarray(line_dir, dtype=np.float64)
    p = np.asarray(line_point, dtype=np.float64)
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    denom = np.dot(d, n)
    if abs(denom) < 1e-12:
        return None
    t = np.dot(np.asarray(plane_origin, dtype=np.float64) - p, n) / denom
    return p + t * d


def average_antiparallel_directions(d1, d2):
    """
    对两个方向向量做平均，自动处理符号歧义（d 与 -d 视为同一方向）。
    """
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)
    if np.dot(d1, d2) < 0:
        d2 = -d2
    avg = d1 + d2
    return avg / (np.linalg.norm(avg) + 1e-12)


# ------------------------------------------------------------------
#  基于水密网格体素化的对称平面估计
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
#  新增：长方体/盒状物体体素化 OBB 坐标系估计
# ------------------------------------------------------------------

def normalize(v):
    """返回单位向量；零向量返回原数组。"""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def angle_between(v1, v2):
    """返回两向量间夹角（弧度），范围 [0, pi]。"""
    v1 = normalize(v1)
    v2 = normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)


def is_right_handed(u_x, u_y, u_z):
    """判断三轴是否为右手系。"""
    return np.dot(np.cross(normalize(u_x), normalize(u_y)), normalize(u_z)) > 0


def make_right_handed(u_x, u_y):
    """由 x, y 轴生成右手系的 z 轴。"""
    u_x = normalize(u_x)
    u_y = normalize(u_y - np.dot(u_y, u_x) * u_x)
    u_z = np.cross(u_x, u_y)
    nz = np.linalg.norm(u_z)
    if nz < 1e-12:
        raise ValueError("u_x 与 u_y 共线，无法构造右手系")
    return u_x, u_y, u_z / nz


def orthogonalize_axes_from_triad(u1, u2):
    """
    从两个近似正交的方向构造右手正交基。
    u1 -> x 轴，u2 投影到 u1 正交补后 -> y 轴，z = x × y。
    """
    u1 = normalize(u1)
    u2 = normalize(u2 - np.dot(u2, u1) * u1)
    u3 = np.cross(u1, u2)
    return np.vstack([u1, u2, u3])


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


# ------------------------------------------------------------------
#  新增辅助函数：侧面检测与合并
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
#  新增：基于网格几何的平面检测 + 侧壁 RANSAC 建立长方体坐标系
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
#  新增：基于网格几何的平面检测 + 二维 OBB 建立长方体坐标系（原函数保留）
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
#  新增：网格缺陷检测与修复工具
# ------------------------------------------------------------------

def compute_mesh_stats(mesh):
    """返回网格基本统计信息字典。"""
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    stats = {}
    stats['vertices'] = mesh.vertices.shape[0]
    stats['faces'] = faces.shape[0]
    stats['edges'] = mesh.edges_unique.shape[0]
    edge_face_map = {}
    for face_idx, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(face_idx)
    boundary_edges = sum(1 for faces_list in edge_face_map.values()
                         if len(faces_list) == 1)
    stats['boundary_edges'] = boundary_edges
    stats['is_watertight'] = mesh.is_watertight
    return stats


def analyze_mesh_defects(mesh):
    """
    分析网格拓扑缺陷：开放边、非流形边，以及涉及的面片。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    edge_face_map = {}
    for face_idx, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(face_idx)

    open_edges = []
    manifold_edges = []
    nonmanifold_edges = []
    for edge, face_list in edge_face_map.items():
        k = len(face_list)
        if k == 1:
            open_edges.append(edge)
        elif k == 2:
            manifold_edges.append(edge)
        else:
            nonmanifold_edges.append(edge)

    open_face_mask = np.zeros(len(faces), dtype=bool)
    nonmanifold_face_mask = np.zeros(len(faces), dtype=bool)

    for edge in open_edges:
        for fi in edge_face_map[edge]:
            open_face_mask[fi] = True
    for edge in nonmanifold_edges:
        for fi in edge_face_map[edge]:
            nonmanifold_face_mask[fi] = True

    stats = {
        'total_faces': len(faces),
        'raw_edges_count': mesh.edges.shape[0],
        'unique_edges_count': len(edge_face_map),
        'open_edges': len(open_edges),
        'manifold_edges': len(manifold_edges),
        'nonmanifold_edges': len(nonmanifold_edges),
        'open_faces': int(np.sum(open_face_mask)),
        'nonmanifold_faces': int(np.sum(nonmanifold_face_mask)),
        'both_defect_faces': int(np.sum(open_face_mask & nonmanifold_face_mask)),
        'watertight_by_count': (len(open_edges) == 0),
    }
    return stats, open_face_mask, nonmanifold_face_mask


def repair_mesh_by_removing_duplicates(mesh):
    """
    通过去除重复/退化面片来修复网格，消除部分非流形边。
    """
    print("Applying duplicate face removal...")
    orig_faces = mesh.faces.shape[0]

    unique_faces, _ = np.unique(mesh.faces, axis=0, return_inverse=True)
    if unique_faces.shape[0] < orig_faces:
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=unique_faces,
            process=True
        )

    areas = mesh.area_faces
    non_degenerate = areas > 1e-12
    if np.sum(~non_degenerate) > 0:
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[non_degenerate],
            process=True
        )

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()

    print(f"  Faces before: {orig_faces}, after: {mesh.faces.shape[0]}")
    return mesh


def repair_nonmanifold_edges(mesh, max_iterations=10, verbose=True):
    """
    策略2：对每个非流形边，保留法向最一致的两个面，删除其余面片。
    迭代直到没有非流形边（或达到迭代上限）。
    """
    for it in range(max_iterations):
        faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

        edge_face_map = {}
        for fi, face in enumerate(faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
                key = (a, b) if a < b else (b, a)
                edge_face_map.setdefault(key, []).append(fi)

        nonmanifold = {e: fl for e, fl in edge_face_map.items()
                       if len(fl) > 2}
        if not nonmanifold:
            if verbose:
                print(f"  [Iter {it}] No nonmanifold edges remain.")
            break

        if verbose:
            print(f"  [Iter {it}] {len(nonmanifold)} nonmanifold edges, "
                  f"removing extra faces...")

        normals = mesh.face_normals
        areas = mesh.area_faces
        faces_to_remove = set()

        for edge, fl in nonmanifold.items():
            if verbose:
                va, vb = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
                print(f"    edge {edge} at {va} <-> {vb}, "
                      f"shared by {len(fl)} faces")
                for fi in fl:
                    print(f"      face {fi}: area={areas[fi]:.4f}, "
                          f"normal={normals[fi].round(3)}")

            best_pair, best_key = None, -np.inf
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    dot = np.dot(normals[fl[i]], normals[fl[j]])
                    score = dot + 1e-6 * min(areas[fl[i]], areas[fl[j]])
                    if score > best_key:
                        best_key = score
                        best_pair = (fl[i], fl[j])

            for fi in fl:
                if fi not in best_pair:
                    faces_to_remove.add(fi)

        keep = np.ones(len(faces), dtype=bool)
        keep[list(faces_to_remove)] = False
        mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=faces[keep], process=False)

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    return mesh


def fill_small_holes(mesh, max_loop_edges=50, verbose=True):
    """
    用质心扇形三角化封闭小边界环。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

    edge_face_map = {}
    for fi, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(fi)
    boundary_edges = [e for e, fl in edge_face_map.items()
                      if len(fl) == 1]

    if not boundary_edges:
        if verbose:
            print("  No boundary edges, nothing to fill.")
        return mesh

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
            neighbors = [v for v in adjacency[curr] if v != prev]
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

    new_vertices = [mesh.vertices]
    new_faces = [faces]
    for loop in loops:
        if len(loop) > max_loop_edges:
            if verbose:
                print(f"  Skipping large boundary loop ({len(loop)} edges).")
            continue
        loop_pts = mesh.vertices[np.array(loop)]
        centroid = loop_pts.mean(axis=0)
        c_idx = sum(len(v) for v in new_vertices)
        new_vertices.append(centroid[None, :])
        tris = []
        for i in range(len(loop)):
            tris.append([c_idx, loop[i], loop[(i + 1) % len(loop)]])
        new_faces.append(np.array(tris))
        if verbose:
            print(f"  Filled boundary loop with {len(loop)} edges.")

    vertices = np.vstack(new_vertices)
    faces = np.vstack(new_faces)
    out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    out.fix_normals()
    return out


# ------------------------------------------------------------------
#  新增：提取开放边界环
# ------------------------------------------------------------------

def extract_boundary_loops(mesh):
    """
    提取网格的所有开放边界环。

    Returns
    -------
    loops : list of list of int
        每个元素是一个边界环的顶点索引列表。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

    edge_face_map = {}
    for fi, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(fi)

    boundary_edges = [e for e, fl in edge_face_map.items() if len(fl) == 1]

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
            neighbors = [v for v in adjacency[curr] if v != prev]
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
