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
    """返回网格基本统计信息字典，包含顶点、面片、边、边界边及边长分布。"""
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    stats = {}
    stats['vertices'] = mesh.vertices.shape[0]
    stats['faces'] = faces.shape[0]
    stats['edges'] = mesh.edges_unique.shape[0]

    # 边长统计
    if mesh.edges_unique.shape[0] > 0:
        edge_lengths = mesh.edges_unique_length
        stats['mean_edge_length'] = float(np.mean(edge_lengths))
        stats['edge_length_p1'] = float(np.percentile(edge_lengths, 1))
        stats['edge_length_p5'] = float(np.percentile(edge_lengths, 5))
        stats['edge_length_p50'] = float(np.percentile(edge_lengths, 50))
        stats['edge_length_p95'] = float(np.percentile(edge_lengths, 95))
        stats['edge_length_p99'] = float(np.percentile(edge_lengths, 99))
    else:
        stats['mean_edge_length'] = 0.0
        stats['edge_length_p1'] = 0.0
        stats['edge_length_p5'] = 0.0
        stats['edge_length_p50'] = 0.0
        stats['edge_length_p95'] = 0.0
        stats['edge_length_p99'] = 0.0

    # 边界边统计
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
    使用基于边界边集合的 DFS，按方向连续性在分叉处选择下一条边。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

    # 构建 edge -> faces 映射，识别边界边
    edge_face_map = {}
    for fi, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(fi)

    boundary_edges = set(e for e, fl in edge_face_map.items() if len(fl) == 1)

    if not boundary_edges:
        if verbose:
            print("  No boundary edges, nothing to fill.")
        return mesh

    # 稳健地提取所有边界环
    loops = []
    edge_set = set(boundary_edges)

    while edge_set:
        e0 = edge_set.pop()
        v_start, v_curr = e0
        loop = [v_start, v_curr]
        v_prev = v_start

        while True:
            # 找与 v_curr 相连且未访问的边界边
            candidates = [e for e in edge_set if v_curr in e]

            if not candidates:
                # 无法闭合，放弃这条路径
                if verbose and len(loop) > 2:
                    print(f"  Dropped unclosed boundary path ({len(loop)} edges)")
                break

            # 如果有多个候选，按方向连续性选择最自然的延续
            if len(candidates) > 1:
                dir_curr = mesh.vertices[v_curr] - mesh.vertices[v_prev]
                dir_curr = dir_curr / (np.linalg.norm(dir_curr) + 1e-12)

                best_edge = None
                best_score = -np.inf
                for e in candidates:
                    v_next = e[0] if e[1] == v_curr else e[1]
                    dir_next = mesh.vertices[v_next] - mesh.vertices[v_curr]
                    dn = np.linalg.norm(dir_next)
                    if dn < 1e-12:
                        continue
                    dir_next = dir_next / dn

                    # 偏好与当前方向夹角最小的延续
                    dot = np.dot(dir_curr, dir_next)
                    # 惩罚反向转弯
                    score = dot if dot >= 0 else -0.5 * dot
                    if score > best_score:
                        best_score = score
                        best_edge = e
                next_edge = best_edge
            else:
                next_edge = candidates[0]

            edge_set.remove(next_edge)
            v_next = next_edge[0] if next_edge[1] == v_curr else next_edge[1]
            loop.append(v_next)

            if v_next == v_start:
                # 成功闭合
                loops.append(loop[:-1])  # 去掉重复的起点
                break

            v_prev, v_curr = v_curr, v_next

            # 安全上限，防止异常拓扑导致无限循环
            if len(loop) > max(max_loop_edges * 3, 500):
                if verbose:
                    print(f"  Dropped overly long boundary path ({len(loop)} edges)")
                break

    # 扇形封闭找到的边界环
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


# ------------------------------------------------------------------
#  新增：薄板分割核心函数（向量化实现）
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


def compute_g1_deviation(mesh, face_i, face_j):
    """
    计算两个相邻面片间的 G1 光滑偏差（二面角，弧度）。
    """
    n1 = mesh.face_normals[face_i]
    n2 = mesh.face_normals[face_j]
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.arccos(dot)


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


# ==================================================================
#  新增：薄壳处理核心函数 (原 shell.py 中已用)
# ==================================================================

# ------------------------------------------------------------------
#  基于 k-d 树的厚度估计
# ------------------------------------------------------------------

def estimate_shell_thickness(mesh, grid_size=128, margin=1.05, k_neighbors=20):
    """
    基于 k-d 树最近邻搜索估计每个面片的局部厚度。

    对每个面片，查找其最近邻中法向反平行的面片，
    以两中心距离近似壁厚。无法找到时厚度为 NaN。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    grid_size : int
        保留参数（兼容旧接口）。
    margin : float
        保留参数（兼容旧接口）。
    k_neighbors : int
        每个面片检查的最近邻数量（默认20）。

    Returns
    -------
    thickness : (N,) ndarray
    reliability : (N,) bool
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

    Parameters
    ----------
    thickness : (N,) ndarray
        每个面片的厚度估计（可含 NaN）。
    mode : 'adaptive' | 'absolute'
        adaptive 时 threshold 为全局中位数的比例；
        absolute 时 threshold 为绝对厚度值。
    threshold : float
        阈值（比例或绝对值）。
    fallback_median : float or None
        adaptive 模式下，若无法取到中位数时的回退值。

    Returns
    -------
    mask : (N,) bool
        厚度被认为过小的面片掩码。
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

    Returns
    -------
    stats : dict
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

    Parameters
    ----------
    mesh : trimesh.Trimesh
    plate_mask : (N,) bool
        属于某一块板（或合并后的板）的面片掩码。

    Returns
    -------
    loops : list of list of int
        每个边界环的顶点索引列表。
    """
    faces = np.asarray(mesh.faces).reshape(-1, 3)
    mask = np.asarray(plate_mask, dtype=bool)
    plate_indices = np.flatnonzero(mask)

    if len(plate_indices) == 0:
        return []

    # 统计子网格内部每条边的出现次数
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


def extract_crease_lines(mesh, plate_labels, dihedral_thr_deg=30.0):
    """
    提取相邻薄板之间的折痕线（二面角较大的内部边链）。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    plate_labels : (N,) ndarray
        面片的薄板标签。
    dihedral_thr_deg : float
        折痕判定二面角阈值（度）。

    Returns
    -------
    chains : list of list of int
        每条折痕链的顶点索引序列。
    """
    faces = np.asarray(mesh.faces).reshape(-1, 3)
    edge_face_map = {}
    for fi, face in enumerate(faces):
        for j in range(3):
            a, b = int(face[j]), int(face[(j + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(fi)

    angle_thr = np.deg2rad(dihedral_thr_deg)
    crease_edges = []

    for edge, fl in edge_face_map.items():
        if len(fl) != 2:
            continue
        fi, fj = fl
        if plate_labels[fi] == plate_labels[fj]:
            continue
        angle = compute_g1_deviation(mesh, fi, fj)
        if angle > angle_thr:
            crease_edges.append(edge)

    if not crease_edges:
        return []

    adjacency = {}
    for a, b in crease_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    visited = set(crease_edges)
    chains = []

    for edge in crease_edges:
        if edge not in visited:
            continue
        visited.remove(edge)

        chain = [edge[0], edge[1]]

        # 向前延伸
        end = edge[1]
        while True:
            nxt_list = [v for v in adjacency.get(end, []) if v != chain[-2]]
            if len(nxt_list) != 1:
                break
            nxt = nxt_list[0]
            e2 = tuple(sorted((end, nxt)))
            if e2 not in visited:
                break
            visited.remove(e2)
            chain.append(nxt)
            end = nxt

        # 向后延伸
        begin = chain[0]
        while True:
            nxt_list = [v for v in adjacency.get(begin, []) if v != chain[1]]
            if len(nxt_list) != 1:
                break
            nxt = nxt_list[0]
            e2 = tuple(sorted((begin, nxt)))
            if e2 not in visited:
                break
            visited.remove(e2)
            chain.insert(0, nxt)
            begin = nxt

        # 闭合环去重
        if len(chain) > 1 and chain[0] == chain[-1]:
            chain = chain[:-1]

        chains.append(chain)

    return chains


# ------------------------------------------------------------------
#  薄壳处理：P1 边缘规律性分类
# ------------------------------------------------------------------

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


def fit_spline_3d(points, degree=3, num_samples=100):
    """
    三维 B 样条拟合。若 scipy 不可用或点数不足，返回 inf。

    Returns
    -------
    rmse : float
    fitted : (num_samples, 3) ndarray or None
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

        # 用最近拟合点距离近似 RMSE
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

    Returns
    -------
    label : 'line' | 'circle' | 'spline' | 'irregular'
    score : float
        归一化拟合误差。
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

# ------------------------------------------------------------------
#  多尺度法向、边缘检测与基于边缘的区域分割
# ------------------------------------------------------------------

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

    Parameters
    ----------
    mesh : trimesh.Trimesh
    scales : tuple of int
        邻域 ring 数

    Returns
    -------
    scale_normals : list of (N, 3) ndarray
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
                # 退化情况：等权平均
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

    Returns
    -------
    strengths : (n_scales, N) ndarray
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

    只有在多个尺度上都表现出强边缘响应的位置，才被认为是真实边缘。
    同时要求小尺度响应不低于大尺度响应（避免纯噪声）。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    scales : tuple of int
    threshold_ratio : float
        相对阈值，边缘强度超过该比例的最大值才被考虑
    min_consistent_scales : int or None
        要求一致超过阈值的尺度数，默认 ceil(n_scales / 2)

    Returns
    -------
    edge_face_mask : (N,) bool
        位于边缘上的面片掩码
    edge_strengths : (n_scales, N) ndarray
        各尺度边缘强度
    """
    scale_normals = compute_multiscale_face_normals(mesh, scales=scales)
    strengths = compute_edge_strength_multiscale(mesh, scale_normals)

    n_scales, N = strengths.shape
    if min_consistent_scales is None:
        min_consistent_scales = max(2, (n_scales + 1) // 2)

    # 相对阈值：超过最大值的 threshold_ratio
    max_val = np.max(strengths, axis=1, keepdims=True) + 1e-12
    normalized = strengths / max_val

    # 一致性：多个尺度超过阈值
    consistent = np.sum(normalized > threshold_ratio, axis=0) >= min_consistent_scales

    # 单调性：随着尺度增大，边缘响应不增强（真实边缘在所有尺度都明显）
    if n_scales >= 2:
        monotonic = np.all(np.diff(strengths, axis=0) <= 0, axis=0)
    else:
        monotonic = np.ones(N, dtype=bool)

    edge_face_mask = consistent & monotonic

    return edge_face_mask, strengths


def segment_regions_by_edges(mesh, edge_face_mask):
    """
    移除边缘面片后，对剩余面片做连通分量分割。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    edge_face_mask : (N,) bool
        边缘面片掩码

    Returns
    -------
    labels : (N,) ndarray
        区域标签。边缘面片标记为 -1。
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    N = len(mesh.faces)
    adjacency = build_face_adjacency(mesh)

    valid_mask = ~np.asarray(edge_face_mask, dtype=bool)
    valid_indices = np.flatnonzero(valid_mask)
    n_valid = len(valid_indices)

    # 旧索引 -> 新索引映射
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

# ------------------------------------------------------------------
#  代理网格与局部法向聚类薄板分割
# ------------------------------------------------------------------

def build_proxy_mesh(mesh, target_faces=50000, max_edge_length=None,
                     iterations=2, smooth=False):
    """
    从原始扫描网格构建均匀、低分辨率、近似水密的代理网格。
    采用「修复 → 简化 → 可选细分」的单次流程，避免面数爆炸。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    target_faces : int
        目标面片数
    max_edge_length : float or None
        最大允许边长，默认取包围盒对角线的 2%。
    iterations : int   (当前未使用，保留兼容)
    smooth : bool
        是否做轻微 Laplacian 平滑
    """
    from toys3d.geometrics import (
        repair_mesh_by_removing_duplicates,
        repair_nonmanifold_edges,
        fill_small_holes,
        analyze_mesh_defects,
    )

    proxy = mesh.copy()

    # 步骤 0：修复网格（最多 3 次迭代），消除开放边和大部分非流形边
    defect_stats, _, _ = analyze_mesh_defects(proxy)
    if defect_stats['open_edges'] > 0 or defect_stats['nonmanifold_edges'] > 0:
        for _ in range(3):
            proxy = repair_mesh_by_removing_duplicates(proxy)
            proxy = repair_nonmanifold_edges(proxy, verbose=False)
            proxy = fill_small_holes(proxy, max_loop_edges=50, verbose=False)
            defect_stats, _, _ = analyze_mesh_defects(proxy)
            if defect_stats['open_edges'] == 0 and defect_stats['nonmanifold_edges'] == 0:
                break

    # 步骤 1：简化到目标面数
    if len(proxy.faces) > target_faces:
        proxy = proxy.simplify_quadric_decimation(face_count=target_faces)

    # 步骤 2：如果需要且合理，细分一次（避免面数爆炸）
    if max_edge_length is None:
        diag = float(np.linalg.norm(proxy.bounding_box.extents))
        max_edge_length = diag * 0.02
    if max_edge_length > 0 and len(proxy.edges_unique) > 0:
        mean_edge = float(np.mean(proxy.edges_unique_length))
        if mean_edge > max_edge_length * 1.5:
            # 预估细分后面数，防止爆炸
            est_factor = (mean_edge / max_edge_length) ** 2
            if len(proxy.faces) * est_factor < max(target_faces * 4, 200000):
                proxy = proxy.subdivide_to_size(max_edge_length)

    # 步骤 3：清理
    proxy.merge_vertices()
    # 去除重复面片
    faces = np.asarray(proxy.faces, dtype=np.int64)
    unique_faces = np.unique(faces, axis=0)
    if unique_faces.shape[0] < faces.shape[0]:
        proxy = trimesh.Trimesh(vertices=proxy.vertices,
                                faces=unique_faces,
                                process=False)
    # 去除退化面片
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

    对每个面片，以其中心为球心、radius 为半径搜索邻域面片。
    这些面片可能来自薄板中心面的两侧（内/外表面），
    也可能来自相邻薄板。通过无符号法向聚类识别薄板交界。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    radius : float or None
        球邻域半径。默认 = 4 * 厚度中位数。
    cluster_angle_deg : float
        同一法向簇最大夹角（度）
    min_faces : int
        薄板最小面片数

    Returns
    -------
    labels : (N,) ndarray
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

    # 自动估计半径
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
        print(f"    thickness median={med:.4f}, radius={radius:.4f} (1.5x thickness)")

    # 球状邻域查询
    print("  Building k-d tree for face centers...")
    t0 = time.time()
    tree = cKDTree(centers)
    print(f"    k-d tree built in {time.time() - t0:.2f}s")

    print(f"  Querying ball neighbors (radius={radius:.4f})...")
    t0 = time.time()
    neighbors_list = tree.query_ball_point(centers, radius)
    elapsed = time.time() - t0
    avg_neighbors = np.mean([len(n) for n in neighbors_list])
    max_neighbors = max(len(n) for n in neighbors_list)
    print(f"    ball query done in {elapsed:.2f}s, "
          f"avg_neighbors={avg_neighbors:.1f}, max_neighbors={max_neighbors}")

    cos_thr = np.cos(np.deg2rad(cluster_angle_deg))
    boundary_mask = np.zeros(N, dtype=bool)

    print("  Clustering normals per face...")
    t0 = time.time()
    report_interval = max(1, N // 10)
    for fi in range(N):
        if fi % report_interval == 0:
            print(f"    clustering {fi}/{N} ({100 * fi / N:.0f}%) "
                  f"+{time.time() - t0:.2f}s")

        neighbors = neighbors_list[fi]
        if len(neighbors) < 3:
            continue

        n_list = normals[neighbors]
        clusters = []

        for n in n_list:
            n_unit = n / (np.linalg.norm(n) + 1e-12)
            added = False
            for cl in clusters:
                if abs(np.dot(n_unit, cl.mean_axis)) >= cos_thr:
                    cl.add(n_unit)
                    added = True
                    break
            if not added:
                clusters.append(_NormalCluster(n_unit))

        # 两个及以上法向簇：位于薄板交界
        if len(clusters) >= 2:
            boundary_mask[fi] = True

    print(f"    clustering done in {time.time() - t0:.2f}s, "
          f"boundary faces={np.sum(boundary_mask)}")

    # 移除边界后面片做连通分量
    print("  Segmenting connected components...")
    t0 = time.time()
    valid_mask = ~boundary_mask
    valid_indices = np.flatnonzero(valid_mask)
    n_valid = len(valid_indices)

    labels = np.full(N, -1, dtype=int)
    if n_valid > 0:
        # 构建面片邻接图（拓扑邻接，用于连通分量）
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

    print(f"    components done in {time.time() - t0:.2f}s")

    # 将边界面片重新分配到相邻薄板
    print("  Reassigning boundary faces...")
    t0 = time.time()
    adjacency = build_face_adjacency(mesh)
    for fi in range(N):
        if labels[fi] != -1:
            continue

        best_label = -1
        best_sim = -np.inf
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

    print(f"    reassignment done in {time.time() - t0:.2f}s")

    # 合并小区域
    print("  Merging small regions...")
    t0 = time.time()
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

    # 压缩标签
    valid = labels >= 0
    if np.any(valid):
        _, new_labels = np.unique(labels[valid], return_inverse=True)
        labels[valid] = new_labels

    print(f"    merge done in {time.time() - t0:.2f}s, "
          f"n_plates={labels.max() + 1}")

    return labels


# ==================================================================
#  Plane fitting based segmentation (RANSAC in ball)
# ==================================================================

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


def _spatial_split_inliers(centers, inlier_mask, adjacency, ball_face_set):
    """Split inlier faces into spatial connected components using face adjacency list."""
    idx = np.where(inlier_mask)[0]
    if len(idx) <= 1:
        return [inlier_mask]

    local_map = {global_id: i for i, global_id in enumerate(idx)}
    n_in = len(idx)

    rows, cols = [], []
    for i, global_id in enumerate(idx):
        # adjacency is a list of lists
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

    Returns labels (0..k-1), -1 for unassigned.
    """
    from scipy.spatial import cKDTree
    import time

    centers = mesh.triangles_center
    N = len(centers)
    if N == 0:
        return np.zeros(0, dtype=int)

    adjacency = build_face_adjacency(mesh)

    print("  Building k-d tree for centers...")
    t0 = time.time()
    tree = cKDTree(centers)
    print(f"    k-d tree built in {time.time() - t0:.2f}s")

    print(f"  Querying ball neighbors (radius={radius:.4f})...")
    t0 = time.time()
    neighbors_list = tree.query_ball_point(centers, radius)
    elapsed = time.time() - t0
    avg_n = np.mean([len(n) for n in neighbors_list])
    print(f"    ball query done in {elapsed:.2f}s, avg_neighbors={avg_n:.1f}")

    boundary_mask = np.zeros(N, dtype=bool)
    if rng is None:
        rng = np.random.default_rng()

    print("  Detecting boundaries by plane fitting...")
    t0 = time.time()
    report_interval = max(1, N // 10)
    for fi in range(N):
        if fi % report_interval == 0:
            print(f"    processing {fi}/{N} ({100 * fi / N:.0f}%) "
                  f"+{time.time() - t0:.2f}s")
        ball_indices = neighbors_list[fi]
        if len(ball_indices) < 6:
            continue
        boundary_mask[fi] = detect_boundary_ball(
            centers, ball_indices, adjacency,
            max_planes=max_planes,
            inlier_threshold=inlier_threshold, max_iter=300, rng=rng
        )
    print(f"    boundary detection done in {time.time() - t0:.2f}s, "
          f"boundary faces={np.sum(boundary_mask)}")

    # segment connected components of non-boundary faces
    print("  Segmenting non-boundary faces...")
    t0 = time.time()
    valid_mask = ~boundary_mask
    valid_indices = np.flatnonzero(valid_mask)
    n_valid = len(valid_indices)
    labels = np.full(N, -1, dtype=int)

    if n_valid > 0:
        remap = np.full(N, -1, dtype=np.int64)
        remap[valid_indices] = np.arange(n_valid)

        rows, cols = [], []
        for i in valid_indices:
            for j in adjacency[i]:
                if j > i and valid_mask[j]:
                    ii, jj = remap[i], remap[j]
                    rows.extend([ii, jj])
                    cols.extend([jj, ii])

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        if len(rows) > 0:
            graph = csr_matrix((np.ones(len(rows), dtype=np.int8),
                                (rows, cols)), shape=(n_valid, n_valid))
            _, comps = connected_components(graph, directed=False)
            labels[valid_indices] = comps
        else:
            labels[valid_indices] = np.arange(n_valid)
    print(f"    components done in {time.time() - t0:.2f}s")

    # reassign boundary faces to nearest plate
    print("  Reassigning boundary faces...")
    t0 = time.time()
    normals = mesh.face_normals
    for fi in range(N):
        if labels[fi] != -1:
            continue
        best_label = -1
        best_dot = -1.0
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
    print(f"    reassignment done in {time.time() - t0:.2f}s")

    # merge small components
    print("  Merging small regions...")
    t0 = time.time()
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

    # compact labels
    valid = labels >= 0
    if np.any(valid):
        _, new_labels = np.unique(labels[valid], return_inverse=True)
        labels[valid] = new_labels

    n_plates = labels.max() + 1 if np.any(labels >= 0) else 0
    print(f"    merge done in {time.time() - t0:.2f}s, n_plates={n_plates}")
    return labels


# ------------------------------------------------------------------
#  自适应孔洞填补
# ------------------------------------------------------------------

def build_vertex_neighbors(faces, n_vertices):
    """
    从面片索引构建每个顶点的 1-ring 拓扑邻域顶点集合。
    """
    neighbors = [set() for _ in range(n_vertices)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])
    return neighbors


def compute_loop_flatness(mesh, loop):
    """
    计算边界环的平坦度。

    Returns
    -------
    flatness : float  [0, 1]
        越接近 0 表示越平坦（近似共面），越接近 1 表示越弯曲。
    relative_rmse : float
        平面拟合残差相对于边界环包围盒对角线的比例。
    """
    pts = mesh.vertices[np.array(loop)]
    if len(pts) < 3:
        return 1.0, 1.0

    center = pts.mean(axis=0)
    centered = pts - center
    _, s, _ = np.linalg.svd(centered, full_matrices=False)

    if len(s) < 3 or s[0] < 1e-12:
        return 0.0, 0.0

    flatness = float(s[2] / (s.sum() + 1e-12))
    rmse = float(np.sqrt(s[2] / len(pts)))
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    relative_rmse = rmse / (diag + 1e-12)

    return flatness, relative_rmse


def project_loop_to_plane(mesh, loop):
    """
    把边界环投影到最佳拟合切平面，返回 2D 坐标和平面基。
    """
    pts = mesh.vertices[np.array(loop)]
    center = pts.mean(axis=0)
    centered = pts - center
    _, s, vh = np.linalg.svd(centered, full_matrices=False)

    plane_normal = vh[2]
    if np.linalg.norm(plane_normal) < 1e-12:
        plane_normal = np.array([0.0, 0.0, 1.0])

    if abs(plane_normal[2]) < 0.9:
        basis_u = np.cross(plane_normal, [0, 0, 1])
    else:
        basis_u = np.cross(plane_normal, [1, 0, 0])
    basis_u = basis_u / (np.linalg.norm(basis_u) + 1e-12)
    basis_v = np.cross(plane_normal, basis_u)

    pts_2d = np.column_stack([
        np.dot(centered, basis_u),
        np.dot(centered, basis_v)
    ])

    return pts_2d, basis_u, basis_v, center, plane_normal


def _triangle_area_2d(a, b, c):
    """计算 2D 三角形有向面积。"""
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) -
                  (b[1] - a[1]) * (c[0] - a[0]))


def _is_convex_2d(pts_2d, prev, curr, next_idx, ccw=True):
    """判断 curr 是否为凸顶点。"""
    a, b, c = pts_2d[prev], pts_2d[curr], pts_2d[next_idx]
    area = _triangle_area_2d(a, b, c)
    return (area > 1e-12) if ccw else (area < -1e-12)


def _point_in_triangle_2d(p, a, b, c):
    """判断点 p 是否在三角形 abc 内部（含边界）。"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
               (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def ear_clip_polygon(pts_2d, min_area=1e-12):
    """
    对 2D 简单多边形做 Ear Clipping 三角化。

    Returns
    -------
    triangles : list of [i, j, k]
        三角形索引列表（相对于输入 pts_2d 的索引）。
    """
    n = len(pts_2d)
    if n < 3:
        return []
    if n == 3:
        return [[0, 1, 2]]

    indices = list(range(n))
    triangles = []

    signed_area = 0.0
    for i in range(n):
        x1, y1 = pts_2d[i]
        x2, y2 = pts_2d[(i + 1) % n]
        signed_area += (x1 * y2 - x2 * y1)
    ccw = signed_area > 0

    max_steps = n * n
    for _ in range(max_steps):
        n_curr = len(indices)
        if n_curr <= 3:
            break

        ear_found = False
        for i in range(n_curr):
            prev = indices[(i - 1) % n_curr]
            curr = indices[i]
            next_idx = indices[(i + 1) % n_curr]

            a, b, c = pts_2d[prev], pts_2d[curr], pts_2d[next_idx]
            area = abs(_triangle_area_2d(a, b, c))
            if area < min_area:
                continue

            if not _is_convex_2d(pts_2d, prev, curr, next_idx, ccw):
                continue

            has_inside = False
            for j in range(n):
                if j == prev or j == curr or j == next_idx:
                    continue
                if _point_in_triangle_2d(pts_2d[j], a, b, c):
                    has_inside = True
                    break

            if has_inside:
                continue

            triangles.append([prev, curr, next_idx])
            indices.pop(i)
            ear_found = True
            break

        if not ear_found:
            triangles.append([indices[0], indices[1], indices[2]])
            indices.pop(1)

    if len(indices) == 3:
        triangles.append([indices[0], indices[1], indices[2]])

    return triangles


def fill_loop_fan(mesh, loop):
    """用质心扇形填补一个边界环。"""
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    vertices = mesh.vertices.copy()

    loop_pts = vertices[np.array(loop)]
    centroid = loop_pts.mean(axis=0)
    c_idx = len(vertices)
    vertices = np.vstack([vertices, centroid[None, :]])

    tris = []
    for i in range(len(loop)):
        tris.append([c_idx, loop[i], loop[(i + 1) % len(loop)]])

    faces = np.vstack([faces, np.array(tris)])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def fill_loop_earclip(mesh, loop):
    """用局部投影 + Ear Clipping 填补一个边界环。"""
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    vertices = mesh.vertices.copy()

    pts_2d, _, _, _, _ = project_loop_to_plane(mesh, loop)
    tri_indices = ear_clip_polygon(pts_2d)

    tris = []
    for t in tri_indices:
        tris.append([loop[t[0]], loop[t[1]], loop[t[2]]])

    faces = np.vstack([faces, np.array(tris)])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def fill_loop_surface_fit(mesh, loop, num_samples=None, g2=False):
    """
    用局部二次曲面拟合 + 规则采样填补大曲面孔洞。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    loop : list of int
        边界环顶点索引。
    num_samples : int or None
        内部采样点数量。
    g2 : bool
        是否使用 G2 光滑曲面拟合。当前预留接口，未实现；设置 True 时
        抛出 NotImplementedError。
    """
    if g2:
        raise NotImplementedError(
            "G2 smooth surface fitting is reserved but not yet implemented. "
            "Use g2=False for quadratic surface fitting."
        )
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    vertices = mesh.vertices.copy()

    pts_2d, basis_u, basis_v, origin, plane_normal = project_loop_to_plane(mesh, loop)

    adjacency = build_vertex_neighbors(faces, len(vertices))
    neighbor_set = set(loop)
    for v in loop:
        neighbor_set.update(adjacency[v])

    fit_pts = []
    fit_z = []
    for v in neighbor_set:
        vec = vertices[v] - origin
        x = np.dot(vec, basis_u)
        y = np.dot(vec, basis_v)
        z = np.dot(vec, plane_normal)
        fit_pts.append([x, y])
        fit_z.append(z)

    fit_pts = np.array(fit_pts)
    fit_z = np.array(fit_z)

    A = np.column_stack([
        fit_pts[:, 0]**2,
        fit_pts[:, 1]**2,
        fit_pts[:, 0] * fit_pts[:, 1],
        fit_pts[:, 0],
        fit_pts[:, 1],
        np.ones(len(fit_pts))
    ])
    coeffs, *_ = np.linalg.lstsq(A, fit_z, rcond=None)

    if num_samples is None:
        num_samples = max(20, int(len(loop) * 0.5))

    from scipy.spatial import Delaunay
    boundary_2d = pts_2d
    xmin, ymin = boundary_2d.min(axis=0)
    xmax, ymax = boundary_2d.max(axis=0)

    rng = np.random.default_rng(42)
    samples = []
    max_attempts = num_samples * 20
    attempts = 0
    while len(samples) < num_samples and attempts < max_attempts:
        p = rng.uniform([xmin, ymin], [xmax, ymax])
        attempts += 1
        inside = False
        n = len(boundary_2d)
        for i in range(n):
            x1, y1 = boundary_2d[i]
            x2, y2 = boundary_2d[(i + 1) % n]
            if ((y1 > p[1]) != (y2 > p[1])) and \
               (p[0] < (x2 - x1) * (p[1] - y1) / (y2 - y1 + 1e-12) + x1):
                inside = not inside
        if inside:
            samples.append(p)

    if len(samples) < 3:
        return fill_loop_earclip(mesh, loop)

    samples = np.array(samples)

    all_2d = np.vstack([boundary_2d, samples])

    tri = Delaunay(all_2d)

    valid_tris = []
    for simplex in tri.simplices:
        tri_pts = all_2d[simplex]
        centroid_2d = tri_pts.mean(axis=0)
        inside = False
        n = len(boundary_2d)
        for i in range(n):
            x1, y1 = boundary_2d[i]
            x2, y2 = boundary_2d[(i + 1) % n]
            if ((y1 > centroid_2d[1]) != (y2 > centroid_2d[1])) and \
               (centroid_2d[0] < (x2 - x1) * (centroid_2d[1] - y1) /
                (y2 - y1 + 1e-12) + x1):
                inside = not inside
        if inside:
            valid_tris.append(simplex)

    new_vertices = []
    for p in samples:
        z = (coeffs[0] * p[0]**2 + coeffs[1] * p[1]**2 +
             coeffs[2] * p[0] * p[1] + coeffs[3] * p[0] +
             coeffs[4] * p[1] + coeffs[5])
        pos_3d = origin + p[0] * basis_u + p[1] * basis_v + z * plane_normal
        new_vertices.append(pos_3d)

    if new_vertices:
        new_vertices = np.array(new_vertices)
        vertex_offset = len(vertices)
        vertices = np.vstack([vertices, new_vertices])

    tris = []
    for simplex in valid_tris:
        v0 = vertex_offset + simplex[0] - len(loop) if simplex[0] >= len(loop) else loop[simplex[0]]
        v1 = vertex_offset + simplex[1] - len(loop) if simplex[1] >= len(loop) else loop[simplex[1]]
        v2 = vertex_offset + simplex[2] - len(loop) if simplex[2] >= len(loop) else loop[simplex[2]]
        tris.append([v0, v1, v2])

    faces = np.vstack([faces, np.array(tris)])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def fill_holes_adaptive(mesh,
                        strategy='flatness',
                        max_fan_edges=15,
                        max_fan_flatness=0.05,
                        max_earclip_flatness=0.15,
                        max_surface_fit_edges=500,
                        max_surface_fit_flatness=0.40,
                        edge_count_small_p=50.0,
                        edge_count_large_p=95.0,
                        g2=False,
                        verbose=True):
    """
    自适应孔洞填补：按平坦度和边界范围选择策略。
    """
    loops = extract_boundary_loops(mesh)

    fan_loops = []
    earclip_loops = []
    surface_fit_loops = []
    skipped_loops = []

    if strategy == 'edge-count':
        if len(loops) == 0:
            small_thr = max(max_fan_edges, 3)
            large_thr = max(max_surface_fit_edges, small_thr + 1)
        else:
            lengths = np.array([len(loop) for loop in loops], dtype=float)
            small_thr = max(int(np.percentile(lengths, edge_count_small_p,
                                              method='lower')), 3)
            large_thr = max(int(np.percentile(lengths, edge_count_large_p,
                                              method='lower')), small_thr + 1)

        for loop in loops:
            n = len(loop)
            if n <= small_thr:
                fan_loops.append(loop)
            elif n <= large_thr:
                earclip_loops.append(loop)
            elif n <= max_surface_fit_edges:
                surface_fit_loops.append(loop)
            else:
                skipped_loops.append(loop)

        if verbose:
            print(f"  Strategy: edge-count")
            print(f"  Thresholds: small<={small_thr}, "
                  f"medium<={large_thr}, large<={max_surface_fit_edges}")

    else:  # flatness
        for loop in loops:
            n_edges = len(loop)
            flatness, _ = compute_loop_flatness(mesh, loop)

            if n_edges <= max_fan_edges and flatness <= max_fan_flatness:
                fan_loops.append(loop)
            elif flatness <= max_earclip_flatness and n_edges <= max_surface_fit_edges:
                earclip_loops.append(loop)
            elif flatness <= max_surface_fit_flatness and n_edges <= max_surface_fit_edges:
                surface_fit_loops.append(loop)
            else:
                skipped_loops.append(loop)

        if verbose:
            print(f"  Strategy: flatness")

    if verbose:
        print(f"  Boundary loops: {len(loops)}")
        print(f"    fan fill: {len(fan_loops)}")
        print(f"    ear clip: {len(earclip_loops)}")
        print(f"    surface fit: {len(surface_fit_loops)}")
        print(f"    skipped: {len(skipped_loops)}")

    result = mesh.copy()
    for loop in fan_loops:
        result = fill_loop_fan(result, loop)
    for loop in earclip_loops:
        result = fill_loop_earclip(result, loop)
    for loop in surface_fit_loops:
        result = fill_loop_surface_fit(result, loop, g2=g2)

    result = result.copy()
    result.remove_unreferenced_vertices()
    result.fix_normals()
    return result


# ------------------------------------------------------------------
#  孤立/准孤立结构移除
# ------------------------------------------------------------------

def remove_isolated_components(mesh, min_faces=20, min_area=None,
                               min_ratio=0.001, verbose=True):
    """
    删除与主体不连通的小型孤立结构。
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    N = len(mesh.faces)
    if N <= 1:
        return mesh.copy()

    adjacency = build_face_adjacency(mesh)

    rows, cols = [], []
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if j > i:
                rows.extend([i, j])
                cols.extend([j, i])

    data = np.ones(len(rows), dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(N, N))
    _, labels = connected_components(graph, directed=False)

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) <= 1:
        return mesh.copy()

    largest_label = unique_labels[np.argmax(counts)]

    area_faces = mesh.area_faces
    delete_mask = np.zeros(N, dtype=bool)

    for lbl in unique_labels:
        if lbl == largest_label:
            continue

        comp_idx = np.flatnonzero(labels == lbl)
        count = len(comp_idx)
        area = float(np.sum(area_faces[comp_idx]))
        ratio = count / N

        if (count < min_faces or
                (min_area is not None and area < min_area) or
                ratio < min_ratio):
            delete_mask[comp_idx] = True

    if verbose:
        n_removed = int(np.sum(delete_mask))
        print(f"  Isolated components removed: "
              f"{len(unique_labels) - 1}, faces deleted: {n_removed}")

    if not np.any(delete_mask):
        return mesh.copy()

    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[~delete_mask],
        process=False,
    )
    new_mesh.remove_unreferenced_vertices()
    return new_mesh


def remove_quasi_isolated_components(mesh, radius=None, n_sample=2000,
                                     min_faces=30, max_ratio=0.05,
                                     remove_bridge=False, rng=None,
                                     verbose=True):
    """
    删除准孤立结构。

    准孤立结构定义：
        存在一个球状邻域 B(c,r)，如果删除 B 内所有面片，
        会使得一个原本与主体连通的小结构 S 变成孤立结构。
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    import time

    N = len(mesh.faces)
    if N <= min_faces:
        return mesh.copy()

    adjacency = build_face_adjacency(mesh)
    rows, cols = [], []
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if j > i:
                rows.extend([i, j])
                cols.extend([j, i])
    data = np.ones(len(rows), dtype=np.int8)
    full_adj = csr_matrix((data, (rows, cols)), shape=(N, N))

    centers = mesh.triangles_center

    if radius is None:
        median_edge = float(np.median(mesh.edges_unique_length)) \
            if len(mesh.edges_unique_length) > 0 else 1.0
        bbox_diag = float(np.linalg.norm(mesh.bounding_box.extents))
        radius = max(5.0 * median_edge, 3.0 * bbox_diag / 1000.0)

    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")

    rng = np.random.default_rng(rng) if rng is None else rng
    n_sample = min(n_sample, N)
    sample_idx = rng.choice(N, size=n_sample, replace=False)

    tree = cKDTree(centers)
    max_bridge_ratio = 0.2
    max_bridge_faces = int(max_bridge_ratio * N)
    min_bridge_faces = 3

    delete_mask = np.zeros(N, dtype=bool)
    bridge_delete_mask = np.zeros(N, dtype=bool)
    candidate_count = 0

    if verbose:
        print(f"  Sampling {n_sample} candidate balls "
              f"(radius={radius:.4f})...")
        t0 = time.time()
        report_interval = max(1, n_sample // 10)

    for k, fi in enumerate(sample_idx):
        if verbose and k % report_interval == 0:
            print(f"    evaluating {k}/{n_sample} "
                  f"({100 * k / n_sample:.0f}%) +{time.time() - t0:.2f}s")

        ball = tree.query_ball_point(centers[fi], r=radius)
        nb = len(ball)
        if nb < min_bridge_faces or nb > max_bridge_faces:
            continue

        remaining_mask = np.ones(N, dtype=bool)
        remaining_mask[list(ball)] = False
        remaining_idx = np.flatnonzero(remaining_mask)
        if len(remaining_idx) == 0:
            continue

        sub_adj = full_adj[remaining_idx][:, remaining_idx]
        _, comps = connected_components(sub_adj, directed=False)

        unique_comps, comp_counts = np.unique(comps, return_counts=True)
        if len(unique_comps) <= 1:
            continue

        main_comp = unique_comps[np.argmax(comp_counts)]

        for comp in unique_comps:
            if comp == main_comp:
                continue
            count = int(comp_counts[comp])
            if count < min_faces or (count / N) > max_ratio:
                continue

            comp_original_faces = remaining_idx[comps == comp]
            delete_mask[comp_original_faces] = True
            candidate_count += 1

            if remove_bridge:
                bridge_delete_mask[list(ball)] = True

    if verbose:
        print(f"  Candidate quasi-isolated structures found: "
              f"{candidate_count}")

    if remove_bridge:
        delete_mask |= bridge_delete_mask

    if not np.any(delete_mask):
        return mesh.copy()

    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[~delete_mask],
        process=False,
    )
    new_mesh.remove_unreferenced_vertices()
    return new_mesh
