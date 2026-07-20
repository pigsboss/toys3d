import numpy as np

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
