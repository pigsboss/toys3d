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
