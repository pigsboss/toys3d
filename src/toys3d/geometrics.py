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
