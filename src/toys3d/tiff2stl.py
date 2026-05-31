#!/usr/bin/env python
import tifffile
import numpy as np
import trimesh
import sys
import os
from showtiff import load_tiff
import argparse
from scipy.interpolate import griddata

def sphere_height(Z, quadrangle, radius, verbose=False):
    # 1. 找到标准分幅中心子午线，确定与中心子午面正交的子午面，作为测高基准
    lon_W, lon_E, lat_S, lat_N = quadrangle
    lon_C = 0.5 * (lon_W + lon_E)
    nrows, ncols = Z.shape
    lon = np.deg2rad(np.linspace(lon_W, lon_E, ncols) - lon_C)
    lat = np.deg2rad(np.linspace(lat_S, lat_N, nrows))
    Lon, Lat = np.meshgrid(lon, lat)
    if verbose:
        print("  标准分幅: {:f} (W) -- {:f} (E), {:f} (S) -- {:f} (N)".format(lon_W, lon_E, lat_S, lat_N))
        print("  初始网格: {:d} (LON) x {:d} (LAT)".format(ncols, nrows))
    h = (Z+radius)*np.cos(Lat)*np.cos(Lon) # 初始格点相对正交子午面的高度
    y = (Z+radius)*np.sin(Lat) # 初始格点在正交子午面投影的纵坐标（平行Y轴）
    x = (Z+radius)*np.cos(Lat)*np.sin(Lon) # 初始格点在正交子午面投影的横坐标（平行X轴）
    hmin = np.min(h.ravel())
    hmax = np.max(h.ravel())
    ymin = np.min(y.ravel())
    ymax = np.max(y.ravel())
    xmin = np.min(x.ravel())
    xmax = np.max(x.ravel())
    if verbose:
        print("  正交子午面基准初始格点高度：{:f} km (min) -- {:f} km (max)".format(hmin/1e3, hmax/1e3))
        print("  正交子午面投影初始横向跨度：{:f} km (min) -- {:f} km (max)".format(xmin/1e3, xmax/1e3))
        print("  正交子午面投影初始纵向跨度：{:f} km (min) -- {:f} km (max)".format(ymin/1e3, ymax/1e3))
    # 2. 确定初始格点在正交子午面的投影范围，重新划分规则、边缘整齐的网格
    x_W = np.max(x[:, 0])
    x_E = np.min(x[:,-1])
    y_S = np.max(y[ 0,:])
    y_N = np.min(y[-1,:])
    u = np.linspace(x_W, x_E, ncols)
    v = np.linspace(y_S, y_N, nrows)
    grid_x, grid_y = np.meshgrid(u, v)
    if verbose:
        print("  正交子午面最大矩形包络：{:f} km (W) -- {:f} km (E), {:f} km (S) -- {:f} km (N)".format(x_W/1e3, x_E/1e3, y_S/1e3, y_N/1e3))
        print("  正交子午面优化网格差值...")
    H = griddata(np.column_stack((x.ravel(), y.ravel())), h.ravel(), (grid_x, grid_y), method='linear')
    assert not np.any(np.isnan(H)), "插值越界！"
    if verbose:
        print("  正交子午面基准重采样高度范围：{:f} km (min) -- {:f} km (max)".format(np.min(H.ravel())/1e3, np.max(H.ravel())/1e3))
        print("  正交子午面投影重采样横向跨度：{:f} km (min) -- {:f} km (max)".format(np.min(grid_x.ravel())/1e3, np.max(grid_x.ravel())/1e3))
        print("  正交子午面投影重采样纵向跨度：{:f} km (min) -- {:f} km (max)".format(np.min(grid_y.ravel())/1e3, np.max(grid_y.ravel())/1e3))
    return H, grid_x, grid_y

def generate_terrain_solid(tiff_input, stl_output, down_sampling=1, base_z=-1.0, sphere=False, verbose=False):
    """
    将二维高度场向下挤出并封底，生成水密实体 STL
    
    :param x_range: X 轴范围 (min, max)
    :param y_range: Y 轴范围 (min, max)
    :param resolution: 网格分辨率 (点数)
    :param base_z: 挤出底座的绝对 Z 坐标 (必须小于高度场的最小值)
    :param out_file: 导出的 STL 文件名
    """
    Z, info = load_tiff(tiff_input)
    if verbose:
        print("输入网格数据类型：{}".format(Z.dtype))
    if sphere:
        print("将海平面基准高度转换为子午面基准高度...")
        lon_W = info['corner_coords'][0][0]
        lon_E = info['corner_coords'][1][0]
        lat_S = info['corner_coords'][2][1]
        lat_N = info['corner_coords'][0][1]
        Z, X, Y = sphere_height(Z, (lon_W, lon_E, lat_S, lat_N), info['radius'], verbose=verbose)
        Z = Z[::down_sampling, ::down_sampling]
        X = X[::down_sampling, ::down_sampling]
        Y = Y[::down_sampling, ::down_sampling]
        print("子午面基准高度范围: {:f} km (min) -- {:f} km (max)".format(np.min(Z.ravel())/1e3, np.max(Z.ravel())/1e3))
        print("子午面投影横向跨度: {:f} km".format(np.max(X.ravel())/1e3 - np.min(X.ravel())/1e3))
        print("子午面投影纵向跨度: {:f} km".format(np.max(Y.ravel())/1e3 - np.min(Y.ravel())/1e3))
        rows, cols = Z.shape
        num_points = rows * cols
    else:
        Z = Z[::down_sampling, ::down_sampling]
        rows, cols = Z.shape
        num_points = rows * cols
        x = np.arange(cols) * info['scale_x'] * down_sampling
        y = np.arange(rows) * info['scale_y'] * down_sampling
        X, Y = np.meshgrid(x, y)
    
    # ================= 2. 构建所有顶点 (Vertices) =================
    # 顶部顶点：原始的山体高度
    V_top = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # 底部顶点：保持 XY 不变，Z 强行拉平到 base_z
    V_bot = np.column_stack((X.ravel(), Y.ravel(), np.full(num_points, base_z)))
    
    # 合并顶点阵列 (索引 0 到 num_points-1 是顶部，num_points 到 2*num_points-1 是底部)
    vertices = np.vstack((V_top, V_bot))
    offset = num_points 
    
    # ================= 3. 构建表面多边形面 (Faces) =================
    # 使用 NumPy 向量化快速生成网格面的索引
    r = np.arange(rows - 1)
    c = np.arange(cols - 1)
    R, C = np.meshgrid(r, c, indexing='ij')
    
    P1 = R * cols + C        # 左下角
    P2 = P1 + 1              # 右下角
    P3 = P1 + cols           # 左上角
    P4 = P3 + 1              # 右上角
    
    # 3.1 顶部表面 (法线朝上)
    T1 = np.column_stack((P1.ravel(), P2.ravel(), P4.ravel()))
    T2 = np.column_stack((P1.ravel(), P4.ravel(), P3.ravel()))
    
    # 3.2 底部封口表面 (映射到底部顶点，并翻转顶点顺序以使法线朝下)
    B1 = np.column_stack((P1.ravel() + offset, P4.ravel() + offset, P2.ravel() + offset))
    B2 = np.column_stack((P1.ravel() + offset, P3.ravel() + offset, P4.ravel() + offset))
    
    faces = [T1, T2, B1, B2]
    
    # 3.3 构建四周垂直的侧壁 (缝合顶部边缘和底部边缘)
    # 南边 (r=0)
    s_p1 = np.arange(cols - 1)
    s_p2 = s_p1 + 1
    faces.append(np.column_stack((s_p1, s_p2, s_p1 + offset)))
    faces.append(np.column_stack((s_p2, s_p2 + offset, s_p1 + offset)))
    
    # 北边 (r=rows-1)
    n_p1 = (rows - 1) * cols + np.arange(cols - 1)
    n_p2 = n_p1 + 1
    faces.append(np.column_stack((n_p1, n_p1 + offset, n_p2)))
    faces.append(np.column_stack((n_p2, n_p1 + offset, n_p2 + offset)))

    # 西边 (c=0)
    w_p1 = np.arange(rows - 1) * cols
    w_p3 = w_p1 + cols
    faces.append(np.column_stack((w_p1, w_p1 + offset, w_p3)))
    faces.append(np.column_stack((w_p3, w_p1 + offset, w_p3 + offset)))
    
    # 东边 (c=cols-1)
    e_p1 = np.arange(rows - 1) * cols + cols - 1
    e_p3 = e_p1 + cols
    faces.append(np.column_stack((e_p1, e_p3, e_p1 + offset)))
    faces.append(np.column_stack((e_p3, e_p3 + offset, e_p1 + offset)))
    
    # ================= 4. 生成与导出 =================
    all_faces = np.vstack(faces)
    
    # 生成 Trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces)
    
    # 修复法线朝向 (极其重要：确保所有的三角面法线严格朝外)
    mesh.fix_normals()
    
    # 检查水密性
    if mesh.is_watertight:
        print(f"✅ 成功生成水密实体！共包含 {len(mesh.vertices)} 个顶点, {len(mesh.faces)} 个面。")
        # 导出为 STL 格式
        mesh.export(stl_output)
        print(f"💾 文件已保存为: {stl_output}")
    else:
        print("❌ 警告：生成的网格存在开放边缘 (Open Edges)，请检查逻辑。")

def main():
    parser = argparse.ArgumentParser(description="TIFF转STL参数解析")
    parser.add_argument(
        "tiff_input",
        type=str,
        help="需要处理的输入TIFF图像文件路径"
    )
    parser.add_argument(
        "stl_output",
        type=str,
        help="需要处理的输入TIFF图像文件路径"
    )
    parser.add_argument(
        "-d", "--down_sampling",
        dest="down_sampling",
        default=10,
        type=int,
        metavar="DOWN_SAMPLING",
        help="Down-sampling step size"
    )
    parser.add_argument(
        "-b", "--base_z",
        dest="base_z",
        type=float,
        default=0.0,
        metavar="BASE_Z",
        help="Z-coordinate of base layer"
    )
    parser.add_argument(
        "--sphere",
        action="store_true",
        help="以球面为基准"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="增加调试消息输出"
    )
    args = parser.parse_args()
    generate_terrain_solid(
        args.tiff_input,
        args.stl_output,
        down_sampling=args.down_sampling,
        base_z=args.base_z,
        sphere=args.sphere,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
