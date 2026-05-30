#!/usr/bin/env python
import os
import sys
import trimesh
import numpy as np
from scipy.interpolate import griddata
import tifffile
import argparse
from xyz2tiff import save_tiff
import matplotlib.pyplot as plt

def load_top_mesh(stl_path):
    print(f"正在加载 STL 模型: {stl_path} ...")
    mesh = trimesh.load_mesh(stl_path)
    norm = mesh.face_normals
    tidx = np.where(norm[:, 2] > 1e-6)[0]
    tmesh = mesh.submesh([tidx], append=True)
    vertices = tmesh.vertices
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)
    print(f"模型边界: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
    return vertices

def mesh_to_dem(vertices, voxel_size=1000., resolution=400):
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    scale_x = ((x_max-x_min)+1.) * voxel_size / resolution
    scale_y = ((y_max-y_min)+1.) * voxel_size / resolution
    print(f"正在执行插值运算，目标分辨率: {resolution} ...")
    points = vertices[:, :2]  # 输入的 XY 坐标
    values = vertices[:,  2]  # 输入的 Z 坐标(高程)
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    assert not np.any(np.isnan(grid_z))
    grid_z = grid_z * voxel_size
    info = {
        'scale_x': scale_x,
        'scale_y': scale_y,
        'ref_x': x_min,
        'ref_y': y_min
    }
    return grid_z, info

def quadrangle_type(arg_string):
    """
    自定义解析函数：将 'x,y,width,height' 字符串解析并校验为整数元组
    """
    try:
        # 尝试按逗号分割，并将每部分转换为整数
        lon_W, lon_E, lat_S, lat_N = map(float, arg_string.split(','))
        return (lon_W, lon_E, lat_S, lat_N)
    except ValueError:
        # 抛出 ArgumentTypeError，argparse 会自动捕获并生成漂亮的报错信息
        raise argparse.ArgumentTypeError(
            f"裁切参数格式错误: '{arg_string}'。必须是 4 个由逗号分隔的数值，例如 '10,20,100,200'。"
        )

def main():
    parser = argparse.ArgumentParser(description="trek.nasa.gov STL 文件转换 TIFF 图像")
    parser.add_argument(
        "stl_input",
        type=str,
        help="需要读取的 STL 文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        dest="tiff_output",
        type=str,
        metavar="TIFF_OUTPUT",
        help="TIFF 图像输出路径"
    )
    parser.add_argument(
        "-s", "--voxel_size",
        dest="voxel_size",
        type=float,
        default=1000.,
        metavar="VOXEL_SIZE",
        help="STL 体素尺寸（单位：米）"
    )
    parser.add_argument(
        "-r", "--resolution",
        dest="resolution",
        type=int,
        default=400,
        metavar="RESOLUTION",
        help="采样分辨率（单位：像素）"
    )
    parser.add_argument(
        "-R", "--radius",
        dest="radius",
        type=float,
        default=3390.0, # Mars mean radius in km
        metavar="RADIUS",
        help="行星半径（单位：千米）"
    )
    parser.add_argument(
        "-Q", "--quadrangle",
        dest="quadrangle",
        type=quadrangle_type,
        metavar="QUADRANGLE",
        help="标准分幅（左边经度，右边经度，下边纬度，上边纬度）"
    )
    parser.add_argument(
        "--sphere",
        action="store_true",
        help="补偿球面高度"
    )
    parser.add_argument(
        "--flop",
        action="store_true",
        help="水平翻转之后输出图像"
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="垂直翻转之后输出图像"
    )
    parser.add_argument(
        "-e", "--exaggerate",
        dest="exaggerate",
        type=float,
        default=1.0,
        metavar="EXAGGERATE",
        help="垂直夸张系数"
    )
    args = parser.parse_args()
    vertices = load_top_mesh(os.path.abspath(os.path.normpath(args.stl_input)))
    gridz, info = mesh_to_dem(vertices, args.voxel_size, args.resolution)
    if args.quadrangle:
        lon_W, lon_E, lat_S, lat_N = args.quadrangle
        info['corner_coords'] = np.array([
            [lon_W, lat_N],
            [lon_E, lat_N],
            [lon_E, lat_S],
            [lon_W, lat_S]
        ], dtype=np.float64)
        print("包含标准分幅信息")
    if args.radius:
        info['radius'] = args.radius
        print("包含行星平均半径")
    if args.flop:
        print("水平翻转图像")
        gridz[:,:] = gridz[:,::-1]
    else:
        print("水平保持原状")
    if args.flip:
        print("垂直翻转图像")
        gridz[:,:] = gridz[::-1,:]
    else:
        print("垂直保持原状")
    if args.exaggerate:
        print("垂直夸张系数：{:f}".format(args.exaggerate))
        base_z = np.min(gridz.ravel())
        gridz = (gridz-base_z) * args.exaggerate + base_z
    if args.tiff_output:
        tiff_path = os.path.abspath(os.path.normpath(args.tiff_output))
        print(f"正在保存 DEM 至: {tiff_path} ...")
        save_tiff(tiff_path, gridz, info)
        print("✅ 转换完成！")
    else:
        print("显示高度图...")
        plt.imshow(gridz)
        plt.show

if __name__ == "__main__":
    main()
