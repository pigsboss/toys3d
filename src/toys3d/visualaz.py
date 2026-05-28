#!/usr/bin/env python
"""Visualize .laz file(s) from swisstopo (e.g., swissSURFACE3D).
"""
import numpy as np
import laspy
import pyvista as pv
import sys
import os
from os import path

def visualize_swiss_surface_3d(file_path, class_id, sampling_rate=1.0):
    """
    使用 PyVista 高效可视化 swissSURFACE3D .laz 点云数据
    
    :param file_path: .laz 文件路径
    :param sampling_rate: 抽样比例 (0.0 到 1.0)。若单张图切片点数过多出现卡顿，可调整至 0.2 (即保留20%的点)
    """
    print(f"正在读取并解压 LAZ 文件: {file_path} ...")
    
    # 1. 使用 laspy 读取点云
    with laspy.open(file_path) as fh:
        print(f"原始点云总数: {fh.header.point_count:,}")
        las_data = fh.read()
    
    # 2. 提取点云的 X, Y, Z 三维坐标
    # laspy 内部已根据 header 的 scale 和 offset 自动转换为了真实的米制地理坐标
    x = np.array(las_data.x)
    y = np.array(las_data.y)
    z = np.array(las_data.z)
    c = np.array(las_data.classification)
    m = np.zeros(c.shape, dtype='bool')
    for i in class_id:
        m = np.logical_or(m, c==i)
    points = np.vstack((x[m], y[m], z[m])).T

    # 3. 大数字地理坐标归零（!!关键优化!!）
    # 瑞士坐标系（MN95）的数值非常大（如 X=2,600,000, Y=1,200,000），
    # 单精度浮点数（显卡标准）处理大基数会导致画面严重抖动、坐标错位。因此减去中心点转换为相对坐标
    center_offset = np.mean(points, axis=0)
    points_centered = points - center_offset
    print(f"坐标已中心化，偏移量（X, Y, Z）: {center_offset}")

    # 4. 可选：针对极大规模数据进行均匀抽样（应对单图数千万点的情况）
    if sampling_rate < 1.0:
        indices = np.random.choice(len(points_centered), size=int(len(points_centered) * sampling_rate), replace=False)
        points_centered = points_centered[indices]
        # 同步抽样属性字段
        intensity = np.array(las_data.intensity[m])[indices]
        classification = np.array(las_data.classification[m])[indices]
        z_values = z[indices]
        print(f"抽样完成，当前渲染点数: {len(points_centered):,}")
    else:
        intensity = np.array(las_data.intensity[m])
        classification = np.array(las_data.classification[m])
        z_values = z[m]

    # 5. 构筑 PyVista 点云对象
    point_cloud = pv.PolyData(points_centered)

    # 6. 将各种 LiDAR 属性注入 PyVista 数据集，方便后续切换着色模式
    point_cloud["Elevation (m)"] = z_values          # 按真实绝对海拔着色
    point_cloud["Intensity"] = intensity             # 按激光反射强度着色
    point_cloud["Classification"] = classification   # 按植被、建筑物、地面等分类代码着色

    # 7. 配置 PyVista 渲染器
    plotter = pv.Plotter(window_size=[1240, 768])
    plotter.background_color = "black"  # 黑色背景能让激光点云细节更醒目

    # 添加点云数据到画布
    # 提示：若电脑配置较高，可设置 render_points_as_spheres=True 让每个点变成3D微球体
    plotter.add_mesh(
        point_cloud,
        scalars="Classification",       # 默认渲染的属性，可替换为 "Intensity" 或 "Classification"
        cmap="terrain",                 # 地形常用色带，可选 'viridis', 'jet', 'plasma' 等
        point_size=1.5,                 # 调整点的大小，数值越小越显精致
        render_points_as_spheres=False, # 设为 False 渲染速度极快，适合海量点
        scalar_bar_args={"title": "Classification", "color": "white"} # 色条配置
    )

    # 添加 3D 坐标轴刻度线（由于中心化了，我们将轴标签隐藏或参考相对值）
    plotter.show_grid(color="gray", fmt="%.1f")
    plotter.add_scalar_bar()
    
    print("正在打开渲染窗口，您可以使用鼠标拖动旋转、滚轮缩放...")
    plotter.show()

if __name__ == "__main__":
    # 请将此处替换为您的真实 swissSURFACE3D .laz 文件路径
    # 示例文件名如: C_CH1903Plus_LV95_2600_1200.laz
    file_path = path.abspath(path.normpath(path.realpath(sys.argv[1])))
    class_id  = eval(sys.argv[2])
    try:
       iter(class_id)
    except:
        class_id = (class_id, )
    # 执行可视化，若单张图点数超过3000万导致拖动卡顿，可将 sampling_rate 改为 0.5 或 0.3
    visualize_swiss_surface_3d(file_path, class_id, sampling_rate=1.0)
