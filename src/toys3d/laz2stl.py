#!/usr/bin/env python
import sys
import laspy
import open3d as o3d
import numpy as np
import os
import argparse
from laz2tiff import CLASSES

def extract_pcd_by_class(las, class_codes):
    """根据分类码提取点云，并转换为 Open3D 格式"""
    mask = np.isin(las.classification[::10], class_codes)
    
    # 提取 XYZ 坐标
    x = las.x[::10][mask]
    y = las.y[::10][mask]
    z = las.z[::10][mask]
    
    if len(x) == 0:
        return None
        
    points = np.vstack((x, y, z)).transpose()
    
    # 构建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def mesh_poisson(pcd, depth=9, density_threshold=0.05):
    """
    泊松表面重建 (Poisson Surface Reconstruction)
    适用: 建筑物、桥梁、码头 (需要水密、平滑的人造结构)
    """
    # 1. 估算点云法线 (泊松重建的前提)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # 2. 泊松重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    # 3. 剔除低密度区域 (去除泊松算法产生的多余“气泡”包裹面)
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    mesh.compute_vertex_normals()
    return mesh

def mesh_alpha_shape(pcd, alpha=1.5):
    """
    阿尔法形状 (Alpha Shapes)
    适用: 植被、散乱的水体 (不需要完美平滑，保留点云边界特征)
    """
    # 下采样以提高计算速度 (对于庞大的树冠极其重要)
    pcd_down = pcd.voxel_down_sample(voxel_size=0.5)
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_down, alpha)
    mesh.compute_vertex_normals()
    return mesh

# ================= 主处理流程 =================

def process_laz_to_stl(laz_path, stl_path):
    print(f"正在读取文件: {laz_path} ...")
    las = laspy.read(laz_path)
    
    for name, codes in CLASSES.items():
        print(f"[{name}] 正在提取分类码 {codes} 的点云...")
        pcd = extract_pcd_by_class(las, codes)
        
        if pcd is None or len(pcd.points) < 100:
            print(f" -> {name} 数据量过少，跳过。")
            continue
            
        print(f" -> 提取到 {len(pcd.points)} 个点。开始重建网格...")
        
        # 根据地物类型选择不同的三角化策略
        if name in ["Buildings", "Bridges_Docks"]:
            # 建筑物和桥梁：使用泊松重建获得平滑表面
            # depth 越大细节越多，但计算越慢
            mesh = mesh_poisson(pcd, depth=10, density_threshold=0.03)
            
        elif name in ["Vegetation"]:
            # 植被：使用 Alpha Shape 生成包裹团块
            # alpha 值越小越紧贴散点，越大越像一个凸包
            mesh = mesh_alpha_shape(pcd, alpha=2.0)
            
        elif name in ["Terrain"]:
            # 地形：通常使用 Alpha Shape 获取表皮
            # 进阶操作：也可以用泊松，但需要极高的密度过滤
            continue
            mesh = mesh_alpha_shape(pcd, alpha=3.0)
            
        elif name in ["Water"]:
            # 水体：LiDAR 扫水体往往只有零星散点
            # 增加 alpha 值强行将稀疏的点连成片
            mesh = mesh_alpha_shape(pcd, alpha=5.0)
        
        # 导出为 STL
        out_file = os.path.join(stl_path, f"{name}.stl")
        o3d.io.write_triangle_mesh(out_file, mesh)
        print(f" -> 成功导出 STL: {out_file}\n")

if __name__ == "__main__":
    laz_input = os.path.abspath(os.path.normpath(sys.argv[1]))
    stl_output = os.path.abspath(os.path.normpath(sys.argv[2]))
    if not os.path.exists(stl_output):
        os.makedirs(stl_output)
    process_laz_to_stl(laz_input, stl_output)
    print("全部处理完成！")
