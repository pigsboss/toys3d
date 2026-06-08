#!/usr/bin/env python
"""swissSURFACE3D LiDAR DSM TIFF to STL
"""
import numpy as np
import tifffile
import trimesh
import os
import argparse
import trimesh
import cv2
from laz2tiff import basename_without_all_extensions
from ast import literal_eval
from scipy.spatial import Delaunay
from laz2tiff import CLASSES

def extrude_object_solid(X, Y, terrain_height, obj_counts, obj_height, obj_area_threshold=3.0, weld_thickness=0.1, verbose=False, use_convex_hull=False):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(obj_counts > 0),
        4,
        cv2.CV_32S
    )
    if verbose:
        print("  {} objects extracted.".format(num_labels))
    meshes = []
    rows, cols = X.shape
    scale_x = np.mean(np.diff(X, axis=1))
    scale_y = np.mean(np.diff(Y, axis=0))
    
    for i in range(1, num_labels):
        if (stats[i, 4] * scale_x * scale_y < obj_area_threshold):
            continue
        if verbose:
            print("  Object {}, area = {} pixels".format(i, stats[i, 4]))
            
        # 修正1：解决 np.bool 废弃报错
        mask = (labels == i)
        
        # 获取顶部所有顶点
        vtx_top = np.column_stack((X[mask].ravel(), Y[mask].ravel(), obj_height[mask].ravel()))
        if verbose:
            print("  Top mesh of object {}: Z_min = {}, Z_max = {}, Z_mean = {}, Z_std = {}".format(
                i, np.min(obj_height[mask]), np.max(obj_height[mask]), np.mean(obj_height[mask]), np.std(obj_height[mask])))
        pts_top = np.column_stack((X[mask].ravel(), Y[mask].ravel()))
        
        try:
            tri_top = Delaunay(pts_top)
        except:
            print("  Object {} is skipped (Delaunay failed).".format(i))
            continue
            
        faces_top = tri_top.simplices
        
        if use_convex_hull:
            # 凸包模式：保留所有三角形，不进行重心过滤
            valid_faces_top = faces_top
            if verbose:
                print("  Using convex hull top mesh ({} triangles)".format(len(valid_faces_top)))
        else:
            # 重心过滤法剔除凹包（与原逻辑一致）
            cx = np.mean(pts_top[faces_top, 0], axis=1)
            cy = np.mean(pts_top[faces_top, 1], axis=1)
            c = np.clip(np.int32((cx - X[0,0])/scale_x), 0, cols - 1)
            r = np.clip(np.int32((cy - Y[0,0])/scale_y), 0, rows - 1)
            valid_faces_top = faces_top[mask[r, c]]
            if verbose:
                print("  After centroid filtering: {} triangles".format(len(valid_faces_top)))
        
        # =========================================================
        # 核心修复：完全抛弃二次 Delaunay，使用纯 NumPy 构建无缝拓扑
        # =========================================================
        
        # 1. 纯 NumPy 提取外边界，保证索引与原始 vtx_top 绝对绑定
        edges = np.vstack((
            valid_faces_top[:, [0, 1]],
            valid_faces_top[:, [1, 2]],
            valid_faces_top[:, [2, 0]]
        ))
        sorted_edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
        # 只被引用1次的边即为悬空外边界
        boundary_edges = unique_edges[counts == 1]
        
        if len(boundary_edges) < 3:
            print("  Object {} is skipped (Not enough boundary edges).".format(i))
            continue

        # 2. 生成底面顶点：因共用相同的XY栅格，直接生成等量的底面顶点
        vtx_bot = np.column_stack((
            X[mask].ravel(),
            Y[mask].ravel(),
            terrain_height[mask].ravel() - weld_thickness
        ))
        if verbose:
            print("  Bottom mesh of object {}: Z_min = {}, Z_max = {}, Z_mean = {}, Z_std = {}".format(
                i, np.min(terrain_height[mask]), np.max(terrain_height[mask]), np.mean(terrain_height[mask]), np.std(terrain_height[mask])))
        
        # 3. 复用顶面拓扑作为底面拓扑，只需要加上偏移量，并反转绕序 (保证法线朝下)
        num_vtx = len(vtx_top)
        valid_faces_bot = valid_faces_top[:, ::-1] + num_vtx
        
        # 4. 生成侧墙：精确连接上下对应的边界顶点
        faces_side = []
        for edge in boundary_edges:
            top_v1, top_v2 = edge[0], edge[1]
            bot_v1 = top_v1 + num_vtx
            bot_v2 = top_v2 + num_vtx
            
            # 将四边形切为两个三角形
            faces_side.append([top_v1, bot_v1, bot_v2])
            faces_side.append([top_v1, bot_v2, top_v2])
            
        faces_side = np.array(faces_side)
        
        # 5. 组装实体模型：此时交给 Trimesh 打包，它会自动清理没用到的孤立点
        all_vertices = np.vstack((vtx_top, vtx_bot))
        all_faces = np.vstack((valid_faces_top, faces_side, valid_faces_bot))
        
        solid_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
        solid_mesh.remove_unreferenced_vertices()
        solid_mesh.process()
        if not solid_mesh.is_watertight:
            solid_mesh.fill_holes()
            solid_mesh.remove_unreferenced_vertices()
        
        # 强制修复侧墙可能存在的法线倒置
        solid_mesh.fix_normals()

        if not solid_mesh.is_watertight:
            print("  Try fill holes...")
            solid_mesh.fill_holes() # 尝试自动缝合破洞
        if solid_mesh.is_watertight:
            if verbose:
                print("  Watertight solid of object {} is generated including {} vertices and {} faces.".format(
                    i, len(solid_mesh.vertices), len(solid_mesh.faces)))
            meshes.append(solid_mesh)
        else:
            print(f"  Open edges detected on object {i}")
            
    return meshes

def generate_terrain_solid(X, Y, Z, base_z):
    rows, cols = X.shape
    num_points = rows * cols
    V_top = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    V_bot = np.column_stack((X.ravel(), Y.ravel(), np.full(num_points, base_z)))
    vertices = np.vstack((V_top, V_bot))
    offset = num_points 
    r = np.arange(rows - 1)
    c = np.arange(cols - 1)
    R, C = np.meshgrid(r, c, indexing='ij')
    P1 = R * cols + C
    P2 = P1 + 1
    P3 = P1 + cols
    P4 = P3 + 1
    T1 = np.column_stack((P1.ravel(), P2.ravel(), P4.ravel()))
    T2 = np.column_stack((P1.ravel(), P4.ravel(), P3.ravel()))
    B1 = np.column_stack((P1.ravel() + offset, P4.ravel() + offset, P2.ravel() + offset))
    B2 = np.column_stack((P1.ravel() + offset, P3.ravel() + offset, P4.ravel() + offset))
    faces = [T1, T2, B1, B2]
    s_p1 = np.arange(cols - 1)
    s_p2 = s_p1 + 1
    faces.append(np.column_stack((s_p1, s_p2, s_p1 + offset)))
    faces.append(np.column_stack((s_p2, s_p2 + offset, s_p1 + offset)))
    n_p1 = (rows - 1) * cols + np.arange(cols - 1)
    n_p2 = n_p1 + 1
    faces.append(np.column_stack((n_p1, n_p1 + offset, n_p2)))
    faces.append(np.column_stack((n_p2, n_p1 + offset, n_p2 + offset)))
    w_p1 = np.arange(rows - 1) * cols
    w_p3 = w_p1 + cols
    faces.append(np.column_stack((w_p1, w_p1 + offset, w_p3)))
    faces.append(np.column_stack((w_p3, w_p1 + offset, w_p3 + offset)))
    e_p1 = np.arange(rows - 1) * cols + cols - 1
    e_p3 = e_p1 + cols
    faces.append(np.column_stack((e_p1, e_p3, e_p1 + offset)))
    faces.append(np.column_stack((e_p3, e_p3 + offset, e_p1 + offset)))
    all_faces = np.vstack(faces)
    terrain = trimesh.Trimesh(vertices=vertices, faces=all_faces)
    terrain.fix_normals()
    return terrain

def main():
    parser = argparse.ArgumentParser(description="swissSURFACE3D LiDAR DSM TIFF 图像转 STL 3D 模型")
    parser.add_argument(
        "tiff_input",
        type=str,
        metavar="TIFF_INPUT",
        help="输入 TIFF 文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        dest="stl_output",
        type=str,
        metavar="STL_OUTPUT",
        help="输出 STL 文件路径前缀"
    )
    parser.add_argument(
        "-b", "--base_z",
        dest="base_z",
        type=float,
        metavar="BASE_Z",
        default=0.0,
        help="几何体底面 Z 坐标"
    )
    parser.add_argument(
        "-w", "--weld_thickness",
        dest="weld_thickness",
        type=float,
        metavar="WELD_THICKNESS",
        default=0.1,
        help="地物沉入地面的厚度"
    )
    parser.add_argument(
        "-a", "--object_area_threshold",
        dest="object_area_threshold",
        type=float,
        metavar="OBJECT_AREA_THRESHOLD",
        default=3.0,
        help="地物提取面积阈值"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="输出详细调试信息"
    )
    parser.add_argument(
        "--convex_hull",
        dest="convex_hull",
        action="store_true",
        help="使用凸包近似顶面（牺牲凹形精度换取水密性）"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="静默运行"
    )
    args = parser.parse_args()
    tiff_input = os.path.abspath(os.path.normpath(args.tiff_input))
    if args.stl_output:
        stl_output = os.path.abspath(os.path.normpath(args.stl_output))
    else:
        stl_output = os.path.abspath(os.path.normpath(
            basename_without_all_extensions(args.tiff_input)))
    surface_height = {}
    surface_intensity = {}
    surface_counts = {}
    surface_classes = []
    with tifffile.TiffFile(tiff_input) as tf:
        scale_x, scale_y, _ = tf.pages[0].tags[33550].value
        rows, cols = tf.pages[0].shape
        num_points = rows * cols
        x = np.arange(cols) * scale_x
        y = np.arange(rows) * scale_y
        for p in tf.pages:
            metadata = literal_eval(p.description)
            if metadata['class_name'].lower() != 'terrain':
                surface_classes.append(metadata['class_name'])
            if metadata['data'].lower() == 'counts':
                surface_counts[metadata['class_name']] = p.asarray()
            elif metadata['data'].lower() == 'height':
                surface_height[metadata['class_name']] = p.asarray()
            elif metadata['data'].lower() == 'intensity':
                surface_intensity[metadata['class_name']] = p.asarray()
    X, Y = np.meshgrid(x, y)
#    terrain_mesh = generate_terrain_solid(X, Y, surface_height['Terrain'], args.base_z)
#    terrain_output = stl_output + '_Terrain.stl'
#    if terrain_mesh.is_watertight:
#        print("Watertight terrain component is generated including {} vertices and {} faces.".format(
#            len(terrain_mesh.vertices), len(terrain_mesh.faces)))
#        terrain_mesh.export(terrain_output)
#        print(f"Terrain component is saved to {terrain_output}")
#    else:
#        print("Open edges detected.")
    scene = trimesh.Scene()
#    scene.graph.update(frame_to='world', frame_from='Terrain', geometry=terrain_mesh)
    for class_name in ['Water', 'Buildings']:
        if class_name.lower() == 'unclassified':
            continue
        meshes = extrude_object_solid(
            X, Y, surface_height['Terrain'],
            surface_counts[class_name],
            surface_height[class_name],
            obj_area_threshold = args.object_area_threshold,
            weld_thickness = args.weld_thickness,
            verbose = args.verbose,
            use_convex_hull = args.convex_hull
        )
        print("  {} generated {} solid objects.".format(class_name, len(meshes)))
        for i in range(len(meshes)):
            node_name = f'{class_name}_Obj_{i}'
            # 直接作为 'world' 的子节点添加，避免组节点导致导出错误
            scene.add_geometry(meshes[i], node_name=node_name, parent_node_name='world')
            obj_stl_output = stl_output + f'_{class_name}_Obj_{i}.stl'
            print(obj_stl_output)
            meshes[i].export(obj_stl_output)
    if scene.is_empty:
        print("Scene is empty!")
    else:
        glb_output = stl_output + '_Scene.glb'
        scene.export(glb_output, merge_primitives=False)
        print(glb_output)

if __name__=='__main__':
    main()
