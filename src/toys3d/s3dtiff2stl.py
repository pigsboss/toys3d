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

def extrude_object_solid(X, Y, terrain_height, obj_counts, obj_height, obj_area_threshold=3.0, weld_thickness=0.1, verbose=False):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(obj_counts>0),
        8,
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
        mask = np.bool(labels == i)
        vtx_top = np.column_stack((X[mask].ravel(), Y[mask].ravel(), obj_height[mask].ravel()))
        pts_top = np.column_stack((X[mask].ravel(), Y[mask].ravel()))
        try:
            tri_top = Delaunay(pts_top)
        except:
            print("  Object {} is skipped.".format(i))
            continue
        faces_top = tri_top.simplices
        cx = np.mean(pts_top[faces_top, 0], axis=1)
        cy = np.mean(pts_top[faces_top, 1], axis=1)
        c = np.clip(np.int32((cx - X[0,0])/scale_x), 0, cols - 1)
        r = np.clip(np.int32((cy - Y[0,0])/scale_y), 0, rows - 1)
        valid_faces_top = faces_top[mask[r, c]]
        mesh_top = trimesh.Trimesh(vertices=vtx_top, faces=valid_faces_top)
        sorted_edges = np.sort(mesh_top.edges, axis=1)
        unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        # 提取所有边界顶点的唯一索引 (这些点在 vtx_top 中的位置)
        boundary_vertex_indices = np.unique(boundary_edges)
        if len(boundary_edges) < 4:
            print("  Object {} is skipped.".format(i))
            continue
        # 建立映射字典：顶面边界顶点索引 -> 将要在总顶点数组中生成的底面顶点索引
        num_vtx_top = len(vtx_top)
        top_to_bottom_idx = {top_idx: num_vtx_top + i for i, top_idx in enumerate(boundary_vertex_indices)}
        # 生成底面顶点 (投影到底部基准面)
        vtx_bot = np.column_stack((X[mask].ravel(), Y[mask].ravel(), terrain_height[mask].ravel()-weld_thickness))[boundary_vertex_indices]
        # 开始构建侧墙面片
        faces_side = []
        for edge in boundary_edges:
            top_v1, top_v2 = edge[0], edge[1]
            bot_v1 = top_to_bottom_idx[top_v1]
            bot_v2 = top_to_bottom_idx[top_v2]
            # 将一个四边形侧墙劈成两个三角形
            # 注意此处的缠绕顺序 (Winding Order) 尽量保持一致
            faces_side.append([top_v1, bot_v1, bot_v2])
            faces_side.append([top_v1, bot_v2, top_v2])
        faces_side = np.array(faces_side)
        # 提取底面顶点的 2D 坐标 (X, Y)
        pts_bot = vtx_bot[:, :2]
        # 对底部点集重新进行 Delaunay 剖分
        try:
            tri_bot = Delaunay(pts_bot)
        except:
            print("  Object {} is skipped.".format(i))
            continue
        faces_bot = tri_bot.simplices
        # 同样使用重心过滤法剔除底座的凹包
        cx = np.mean(pts_bot[faces_bot, 0], axis=1)
        cy = np.mean(pts_bot[faces_bot, 1], axis=1)
        c = np.clip(np.int32((cx - X[0,0])/scale_x), 0, cols - 1)
        r = np.clip(np.int32((cy - Y[0,0])/scale_y), 0, rows - 1)
        valid_faces_bot_local = faces_bot[mask[r, c]]
        # 此时的 valid_faces_bot_local 是基于 vtx_bot 的局部索引 (0, 1, 2...)
        # 我们需要将它们映射回全局顶点数组的索引 (num_vtx_top + i)
        faces_bot_global = np.zeros_like(valid_faces_bot_local)
        for j in range(3):
            # valid_bot_faces_local 里的值对应的是 boundary_vertex_indices 数组的位置
            local_idx_array = valid_faces_bot_local[:, j]
            # 获取对应的顶面索引，再通过字典查到底面全局索引
            faces_bot_global[:, j] = [top_to_bottom_idx[boundary_vertex_indices[idx]] for idx in local_idx_array]
        # [关键细节] 必须将底面的三角形顶点顺序反转 (A,B,C -> C,B,A)
        # 这样底面的法线才会统一下方，而不是向着模型内部
        faces_bot_global = faces_bot_global[:, ::-1]
        # 合并所有的顶点
        all_vertices = np.vstack((vtx_top, vtx_bot))
        # 合并所有的面片
        all_faces = np.vstack((valid_faces_top, faces_side, faces_bot_global))
        # 创建最终的 3D 实体
        solid_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        # [防呆机制] 让 Trimesh 自动处理任何细微的法线方向错乱
        solid_mesh.fix_normals()
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
    terrain_mesh = generate_terrain_solid(X, Y, surface_height['Terrain'], args.base_z)
    terrain_output = stl_output + '_Terrain.stl'
    if terrain_mesh.is_watertight:
        print("Watertight terrain component is generated including {} vertices and {} faces.".format(
            len(terrain_mesh.vertices), len(terrain_mesh.faces)))
        terrain_mesh.export(terrain_output)
        print(f"Terrain component is saved to {terrain_output}")
    else:
        print("Open edges detected.")
    scene = trimesh.Scene()
    scene.graph.update(frame_to='world', frame_from='Terrain', geometry=terrain_mesh)
    for class_name in surface_classes:
        if class_name.lower() == 'unclassified':
            continue
        grp_name = class_name + '_Group'
        scene.graph.update(frame_to='world', frame_from=grp_name)
        meshes = extrude_object_solid(
            X, Y, surface_height['Terrain'],
            surface_counts[class_name],
            surface_height[class_name],
            obj_area_threshold = args.object_area_threshold,
            weld_thickness = args.weld_thickness,
            verbose = args.verbose
        )
        print("  {} generated {} solid objects.".format(class_name, len(meshes)))
        for i in range(len(meshes)):
            scene.graph.update(frame_to=grp_name, frame_from=class_name+'_Obj_{:d}'.format(i), geometry=meshes[i])
    print(scene.graph)
    scene.export(stl_output + '_Scene.glb')

if __name__=='__main__':
    main()
