#!/usr/bin/env python
"""swissSURFACE3D LiDAR DSM TIFF to STL
"""
import numpy as np
import tifffile
import trimesh
import os
import random
import argparse
import trimesh
import cv2
from laz2tiff import basename_without_all_extensions
from ast import literal_eval
from laz2tiff import CLASSES
from scipy.interpolate import griddata, RegularGridInterpolator
from skimage.restoration import inpaint_biharmonic
from collections import Counter

x_junc_avoid = 0.01

def extrude_object_solid(X, Y, terrain_height, obj_counts, obj_height, obj_area_threshold=3.0, weld_thickness=0.1, verbose=False):
    scale_x = np.mean(np.diff(X, axis=1))
    scale_y = np.mean(np.diff(Y, axis=0))
    origin_x = X[0, 0]
    origin_y = Y[0, 0]
    rows, cols = X.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(obj_counts > 0), 4, cv2.CV_32S
    )
    if verbose:
        print(f"  {num_labels} objects extracted.")

    meshes = []

    for i in range(1, num_labels):
        num_pixels = stats[i, cv2.CC_STAT_AREA]
        obj_area   = num_pixels * scale_x * scale_y
        if obj_area < obj_area_threshold:
            continue
        if verbose:
            print(f"  Object {i}, area = {num_pixels} pixels ({obj_area} sq.m)")
        mask = (labels == i)
        c0 = max(0, stats[i, cv2.CC_STAT_LEFT])
        r0 = max(0, stats[i, cv2.CC_STAT_TOP])
        c1 = min(cols, stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH])
        r1 = min(rows, stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT])
        y_local = np.linspace(Y[r0, 0]-scale_y, Y[r1-1, 0]+scale_y, r1-r0+2)
        x_local = np.linspace(X[0, c0]-scale_x, X[0, c1-1]+scale_x, c1-c0+2)
        if verbose:
            print(f"  Object {i}, rectangle: row {r0} to {r1}, column {c0} to {c1}")
            print(f"  Object {i}, rectangle: x = {x_local[0]}, {x_local[1]}, ... {x_local[-1]} ({len(x_local)} points); y = {y_local[0]}, {y_local[1]}, ... {y_local[-1]} ({len(y_local)} points)")
        obj_height_inpaint = inpaint_biharmonic(
            np.pad(
                obj_height[r0:r1, c0:c1],
                ((1,1), (1,1)),
                mode='constant',
                constant_values=((0., 0.), (0., 0.))),
            np.pad(
                np.logical_not(mask[r0:r1, c0:c1]),
                ((1,1), (1,1)),
                mode='constant',
                constant_values=((1,1), (1,1)))
        )
        if verbose:
            print(f"  Object {i}, local DEM (inpainted): {obj_height_inpaint.shape[0]} (rows) x {obj_height_inpaint.shape[1]} (columns)")
        interp_top = RegularGridInterpolator(
            (y_local, x_local),
            obj_height_inpaint,
            method='linear'
        )
        interp_bot = RegularGridInterpolator(
            (y_local, x_local),
            inpaint_biharmonic(
                np.pad(
                    terrain_height[r0:r1, c0:c1],
                    ((1,1), (1,1)),
                    mode='constant',
                    constant_values=((0., 0.), (0., 0.))),
                np.pad(
                    np.zeros((r1-r0, c1-c0), dtype='uint8'),
                    ((1,1), (1,1)),
                    mode='constant',
                    constant_values=((1,1), (1,1)))
            ),
            method='linear'
        )
        xc = X[mask].ravel()
        yc = Y[mask].ravel()
        x0 = xc - 0.5 * scale_x # south west vertice
        x1 = xc + 0.5 * scale_x # south east vertice
        x2 = xc - 0.5 * scale_x # north west vertice
        x3 = xc + 0.5 * scale_x # north east vertice
        y0 = yc - 0.5 * scale_y # south west vertice
        y1 = yc - 0.5 * scale_y # south east vertice
        y2 = yc + 0.5 * scale_y # north west vertice
        y3 = yc + 0.5 * scale_y # north east vertice
        mask_local = np.pad(
            mask[r0:r1, c0:c1],
            ((1,1), (1,1)),
            mode='constant',
            constant_values=((0,0), (0,0)))
        is_wall_w = np.logical_and(mask_local[1:-1, 1:-1], ~mask_local[1:-1,  :-2])
        is_wall_e = np.logical_and(mask_local[1:-1, 1:-1], ~mask_local[1:-1, 2:  ])
        is_wall_s = np.logical_and(mask_local[1:-1, 1:-1], ~mask_local[ :-2, 1:-1])
        is_wall_n = np.logical_and(mask_local[1:-1, 1:-1], ~mask_local[2:  , 1:-1])
        k_wall_w = np.argwhere(is_wall_w[mask_local[1:-1, 1:-1]].ravel()).ravel().astype('int32')
        k_wall_e = np.argwhere(is_wall_e[mask_local[1:-1, 1:-1]].ravel()).ravel().astype('int32')
        k_wall_s = np.argwhere(is_wall_s[mask_local[1:-1, 1:-1]].ravel()).ravel().astype('int32')
        k_wall_n = np.argwhere(is_wall_n[mask_local[1:-1, 1:-1]].ravel()).ravel().astype('int32')
        x_junc_sw = mask_local[1:-1, 1:-1] & ~mask_local[1:-1,  :-2] & mask_local[ :-2,  :-2] & ~mask_local[ :-2, 1:-1]
        x_junc_se = mask_local[1:-1, 1:-1] & ~mask_local[1:-1, 2:  ] & mask_local[ :-2, 2:  ] & ~mask_local[ :-2, 1:-1]
        x_junc_nw = mask_local[1:-1, 1:-1] & ~mask_local[1:-1,  :-2] & mask_local[2:  ,  :-2] & ~mask_local[2:  , 1:-1]
        x_junc_ne = mask_local[1:-1, 1:-1] & ~mask_local[1:-1, 2:  ] & mask_local[2:  , 2:  ] & ~mask_local[2:  , 1:-1]
        num_x_junc_sw = np.sum(x_junc_sw)
        num_x_junc_se = np.sum(x_junc_se)
        num_x_junc_nw = np.sum(x_junc_nw)
        num_x_junc_ne = np.sum(x_junc_ne)
        if verbose:
            print(f"  Object {i}, X-junctions: {num_x_junc_sw} (SW), {num_x_junc_se} (SE), {num_x_junc_nw} (NW), {num_x_junc_ne} (NE).")
            print(f"  {k_wall_w.size} west wall pixels detected.")
            print(f"  {k_wall_e.size} east wall pixels detected.")
            print(f"  {k_wall_s.size} south wall pixels detected.")
            print(f"  {k_wall_n.size} north wall pixels detected.")
        mask_x_sw = x_junc_sw[mask_local[1:-1, 1:-1]].ravel()
        mask_x_se = x_junc_se[mask_local[1:-1, 1:-1]].ravel()
        mask_x_nw = x_junc_nw[mask_local[1:-1, 1:-1]].ravel()
        mask_x_ne = x_junc_ne[mask_local[1:-1, 1:-1]].ravel()
        x0[mask_x_sw] +=  x_junc_avoid * scale_x
        y0[mask_x_sw] +=  x_junc_avoid * scale_y
        x1[mask_x_se] += -x_junc_avoid * scale_x
        y1[mask_x_se] +=  x_junc_avoid * scale_y
        x2[mask_x_nw] +=  x_junc_avoid * scale_x
        y2[mask_x_nw] += -x_junc_avoid * scale_y
        x3[mask_x_ne] += -x_junc_avoid * scale_x
        y3[mask_x_ne] += -x_junc_avoid * scale_y
        xv = np.concatenate((x0, x1, x2, x3))
        yv = np.concatenate((y0, y1, y2, y3))
        if verbose:
            print(f"  Object {i}, vertices on top surface: x_min = {np.min(xv)}, x_max = {np.max(xv)}, y_min = {np.min(yv)}, y_max = {np.max(yv)}")
        zv_obj = interp_top((yv, xv))
        zv_terrain = interp_bot((yv, xv)) - weld_thickness
        vtx_top = np.column_stack((xv, yv, zv_obj))
        vtx_bot = np.column_stack((xv, yv, zv_terrain))
        vertices = np.vstack((vtx_top, vtx_bot))
        k_top = np.arange(num_pixels).astype('int32')
        tri_top_d = np.column_stack((k_top               , k_top +   num_pixels, k_top + 3*num_pixels))
        tri_top_u = np.column_stack((k_top               , k_top + 3*num_pixels, k_top + 2*num_pixels))
        tri_bot_d = np.column_stack((k_top + 4*num_pixels, k_top + 6*num_pixels, k_top + 5*num_pixels))
        tri_bot_u = np.column_stack((k_top + 6*num_pixels, k_top + 7*num_pixels, k_top + 5*num_pixels))
        tri_w_w_d = np.column_stack((k_wall_w + 6*num_pixels, k_wall_w + 4 * num_pixels, k_wall_w               ))
        tri_w_w_u = np.column_stack((k_wall_w + 6*num_pixels, k_wall_w                 , k_wall_w + 2*num_pixels))
        tri_w_e_d = np.column_stack((k_wall_e + 5*num_pixels, k_wall_e + 7 * num_pixels, k_wall_e +   num_pixels))
        tri_w_e_u = np.column_stack((k_wall_e + 7*num_pixels, k_wall_e + 3 * num_pixels, k_wall_e +   num_pixels))
        tri_w_s_d = np.column_stack((k_wall_s + 4*num_pixels, k_wall_s + 5 * num_pixels, k_wall_s +   num_pixels))
        tri_w_s_u = np.column_stack((k_wall_s + 4*num_pixels, k_wall_s +     num_pixels, k_wall_s               ))
        tri_w_n_d = np.column_stack((k_wall_n + 7*num_pixels, k_wall_n + 6 * num_pixels, k_wall_n + 3*num_pixels))
        tri_w_n_u = np.column_stack((k_wall_n + 6*num_pixels, k_wall_n + 2 * num_pixels, k_wall_n + 3*num_pixels))
        faces = np.vstack((
            tri_top_d,
            tri_top_u,
            tri_bot_d,
            tri_bot_u,
            tri_w_w_d,
            tri_w_w_u,
            tri_w_e_d,
            tri_w_e_u,
            tri_w_s_d,
            tri_w_s_u,
            tri_w_n_d,
            tri_w_n_u
        ))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        if mesh.is_watertight:
            meshes.append(mesh)
            if verbose:
                print(f"  Watertight solid of object {i} generated: {len(mesh.vertices)} vertices ({8*num_pixels} vertices added), {len(mesh.faces)} faces.")
        else:
            print(f"  Open edges detected on object {i}: {len(mesh.vertices)} vertices ({8*num_pixels} vertices added), {len(mesh.faces)} faces.")
            mesh.merge_vertices()
            print(f"    {len(mesh.vertices)} vertices remain after merging.")
            meshes.append(mesh)
            edge_counter = Counter()
            for face in mesh.faces:                                                                                                                                                                                        
                edge_counter[tuple(sorted([face[0], face[1]]))] += 1
                edge_counter[tuple(sorted([face[1], face[2]]))] += 1
                edge_counter[tuple(sorted([face[2], face[0]]))] += 1
            num_open_edges = 0
            num_complex_edges = 0
            num_kissing_edges = 0
            for e, cnt in edge_counter.items():
                if cnt < 2:
                    num_open_edges += 1
                elif cnt > 2:
                    num_complex_edges += 1
                    if cnt == 4:
                        num_kissing_edges += 1
                    v1 = mesh.vertices[e[0]]
                    v2 = mesh.vertices[e[1]]
                    assert np.allclose(v1[:2], v2[:2])
                    #print(f"      Edge: ({v1[0]:.4f}, {v1[1]:.4f}, {v1[2]:.4f}) - ({v2[0]:.4f}, {v2[1]:.4f}, {v2[2]:.4f})")
            print(f"    {num_open_edges} open edges found, {num_complex_edges} non-manifold edges found, {num_kissing_edges} kissing edges found.")
            assert num_complex_edges == num_kissing_edges
            assert num_kissing_edges == num_x_junc_se + num_x_junc_sw
    return meshes

def generate_terrain_solid_optimized(X, Y, Z, base_z):
    rows, cols = X.shape
    num_points = rows * cols
    
    # 1. 构建顶面 (完全保留你的高效逻辑)
    V_top = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    r = np.arange(rows - 1)
    c = np.arange(cols - 1)
    R, C = np.meshgrid(r, c, indexing='ij')
    P1 = R * cols + C
    P2 = P1 + 1
    P3 = P1 + cols
    P4 = P3 + 1
    
    # 保持顶面拓扑
    T1 = np.column_stack((P1.ravel(), P2.ravel(), P4.ravel()))
    T2 = np.column_stack((P1.ravel(), P4.ravel(), P3.ravel()))
    valid_faces_top = np.vstack((T1, T2))
    
    # 2. 提取完美的外圈边界索引 (顺时针或逆时针环绕一圈)
    # 顺序为：上边缘 -> 右边缘 -> 下边缘(倒序) -> 左边缘(倒序)
    top_edge = np.arange(cols)
    right_edge = np.arange(1, rows) * cols + (cols - 1)
    bottom_edge = (rows - 1) * cols + np.arange(cols - 2, -1, -1)
    left_edge = np.arange(rows - 2, 0, -1) * cols
    
    # 将边界连成一个完整的闭合环
    boundary_indices = np.concatenate([top_edge, right_edge, bottom_edge, left_edge])
    num_boundary = len(boundary_indices)
    
    # 3. 仅为边界点生成底面顶点 (极度节省内存)
    V_bot = V_top[boundary_indices].copy()
    V_bot[:, 2] = base_z
    
    # 4. 构建侧墙 (严格控制逆时针绕序，法线绝对朝外)
    faces_side = []
    top_offset = num_points # 底面顶点在全局数组中的起始偏移量
    
    for i in range(num_boundary):
        # 当前边界点和下一个边界点 (环状连接)
        next_i = (i + 1) % num_boundary 
        
        # 顶面对应的全局索引
        t_curr = boundary_indices[i]
        t_next = boundary_indices[next_i]
        
        # 底面对应的全局索引
        b_curr = top_offset + i
        b_next = top_offset + next_i
        
        # 将矩形墙面劈成两个三角形 (严格按照外视逆时针方向)
        faces_side.append([t_curr, b_curr, b_next])
        faces_side.append([t_curr, b_next, t_next])
        
    faces_side = np.array(faces_side)
    
    # 5. 极简底盖剖分 (仅对这几百个边界点进行 Delaunay)
    pts_bot_2d = V_bot[:, :2]
    tri_bot = Delaunay(pts_bot_2d)
    
    # 底面三角形索引加上偏移量，并翻转法线朝下 (::-1)
    faces_bot = tri_bot.simplices + top_offset
    faces_bot = faces_bot[:, ::-1]
    
    # 6. 最终拼装
    all_vertices = np.vstack((V_top, V_bot))
    all_faces = np.vstack((valid_faces_top, faces_side, faces_bot))
    
    terrain = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=True)
    terrain.fix_normals()
    
    return terrain

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
    # 定义经典色卡调色板
    CLASS_PALETTES = {
        'Water': [
            (  0, 119, 190),   # 海蓝
            (  0, 191, 255),   # 湖蓝
            ( 25,  25, 112),   # 深蓝
            ( 70, 130, 180),   # 钢蓝
            (100, 149, 237),   # 矢车菊蓝
        ],
        'Buildings': [
            (176, 176, 176),   # 水泥灰
            (178,  34,  34),   # 砖红
            (194, 178, 128),   # 沙褐
            (128, 128, 128),   # 灰
            ( 85,  85,  85),   # 深灰
            (205, 133,  63),   # 铜色
        ],
    }

    def get_class_color(palette):
        """从调色板随机选一种颜色，并施加不可察觉的微小抖动"""
        base = random.choice(palette)
        r = max(0, min(255, base[0] + random.randint(-3, 3)))
        g = max(0, min(255, base[1] + random.randint(-3, 3)))
        b = max(0, min(255, base[2] + random.randint(-3, 3)))
        a = max(0, min(255, 255 + random.randint(-10, 10)))  # 抖动 alpha
        return [r, g, b, a]

    random.seed()  # 可复现（可选：设置固定数字如 random.seed(42)）

    for class_name in ['Water', 'Buildings', 'Vegetation']:
        if class_name.lower() == 'unclassified':
            continue
        palette = CLASS_PALETTES.get(class_name, [(128, 128, 128)])
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
            color = get_class_color(palette)
            meshes[i].visual.face_colors = color
            node_name = f'{class_name}_Obj_{i}'
            scene.add_geometry(meshes[i], node_name=node_name, parent_node_name='world')
            obj_stl_output = stl_output + f'_{class_name}_Obj_{i}.stl'
            print(obj_stl_output)
            meshes[i].export(obj_stl_output)
    if scene.is_empty:
        print("Scene is empty!")
    else:
        glb_output = stl_output + '_Scene.3mf'
        scene.export(glb_output)
        print(glb_output)

if __name__=='__main__':
    main()
