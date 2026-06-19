#!/usr/bin/env python
"""swissSURFACE3D LiDAR DSM TIFF to STL/OBJ/GLB/3MF
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
from scipy.spatial import Delaunay
from skimage.filters import frangi

x_junc_avoid = 0.01

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

def frangi_response(intensity, counts, min_width, max_width, frangi_beta=0.5):
    """
    使用基于 Hessian 矩阵的多尺度 Frangi 滤波器提取道路实体。
    
    参数:
    min_width_m, max_width_m: 道路在现实世界中的物理宽度范围 (米)
    """
    # 1. 物理尺度到像素尺度的转换 (Sigma 对应高斯核的标淮差)
    # 假设你的 scale_x 是 0.5m/pixel，那么 12 米的道路宽度大约是 24 个像素
    sigmas = np.arange(min_width, max_width, 1.0)
    
    # 2. 预处理：对比度拉伸，排除极端值
    p2, p98 = np.percentile(intensity[counts > 0], (2, 98))
    intensity_norm = np.clip((intensity - p2) / (p98 - p2), 0, 1)
    
    # 注意：Frangi 默认寻找“亮”的管状物。
    # 柏油路在 Intensity 图中通常是黑色的，所以我们需要将图像反转 (Invert)
    intensity_inv = 1.0 - intensity_norm
    
    # 3. 多尺度管状结构滤波 (核心降维打击)
    # black_ridges=False 因为我们已经反转了图像，寻找白色的脊
    return intensity_inv, frangi(
        intensity_inv,
        sigmas=sigmas,
        beta=frangi_beta,
        black_ridges=False
    )

def extrude_object_solid(X, Y, terrain_height, obj_counts, obj_height, obj_area_threshold=3.0, weld_thickness=0.1, verbose=False):
    scale_x = np.mean(np.diff(X, axis=1))
    scale_y = np.mean(np.diff(Y, axis=0))
    origin_x = X[0, 0]
    origin_y = Y[0, 0]
    rows, cols = X.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(obj_counts > 0), 4, cv2.CV_32S)
    if verbose:
        print(f"  {num_labels} objects extracted.")

    meshes = []

    for i in range(1, num_labels):
        num_pixels = stats[i, cv2.CC_STAT_AREA]
        obj_area   = num_pixels * scale_x * scale_y
        if obj_area < obj_area_threshold:
            continue
        if verbose:
            print(f"  Object {i}:"
                  f"    area = {num_pixels} pixels ({obj_area} sq.m)")
        mask = (labels == i)
        c0 = max(0, stats[i, cv2.CC_STAT_LEFT])
        r0 = max(0, stats[i, cv2.CC_STAT_TOP])
        c1 = min(cols, stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH])
        r1 = min(rows, stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT])
        y_local = np.linspace(Y[r0, 0]-scale_y, Y[r1-1, 0]+scale_y, r1-r0+2)
        x_local = np.linspace(X[0, c0]-scale_x, X[0, c1-1]+scale_x, c1-c0+2)
        if verbose:
            print("    rectangle:")
            print(f"      row {r0} to {r1}, column {c0} to {c1}")
            print(f"      x = {x_local[0]}, {x_local[1]}, ... {x_local[-1]} ({len(x_local)} points)")
            print(f"      y = {y_local[0]}, {y_local[1]}, ... {y_local[-1]} ({len(y_local)} points)")
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
                constant_values=((1,1), (1,1))))
        if verbose:
            print(f"    local DEM (inpainted): {obj_height_inpaint.shape[0]} (rows) "
                  f"x {obj_height_inpaint.shape[1]} (columns)")
        interp_top = RegularGridInterpolator(
            (y_local, x_local),
            obj_height_inpaint,
            method='linear')
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
                    constant_values=((1,1), (1,1)))),
            method='linear')
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
            print(f"    X-junctions detected: {num_x_junc_sw} (SW), "
                  f"{num_x_junc_se} (SE), "
                  f"{num_x_junc_nw} (NW), "
                  f"{num_x_junc_ne} (NE).")
            print(f"    side wall pixels: {k_wall_w.size} (W) "
                  f"{k_wall_e.size} (E) "
                  f"{k_wall_s.size} (S) "
                  f"{k_wall_n.size} (N)")
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
            print("    vertices on top/bottom surface:")
            print(f"      x_min = {np.min(xv)}, x_max = {np.max(xv)}")
            print(f"      y_min = {np.min(yv)}, y_max = {np.max(yv)}")
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
            tri_w_n_u))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        if mesh.is_watertight:
            meshes.append(mesh)
            if verbose:
                print(f"      ✅ Watertight solid generated: {len(mesh.vertices)} "
                      f"vertices, {len(mesh.faces)} faces.")
        else:
            print("     ⚠  Non-watertight mesh generated.")
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
            print(f"      🔬 Diagnosis: {num_open_edges} open edges found, "
                  f"{num_complex_edges} non-manifold edges found, "
                  f"{num_kissing_edges} kissing edges found.")
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

def extract_asphalt(terrain_counts, terrain_height, terrain_intensity, pixel_scale, asphalt_thickness=1., segments_acceptance=0.05, frangi_threshold=0.01, min_width_m=2.0, max_width_m=7.0, frangi_beta=0.5, verbose=False):
    if verbose:
        print('  Extracting asphalt with frangi filters...')
    _, resp = frangi_response(terrain_intensity, terrain_counts, min_width_m/pixel_scale, max_width_m/pixel_scale, frangi_beta=frangi_beta)
    fmask = np.uint8(resp > frangi_threshold)
    # 优化点：形态学闭运算，防止路面因为斑马线、汽车、落叶而产生空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fmask_closed = cv2.morphologyEx(fmask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fmask_closed, 4, cv2.CV_32S)
    if verbose:
        print(f'  {num_labels-1} asphalt segments detected.')
    mask = np.isin(labels, np.argwhere(stats[1:,cv2.CC_STAT_AREA]>np.percentile(stats[1:,cv2.CC_STAT_AREA],100.*(1-segments_acceptance)))+1)
    asphalt_counts = np.zeros_like(terrain_counts)
    asphalt_counts[mask] = terrain_counts[mask]
    asphalt_height = np.zeros_like(terrain_height)
    asphalt_height[mask] = terrain_height[mask]
    terrain_height[mask] = terrain_height[mask] - asphalt_thickness
    asphalt_intensity = terrain_intensity.copy()
    asphalt_intensity[mask] = terrain_intensity[mask]
    return terrain_height, asphalt_counts, asphalt_height, asphalt_intensity

def main():
    parser = argparse.ArgumentParser(description="swissSURFACE3D LiDAR DSM TIFF 图像转 STL/GLB/OBJ/3MF 3D 模型")
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
        default=1.0,
        help="地物沉入地面的厚度"
    )
    parser.add_argument(
        "-a", "--object_area_threshold",
        dest="object_area_threshold",
        type=float,
        metavar="OBJECT_AREA_THRESHOLD",
        default=50.0,
        help="地物提取面积阈值"
    )
    parser.add_argument(
        "--asphalt_thickness",
        dest="asphalt_thickness",
        type=float,
        metavar="ASPHALT_THICKNESS",
        default=1.0,
        help="铺装路面厚度"
    )
    parser.add_argument(
        "--asphalt_min_width",
        dest="asphalt_min_width",
        type=float,
        metavar="ASPHALT_MIN_WIDTH",
        default=2.0,
        help="铺装路面最小宽度（单位：米）"
    )
    parser.add_argument(
        "--asphalt_max_width",
        dest="asphalt_max_width",
        type=float,
        metavar="ASPHALT_MAX_WIDTH",
        default=7.0,
        help="铺装路面最大宽度（单位：米）"
    )
    parser.add_argument(
        "--asphalt_segments_acceptance",
        dest="asphalt_segments_acceptance",
        type=float,
        metavar="ASPHALT_SEGMENTS_ACCEPTANCE",
        default=0.05,
        help="铺装路段接受比例（0.0 - 1.0）"
    )
    parser.add_argument(
        "--asphalt_frangi_threshold",
        dest="asphalt_frangi_threshold",
        type=float,
        metavar="ASPHALT_FRANGI_THRESHOLD",
        default=0.01,
        help="铺装路面提取滤波算法阈值"
    ) 
    parser.add_argument(
        "--asphalt_frangi_beta",
        dest="asphalt_frangi_beta",
        type=float,
        metavar="ASPHALT_FRANGI_BETA",
        default=0.5,
        help="铺装路面提取滤波算法 Beta 参数"
    )
    parser.add_argument(
        "-C", "--classes",
        dest="classes",
        nargs='+',
        default=['Water','Buildings','Vegetation'],
        help="提取地物类别"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="输出详细调试信息"
    )
    parser.add_argument(
        "-e", "--extract_asphalt",
        action="store_true",
        help="提取铺装路面"
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
    scene = trimesh.Scene()
    X, Y = np.meshgrid(x, y)
    if args.extract_asphalt:
        surface_height['Terrain'], asphalt_counts, asphalt_height, asphalt_intensity = extract_asphalt(
            surface_counts['Terrain'],
            surface_height['Terrain'],
            surface_intensity['Terrain'],
            min(scale_x, scale_y),
            asphalt_thickness=args.asphalt_thickness,
            segments_acceptance=args.asphalt_segments_acceptance,
            frangi_threshold=args.asphalt_frangi_threshold,
            min_width_m=args.asphalt_min_width,
            max_width_m=args.asphalt_max_width,
            frangi_beta=args.asphalt_frangi_beta,
            verbose=args.verbose)
        surface_counts['Asphalt'] = asphalt_counts
        surface_height['Asphalt'] = asphalt_height
        surface_intensity['Asphalt'] = asphalt_intensity
        args.classes.append('Asphalt')
    terrain_mesh = generate_terrain_solid_optimized(X, Y, surface_height['Terrain'], args.base_z)
    terrain_output = stl_output + '_Terrain.stl'
    if terrain_mesh.is_watertight:
        print("Watertight terrain component is generated including {} vertices and {} faces.".format(
            len(terrain_mesh.vertices), len(terrain_mesh.faces)))
        terrain_mesh.export(terrain_output)
        print(f"Terrain component is saved to {terrain_output}")
    else:
        print("Open or non-manifold edges detected.")
    scene.add_geometry(terrain_mesh, node_name='Terrain', parent_node_name='world')

    for class_name in args.classes:
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
        print("{} generated {} solid objects.".format(class_name, len(meshes)))
        for i in range(len(meshes)):
            color = get_class_color(palette)
            meshes[i].visual.face_colors = color
        obj_mesh = trimesh.util.concatenate(meshes)
        obj_mesh.visual.face_colors = get_class_color(palette)
        obj_output = stl_output + '_{}.stl'.format(class_name)
        obj_mesh.export(obj_output)
        print(f"{class_name} component is saved to {obj_output}")
        scene.add_geometry(obj_mesh, node_name=class_name, parent_node_name='world')

    if scene.is_empty:
        print("Scene is empty!")
    else:
        glb_output = stl_output + '_Scene.3mf'
        scene.export(glb_output)
        print(glb_output)
        glb_output = stl_output + '_Scene.obj'
        scene.export(glb_output)
        print(glb_output)
        glb_output = stl_output + '_Scene.glb'
        scene.export(glb_output)
        print(glb_output)

if __name__=='__main__':
    main()
