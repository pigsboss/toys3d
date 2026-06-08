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
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.validation import make_valid
from scipy.interpolate import RegularGridInterpolator
from collections import Counter

def extrude_object_solid(X, Y, terrain_height, obj_counts, obj_height,
                         obj_area_threshold=3.0, weld_thickness=0.1,
                         verbose=False, use_convex_hull=False):
    """
    使用多边形挤出生成水密实体，支持内部孔洞。
    
    参数：
        X, Y: ndarray (rows, cols) 世界坐标网格
        terrain_height: 地面高程网格
        obj_counts: 分类计数网格（用于连通域）
        obj_height: 地物高程网格
        weld_thickness: 地物沉入地面厚度
        use_convex_hull: 忽略（兼容旧参数）
    
    返回：
        meshes: list of trimesh.Trimesh
    """
    # 连通域提取（使用原始计数网格）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(obj_counts > 0), 4, cv2.CV_32S
    )
    if verbose:
        print(f"  {num_labels} objects extracted.")
    
    meshes = []
    rows, cols = X.shape

    # 像素坐标到世界坐标的变换参数
    # X, Y 是均匀网格，可直接通过索引映射
    scale_x = X[0, 1] - X[0, 0] if cols > 1 else 1.0
    scale_y = Y[1, 0] - Y[0, 0] if rows > 1 else 1.0
    origin_x = X[0, 0]
    origin_y = Y[0, 0]

    for i in range(1, num_labels):
        area_pixels = stats[i, cv2.CC_STAT_AREA]
        if area_pixels * scale_x * scale_y < obj_area_threshold:
            continue
        if verbose:
            print(f"  Object {i}, area = {area_pixels} pixels")

        # mask 和边界矩形
        mask = (labels == i)
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        # 裁剪到图像边界内
        y0, y1 = max(0, y), min(rows, y + h)
        x0, x1 = max(0, x), min(cols, x + w)
        sub_mask = mask[y0:y1, x0:x1]

        # 提取轮廓（层次树模式）
        sub_mask_uint8 = sub_mask.astype(np.uint8) * 255
        # 使用 RETR_TREE 以获取内外轮廓关系
        contours, hierarchy = cv2.findContours(sub_mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None or len(contours) == 0:
            if verbose:
                print(f"  Object {i} skipped (no contours)")
            continue

        # 对于每个外轮廓（hierarchy[0][][3] == -1），构建带孔的多边形
        for idx, cnt in enumerate(contours):
            if hierarchy is None or hierarchy[0][idx][3] != -1:
                continue   # 只处理外轮廓

            # 外轮廓顶点（像素坐标）
            outer_poly = cnt.squeeze()
            if len(outer_poly.shape) != 2 or outer_poly.shape[0] < 3:
                continue
            # 多边形简化（基于周长）
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx_outer = cv2.approxPolyDP(cnt, epsilon, True).squeeze()
            if len(approx_outer.shape) != 2 or approx_outer.shape[0] < 3:
                approx_outer = outer_poly  # fallback

            # 转换为世界坐标（像素坐标 + 偏移）
            outer_world = approx_outer.copy().astype(np.float64)
            outer_world[:, 0] = outer_world[:, 0] * scale_x + origin_x + (x0 * scale_x)
            outer_world[:, 1] = outer_world[:, 1] * scale_y + origin_y + (y0 * scale_y)

            # 收集内部孔洞
            holes_world = []
            for j, hcnt in enumerate(contours):
                if hierarchy[0][j][3] == idx:  # 子轮廓（内孔）
                    hole_poly = hcnt.squeeze()
                    if len(hole_poly.shape) != 2 or hole_poly.shape[0] < 3:
                        continue
                    epsilon_h = 0.02 * cv2.arcLength(hcnt, True)
                    approx_hole = cv2.approxPolyDP(hcnt, epsilon_h, True).squeeze()
                    if len(approx_hole.shape) != 2 or approx_hole.shape[0] < 3:
                        approx_hole = hole_poly
                    hole_world = approx_hole.copy().astype(np.float64)
                    hole_world[:, 0] = hole_world[:, 0] * scale_x + origin_x + (x0 * scale_x)
                    hole_world[:, 1] = hole_world[:, 1] * scale_y + origin_y + (y0 * scale_y)
                    # 确保方向为顺时针（shapely 要求内孔顺时针）
                    holes_world.append(hole_world)

            # 构建 shapely 多边形
            try:
                poly = Polygon(outer_world, holes_world)
                if not poly.is_valid:
                    poly = make_valid(poly)
                poly = orient(poly, sign=1.0)  # 外逆时针，内顺时针
            except Exception as e:
                if verbose:
                    print(f"      Failed to create polygon: {e}")
                continue

            # 计算物体平均高度和地面平均高度（在完整 mask 范围内）
            mask_obj_height = obj_height[mask]
            mask_terrain_height = terrain_height[mask]
            if mask_obj_height.size == 0:
                continue
            avg_obj_z = np.mean(mask_obj_height)
            avg_terrain_z = np.mean(mask_terrain_height)
            extrude_height = avg_obj_z - (avg_terrain_z - weld_thickness)
            if extrude_height <= 0:
                if verbose:
                    print(f"      Object {i} skipped (extrude height <= 0)")
                continue

            # 挤出生成实体
            try:
                solid_mesh = trimesh.creation.extrude_polygon(poly, height=extrude_height)
            except Exception as e:
                if verbose:
                    print(f"      Extrude failed: {e}")
                continue

            # 平移 mesh 使底面位于地面高程 - weld_thickness
            # extrude_polygon 默认底面在 Z=0，顶面在 Z=height
            # 我们需要将 Z 平移到 avg_terrain_z - weld_thickness
            solid_mesh.apply_translation([0, 0, avg_terrain_z - weld_thickness])
            # 法线修复
            solid_mesh.fix_normals()
            # 可选：简化网格以减少面数（但可能影响水密性）
            # solid_mesh = solid_mesh.simplify_quadratic_decimation(face_count=some)
            
            # 检查水密性
            if not solid_mesh.is_watertight:
                solid_mesh.fill_holes()
                solid_mesh.remove_unreferenced_vertices()
                solid_mesh.fix_normals()
            
            if solid_mesh.is_watertight:
                if verbose:
                    print(f"  Watertight solid of object {i} is generated including "
                          f"{len(solid_mesh.vertices)} vertices and {len(solid_mesh.faces)} faces.")
                meshes.append(solid_mesh)
            else:
                if verbose:
                    print(f"  Open edges detected on object {i} (non-manifold after all)")
                    # 调试信息
                    edges = solid_mesh.edges_sorted
                    edge_count = Counter(tuple(e) for e in edges)
                    boundary = [e for e, cnt in edge_count.items() if cnt == 1]
                    non_manifold = [e for e, cnt in edge_count.items() if cnt > 2]
                    print(f"      boundary edges: {len(boundary)}, non-manifold edges: {len(non_manifold)}")
        # end for each outer contour
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

    for class_name in ['Water']:
        if class_name.lower() == 'unclassified':
            continue
        palette = CLASS_PALETTES.get(class_name, [(128, 128, 128)])
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
