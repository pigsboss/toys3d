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
from shapely.geometry import Point
from shapely.validation import make_valid
from shapely.ops import triangulate
from scipy.interpolate import RegularGridInterpolator
from collections import Counter

def extrude_object_solid(X, Y, terrain_height, obj_counts, obj_height,
                         obj_area_threshold=3.0, weld_thickness=0.1,
                         verbose=False, use_convex_hull=False):
    """
    使用 trimesh.creation.extrude_polygon 生成水密实体，再通过插值调整顶面高程。
    """
    import cv2
    import numpy as np
    import trimesh
    from scipy.interpolate import RegularGridInterpolator
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    from shapely.geometry.polygon import orient

    scale_x = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
    scale_y = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
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
        area_pixels = stats[i, cv2.CC_STAT_AREA]
        if area_pixels * scale_x * scale_y < obj_area_threshold:
            continue
        if verbose:
            print(f"  Object {i}, area = {area_pixels} pixels")

        mask = (labels == i)
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        y0, y1 = max(0, y), min(rows, y + h)
        x0, x1 = max(0, x), min(cols, x + w)
        sub_mask = mask[y0:y1, x0:x1]
        if sub_mask.sum() == 0:
            continue

        # 创建局部插值器 (y 递增, x 递增)
        y_local = np.linspace(Y[y0, 0], Y[y1-1, 0], h)
        x_local = np.linspace(X[0, x0], X[0, x1-1], w)

        interp_top = RegularGridInterpolator(
            (y_local, x_local),
            obj_height[y0:y1, x0:x1],
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        interp_bot = RegularGridInterpolator(
            (y_local, x_local),
            terrain_height[y0:y1, x0:x1],
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # 提取轮廓
        sub_mask_uint8 = (sub_mask.astype(np.uint8) * 255)
        contours, hierarchy = cv2.findContours(sub_mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None or len(contours) == 0:
            continue

        # 处理每个外轮廓
        for idx, cnt in enumerate(contours):
            if hierarchy is None or hierarchy[0][idx][3] != -1:
                continue

            # 外轮廓简化
            epsilon = 0.0001 * cv2.arcLength(cnt, True)
            approx_outer = cv2.approxPolyDP(cnt, epsilon, True).squeeze()
            if len(approx_outer.shape) != 2 or approx_outer.shape[0] < 3:
                continue
            outer_world = approx_outer.astype(np.float64).copy()
            outer_world[:, 0] = outer_world[:, 0] * scale_x + origin_x + (x0 * scale_x)
            outer_world[:, 1] = outer_world[:, 1] * scale_y + origin_y + (y0 * scale_y)

            # 收集内孔
            holes_world = []
            for j, hcnt in enumerate(contours):
                if hierarchy[0][j][3] == idx:
                    epsilon_h = 0.001 * cv2.arcLength(hcnt, True)
                    approx_hole = cv2.approxPolyDP(hcnt, epsilon_h, True).squeeze()
                    if len(approx_hole.shape) != 2 or approx_hole.shape[0] < 3:
                        continue
                    hole_world = approx_hole.astype(np.float64).copy()
                    hole_world[:, 0] = hole_world[:, 0] * scale_x + origin_x + (x0 * scale_x)
                    hole_world[:, 1] = hole_world[:, 1] * scale_y + origin_y + (y0 * scale_y)
                    holes_world.append(hole_world)

            # 构建 shapely polygon
            try:
                poly = Polygon(outer_world, holes_world)
                if not poly.is_valid:
                    poly = make_valid(poly)
                poly = orient(poly, sign=1.0)
            except Exception as e:
                if verbose:
                    print(f"      Polygon creation failed: {e}")
                continue

            # 提取所有 Polygon 子几何体
            if poly.geom_type == 'MultiPolygon':
                polys = list(poly.geoms)
            elif poly.geom_type == 'GeometryCollection':
                polys = [geom for geom in poly.geoms if geom.geom_type == 'Polygon']
            elif poly.geom_type == 'Polygon':
                polys = [poly]
            else:
                if verbose:
                    print(f"      Unexpected geometry type: {poly.geom_type}, skip")
                continue

            for sub_poly in polys:
                if sub_poly.geom_type != 'Polygon':
                    continue
                if sub_poly.area < obj_area_threshold:
                    continue

                # --- 使用 extrude_polygon 生成水密实体 ---
                # 先以一个临时高度挤出（例如 1.0 米），后面调整顶点
                approx_height = 1.0
                mesh = trimesh.creation.extrude_polygon(sub_poly, height=approx_height)

                # 获取顶点
                verts = mesh.vertices.copy()
                n_verts = len(verts)
                # extrude_polygon 先输出底面顶点（逆时针），再输出顶面顶点（相同数量，相同顺序）
                n_base = n_verts // 2
                if n_verts % 2 != 0:
                    if verbose:
                        print(f"      Unexpected vertex count {n_verts}, skip")
                    continue

                # 获取所有轮廓点的世界坐标（用于插值）
                # 收集外环 + 所有内环的点（按 shapely 顺序）
                all_contour_pts = []
                all_contour_pts.extend(list(sub_poly.exterior.coords)[:-1])  # 不重复闭合点
                for hole in sub_poly.interiors:
                    all_contour_pts.extend(list(hole.coords)[:-1])

                # 确保数量匹配
                if len(all_contour_pts) != n_base:
                    if verbose:
                        print(f"      Contour points mismatch: {len(all_contour_pts)} vs {n_base}, skip")
                    continue

                # 对每个轮廓点插值得到顶面和底面的高度
                for j, pt in enumerate(all_contour_pts):
                    top_z = interp_top((pt[1], pt[0]))
                    bot_z = interp_bot((pt[1], pt[0])) - weld_thickness
                    # 处理 NaN
                    if np.isnan(top_z):
                        top_z = 0.0
                    if np.isnan(bot_z):
                        bot_z = 0.0
                    # 底面顶点（前一半）
                    verts[j, 2] = bot_z
                    # 顶面顶点（后一半）
                    verts[j + n_base, 2] = top_z

                mesh.vertices = verts
                mesh.fix_normals()

                # 验证水密性
                if not mesh.is_watertight:
                    mesh.fill_holes()
                    mesh.remove_unreferenced_vertices()
                    mesh.fix_normals()

                if mesh.is_watertight and len(mesh.vertices) > 0:
                    meshes.append(mesh)
                    if verbose:
                        print(f"  Watertight solid of object {i} generated: "
                              f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")
                else:
                    if verbose:
                        print(f"  Object {i} mesh not watertight after fix.")

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
