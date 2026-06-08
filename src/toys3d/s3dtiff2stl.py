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
    使用双线性插值恢复真实高程的水密实体生成。
    """
    import cv2
    import numpy as np
    import trimesh
    from scipy.interpolate import RegularGridInterpolator
    from scipy.spatial import KDTree
    from shapely.geometry import Polygon, Point
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

            # 提取所有 Polygon 子几何体（支持 MultiPolygon 和 GeometryCollection）
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

                # === 仅使用多边形简化轮廓点（外轮廓 + 内孔） ===
                bnd_pts = outer_world                               # shape (N,2)
                hole_pts_list = holes_world                         # list of (M,2)
                parts = [bnd_pts] + hole_pts_list
                if not parts:
                    continue
                all_pts = np.concatenate(parts, axis=0)
                # 去重（容差1e-10）
                _, uniq_idx = np.unique(np.round(all_pts, decimals=10), axis=0, return_index=True)
                all_pts = all_pts[np.sort(uniq_idx)]

                # 使用 shapely.ops.triangulate 进行约束三角剖分
                tri_polys = triangulate(sub_poly)

                # 1. 先插入所有边界点（确保索引与侧面完全一致）
                verts_dict = {}
                for ring in [sub_poly.exterior] + list(sub_poly.interiors):
                    for coord in ring.coords[:-1]:  # 忽略闭合重复点
                        key = (round(coord[0], 10), round(coord[1], 10))
                        if key not in verts_dict:
                            verts_dict[key] = len(verts_dict)

                # 2. 再插入 triangulate 生成的顶点（仅新点）
                tri_verts = []
                for tri_poly in tri_polys:
                    idx_tri = []
                    for coord in tri_poly.exterior.coords[:3]:
                        key = (round(coord[0], 10), round(coord[1], 10))
                        if key not in verts_dict:
                            verts_dict[key] = len(verts_dict)
                        idx_tri.append(verts_dict[key])
                    tri_verts.append(idx_tri)

                # 3. 从 verts_dict 构建顶点坐标数组
                all_pts = np.array([list(k) for k in verts_dict.keys()])

                # 插值得到每个顶点的高度
                top_z = interp_top((all_pts[:, 1], all_pts[:, 0]))
                bot_z = interp_bot((all_pts[:, 1], all_pts[:, 0])) - weld_thickness
                top_z = np.nan_to_num(top_z, nan=0.0)
                bot_z = np.nan_to_num(bot_z, nan=0.0)

                N = len(all_pts)
                vertices_top = np.column_stack((all_pts[:, 0], all_pts[:, 1], top_z))
                vertices_bot = np.column_stack((all_pts[:, 0], all_pts[:, 1], bot_z))
                vertices = np.vstack((vertices_top, vertices_bot))

                # 顶面三角形（逆时针，保持 tri_verts 顺序）
                faces_top = np.array(tri_verts)
                # 底面三角形（顺时针）
                faces_bot = np.array(tri_verts)[:, [0, 2, 1]] + N

                # === 侧面：沿多边形的外轮廓和内孔轮廓（直接从 sub_poly 获取顶点索引） ===
                # 利用 verts_dict 获取边界点的精确索引
                def get_contour_indices(ring):
                    indices = []
                    for coord in ring.coords[:-1]:  # 忽略最后一个重复点
                        key = (round(coord[0], 10), round(coord[1], 10))
                        if key in verts_dict:
                            indices.append(verts_dict[key])
                        else:
                            # 理论上不会发生，因为 triangulate 包含了边界点
                            # 若发生则用最近邻（极少见）
                            from scipy.spatial import KDTree
                            tree = KDTree(all_pts)
                            _, idx = tree.query(coord)
                            indices.append(idx)
                    # 去除连续重复（避免退化三角形）
                    unique = []
                    for idx in indices:
                        if not unique or idx != unique[-1]:
                            unique.append(idx)
                    return unique

                side_faces = []

                # 外轮廓
                outer_ring = sub_poly.exterior
                outer_idx = get_contour_indices(outer_ring)
                if len(outer_idx) >= 3:
                    for k in range(len(outer_idx) - 1):
                        a = outer_idx[k]
                        b = outer_idx[k+1]
                        side_faces.append([a, b, b + N])
                        side_faces.append([a, b + N, a + N])
                    # 闭合
                    a = outer_idx[-1]
                    b = outer_idx[0]
                    side_faces.append([a, b, b + N])
                    side_faces.append([a, b + N, a + N])

                # 内孔轮廓
                for interior_ring in sub_poly.interiors:
                    hole_idx = get_contour_indices(interior_ring)
                    if len(hole_idx) >= 3:
                        for k in range(len(hole_idx) - 1):
                            a = hole_idx[k]
                            b = hole_idx[k+1]
                            side_faces.append([a, b, b + N])
                            side_faces.append([a, b + N, a + N])
                        a = hole_idx[-1]
                        b = hole_idx[0]
                        side_faces.append([a, b, b + N])
                        side_faces.append([a, b + N, a + N])

                # 组装网格
                all_faces = np.vstack((faces_top, faces_bot, np.array(side_faces)))
                # 调试输出：顶点数、各组成部分面数、索引合法性
                if verbose:
                    print(f"    Debug: N={N}")
                    print(f"    Debug: faces_top.shape={faces_top.shape}, faces_bot.shape={faces_bot.shape}, side_faces.shape={np.array(side_faces).shape}")
                    print(f"    Debug: all_faces.shape={all_faces.shape}, max index={all_faces.max()}, vertices shape={vertices.shape}")
                    print(f"    Debug: any index >= 2*N? {np.any(all_faces >= 2*N)}")
                    # 检查退化三角形（两个顶点相同）
                    dup_edges = (all_faces[:, 0] == all_faces[:, 1]) | (all_faces[:, 1] == all_faces[:, 2]) | (all_faces[:, 0] == all_faces[:, 2])
                    print(f"    Debug: degenerate faces count={dup_edges.sum()}")
                mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces)
                if verbose:
                    print(f"    Debug after Trimesh construction: watertight? {mesh.is_watertight}")
                mesh.fix_normals()
                if verbose:
                    print(f"    Debug after fix_normals: watertight? {mesh.is_watertight}, euler={mesh.euler_number}, winding_consistent? {mesh.is_winding_consistent}")
                if not mesh.is_watertight:
                    mesh.fill_holes()
                    mesh.remove_unreferenced_vertices()
                    mesh.fix_normals()
                    if verbose:
                        print(f"    Debug after fill_holes+remove_unreferenced: watertight? {mesh.is_watertight}, euler={mesh.euler_number}")
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
