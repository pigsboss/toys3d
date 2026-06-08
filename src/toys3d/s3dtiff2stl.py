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
from scipy.spatial import Delaunay
from laz2tiff import CLASSES

def extrude_object_solid(X, Y, terrain_height, obj_counts, obj_height, obj_area_threshold=3.0, weld_thickness=0.1, verbose=False, use_convex_hull=False):
    _ = use_convex_hull  # 忽略旧参数
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(obj_counts > 0), 4, cv2.CV_32S
    )
    if verbose:
        print("  {} objects extracted.".format(num_labels))
    meshes = []
    rows, cols = X.shape
    scale_x = np.mean(np.diff(X, axis=1))
    scale_y = np.mean(np.diff(Y, axis=0))

    for i in range(1, num_labels):
        if stats[i, 4] * scale_x * scale_y < obj_area_threshold:
            continue
        if verbose:
            print("  Object {}, area = {} pixels".format(i, stats[i, 4]))

        mask = (labels == i)
        idx = np.argwhere(mask)          # (N, 2)
        N = idx.shape[0]
        if N < 3:
            continue

        # ---- 顶面顶点 ----
        vtx_top = np.empty((N, 3), dtype=np.float64)
        vtx_top[:, 0] = X[mask].ravel()
        vtx_top[:, 1] = Y[mask].ravel()
        vtx_top[:, 2] = obj_height[mask].ravel()

        # ---- 底面顶点 ----
        vtx_bot = np.empty((N, 3), dtype=np.float64)
        vtx_bot[:, 0] = vtx_top[:, 0]
        vtx_bot[:, 1] = vtx_top[:, 1]
        vtx_bot[:, 2] = terrain_height[mask].ravel() - weld_thickness

        # ---- 像素 -> 顶点索引映射 ----
        vtx_idx = np.full((rows, cols), -1, dtype=np.int32)
        vtx_idx[mask] = np.arange(N, dtype=np.int32)

        # ---- 顶面三角形：基于像素网格，覆盖所有 mask 像素 ----
        top_faces = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                p00 = mask[r, c]
                p01 = mask[r, c+1]
                p10 = mask[r+1, c]
                p11 = mask[r+1, c+1]
                # 四个角点的顶点索引
                idx = [
                    vtx_idx[r, c]     if p00 else -1,
                    vtx_idx[r, c+1]   if p01 else -1,
                    vtx_idx[r+1, c]   if p10 else -1,
                    vtx_idx[r+1, c+1] if p11 else -1
                ]
                valid = [p00, p01, p10, p11]
                count = sum(valid)
                if count == 4:
                    # 四个有效 → 两个三角形
                    top_faces.append([idx[0], idx[1], idx[3]])
                    top_faces.append([idx[0], idx[3], idx[2]])
                elif count == 3:
                    # 三个有效 → 一个三角形
                    missing = next(i for i, v in enumerate(valid) if not v)
                    tri = [idx[i] for i in range(4) if i != missing]
                    top_faces.append(tri)
                # count <= 2 不生成三角形
        if len(top_faces) == 0:
        if len(top_faces) == 0:
            if verbose:
                print("  Object {} is skipped (no valid grid cells).".format(i))
            continue
        top_faces = np.array(top_faces, dtype=np.int32)
        # 底面三角形（反转绕序）
        bot_faces = top_faces[:, ::-1] + N

        # ---- 侧墙：通过轮廓提取边界 ----
        contours, hierarchy = cv2.findContours(
            np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        side_faces = []
        for contour in contours:
            contour = contour[:, 0, :]  # shape (M, 2)，每行 (c, r)
            # 将轮廓点转为顶点索引
            pts = []
            for (c, r) in contour:
                if 0 <= r < rows and 0 <= c < cols and mask[r, c]:
                    vid = vtx_idx[r, c]
                    if vid >= 0:
                        pts.append(vid)
            if len(pts) < 2:
                continue
            # 沿轮廓生成侧墙四边形
            for j in range(len(pts) - 1):
                t1, t2 = pts[j], pts[j+1]
                b1, b2 = t1 + N, t2 + N
                side_faces.append([t1, b1, b2])
                side_faces.append([t1, b2, t2])
            # 闭合最后一点到第一点
            if len(pts) > 2:
                t1, t2 = pts[-1], pts[0]
                b1, b2 = t1 + N, t2 + N
                side_faces.append([t1, b1, b2])
                side_faces.append([t1, b2, t2])

        if len(side_faces) == 0:
            if verbose:
                print("  Object {} is skipped (no side walls).".format(i))
            continue
        side_faces = np.array(side_faces, dtype=np.int32)

        # ---- 组装 ----
        all_vertices = np.vstack((vtx_top, vtx_bot))
        all_faces = np.vstack((top_faces, bot_faces, side_faces))

        solid_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
        solid_mesh.remove_unreferenced_vertices()
        solid_mesh.process()
        if not solid_mesh.is_watertight:
            solid_mesh.fill_holes()
            solid_mesh.remove_unreferenced_vertices()
        solid_mesh.fix_normals()

        if solid_mesh.is_watertight:
            if verbose:
                print("  Watertight solid of object {} is generated including {} vertices and {} faces.".format(
                    i, len(solid_mesh.vertices), len(solid_mesh.faces)))
            meshes.append(solid_mesh)
        else:
            if verbose:
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
