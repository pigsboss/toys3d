#!/usr/bin/env python
"""Multiple Object TIFF to STL
"""
import numpy as np
import tifffile
import trimesh
import os
import argparse
import trimesh
from laz2tiff import basename_without_all_extensions
from ast import literal_eval

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
    parser = argparse.ArgumentParser(description="多目标 TIFF 图像转 STL 3D 模型")
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
        stl_output = os.path.abspath(os.path.normpath(basename_without_all_extensions(args.tiff_input)))
    surface_height = {}
    surface_intensity = {}
    surface_counts = {}
    with tifffile.TiffFile(tiff_input) as tf:
        scale_x, scale_y, _ = tf.pages[0].tags[33550].value
        rows, cols = tf.pages[0].shape
        num_points = rows * cols
        x = np.arange(cols) * scale_x
        y = np.arange(rows) * scale_y
        for p in tf.pages:
            metadata = literal_eval(p.description)
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
        print(f"Watertight terrain component is generated including {len(terrain_mesh.vertices)} vertices and {len(terrain_mesh.faces)} faces.")
        mesh.export(terrain_output)
        print(f"Terrain component is saved to {terrain_output}")
    else:
        print("Open edges detected.")
    
if __name__=='__main__':
    main()
