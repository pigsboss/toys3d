#!/usr/bin/env python
"""swissSURFACE3D LiDAR DSM TIFF to RGBA 4-channel PNG
"""
import numpy as np
import tifffile
import os
import argparse
import cv2
from laz2tiff import basename_without_all_extensions
from ast import literal_eval
from laz2tiff import CLASSES
from time import time
from s3dtiff2stl import extract_asphalt

CLASS_PALETTES = {
    'Terrain':          ((0.431, 0.333, 0.235), (0.612, 0.647, 0.529)), # 泥土 -- 灰苔
    'Water':            ((0.000, 0.000, 0.545), (0.125, 0.698, 0.667)),
    'Buildings':        ((0.545, 0.227, 0.169), (0.824, 0.706, 0.549)), # Geneva theme
    'Vegetation':       ((0.333, 0.420, 0.184), (0.604, 0.804, 0.196)),
    'Asphalt':          ((0.176, 0.176, 0.176), (0.314, 0.314, 0.314)),
    'Wire':             ((0.439, 0.502, 0.565), (0.663, 0.663, 0.663)),
    'Masts':            ((0.169, 0.169, 0.169), (0.184, 0.310, 0.310)),
    'Bridge_Deck':      ((0.290, 0.290, 0.290), (0.314, 0.314, 0.314)),
    'Building_facades': ((0.627, 0.322, 0.176), (0.961, 0.961, 0.863)),
    'Bridge_piers':     ((0.753, 0.753, 0.753), (0.827, 0.824, 0.780)),
}

def main():
    parser = argparse.ArgumentParser(description="swissSURFACE3D LiDAR DSM 多页 TIFF 图像转 RGBA 4 通道彩色 PNG 图像")
    parser.add_argument(
        "tiff_input",
        type=str,
        metavar="TIFF_INPUT",
        help="输入 TIFF 文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        dest="png_output",
        type=str,
        metavar="PNG_OUTPUT",
        help="输出 PNG 文件路径前缀"
    )
    parser.add_argument(
        "-C", "--classes",
        dest="classes",
        nargs='+',
        default=['Water','Buildings','Vegetation', 'Wire', 'Masts', 'Bridge_Deck', 'Building_facades', 'Bridge_piers'],
        help="绘制地物类型"
    )
    parser.add_argument(
        "-m", "--mapping",
        dest="mapping",
        type=str,
        default='hybrid',
        help="高程/强度/混合映射"
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
        "-q", "--quiet",
        action="store_true",
        help="静默运行"
    )
    args = parser.parse_args()
    tiff_input = os.path.abspath(os.path.normpath(args.tiff_input))
    if args.png_output:
        png_output = os.path.abspath(os.path.normpath(args.png_output))
    else:
        png_output = os.path.abspath(os.path.normpath(
            basename_without_all_extensions(args.tiff_input))) + '.png'
    surface_height = {}
    surface_intensity = {}
    surface_counts = {}
    surface_classes = []
    with tifffile.TiffFile(tiff_input) as tf:
        scale_x, scale_y, _ = tf.pages[0].tags[33550].value
        rows, cols = tf.pages[0].shape
        num_points = rows * cols
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

    if args.extract_asphalt:
        surface_height['Terrain'], counts, height, intensity = extract_asphalt(
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
        surface_counts['Asphalt'] = counts
        surface_height['Asphalt'] = height
        surface_intensity['Asphalt'] = intensity
        surface_classes.append('Asphalt')
        args.classes.insert(0, 'Asphalt')

    terrain_mask = (surface_counts['Terrain'] > 0)
    rgb = np.zeros((rows, cols, 3), dtype = 'uint16')
    for class_name in args.classes:
        if class_name.lower() == 'unclassified':
            continue
        if class_name not in surface_classes:
            print(f'{class_name} not found.')
            continue
        rgb_min, rgb_max = CLASS_PALETTES[class_name]
        if args.mapping.lower().startswith('i'):
            # intensity map
            v = surface_intensity[class_name]
        elif args.mapping.lower().startswith('he'):
            # height map
            if class_name == 'Water':
                v = surface_height['Water'] - surface_height['Terrain']
            else:
                v = surface_height[class_name]
        elif args.mapping.lower().startswith('hy'):
            # hybrid map
            if class_name == 'Water':
                v = surface_height['Water'] - surface_height['Terrain']
            else:
                v = surface_intensity[class_name]
        else:
            raise TypeError(f'{args.mapping} is not supported.')
        m = (surface_counts[class_name] > 0)
        terrain_mask = np.logical_and(terrain_mask, ~m)
        if not np.any(m):
            continue
        try:
            v2, v98 = np.percentile(v[m], (2, 98))
        except:
            print(f'{np.size(v[m])}')
            v2 = np.min(v[m])
            v98 = np.max(v[m])
        for c in range(3):
            rgb[m, c] = np.uint16((np.clip((v[m]-v2) / (v98-v2), 0., 1.) * (rgb_max[c]-rgb_min[c]) + rgb_min[c]) * 65535.5)

    rgb_min, rgb_max = CLASS_PALETTES['Terrain']
    if args.mapping.lower().startswith('i'):
        # intensity map
        v = surface_intensity['Terrain']
    elif args.mapping.lower().startswith('he'):
        # height map
        v = surface_height['Terrain']
    elif args.mapping.lower().startswith('hy'):
        # hybrid map
        v = surface_intensity['Terrain']
    else:
        raise TypeError(f'{args.mapping} is not supported.')
    try:
        v2, v98 = np.percentile(v[terrain_mask], (2, 98))
    except:
        print(f'{np.size(v[terrain_mask])}')
        v2 = np.min(v[terrain_mask])
        v98 = np.max(v[terrain_mask])
    for c in range(3):
        rgb[terrain_mask, c] = np.uint16((np.clip((v[terrain_mask]-v2) / (v98-v2), 0., 1.) * (rgb_max[c]-rgb_min[c]) + rgb_min[c]) * 65535.5)

    if args.verbose:
        print("  RGB map complete.")
    cv2.imwrite(
        png_output,
        rgb[::-1,:,::-1],
        [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not args.quiet:
        print("  {} saved.".format(png_output))
    
if __name__=='__main__':
    main()
