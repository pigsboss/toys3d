#!/usr/bin/env python
import laspy
import tifffile
import argparse
import numpy as np
import cv2
import os

# swissSURFACE3D (ASPRS标准) 分类码C
CLASSES = {
    "Unclassified": [1],      #  1: 未分类
    "Terrain": [2],           #  2: 地面
    "Vegetation": [3, 4, 5],  #  3,4,5: 低/中/高植被
    "Buildings": [6],         #  6: 建筑物
    "Water": [9],             #  9: 水体
    "Wire": [14],             # 14: 电线
    "Masts": [15],            # 15: 塔架
    "Bridge_Deck": [17],      # 17: 桥面
    "Building_facades": [26], # 26: 建筑物外立面
    "Bridge_piers": [27]      # 27: 桥墩
}

laz_xdim = 100000  # LAZ x-axis dimension in cm
laz_ydim = 100000  # LAZ y-axis dimension in cm

def basename_without_all_extensions(path):
    basename = os.path.basename(path)  # 仅文件名部分，不含路径
    while True:
        basename, ext = os.path.splitext(basename)
        if not ext:          # 没有扩展名了
            break
    return basename

def build_meshgrid(laz_data, resolution):
    info = {
        'ref_x': laz_data.X.min() / 100.,
        'ref_y': laz_data.Y.min() / 100.,
        'scale_x': resolution,
        'scale_y': resolution
    }
    return info

def xy2pixel(x, y, info):
    row = np.int32((y-info['ref_y'])/info['scale_y'])
    col = np.int32((x-info['ref_x'])/info['scale_x'])
    pid = np.int32(row*laz_xdmin/100./info['scale_x'])
    return row, col, pid

def rasterize_laz(laz_data, class_ids):
    pass

def main():
    parser = argparse.ArgumentParser(description="LAZ 激光测高点云文件转换 TIFF 图像")
    parser.add_argument(
        "laz_input",
        type=str,
        metavar="LAZ_INPUT",
        help="输入 LAZ 点云文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        dest="tiff_output",
        type=str,
        metavar="TIFF_OUTPUT",
        help="输出 TIFF 图像文件路径"
    )
    parser.add_argument(
        "-r", "--resolution",
        dest="resolution",
        type=float,
        default=1.0,
        metavar="RESOLUTION",
        help="栅格化分辨率（单位：米）"
    )
    parser.add_argument(
        "-c", "--classes",
        dest="classes",
        nargs="+",
        type=int,
        default=[2,3,4,5,6,9],
        metavar="CLASSES",
        help="提取地物种类"
    )
    parser.add_argument(
        "-p", "--png_output",
        action="store_true",
        help="保存 PNG 格式用于快速预览"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="输出详细调试信息"
    )
    args = parser.parse_args()
    laz_input = os.path.abspath(os.path.normpath(args.laz_input))
    if args.tiff_output:
        tiff_output = os.path.abspath(os.path.normpath(args.tiff_output))
    else:
        tiff_output = os.path.abspath(os.path.normpath(
            os.path.join(
                os.path.dirname(laz_input),
                basename_without_all_extensions(laz_input) + '.tiff')))
    with laspy.open(laz_input) as laz:
        laz_data = laz.read()
    print(args.classes)
    print(tiff_output)
    info = build_meshgrid(laz_data, args.resolution)
    print(info)

if __name__=='__main__':
    main()
