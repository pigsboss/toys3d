#!/usr/bin/env python
import laspy
import tifffile
import argparse
import numpy as np
import cv2
import os
import ast
import reverse_geocoder as rg
import zipfile
from showtiff import convert_swiss_xyz_to_wgs84
from skimage.restoration import inpaint_biharmonic

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
    basename = os.path.basename(path)
    while True:
        basename, ext = os.path.splitext(basename)
        if not ext:
            break
    return basename

def build_meshgrid(laz_data, resolution):
    grid_info = {
        'ref_x': np.floor(np.min(laz_data.xyz[:, 0])/resolution)*resolution,
        'ref_y': np.floor(np.min(laz_data.xyz[:, 1])/resolution)*resolution,
        'scale_x': resolution,
        'scale_y': resolution
    }
    return grid_info

def xy2pixel(x, y, info):
    row = np.int32((y-info['ref_y'])/info['scale_y'])
    col = np.int32((x-info['ref_x'])/info['scale_x'])
    pid = np.int32(row*laz_xdim/100./info['scale_x']+col)
    return row, col, pid

def rasterize_laz(laz_data, grid_info, class_ids, verbose=False):
    mask = np.isin(laz_data.classification, class_ids)
    if verbose:
        print("  Class IDs {}: {:d} points".format(class_ids, np.sum(mask)))
    row, col, pid = xy2pixel(laz_data.xyz[mask, 0], laz_data.xyz[mask, 1], grid_info)
    ncols = int(laz_xdim/100./grid_info['scale_x'])
    nrows = int(laz_ydim/100./grid_info['scale_y'])
    pixels = nrows * ncols
    counts = np.bincount(pid, minlength=pixels).reshape((nrows, ncols)).astype('float64')
    height = np.bincount(pid, weights=laz_data.xyz[mask, 2], minlength=pixels).reshape((nrows, ncols)) / np.clip(counts, 1.0, None)
    intensity = np.bincount(pid, weights=laz_data.intensity[mask], minlength=pixels).reshape((nrows, ncols)) / np.clip(counts, 1.0, None)
    return counts, height, intensity

def main():
    parser = argparse.ArgumentParser(description="LAZ/LAS 激光测高点云文件转换 TIFF 图像")
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
        default=[1,2,3,4,5,6,9,14,15,17,26,27], # all classes
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
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="静默运行"
    )
    parser.add_argument(
        "-i", "--inpaint",
        action="store_true",
        help="填充地物缺失像素"
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
    if laz_input.lower().endswith('.laz'):
        with laspy.open(laz_input) as laz:
            laz_data = laz.read()
    elif laz_input.lower().endswith('.las.zip'):
        with zipfile.ZipFile(laz_input, 'r') as zf:
            all_files = zf.namelist()
            assert all_files[0].lower().endswith('.las')
            laz_data = laspy.read(zf.read(all_files[0]))
    else:
        raise TypeError('{} is not supported.')
    if not args.quiet:
        print("LAZ data read from {}".format(laz_input))
    grid_info = build_meshgrid(laz_data, args.resolution)
    if args.verbose:
        lon, lat = convert_swiss_xyz_to_wgs84(grid_info['ref_x']+laz_xdim/100., grid_info['ref_y']+laz_ydim/100.)
        print("  Reference [0,0]: X = {} m, Y = {} m".format(grid_info["ref_x"], grid_info["ref_y"]))
        print("  Center: LON = {}, LAT = {}".format(lon, lat))
        results = rg.search((lat, lon))
        for res in results:
            print("  Country: {}, City: {}, Province: {}".format(res["cc"], res["name"], res["admin1"]))
    tags = [
        (33550, 'd', 3, (grid_info['scale_x'], grid_info['scale_y'], 0.0), True),
        (33922, 'd', 6, (0.0, 0.0, 0.0, grid_info['ref_x'], grid_info['ref_y'], 0.0), True)]
    with tifffile.TiffWriter(tiff_output) as tif:
        for class_name in CLASSES:
            class_ids = []
            for i in CLASSES[class_name]:
                if i in args.classes:
                    class_ids.append(i)
            if not class_ids:
                continue
            counts, height, intensity = rasterize_laz(laz_data, grid_info, class_ids, args.verbose)
            if args.inpaint:
                if class_name.lower() == 'terrain':
                    if args.verbose:
                        print("  Terrain map inpainting...")
                    mask = np.uint8(counts==0) # terrain missing pixels
                    height[:] = inpaint_biharmonic(height, mask)
                    intensity[:] = inpaint_biharmonic(intensity, mask)
                elif class_name.lower() in ['buildings', 'vegetation', 'water', 'building_facades']:
                    if args.verbose:
                        print(f"  {class_name} map inpainting...")
                    noobj = np.uint8((counts<1)) # object missing pixels
                    # 背景：有测量值的区域；前景：无测量值的区域。
                    # 图像分割之后，有测量值的区域标签为0，无测量值的区域标签从1开始。
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(noobj, 4, cv2.CV_32S)
                    mask = np.zeros(noobj.shape, dtype='uint8')
                    for i in range(1, num_labels):
                        if args.verbose:
                            print("  {} map missing component {}: X={}, Y={}, Width={}, Height={}, Area={}".format(
                                class_name, i, stats[i,0], stats[i,1], stats[i,2], stats[i,3], stats[i,4]))
                        if (stats[i,2] < 3) or (stats[i,3] < 3) or (stats[i,4] < 7):
                            mask = np.logical_or(mask, labels==i)
                            if args.verbose:
                                print("  Add component {} to inpainting list".format(i))
                    height[:] = inpaint_biharmonic(height, mask)
                    intensity[:] = inpaint_biharmonic(intensity, mask)
                    # counts[mask] = 1
            tif.write(
                counts.astype('uint16'),
                compression='lzw',
                photometric='minisblack',
                metadata={
                    'class_name':class_name,
                    'class_ids':class_ids,
                    'data':'counts'},
                extratags=tags
            )
            if not args.quiet:
                print("  {}:{} counts saved to {}".format(class_name, class_ids, tiff_output))
            tif.write(
                height,
                compression='lzw',
                photometric='minisblack',
                metadata={
                    'class_name':class_name,
                    'class_ids':class_ids,
                    'data':'height'},
                extratags=tags
            )
            if not args.quiet:
                print("  {}:{} height saved to {}".format(class_name, class_ids, tiff_output))
            tif.write(
                intensity,
                compression='lzw',
                photometric='minisblack',
                metadata={
                    'class_name':class_name,
                    'class_ids':class_ids,
                    'data':'intensity'},
                extratags=tags
            )
            if not args.quiet:
                print("  {}:{} intensity saved to {}".format(class_name, class_ids, tiff_output))
            if args.png_output:
                png_output = os.path.splitext(tiff_output)[0]+'_{}'.format(class_name)+'_counts.png'
                cv2.imwrite(
                    png_output,
                    (((counts - counts.min()) / max(1, counts.max() - counts.min())) * 65535.).astype(np.uint16),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not args.quiet:
                    print("  {} saved.".format(png_output))
                png_output = os.path.splitext(tiff_output)[0]+'_{}'.format(class_name)+'_height.png'
                cv2.imwrite(
                    png_output,
                    (((height - height.min()) / max(1, height.max() - height.min())) * 65535.).astype(np.uint16),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not args.quiet:
                    print("  {} saved.".format(png_output))
                png_output = os.path.splitext(tiff_output)[0]+'_{}'.format(class_name)+'_intensity.png'
                cv2.imwrite(
                    png_output,
                    (((intensity - intensity.min()) / max(1, intensity.max() - intensity.min())) * 65535.).astype(np.uint16),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not args.quiet:
                    print("  {} saved.".format(png_output))

if __name__=='__main__':
    main()
