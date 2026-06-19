#!/usr/bin/env python
"""Merge swissSURFACE3D LiDAR DSM TIFF
"""
import numpy as np
import tifffile
import os
import argparse
from pathlib import Path
from laz2tiff import basename_without_all_extensions
from ast import literal_eval
from laz2tiff import CLASSES

def main():
    parser = argparse.ArgumentParser(description="swissSURFACE3D LiDAR DSM 多页 TIFF 图像合并")
    parser.add_argument(
        "tiff_dir",
        type=str,
        metavar="TIFF_DIR",
        help="输入 TIFF 文件路径"
    )
    parser.add_argument(
        "-p", "--input_prefix",
        dest="input_prefix",
        type=str,
        default="swisssurface3d_",
        metavar="INPUT_PREFIX",
        help="输入 TIFF 文件前缀"
    )
    parser.add_argument(
        "-o", "--output",
        dest="tiff_output",
        type=str,
        metavar="TIFF_OUTPUT",
        help="输出 TIFF 文件路径"
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
    tiff_dir = Path(os.path.abspath(os.path.normpath(args.tiff_dir)))
    tiff_list = []
    for file_path in tiff_dir.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() == '.tiff' and file_path.name.lower().startswith(args.input_prefix.lower()):
                tiff_list.append(file_path.absolute())
    nframes = len(tiff_list)
    if args.verbose:
        print(f'{nframes} TIFF files found.')
    ref_x_list = []
    ref_y_list = []
    scale_x_list = []
    scale_y_list = []
    rows_list = []
    cols_list = []
    num_pages_list = []
    for tiff_path in tiff_list:
        with tifffile.TiffFile(tiff_path) as tf:
            scale_x, scale_y, _ = tf.pages[0].tags[33550].value
            _, _, _, ref_x, ref_y, _ = tf.pages[0].tags[33922].value
            rows, cols = tf.pages[0].shape
            num_pages = len(tf.pages)
            if args.verbose:
                print(f'  {os.path.basename(tiff_path)}: {num_pages} pages, {rows} x {cols}, ref_x = {ref_x}, ref_y = {ref_y}, scale_x = {scale_x}, scale_y = {scale_y}')
        ref_x_list.append(ref_x)
        ref_y_list.append(ref_y)
        scale_x_list.append(scale_x)
        scale_y_list.append(scale_y)
        rows_list.append(rows)
        cols_list.append(cols)
        num_pages_list.append(num_pages)
    ref_x = np.min(ref_x_list)
    ref_y = np.min(ref_y_list)
    frame_rows = int((np.max(ref_y_list) - ref_y) / scale_y / rows + 1)
    frame_cols = int((np.max(ref_x_list) - ref_x) / scale_x / cols + 1)
    if args.verbose:
        print(f'{num_pages} pages, {frame_rows} x {frame_cols} frames, ref_x = {ref_x}, ref_y = {ref_y}')
    tags = [
        (33550, 'd', 3, (scale_x, scale_y, 0.0), True),
        (33922, 'd', 6, (0.0, 0.0, 0.0, ref_x, ref_y, 0.0), True)]
    dtype_list = []
    class_name_list = []
    class_ids_list = []
    data_list = []
    with tifffile.TiffFile(tiff_list[0]) as tf:
        for i in range(num_pages):
            metadata = literal_eval(tf.pages[i].description)
            dtype_list.append(tf.pages[i].dtype)
            class_name_list.append(metadata['class_name'])
            class_ids_list.append(metadata['class_ids'])
            data_list.append(metadata['data'])
    if args.verbose:
        print('TIFF Pages definition:')
        for i in range(num_pages):
            print(f'  page {i}: {class_name_list[i]} ({class_ids_list[i]}), {data_list[i]} ({dtype_list[i]})')

    with tifffile.TiffWriter(args.tiff_output) as tf_out:
        for i in range(num_pages):
            if args.verbose:
                print(f'merging page {i}...')
            ar = np.zeros((frame_rows * rows, frame_cols * cols), dtype=dtype_list[i])
            if args.verbose:
                print(f'  {ar.shape[0]} x {ar.shape[1]} {ar.dtype} array initiated.')
            for tiff_path in tiff_list:
                if args.verbose:
                    print(f'    loading {os.path.basename(tiff_path)}...')
                with tifffile.TiffFile(tiff_path) as tf_in:
                    _, _, _, local_ref_x, local_ref_y, _ = tf_in.pages[0].tags[33922].value
                    r = int((local_ref_y - ref_y) / scale_y)
                    c = int((local_ref_x - ref_x) / scale_x)
                    R = r // rows
                    C = c // cols
                    ar[r:r+rows,c:c+cols] = tf_in.pages[i].asarray()
                if args.verbose:
                    print(f'    frame ({R}, {C}) filled.')
            if args.verbose:
                print(f'  page {i} array filled.')
            tf_out.write(
                ar,
                compression='lzw',
                photometric='minisblack',
                metadata={
                    'class_name':class_name_list[i],
                    'class_ids':class_ids_list[i],
                    'data':data_list[i]},
                extratags=tags)

if __name__=='__main__':
    main()
