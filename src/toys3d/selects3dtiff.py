#!/usr/bin/env python
"""Select swissSURFACE3D LiDAR DSM TIFF
"""
import numpy as np
import tifffile
import os
import argparse
from laz2tiff import basename_without_all_extensions
from ast import literal_eval
from laz2tiff import CLASSES

def rectangle_type(arg_string):
    """
    自定义解析函数：将 'x,y,width,height' 字符串解析并校验为整数元组
    """
    try:
        x, y, w, h = map(int, arg_string.split(','))
        return (x, y, w, h)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"裁切参数格式错误: '{arg_string}'。必须是 4 个由逗号分隔的数值，例如 '10,20,100,200'。"
        )

def main():
    parser = argparse.ArgumentParser(description="swissSURFACE3D LiDAR DSM 多页 TIFF 图像合并")
    parser.add_argument(
        "tiff_input",
        type=str,
        metavar="TIFF_DIR",
        help="输入 TIFF 文件路径"
    )
    parser.add_argument(
        "-r", "--rectangle",
        dest="rectangle",
        type=rectangle_type,
        metavar="RECTANGLE",
        help="选取矩形区域（x, y, width, height），单位：像素"
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
    tiff_input = os.path.abspath(os.path.normpath(args.tiff_input))
    x, y, w, h = args.rectangle
    dtype_list = []
    class_name_list = []
    class_ids_list = []
    data_list = []
    with tifffile.TiffFile(tiff_input) as tf:
        scale_x, scale_y, _ = tf.pages[0].tags[33550].value
        _, _, _, ref_x, ref_y, _ = tf.pages[0].tags[33922].value
        rows, cols = tf.pages[0].shape
        num_pages = len(tf.pages)
        if args.verbose:
            print(f'{os.path.basename(tiff_input)}: {num_pages} pages, {rows} x {cols}, ref_x = {ref_x}, ref_y = {ref_y}, scale_x = {scale_x}, scale_y = {scale_y}')
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
        tags = [
            (33550, 'd', 3, (scale_x, scale_y, 0.0), True),
            (33922, 'd', 6, (0.0, 0.0, 0.0, ref_x + x/scale_x, ref_y + y/scale_y, 0.0), True)]
        with tifffile.TiffWriter(args.tiff_output) as tf_out:
            for i in range(num_pages):
                ar = np.zeros((h, w), dtype=dtype_list[i])
                ar[:,:] = tf.pages[i].asarray()[y:y+h, x:x+w]
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
