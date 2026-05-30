#!/usr/bin/env python
import numpy as np
import sys
import os
from xyz2tiff import save_tiff
from showtiff import load_tiff
import argparse
import matplotlib.pyplot as plt

def crop_type(arg_string):
    """
    自定义解析函数：将 'x,y,width,height' 字符串解析并校验为整数元组
    """
    try:
        # 尝试按逗号分割，并将每部分转换为整数
        x, y, w, h = map(int, arg_string.split(','))
        return (x, y, w, h)
    except ValueError:
        # 抛出 ArgumentTypeError，argparse 会自动捕获并生成漂亮的报错信息
        raise argparse.ArgumentTypeError(
            f"裁切参数格式错误: '{arg_string}'。必须是 4 个由逗号分隔的整数，例如 '10,20,100,200'。"
        )

def main():
    parser = argparse.ArgumentParser(description="图像处理参数解析示例")
    parser.add_argument(
        "input_file",             # 在代码中通过 args.input_file 调用
        type=str,
        help="需要处理的输入图像文件路径"
    )
    parser.add_argument(
        "-s", "--save",
        dest="save_path",      # 存储到 args.save_path
        type=str,
        metavar="PATH",        # 在 help 信息中显示的占位符
        help="指定保存文件的路径"
    )
    parser.add_argument(
        "-c", "--crop",
        dest="crop_region",    # 存储到 args.crop_region
        type=crop_type,        # 核心：使用自定义的校验和转换逻辑
        metavar="X,Y,W,H",
        help="指定图像裁切区域，格式为 x,y,width,height（例如：10,20,100,200）"
    )
    parser.add_argument(
        "-e", "--exaggerate",
        dest="exaggerate",
        type=float,
        default=1.0,
        metavar="EXAGGERATE",
        help="垂直夸张系数"
    )
    args = parser.parse_args()
    if args.crop_region:
        x, y, w, h = args.crop_region
        print(f"裁切区域 (Crop): X={x}, Y={y}, 宽度={w}, 高度={h}")
    else:
        print("未指定裁切区域。")
    tiff_input = os.path.abspath(os.path.normpath(sys.argv[1]))
    cdata, info = load_tiff(tiff_input)
    z = cdata[y:y+h,x:x+w]
    if args.exaggerate:
        print("垂直夸张系数：{:f}".format(args.exaggerate))
        base_z = np.min(z.ravel())
        z = (z-base_z) * args.exaggerate + base_z
    if args.save_path:
        print(f"保存路径 (Save Path): {args.save_path}")
        tiff_output = os.path.abspath(os.path.normpath(args.save_path))
        save_tiff(tiff_output, z, {
            'scale_x': info['scale_x'],
            'scale_y': info['scale_y'],
            'ref_x': info['ref_x'] + info['scale_x']*(x-info['ref_col']),
            'ref_y': info['ref_y'] + info['scale_y']*(y-info['ref_row'])
        })
    else:
        print("未指定保存路径。")
        plt.imshow(z)
        plt.show()

if __name__ == "__main__":
    main()
