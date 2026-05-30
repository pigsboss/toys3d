#!/usr/bin/env python
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pyproj import Transformer
import reverse_geocoder as rg
import argparse

def convert_swiss_xyz_to_wgs84(x, y, is_lv95=True):
    """
    将瑞士坐标系的 XYZ 文件转换为 WGS84 经纬度文件 (Lon, Lat, Z)
    """
    # 1. 定义源坐标系
    source_crs = "EPSG:2056" if is_lv95 else "EPSG:21781"
    target_crs = "EPSG:4326" # WGS84 经纬度

    print(f"正在配置转换器: {source_crs} -> {target_crs}")
    
    # 2. 初始化转换器
    # ⚠️ 极其关键的参数：always_xy=True 
    # 它确保输出顺序始终是 (经度Longitude, 纬度Latitude)，否则 pyproj 默认可能会输出 (纬度, 经度)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    # 4. 批量执行转换 (NumPy 向量化运算，速度极快)
    print("正在执行坐标投影转换...")
    lon, lat = transformer.transform(x, y)
    return lon, lat

def load_tiff(tifffile_input):
    with tifffile.TiffFile(tifffile_input) as tif:
        info = {}
        nrows, ncols = tif.pages[0].shape
        # 获取第一页的所有标签
        tags = tif.pages[0].tags
        # 1. 解析像素间隔 / 分辨率 (Tag 33550: ModelPixelScaleTag)
        if 33550 in tags:
            # 数据格式为 (ScaleX, ScaleY, ScaleZ)
            scale_x, scale_y, scale_z = tags[33550].value
            info["scale_x"] = scale_x
            info["scale_y"] = scale_y
        else:
            print("未找到像素间隔标签 (33550)。")
        # 2. 解析参考点 / 锚点 (Tag 33922: ModelTiepointTag)
        if 33922 in tags:
            # 数据格式为 6 个值构成的一组: (I, J, K, X, Y, Z)
            # I, J, K 是图像的列(X)、行(Y)、深度(Z)像素索引
            # X, Y, Z 是对应的真实地理坐标
            tie_point = tags[33922].value
            ref_col = tie_point[0]
            ref_row = tie_point[1]
            ref_x = tie_point[3]
            ref_y = tie_point[4]
            ctr_x = ref_x + (ncols//2 - ref_col)*scale_x
            ctr_y = ref_y + (nrows//2 - ref_row)*scale_y
            lon, lat = convert_swiss_xyz_to_wgs84([ref_x, ctr_x], [ref_y, ctr_y], is_lv95=True)
            info["ref_row"] = ref_row
            info["ref_col"] = ref_col
            info["ref_x"] = ref_x
            info["ref_y"] = ref_y
            info["ref_lon"] = lon[0]
            info["ref_lat"] = lat[0]
            info["ctr_lon"] = lon[1]
            info["ctr_lat"] = lat[1]
        else:
            print("未找到参考点标签 (33922)。")
        if 65000 in tags:
            corner_coords = tags[65000].value
            info['corner_coords'] = np.reshape(corner_coords, (4,-1))
        if 65001 in tags:
            info['radius'] = tags[65001].value
        cdata = tif.asarray()
    return cdata, info

def main():
    parser = argparse.ArgumentParser(description="TIFF图像显示参数解析示例")
    parser.add_argument(
        "input_file",
        type=str,
        help="需要处理的输入图像文件路径"
    )
    parser.add_argument(
        "-c", "--colormap",
        dest="cmap",
        type=str,
        metavar="COLORMAP",
        help="指定伪彩色映射"
    )
    parser.add_argument(
        "-d", "--down_sampling",
        dest="down_sampling",    # 存储到 args.crop_region
        type=int,        # 核心：使用自定义的校验和转换逻辑
        metavar="DOWN_SAMPLING",
        help="指定图像下采样倍数"
    )
    args = parser.parse_args()
    file_path = os.path.abspath(os.path.normpath(args.input_file))
    cdata, info = load_tiff(file_path)
    # 查看数组信息
    print("数据类型 (dtype): {}".format(cdata.dtype))
    print("数组形状 (shape): {}".format(cdata.shape))
    if "radius" in info:
        print("行星平均半径: {:f} km".format(info['radius']))
    if "corner_coords" in info:
        lon_W = info['corner_coords'][0][0]
        lon_E = info['corner_coords'][1][0]
        lat_S = info['corner_coords'][2][1]
        lat_N = info['corner_coords'][0][1]
        print("标准分幅边界: LON_W={:f}, LON_E={:f}, LAT_S={:f}, LAT_N={:f}".format(lon_E, lon_W, lat_S, lat_N))
    if "scale_x" in info:
        print("像素间隔 (Scale X, Y): {}, {}".format(info["scale_x"], info["scale_y"]))
        nrows, ncols = cdata.shape
        print("图像尺寸: {:f} km, {:f} km".format(abs(info["scale_x"])*ncols/1e3, abs(info["scale_y"])*nrows/1e3))
    print("海拔高度：{:f} m (min) -- {:f} m (max)".format(np.min(cdata.ravel()), np.max(cdata.ravel())))
    if "ref_col" in info:
        print("参考点映射关系:")
        print(" -> 参考像素: 第 {} 行, 第 {} 列".format(info["ref_row"], info["ref_col"]))
        print(" -> 真实坐标: X={}, Y={}, LON={}, LAT={}".format(info["ref_x"], info["ref_y"], info["ref_lon"], info["ref_lat"]))
        print("图像中心位置: LON={}, LAT={}".format(info["ctr_lon"], info["ctr_lat"]))
        results = rg.search((info["ctr_lat"], info["ctr_lon"]))
        for res in results:
            print(" -> 国家: {}, 城市: {}, 省份: {}".format(res["cc"], res["name"], res["admin1"]))
    if args.cmap:
        color_map = args.cmap
    else:
        color_map = 'gray'
    if args.down_sampling:
        cdata = cdata[::args.down_sampling, ::args.down_sampling]
    plt.imshow(cdata, cmap=color_map)
    plt.show()

if __name__ == "__main__":
    main()
