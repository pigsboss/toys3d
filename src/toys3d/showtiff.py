#!/usr/bin/env python

import tifffile
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

file_path = os.path.abspath(os.path.normpath(sys.argv[1]))
color_map = sys.argv[2]
downsample = int(eval(sys.argv[3]))
image_array = tifffile.imread(file_path)

# 查看数组信息
print(f"数据类型 (dtype): {image_array.dtype}")
print(f"数组形状 (shape): {image_array.shape}")

plt.imshow(image_array[::downsample,::downsample], cmap=color_map)
plt.show()
