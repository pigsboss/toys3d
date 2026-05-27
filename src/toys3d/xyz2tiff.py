#!/usr/bin/env python
"""xyz2tiff.py Load and convert swisstopo .xyz.zip files (e.g., swissALTIregio, swissALTI3D) to TIFF.
"""
import zipfile
import sys
from os import path
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import gc
from multiprocessing import Pool, cpu_count

def load_xyz(zip_input, npixels):
    x = np.empty((npixels, npixels), dtype=np.float32)
    y = np.empty_like(x)
    z = np.empty_like(x)
    with zipfile.ZipFile(zip_input.absolute(), 'r') as zf:
        all_files = zf.namelist()
        target_file = all_files[0]
        print("  {} loaded.".format(target_file))
        bytes_content = zf.read(target_file)
        text_content = bytes_content.decode('utf-8')
        xyzlst = text_content.split('\n')
        assert len(xyzlst) > npixels*npixels, '{} file number of rows ({:d}) mismatch.'.format(target_file, len(xyzlst))
        for j in range(npixels):
            for k in range(npixels):
                x[j,k], y[j,k], z[j,k] = map(float, xyzlst[k+j*npixels+1].split(' '))
    return x, y, z

zip_dir = Path(sys.argv[1])
tiff_output = path.join(zip_dir.absolute(), 'preview.tiff')
png_output = path.join(zip_dir.absolute(), 'preview.png')
zip_list = []
for file_path in zip_dir.iterdir():
    if file_path.is_file():
        if file_path.suffix.lower() == '.zip' and file_path.name.lower().startswith('swissalti'):
            zip_list.append(file_path.absolute())
nframes = len(zip_list)
if zip_list[0].name.lower().startswith('swissaltiregio'):
    npixels = 1000
elif zip_list[0].name.lower().startswith('swissalti3d'):
    npixels = 2000
else:
    raise TypeError('unrecognized zip file.')
print('{:d} zip files found in {}.'.format(nframes, zip_dir.absolute()))
nworkers = max(1, cpu_count()-1)
args = ()
for zip_input in zip_list:
    args += ((zip_input, npixels), )
with Pool(processes = nworkers) as pool:
    results = pool.starmap(load_xyz, args)
xmin =  np.inf
xmax = -np.inf
ymin =  np.inf
ymax = -np.inf
for x,y,z in results:
    xmin = min(xmin, np.min(x.ravel()))
    xmax = max(xmax, np.max(x.ravel()))
    ymin = min(ymin, np.min(y.ravel()))
    ymax = max(ymax, np.max(y.ravel()))
xinc = results[0][0][0,1] - results[0][0][0,0]
yinc = results[0][1][1,0] - results[0][1][0,0]
xdim = xmax - xmin
ydim = ymax - ymin
xpts = int((xmax - xmin)/abs(xinc) + 1.)
ypts = int((ymax - ymin)/abs(yinc) + 1.)
print('full frame: {:f} km x {:f} km ({:d} pixels x {:d} pixels)'.format(xdim/1e3, ydim/1e3, xpts, ypts))
xfull = np.empty((ypts, xpts), dtype=np.float32)
yfull = np.empty_like(xfull)
zfull = np.empty_like(xfull)
for i in range(nframes):
    if yinc > 0:
        ridx = int(np.floor((results[i][1][0,0] - ymin)/yinc/npixels))
    else:
        ridx = int(np.floor((results[i][1][0,0] - ymax)/yinc/npixels))
    if xinc > 0:
        cidx = int(np.floor((results[i][0][0,0] - xmin)/xinc/npixels))
    else:
        cidx = int(np.floor((results[i][0][0,0] - xmax)/xinc/npixels))
    xfull[ridx*npixels:(ridx+1)*npixels, cidx*npixels:(cidx+1)*npixels] = results[i][0][:,:]
    yfull[ridx*npixels:(ridx+1)*npixels, cidx*npixels:(cidx+1)*npixels] = results[i][1][:,:]
    zfull[ridx*npixels:(ridx+1)*npixels, cidx*npixels:(cidx+1)*npixels] = results[i][2][:,:]
del results
gc.collect()
# plt.imshow(zfull)
# plt.show()
tifffile.imwrite(
    tiff_output,
    zfull,
    compression='lzw',       # 使用无损的 LZW 压缩
    photometric='minisblack' # 单通道灰度
)
print(f"{tiff_output} preview saved.")
cv2.imwrite(
    png_output,
    (((zfull-zfull.min()) / (zfull.max()-zfull.min())) * 65535.).astype(np.uint16),
    [cv2.IMWRITE_PNG_COMPRESSION, 1])
print(f"{png_output} preview saved.")

