# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 16:49:36 2026

@author: sgholam
"""

import os
path_eventem = r'C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\py5DED\py4DTomo\io_utils'
os.chdir(path_eventem)
import eventem
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import tifffile
import hyperspy.api as hs
from tqdm import tqdm
from scipy.ndimage import center_of_mass
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.patches import Rectangle
from skimage.draw import disk
#%% funcs
def calculate_nav_img_tpx3(fn, scanSize=(512,512), dwellTime=1, 
                   r_in=0, r_out=512, repetitions=1, fn_pattern=None):
    vstem = eventem.vSTEM(repetitions) #arg is number of reperitions in the scan
    vstem.b_cumulative = True
    vstem.set_file(fn)
    vstem.nx = scanSize[0]
    vstem.ny = scanSize[1]
    vstem.inner_radia = [r_in]
    vstem.outer_radia = [r_out]
    vstem.set_dwell_time(dwellTime*1000)
    if fn_pattern:
        vstem.set_pattern_file(fn_pattern)
    # vstem.offsets([256,256]) # VDF
    vstem.run()
    nav_image = np.asarray(vstem.vSTEM_image).reshape(
        scanSize[1],scanSize[0])
    return nav_image

def get_roi(fn_tpx3, dwellTime, roi, bitDepth=8, 
            scanSize=(512,512), fn_pattern=None, fourd=False):
    x,y,w,h = roi
    s = eventem.Roi(repetitions=repetitions, extract_4D=fourd)
    s.set_bitdepth(bitDepth)
    s.nx = scanSize[0]
    s.ny = scanSize[1]
    # s.b_cumulative = True
    s.set_file(fn_tpx3)
    if fn_pattern:
        s.set_pattern_file(fn_pattern)
    s.set_roi(x=x, y=y, width=w, height=h)
    # s.set_roi(x=x, y=y, width=w, height=h)
    s.set_dwell_time(dwellTime*1000)
    s.run()
    # ROI_scan_image = np.asarray(roi.Roi_scan_image)
    # ROI_diffp = np.asarray(roi.Roi_diffraction_pattern).reshape(512, 512)
    return s

def get_dp(fn, fn_pattern, roi, dwellTime, repetitions=1):
    s_pacbed = eventem.Pacbed(repetitions) # repetitions
    
    s_pacbed.set_file(fn)
    s_pacbed.set_pattern_file(fn_pattern)
    s_pacbed.b_cumulative = True
    s_pacbed.set_dwell_time(int(dwellTime*1000))
    
    x, y, w, h = roi
    s_pacbed.nx = x
    s_pacbed.ny = y
    s_pacbed.nx = s_pacbed.nx + w
    s_pacbed.ny = s_pacbed.nx + h
    s_pacbed.run()
    im_2 = np.asarray(s_pacbed.Pacbed_image).reshape(512,512)
    return im_2

def create_virtual_mask(center, r_out, r_in=0, shape=(512,512)):
    cy, cx = center
    mask = np.zeros((512,512), dtype=np.uint8)
    rr, cc = disk((cy, cx), r_out, shape=mask.shape)
    mask[rr, cc] = 1
    mask_in = np.zeros((512,512), dtype=np.uint8)
    rr, cc = disk((cy, cx), r_in, shape=shape)
    mask[rr, cc] = 0
    return mask

def find_dp_center(dp, r=50, plot=False):
    idx = np.argmax(dp)# index in flattened array
    cy, cx = np.unravel_index(idx, dp.shape)
    mask_temp = create_virtual_mask((cy,cx), r)
    
    if plot:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(dp*mask_temp, norm=SymLogNorm(linthresh=1))
        cx, cy = center_of_mass(dp * mask_temp)
        ax[0].set_title('Masked Pattern for\nFinding the Center')
        
        ax[1].imshow(dp, norm=SymLogNorm(linthresh=1))
        ax[1].scatter(cy, cx, color='r')
        ax[1].set_title('Found Center')
    return (cx,cy)
#%% input params
scanSize = (512,512)
dwellTime = 500 * 1000 # ns
det_shape = 512
repetitions = 1
bitDepth = 16
#%% files
# path = r'Z:\emattecnai\Lemuel\smart_1\2026-02-16_16-18-02' # first part
path = r'Z:\emattecnai\Lemuel\smart_1\2026-02-16_16-35-07' # second batch
fns_tpx3 = glob(os.path.join(path, '*.tpx3'))
fns_tif = glob(os.path.join(path, '*.tiff'))
fns_txt = glob(os.path.join(path, '*.txt'))

fns_tpx3.sort()
fns_tif.sort()
fns_txt.sort()
#%% remove unwanted files (with no pattern)
# =============================================================================
# # 1st batch
# fns_tpx3 = fns_tpx3[:-1]
# fns_txt = fns_txt[:69]
# =============================================================================
## second batch
fns_tpx3 = fns_tpx3[1:]
fns_txt = fns_txt[:-3]

# check for empty patterns
empty_pat = []
for i, fn in enumerate(fns_txt):
    pat = np.loadtxt(fn)
    if len(pat) < 1:
        empty_pat.append(i)
for i in empty_pat[::-1]:
    fns_txt.pop(i)
#%% get angle from fns
angles = np.zeros((len(fns_txt)))
for i, _ in enumerate(fns_txt):
    fn_p = os.path.split(fns_txt[i])[1]
    angles[i] = float(fn_p.split('_')[1][5:])
#%% check haadf
# =============================================================================
# ## filter by correct patterns
# bad_haadf = []
# for _, fn in enumerate(fns_tif[::-1]):
#     a = float(os.path.split(fns_tif[i])[1].split('_')[-1][:-5])
#     if a not in angles:
#         bad_haadf.append(i)
# for i in bad_haadf[::-1]:
#     fns_tif.pop(i)
# 
# ## separate quick and slow scans
# fns_tif_s = [fn for fn in fns_tif if fn.split('_')[-1].split('.tiff')[0] == '2']
# fns_tif_q = [fn for fn in fns_tif if fn not in fns_tif_s]
# 
# # =============================================================================
# # ## plot a HAADF image
# # num = 1
# # im = tifffile.imread(fns_tif[num])
# # fig, ax = plt.subplots()
# # cbar = ax.imshow(im)
# # fig.colorbar(cbar)
# # =============================================================================
# 
# ## plot all haadf images
# to_plot = fns_tif_q
# 
# s = np.zeros((len(fns_tif), 512, 512))
# for i, fn in enumerate(tqdm(to_plot)):
#     s[i] = tifffile.imread(fn)
# s = hs.signals.Signal2D(s)
# s.plot(cmap='viridis')
# =============================================================================
#%% calculate navigation image
num = 10
nav_img_test = calculate_nav_img_tpx3(fns_tpx3[num], scanSize=(512,512), 
                            fn_pattern=fns_txt[num], dwellTime=500)
fig, ax = plt.subplots()
ax.imshow(nav_img_test, cmap='viridis')
ax.set_title(f'Num {num}\n{os.path.split(fns_tpx3[num])[1]}')
#%% get roi data
num = 10
roi = [220, 270, 25, 25]

## full 4D
s_roi_test = get_roi(fns_tpx3[num], dwellTime, roi, 
                     bitDepth, fn_pattern=fns_txt[num], fourd=True)
ss_roi_test = hs.signals.Signal2D(np.asarray(s_roi_test.get_4D()))
## plot 4d signal
ss_roi_test.plot(norm='symlog', cmap='inferno', vmin=None)
# plot nav and full dp
fig, ax = plt.subplots(1,3)
ax[0].imshow(nav_img_test, cmap='viridis')
x,y,w,h = roi
rect = Rectangle((x,y), w, h, edgecolor='r', facecolor='none')
ax[0].add_patch(rect)

ax[1].imshow(np.asarray(s_roi_test.Roi_scan_image).reshape(w,h), cmap='inferno')

ax[2].imshow(np.asarray(s_roi_test.Roi_diffraction_pattern).reshape(512,512), 
             norm=SymLogNorm(linthresh=1), cmap='inferno')
ax[0].set_title('Full Virtual Image')
ax[1].set_title('ROI Virtual Image')
ax[2].set_title('DP')
#%% get full dp
num = 10
dp_full_test = get_dp(fns_tpx3[num], fns_txt[num], roi, dwellTime)

fig, ax = plt.subplots()
ax.imshow(dp_full_test, norm=SymLogNorm(linthresh=1), cmap='inferno')
#%% create a virtual disc
r_cc = 50
r_in, r_out = 25, 150
cc = find_dp_center(dp_full_test, r=r_cc, plot=True)
mask = create_virtual_mask(cc, r_out=r_out, r_in=r_in)
fig, ax = plt.subplots()
ax.imshow(dp_full_test*mask, norm=SymLogNorm(linthresh=1), cmap='inferno')
#%% create the arguments for calculating navigation images
fns_in = fns_tpx3
rep = len(fns_in)
scanSizes = [(512,512)] * rep
dwellTime = [500] * rep
rr_in = [r_in] * rep
rr_out = [r_out] * rep
repetitions = [1] * rep
fns_pat_in = fns_txt[:len(fns_in)]

args_zip = zip(fns_in, scanSizes, dwellTime, rr_in, rr_out, 
                             repetitions, fns_pat_in)
results = [None] * rep
#%% calculate all navigation images
## use threading
n_workers = 10
with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = {
        executor.submit(calculate_nav_img_tpx3, *arg): i
        for i, arg in enumerate(args_zip)
    }

    for future in tqdm(as_completed(futures), total=rep):
        idx = futures[future]
        results[idx] = future.result()  # reorders to original submission order

# =============================================================================
# # no threading
# imgs = np.zeros((rep, 512,512))
# tic = perf_counter()
# for i, arg in enumerate(args):
#     imgs[i] = calculate_nav_img_tpx3(*arg)
#     print(i)
# toc = perf_counter()
# print(f'Processing time: {(toc-tic)//60:.0f}:{(toc-tic)%60:.0f}')
# =============================================================================
#%% plot navigation images
s = hs.signals.Signal2D(results)
s.plot(cmap='inferno')
#%% save navigation images
path_save = r'Z:\emattecnai\Lemuel\smart_1\Analysis'
s.save(os.path.join(path_save, 'navigation signal_2.hspy'))
#%% Extract dps with tracked regions


