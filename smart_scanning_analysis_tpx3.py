# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 16:49:36 2026

@author: sgholam
"""

import os
import sys
path_eventem = r'C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\py5DED\py4DTomo\io_utils'
# os.chdir(path_eventem)
sys.path.append(path_eventem)
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
from skimage.filters import threshold_li, threshold_otsu, threshold_yen
from matplotlib.widgets import Slider
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
    s = eventem.Roi(repetitions=1, extract_4D=fourd)
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
dwellTime = 100 * 1000 # ns
det_shape = 512
repetitions = 1
bitDepth = 8
#%% files
path_tpx3 = r'Z:\emattecnai\Lemuel\smart_2\2026-02-16_18-20-02'
path_txt = path_tpx3
path_tif = path_tpx3
fns_tpx3 = glob(os.path.join(path_tpx3, '*.tpx3'))
fns_txt = glob(os.path.join(path_txt, '*.txt'))
fns_tif = glob(os.path.join(path_tif, '*.tiff'))

fns_tpx3.sort()
fns_tif.sort()
fns_txt.sort()
#%% remove unwanted files (with no pattern)
fns_tpx3 = fns_tpx3[1:]
fns_txt = fns_txt[:-2]

# check for empty patterns
empty_pat = []
for i, fn in enumerate(tqdm(fns_txt)):
    pat = np.loadtxt(fn)
    if len(pat) < 1:
        empty_pat.append(i)
for i in empty_pat[::-1]:
    fns_txt.pop(i)
    # fns_tpx3.pop(i)
#%% get angle from fns
angles = np.zeros((len(fns_txt)))
angles_tpx3 = np.zeros((len(fns_tpx3)))
for i, _ in enumerate(tqdm(fns_txt)):
    fn_p = os.path.split(fns_txt[i])[1]
    angles[i] = float(fn_p.split('_')[1][5:])
for i, fn in enumerate(tqdm(fns_tpx3)):
    angles_tpx3[i] = float(os.path.split(fn)[1].split('_')[2])

fig, ax = plt.subplots()
ax.plot(angles)
ax.plot(angles_tpx3)
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
num = 0
nav_img_test = calculate_nav_img_tpx3(fns_tpx3[num], scanSize=(512,512), 
                            fn_pattern=fns_txt[num], dwellTime=500)
# nav_img_test = calculate_nav_img_tpx3(fns_tpx3[num], scanSize=(512,512), 
#                             fn_pattern=fns_txt[num], dwellTime=500)
fig, ax = plt.subplots()
ax.imshow(nav_img_test, cmap='viridis')
ax.set_title(f'Num {num}\n{os.path.split(fns_tpx3[num])[1]}')
#%% get roi data
# num = 10
roi = [280, 180, 60, 50] # select based on the previous plot

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
# num = 10
dp_full_test = get_dp(fns_tpx3[num], fns_txt[num], roi, dwellTime)

fig, ax = plt.subplots()
ax.imshow(dp_full_test, norm=SymLogNorm(linthresh=1), cmap='inferno')
#%% create a virtual disc
r_cc = 50
r_in, r_out = 20, 190
cc = find_dp_center(dp_full_test, r=r_cc, plot=True)
mask = create_virtual_mask(cc, r_out=r_out, r_in=r_in)
fig, ax = plt.subplots()
ax.imshow(dp_full_test*mask, norm=SymLogNorm(linthresh=1), cmap='inferno')
#%% create the arguments for calculating navigation images
fns_in = fns_tpx3
rep = len(fns_in)
scanSizes = [scanSize] * rep
dwellTimes = [dwellTime] * rep
rr_in = [r_in] * rep
rr_out = [r_out] * rep
repetitions = [1] * rep
fns_pat_in = fns_txt[:len(fns_in)]

args_zip = zip(fns_in, scanSizes, dwellTimes, rr_in, rr_out, 
                             repetitions, fns_pat_in)
results = [None] * rep
#%% calculate all navigation images
## use threading
n_workers = 5
with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = {
        executor.submit(calculate_nav_img_tpx3, *arg): i
        for i, arg in enumerate(args_zip)
    }

    for future in tqdm(as_completed(futures), total=rep): #TODO add a print counter
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


## plot navigation images
s_nav = hs.signals.Signal2D(results)
s_nav.plot(cmap='inferno')
#%% save navigation images
path_save = r'Z:\emattecnai\Lemuel\smart_2\Analysis'
s_nav.save(os.path.join(path_save, 'navigation signal.hspy'))
#%% tracking



#%% Extract dps with tracked regions
fn_rois = r'Z:\emattecnai\Lemuel\smart_2\Analysis\5DED Analysis\2026-02-23__14-37-11\roi No 1\output_rois.npy'
rois = np.load(fn_rois)
#%%
num = 10
fig, ax = plt.subplots()
ax.imshow(s_nav.inav[num].data)
x,y,w,h = rois[num]
rect = Rectangle((x,y), w, h, edgecolor='r', facecolor='none')
ax.add_patch(rect)
#%%
# num = 10
im = s_nav.inav[num].data
y,x,h,w = rois[num]
im = im[x:x+w, y:y+h]
th = threshold_otsu(im[im>0])
fig, ax = plt.subplots(1,3)
ax[0].imshow(im)
ax[1].imshow(im>=th)
im_2 = np.where(im<th,0,im)
ax[2].imshow(im_2)
#%%
from scipy.ndimage import generic_filter
from scipy.ndimage import convolve

def fill_zeros_iterative(img, max_iter=100, tol=1e-6):
    img = img.astype(float, copy=True)
    zero_mask = img == 0

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=float)

    for _ in range(max_iter):
        neigh_sum = convolve(img, kernel, mode="mirror")
        neigh_count = convolve((img != 0).astype(float), kernel, mode="mirror")

        update = neigh_sum / np.maximum(neigh_count, 1)
        new_vals = img.copy()
        new_vals[zero_mask & (neigh_count > 0)] = update[zero_mask & (neigh_count > 0)]

        if np.nanmax(np.abs(new_vals - img)) < tol:
            break

        img = new_vals

    return img
#%%
masks = []
masks_applied = []
ims_cropped = []
for i, im in enumerate(tqdm(s_nav.data)):
    # x,y,w,h = rois[i]
    y,x,h,w = rois[i]
    # im = im[x:x+w, y:y+h]
    # ims_cropped.append(im)
# =============================================================================
#     th = threshold_otsu(im[im>0])
#     masks[i] = im>=th
#     masks_applied[i] = np.where(im<th, 0, im)
# =============================================================================
    im_2 = fill_zeros_iterative(im)
    th = threshold_li(im_2)
    th_2 = threshold_otsu(im_2[im_2>th])
    th_2 *= 0.90
    # masks[i] = im>=th_2
    # masks_applied[i] = np.where(im<th_2, 0, im)
    im = im[x:x+w, y:y+h]
    ims_cropped.append(im)
    masks.append(im>=th_2)
    masks_applied.append(np.where(im<th_2, 0, im))

# s_masks = hs.signals.Signal2D(masks)
# s_masks_applied = hs.signals.Signal2D(masks_applied)
# s_masks.plot()
# s_masks_applied.plot()
#%% plot one frame of the mask results
num = 175
fig, ax = plt.subplots(1,3)
ax[0].imshow(ims_cropped[num])
ax[1].imshow(masks[num])
ax[2].imshow(masks_applied[num])
#%% plot all results
def plot_mask(imgs, imgs_crop, masks, masks_applied, rois):
    fig, ax = plt.subplots(1, 4, figsize=(12,4), constrained_layout=True)
    # fig, ax = plt.subplots(2,2, figsize=(10,8), constrained_layout=True)
    ax = ax.flatten()
    slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(
        ax=slider_ax,
        label="Frame",
        valmin=0,
        valmax=len(imgs) - 1,
        valinit=0,
        valstep=1
    )
    
    im1 = ax[0].imshow(imgs[0])
    im2 = ax[1].imshow(imgs_crop[0])
    im3 = ax[2].imshow(masks[0])
    im4 = ax[3].imshow(masks_applied[0])
    
    x,y,w,h = rois[0]
    rect = Rectangle((x,y), w, h, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)
    
    ax[0].set_title('Full image with ROI')
    ax[1].set_title('Tracked ROI image')
    ax[2].set_title('Binary Mask')
    ax[3].set_title('Masked Image')
    
    fig.suptitle(f'Frame 1 / {len(imgs)}')
    
    def update(val):
        i = int(slider.val)
        im1.set_data(imgs[i])
        im2.set_data(imgs_crop[i])
        im3.set_data(masks[i])
        im4.set_data(masks_applied[i])
        
        x,y,w,h = rois[i]
        rect.set_xy((x,y))
        rect.set_width(w)
        rect.set_height(h)
        fig.suptitle(f'Frame {i+1} / {len(imgs)}')
        
        fig.canvas.draw_idle()
    def _on_key(event):
        i = int(slider.val)
        if event.key == "left":
            slider.set_val(max(0, i - 1))
        elif event.key == "right":
            slider.set_val(min(len(imgs) - 1, i + 1))

    slider.on_changed(update)
    fig.canvas.mpl_connect("key_press_event", _on_key)    

plot_mask(s_nav.data, ims_cropped, masks, masks_applied, rois)
#%%
def get_dp_with_mask(num):
    dp_s = get_roi(fns_tpx3[num], dwellTime, rois[num], bitDepth, scanSize, 
                 fn_pattern=fns_txt[num], fourd=True)
    mask = masks[num].flatten()
    dp_ext = dp_s.get_4D().reshape(-1,512,512)[mask].sum(axis=0)
    return dp_ext

dp_test = get_dp_with_mask(0)
fig, ax = plt.subplots()
ax.imshow(dp_test, norm=SymLogNorm(linthresh=1))
#%%
ext_dps = np.zeros((len(masks), 512, 512), dtype='uint32')
# ext_dps = ext_dps[:5]
n_workers = 5
with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = {executor.submit(get_dp_with_mask, i): i
            for i in range(len(ext_dps))
        }

    for i, future in enumerate(tqdm(as_completed(futures), total=len(masks))): #TODO add a print counter
        idx = futures[future]
        print('\n', i)
        ext_dps[idx] = future.result()  # reorders to original submission order

ext_dps = hs.signals.Signal2D(ext_dps)
#%% saving results
import pickle
path_save_3ded = r'Z:\emattecnai\Lemuel\smart_2\Analysis\5DED Analysis\2026-02-23__14-37-11\roi No 1\Extracted'
ext_dps.save(os.path.join(path_save_3ded, '3ded_1.hspy'))
with open(os.path.join(path_save_3ded, "masks.pkl"), "wb") as f:
    pickle.dump(masks, f, protocol=pickle.HIGHEST_PROTOCOL)

path_frames = os.path.join(path_save_3ded, 'frames')
if not os.path.isdir(path_frames):
    os.mkdir(path_frames)
for i, fr in enumerate(tqdm(ext_dps.data)):
    tifffile.imwrite(os.path.join(path_frames, f'{i:04d}_a {angles[i]:.2f}.tif'), fr)
    
ext_dps.plot(norm='symlog')
#%% list of angles and frames for PETS
ls = []
for i, a in enumerate(angles):
    temp = f'"frames/{i:04d}_a {angles[i]:.2f}.tif"' + f'{angles[i]: 10.4f}\n'
    ls.append(temp)

with open(os.path.join(path_save_3ded,'angles.txt'), 'w') as f:
    f.writelines(ls)