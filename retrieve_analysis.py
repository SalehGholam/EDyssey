# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:41:17 2025

@author: sgholam
"""

import os
import hyperspy.api as hs
import numpy as np
from glob import glob
import py4DTomo.io_utils as io
import py4DTomo.tracking_utils as tr
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
from tqdm import tqdm
#%% input
path_analysis = r'C:\My Files\Microscope Data\Tecnai\24-09-11_TiO2_200 kV\S2\5DED Analysis\2025-02-12__13-27-28\roi No 1'
fn_mask = 'segmentation masks_ obj ID 1.npy'
fn_rois = 'roi coords_id 1.npy'
fn_dp = '3DED_id 1.hspy'
path_4d = r'C:\My Files\Microscope Data\Tecnai\24-09-11_TiO2_200 kV\S2\4D Signals'
dtype = '.hdf5'
scanSize = (512,512)
#%%
rois = np.load(os.path.join(path_analysis, fn_rois))
masks = np.load(os.path.join(path_analysis, fn_mask))
s_mask = hs.signals.Signal2D(masks)
s_mask.plot()
s_dp = hs.load(os.path.join(path_analysis, fn_dp))
s_dp.plot(norm='symlog', cmap='inferno')
fns_4d = glob(os.path.join(path_4d, '*.*'))
#%% reload a dataset
num = 87
mask = masks[num]
fn_4d = fns_4d[num]
roi = rois[num]
navImg = io.calculate_nav_signal_hdf5(fn_4d)
s_roi = io.load_signal(fn_4d, roi=roi, scanSize=scanSize, lazy=False)
navImg_roi = s_roi.sum(axis=(2,3)).data
dp_mask = tr.extract_3ded_mask_single_frame(fn_4d, mask=mask, roi=roi, dtype=dtype)

fig, ax = plt.subplots(1,4)
ax[0].imshow(navImg)
ax[1].imshow(mask)
ax[2].imshow(navImg_roi)
ax[3].imshow(dp_mask, norm=SymLogNorm(linthresh=0.1))
titles = ['Full Nav. Img.', 'Segmented Obj', 'ROI nav. Img.', 'DP']
for i, a in enumerate(ax):
    a.set_axis_off()
    a.set_title(titles[i])
fig.tight_layout()
#%% reload several DPs which have failed (are zero)
fr_nos = []
for fr_no, img in enumerate(s_dp.data):
    if np.all((img == 0)):
        fr_nos.append(fr_no)
new_dp = np.zeros((len(fr_nos), 512, 512), dtype='uint32')
for i, fr_no in enumerate(tqdm(fr_nos)):
    new_dp[i] = dp_mask = tr.extract_3ded_mask_single_frame(fns_4d[fr_no], mask=masks[fr_no], 
                                                            roi=rois[fr_no], dtype='.hdf5')
# tailor both DPs
dps = s_dp.data
for i, fr_no in enumerate(fr_nos):
    dps[fr_no] = new_dp[i]
s_dp_new = hs.signals.Signal2D(dps)
s_dp_new.plot(norm='symlog', cmap='inferno')
#%% save new frames
path_save = r'C:\My Files\Microscope Data\Tecnai\24-09-11_TiO2_200 kV\S2\5DED Analysis\2025-02-12__13-27-28\roi No 1\new_frames'
io.create_frames(path_save, s_dp_new)
#%% erosion
from scipy.ndimage import binary_erosion
# structure = np.ones((3,3), dtype=bool)
mask = masks[0]
eroded_masks = np.zeros((5,512,512), dtype='int8')

for i, size in enumerate(range(1,30,6)):
    structure = np.ones((size,size), dtype=bool)
    eroded_mask = binary_erosion(mask, structure=structure)
    eroded_masks[i] = mask & ~eroded_mask

# making shells
shells = []
for i, er_mask in enumerate(eroded_masks):
    if i == 0:
        shells.append(er_mask)
    else:
        shells.append((er_mask - eroded_masks[i-1]) * i)

#%%
fig, axs = plt.subplots(1,5)
axs[0].imshow(mask)
axs[0].set_axis_off()

for i in range(1,5):
    axs[i].imshow(eroded_masks[i])
    axs[i].set_axis_off()
fig.tight_layout()

fig, ax = plt.subplots()
ax.imshow(eroded_masks.sum(axis=(0)))