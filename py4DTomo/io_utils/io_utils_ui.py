# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:58:43 2024

@author: SGholam
"""

import numpy as np
# import pyxem as px
import os
from glob import glob
from tqdm import tqdm
import hyperspy.api as hs
import tifffile
import dask.array as da
import h5py
import cv2
from copy import deepcopy
import dask
# import pyLiveProcessing as pyLP
import eventem
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
from warnings import warn
# import cupy as cp
# from moviepy.editor import VideoClip
# from moviepy.video.io.bindings import mplfig_to_npimage
# from moviepy import VideoClip
# from moviepy.video.io.ffmpeg_reader import ffmpeg_read_image
from matplotlib.animation import FuncAnimation
from typing import Literal
path_ffmpeg = r'C:\Users\sgholam\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'
plt.rcParams['animation.ffmpeg_path'] = path_ffmpeg  # Windows example
#%%
def get_4d_files(path_4d_datasets, dtype):
    fns_4d = glob(os.path.join(path_4d_datasets, f'*.{dtype}'))
    if len(fns_4d) == 0: # search in sub-directories
        fns_4d = glob(os.path.join(path_4d_datasets, '**', f'*.{dtype}'), recursive=True)
    assert len(fns_4d) != 0, "4D Datasets with correct format are not found!"
    return fns_4d

def load_signal(fn, **kwargs):
    try:
        dtype = kwargs.get('dtype')
    except:
        dtype = None
    if dtype is None:
        dtype = os.path.splitext(fn)[1]
    
    if dtype == '.tpx3':
        result = load_tpx3(fn, **kwargs)        
    elif dtype == '.hdf5':
        result = load_hdf5(fn, **kwargs)        
    elif dtype in ['.zspy', '.hspy']:
        result = load_hs(fn, **kwargs)
    elif dtype == '.mib':
        result = load_mib(fn, **kwargs)
    return result

def load_tpx3(fn, roi=None, scanSize=(512,512), dwellTime=1, bitDepth=16,
              det_shape=512, repetitions=1, sum_dp=False, **kwargs):
    if roi is None:
        x=0
        y=0 
        w=scanSize[0]
        y=scanSize[1]
    else:
        x, y, w, h = roi
        
    roi = eventem.Roi(repetitions=repetitions, extract_4D=True)
    roi.set_bitdepth(bitDepth)
    roi.nx = scanSize[0]
    roi.ny = scanSize[1]
    roi.set_file(fn)
    roi.set_roi(x=x, y=y, width=w, height=h)
    roi.set_dwell_time(dwellTime*1000)
    roi.run()
    # ROI_scan_image = np.asarray(roi.Roi_scan_image)
    # ROI_diffp = np.asarray(roi.Roi_diffraction_pattern).reshape(512, 512)
    s = np.asarray(roi.get_4D())
    if sum_dp:
        s = s.sum(axis=(-1,-2))
        return s
    s = hs.signals.Signal2D(s)
    return s

def load_hdf5(fn, roi=None, scanSize=None, chunks=(8,512,512,512), lazy=False, 
              sum_dp=False, **kwargs):
    # with h5py.File(fn, 'r') as f:
    f = h5py.File(fn, 'r')
    arr_dim = len(f['4D'].shape) # data might be flattened
    if arr_dim == 1: # TODO do it only once before running tomo
        det_shape = f['shape'][:][-1] # works on only scquare detectors
        scanSize_written = tuple(f['shape'][:][:2])
        if scanSize is None:
            scanSize = scanSize_written
        else:           
            assert scanSize == scanSize_written, f"Scan size entered does not match to the shape of the hdf5 file: {scanSize} vs {scanSize_written}"
    
    if chunks is None: # TODO not optimum necessarily
        if roi is not None:
            y,x,h,w = roi
            chunks = (h, w, det_shape, det_shape)
        if det_shape == 512:
            chunks = (16, 16, det_shape, det_shape)
        else:
            chunks = (32, 32, det_shape, det_shape)
    
    if arr_dim == 1:
        chunks = np.prod(chunks)
        s = da.from_array(f['4D'], chunks=chunks)
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            s = s.reshape(scanSize[0], scanSize[1], det_shape, det_shape)
        # s = s.map_blocks(cp.asarray)
    else:
        s = da.from_array(f['4D'], chunks=chunks)
        # s = s.map_blocks(cp.asarray)
    
    if np.any(roi): #TODO test
        # x,y,w,h = roi
        y,x,h,w = roi
        s = s[x:x+w, y:y+h]
    
    if sum_dp:
        dp = s.sum(axis=(2,3))
        dp_res = dp.compute()
        # dp_res = cp.asnump(dp_res)
        f.close()
        # cp.get_default_memory_pool().free_all_blocks()
        # del s, dp
        # cp.get_default_memory_pool().free_all_blocks()
        return dp_res
    
    if not lazy:
        s_get = s.compute()
        # s_get = cp.asnumpy(s_get) # turning it to numpy from cupy
        # cp.get_default_memory_pool().free_all_blocks()
        # del s
        f.close()
        s_get = hs.signals.Signal2D(s_get)
        return s_get
    
    else:
        s = hs.signals.Signal2D(s)
        # cp.get_default_memory_pool().free_all_blocks()
        return s, f
    
def load_hs(fn, roi=None, chunks=None, lazy=False, sum_dp=False, **kwargs):
    #TODO add cupy if possible
    s = hs.load(fn, lazy=True)
    if chunks is not None:
        s.rechunk(nav_chunks=chunks[:2], sig_chunks=chunks[2:])
    
    if np.any(roi):
        x,y,w,h = roi
        s = s.inav[x:x+w, y:y+h]
    
    if sum_dp:
        dp = s.sum(axis=(2,3)).data
        return dp.compute()
    
    if not lazy:
        s.compute()
    return s    

def load_mib(fn, roi=None, scanSize=None, chunks=None, lazy=False, sum_dp=False, **kwargs):
    s = hs.load(fn, lazy=True)
    det_shape = s.data.shape[-1]
    if len(s.data.shape) == 3:
        if scanSize is None:
            fld = os.path.split(fn)[0]
            fn_hdr = os.path.join(fld, 'default.hdr')
            scanSize = get_scan_size_mib_hdr(fn_hdr)
        s.reshape(scanSize[0], scanSize[1], det_shape, det_shape)
    
    if chunks is None:
        if det_shape == 512:
            s.rechunk(nav_chunks=(16,16), sig_chunks=(det_shape,det_shape))
        else:
            s.rechunk(nav_chunks=(32,32), sig_chunks=(det_shape,det_shape))
    else:
        s.rechunk(nav_chunks=chunks[:2], sig_chunks=chunks[2:])
        
    if roi is not None:
        x,y,w,h = roi
        s = s.inav[x:x+w, y:y+h]
    
    if sum_dp:
        dp = s.sum(axis=(2,3)).data
        dp.compute()
        return dp
    
    if not lazy:
        s.compute()
    return s

def get_scan_size(fn):
    dtype = os.path.splitext(fn)[-1]
    if dtype == '.hdf5':
        with h5py.File(fn, 'r') as f:
            scanSize = tuple(f['shape'])[:2]
        return scanSize
    else:
        s = load_signal(fn, lazy=True)
        return (s.data.shape[1], s.data.shape[0]) # TODO return y and x?

def get_det_size(fn):
    s = load_signal(fn, lazy=True)
    if type(s) == tuple: # for hdf5
        s, f = s
        f.close()
    det_shape = (s.data.shape[3], s.data.shape[2])
    print('detector shape:', det_shape)
    return det_shape

def get_scan_size_mib_hdr(fn_hdr):
    with open(fn_hdr, 'r') as file:
        hdr = file.readlines()
    fpt = [line for line in hdr if 'Frames per Trigger' in line][0]
    fpt = fpt.split(':')[1]
    fpt = int(fpt)
    
    framesAcq = [line for line in hdr if 'Frames in Acquisition' in line][0]
    framesAcq = framesAcq.split(':')[1]
    framesAcq = int(framesAcq)
    scanSize = (fpt, int(framesAcq/fpt))
    return scanSize

def create_nav_signal_from_haadf (fns, dtype): #TODO fix
    temp = hs.load(fns[0])
    size = temp.data.shape[-1]
    dtype=temp.data.dtype
    s = np.zeros((len(fns), size, size), dtype=dtype)
    
# =============================================================================
#     if dtype in ['tif' or 'tiff']: #TODO fix file names during acquisition
#         seq = [int(os.path.split(fn)[1].split('_')[2]) for fn in fns]
#         seq = np.array(seq)
#         seq = np.argsort(seq)
#         fns = np.array(fns)[seq]
# =============================================================================
    
    for i, fn in enumerate(tqdm(fns)):
        s[i] = hs.load(fn).data
    s = hs.signals.Signals2D(s)
# =============================================================================
#     if dtype == 'tiff': #TODO haadf is flipped at Tecnai
#         s = s.flip_diffraction_x()
# =============================================================================
    return s

def calculate_nav_img_tpx3(fn, scanSize, dwellTime=None, r_in=0, r_out=512, repetitions=1, ):
    vstem = eventem.vSTEM(repetitions) #arg is number of reperitions in the scan
    vstem.b_cumulative = True
    vstem.set_file(fn)
    vstem.nx = scanSize[0]
    vstem.ny = scanSize[1]
    vstem.inner_radia = [r_in]
    vstem.outer_radia = [r_out]
    vstem.set_dwell_time(dwellTime*1000)
    # vstem.offsets([256,256]) # VDF
    vstem.run()
    nav_image = np.asarray(vstem.vSTEM_image).reshape(
        scanSize[1],scanSize[0])
    return nav_image

def calculate_nav_img_hdf5(fn, scanSize=None):
    with h5py.File(fn, 'r') as f:
        if 'dose_image' in f.keys():
            nav_img = f['dose_image'][:]
            return nav_img
        # print('Navigation images is pre-calculated')
    nav_img = load_hdf5(fn, scanSize=scanSize, sum_dp=True)
    return nav_img

def calculate_nav_img(fn, dtype=None, scanSize=None, dwellTime=1):
    if dtype is None:
        dtype = os.path.splitext(fn)[-1]
        
    if dtype == '.mib':
        nav_img = load_signal(fn, scanSize=scanSize, sum_dp=True)
    elif dtype == '.tpx3':
        nav_img = calculate_nav_img_tpx3(fn, scanSize, dwellTime)
    elif dtype == '.hdf5':
        nav_img = calculate_nav_img_hdf5(fn, scanSize)
    elif dtype in ['.zspy', '.hspy']:
        nav_img = load_signal(fn, scanSize=scanSize, sum_dp=True)
    elif dtype in ['.dm2', '.dm3', '.tif', '.tiff']:
        nav_img = hs.load(fn).data
    return nav_img
    
def convert_to_8bit(s):
    s_8bit = deepcopy(s)
    # s_8bit.data[s_8bit.data < 0] = 0
    for i, img in enumerate(s_8bit.data):
        # print(img.min(), img.max())
        s_8bit.data[i] = ((img - img.min()) / (img.max() - img.min())) * 255.0
    s_8bit.data = s_8bit.data.astype(np.uint8)
    return s_8bit

def convert_img_to_8bit(img):
    img_8bit = deepcopy(img)
    # img_8bit[img_8bit < 0] = 0
    img_8bit = ((img - img.min()) / (img.max() - img.min())) * 255.0
    img_8bit = img_8bit.astype(np.uint8)
    return img_8bit

def gaussian_blur(img, kernel_size=3):
    return cv2.GaussianBlur(img, (kernel_size,kernel_size), 0)

def convert_to_rgb(imgs):
    l,w,h = imgs.shape
    # imgs_8bit = convert_to_8bit(imgs)
    imgs_rgb = np.zeros((w,h,3,l), dtype=imgs.dtype)
    for i, img in enumerate(imgs):
        imgs_rgb[:,:,:,i] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return imgs_rgb

def create_tracking_result(s, rois):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.25
    font_thickness = 1
    font_color = (255, 255, 255)  # BGR color format (blue, green, red)
    s_roi = np.zeros(s.data.shape, dtype=np.uint16)
    for i_img, img in enumerate(tqdm(s.data)):
        img = deepcopy(img)
        for i_r, roi in enumerate(rois):
            (x,y,w,h) = roi[i_img]
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 1)
            cv2.putText(img, str(i_r+1), (x-3, y-3), font, font_scale, font_color, font_thickness)
        s_roi[i_img] = img
    s_roi = hs.signals.Signals2D(s_roi)
    return s_roi

def create_array_from_dissimilar_imgs(imgs, mode='constant', signal=True):
    shapes = [[img.shape[0], img.shape[1]] for img in imgs]
    shapes = np.array(shapes)
    w = shapes[:,0].max()
    h = shapes[:,1].max()
    new_arr = np.zeros((len(imgs), w, h), dtype=np.uint8)
    for i, img in enumerate(imgs):
        w2 = w - img.shape[0]
        h2 = h - img.shape[1]
        new_img = np.pad(img, pad_width=[(0,w2), (0,h2)], mode=mode)
        new_arr[i] = new_img
        # new_arr[i][:img.shape[0], :img.shape[1]] = img
    if signal:
        new_arr = hs.signals.Signal2D(new_arr)
    return new_arr

def create_frames(pathSave, frames):
    # pathSave = os.path.join(path, 'frames')
    if os.path.isdir(pathSave):
        [os.remove(os.path.join(pathSave, fn)) for fn in os.listdir(pathSave)]
    else:
        os.mkdir(pathSave)
    # for i, fr in enumerate(tqdm(s.data)):
    for i, fr in enumerate(tqdm(frames)):
        tifffile.imwrite(os.path.join(pathSave, f'{i+1:04d}.tif'), fr)

def create_clip_dp(fn, s, scale=None, dpi=400, fps=None, vmin=None, vmax=None, cmap='inferno'):
    print('Making DP clip...')
    plt.ioff()
    fig, ax = plt.subplots()
    img = ax.imshow(s[0], cmap=cmap, norm=mcolors.SymLogNorm(
        vmin=vmin, vmax=vmax, linthresh=1))
    fig.tight_layout()
    if scale is not None:
        scalebar = ScaleBar(scale*10, '1/nm', dimension='si-length-reciprocal', location='lower left',
                            scale_formatter=lambda value, unit:  f'{value / 10}'r' $\AA^{-1}$', fixed_value=5)

        ax.add_artist(scalebar)
    def update_frame(fr_no):
        img.set_data(s[fr_no])
        return (img, )
    
    if fps is None:
        fps = len(s) // 20 # sec
        if fps == 0:
            fps = 1
    ani = FuncAnimation(fig, update_frame, frames=range(s.shape[0]), blit=True)
    try:
        ani.save(fn + '.mp4', writer='ffmpeg', fps=fps, dpi=dpi)
    except:
        ani.save(fn + '.gif', fps=fps)
        print('No ffmpeg was found to make mp4 clip!')
    plt.ion()
    print('DP clip is created!')
        
def create_clip_tracking(fn, imgs, rois=None, scale=None, dpi=400, 
                         fps=None, duration=None, cmap='viridis'):
    print('Making tracking clip...')
    plt.ioff()
    fig, ax = plt.subplots()
    img = ax.imshow(imgs[0], cmap=cmap)
    if scale is not None:
        scalebar = ScaleBar(scale, 'nm', dimension='si-length', location='lower left')
        ax.add_artist(scalebar)
    if rois is not None:
        patch = []
        x,y,w,h = rois[0]
        rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='r', 
                                 facecolor='none')
        ax.add_patch(rect)
        patch.append(rect)
    
    def update_frame(fr_no):
        img.set_data(imgs[fr_no])
        if rois is not None:
            for p in patch:
                p.remove()
                patch.remove(p)
            x,y,w,h = rois[fr_no]
            rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='r', 
                                     facecolor='none')
            ax.add_patch(rect)
            patch.append(rect)
        return (img, )
    fig.tight_layout()
    
    if fps is None:
        if duration is None:
            fps = len(imgs) // 20 # sec
        else:
            fps = len(imgs) // duration # sec
    if fps == 0:
        fps = 1
            
    ani = FuncAnimation(fig, update_frame, frames=range(imgs.shape[0]), blit=True)
    path, fn_raw = os.path.split(fn)
    fn = os.path.join(path, fn)
    
    try:
        ani.save(fn + '.mp4', writer='ffmpeg', fps=fps, dpi=dpi)
    except:
        ani.save(fn + '.gif', fps=fps)
        print('No ffmpeg was found to make mp4 clip!')
    plt.ion()
    print('Tracking clip is Created!')

def create_clip_tracking_with_mask(fn, imgs, masks, obj_id=1, scale=None, 
                                   dpi=400, fps=None, cmap='viridis'):
    def show_mask(mask, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            # color = np.array([30/255, 144/255, 255/255, 0.6])
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        # mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        img_mask.set_data(mask_image)

    def update_frame(fr_no):
        img.set_data(imgs[fr_no])
        show_mask(masks[fr_no], obj_id)
        return (img, )
    print('Making tracking clip...')
    plt.ioff()
    fig, ax = plt.subplots()
    img = ax.imshow(imgs[0], cmap=cmap)
    img_0 = np.zeros(imgs[0].shape, dtype=bool)
    img_mask = ax.imshow(img_0)
    show_mask(masks[0], obj_id, img_mask)
    if scale is not None:
        scalebar = ScaleBar(scale, 'nm', dimension='si-length', location='lower left')
        ax.add_artist(scalebar)
    fig.tight_layout()
    
    if fps is None:
        fps = len(imgs) // 20 # sec
    ani = FuncAnimation(fig, update_frame, frames=range(imgs.shape[0]), blit=True)
    try:
        ani.save(fn + '.mp4', writer='ffmpeg', fps=fps, dpi=dpi)
    except:
        ani.save(fn + '.gif', fps=fps)
        print('No ffmpeg was found to make mp4 clip!')
    plt.ion()
    print('Tracking clip is created')
# =============================================================================
# if __name__ == '__main__':
#     pass
# =============================================================================
