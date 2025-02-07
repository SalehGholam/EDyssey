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
import eventem as pyLP
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
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

def load_signal(fn, lazy=True, dtype=None, scanSize=None, 
                roi=None, signal=True, det_shape=512, dwellTime=1): #TODO dwell time is not set always
    if dtype is None:
        dtype = os.path.splitext(fn)[1]
    do_cut = False
    if roi is not None:
        do_cut = True
        
    if dtype in ['.zspy', '.hspy']:
        s = hs.load(fn, lazy=True)
        if do_cut:
            x,y,w,h = roi
            s = s.inav[x:x+w, y:y+h]
            s.compute()
        if lazy==False:
            if hasattr(s, 'compute'):
                s.compute()
        return s
    
    elif dtype == '.hdf5': # no lazy, as the file will be closed at the end
        with h5py.File(fn, 'r') as f:
            if scanSize is not None: # checking scan size entered and the one written in the file
                assert scanSize == tuple(f['shape'])[:2], f"Scan size entered does not match to the shape of the hdf5 file: {f['shape'][:2]}"
            else:
                scanSize = tuple(f['shape'])[:2]
            # if lazy: # this doesnt work as the file is closed at the end
            lines = 4096 / scanSize[0]
            chunks=(lines, scanSize[0], det_shape, det_shape)
            chunks1D = np.prod(chunks)
            s = da.from_array(f['4D'], chunks=chunks1D) # only 1D hdf5 files
            if len(s.shape) == 1: # for 1D array
                with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                    s = s.reshape(scanSize[0], scanSize[1], det_shape, det_shape)
                # s = s.rechunk(32,32,32,32) #TODO it might be slow; should be tested
            s = hs.signals.Signal2D(s)
            s = s.as_lazy()
            if do_cut:
                x,y,w,h = roi
                # y, x, h, w = roi
                s = s.inav[x:x+w, y:y+h]
            if not lazy:
                s.compute()
        return s
            
    elif dtype == '.mib':
        # fld = os.path.split(fn)[0]
        # fn_hdr = os.path.join(fld, 'default.hdr')
        # scanSize = get_scan_size_mib_hdr(fn_hdr)
        s = hs.load(fn, lazy=lazy)
        s.rechunk(nav_chunks=(32,32))
        if do_cut:
            x,y,w,h = roi
            s = s.inav[x:x+w, y:y+h]
            s.compute()
        return s
            
    elif dtype == '.tpx3':
        if do_cut:
            x, y, w, h = roi
        else:
            x=0
            y=0 
            w=scanSize[0]
            y=scanSize[1]
        Do_4D = True
        roi = pyLP.Roi(repetitions=1, extract_4D=Do_4D)  
        roi.nx = scanSize[0]
        roi.ny = scanSize[1]
        roi.set_file(fn)
        roi.set_roi(x=x, y=y, width=w, height=h)
        roi.set_dwell_time(dwellTime*1000)
        roi.run()
        # ROI_scan_image = np.asarray(roi.Roi_scan_image)
        # ROI_diffp = np.asarray(roi.Roi_diffraction_pattern).reshape(512, 512)
        s = np.asarray(roi.Roi_4D)
        s = hs.signals.Signal2D(s)
        return s

# =============================================================================
# def load_mib_reshape(fn, scanSize, chunkSize=32):
#     # s = px.load_mib(fn, reshape=False)
#     s = hs.load(fn, lazy=True)
#     # s = px.signals.ElectronDiffraction2D(s.data.reshape(scanSize[1], scanSize[0], 256, 256))
#     # s = s.as_lazy() #TODO check for new pyxem version
#     s.rechunk(nav_chunks=(chunkSize,chunkSize))
#     return s
# =============================================================================

def get_scan_size(fn):
    dtype = os.path.splitext(fn)[-1]
    if dtype == '.hdf5':
        with h5py.File(fn, 'r') as f:
            scanSize = tuple(f['shape'])[:2]
        return scanSize
    else:
        s = load_signal(fn)
        return (s.data.shape[1], s.data.shape[0]) # TODO return y and x?

def get_det_size(fn):
    s = load_signal(fn)
    return (s.data.shape[3], s.data.shape[2])

def calculate_nav_image(s):
    img = s.sum(axis=(-1, -2))
    try:
        img.compute()
    except:
        pass
    img = img.data
    return img

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

def calculate_nav_signal_mib(fn, scanSize=None):
    if scanSize is None:
        fld = os.path.split(fn)[0]
        fn_hdr = os.path.join(fld, 'default.hdr')
        scanSizes = get_scan_size_mib_hdr(fn_hdr)
    assert len(np.unique(scanSizes)) == 1, 'Scan sizes are not the same!'
    scanSize = int(np.unique(scanSizes))
  
    s = hs.load(fn, lazy=True)
    nav_image = calculate_nav_image(s)
    return nav_image

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

def calculate_nav_signal_tpx3(fn, scanSize, dwellTime=None, r_in=0, r_out=512):
    vstem = pyLP.vSTEM(1) #arg is number of reperitions in the scan
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

def calculate_nav_signal_hdf5(fn, scanSize=None): # h5py.File(fn, 'r')['4D']
    # check for scan size and nav image in hdf5
    with h5py.File(fn, 'r') as f:
        if scanSize is None:
            scanSize = f['shape'][:2]
        else:
            assert scanSize == tuple(f['shape'])[:2], f"Scan size entered does not match to the shape of the hdf5 file: {f['shape'][:2]}"

        if 'dose_image' in f.keys():
            nav_img = f['dose_image'][:]
            # print('Navigation images is pre-calculated')
        else:
            nav_img = get_nav_image_hdf5(fn, scanSize)
    return nav_img

def get_nav_image_hdf5(fn, scanSize, det_shape=512):
    with h5py.File(fn, 'r') as f:
        if scanSize is not None: # checking scan size entered and the one written in the file
            assert scanSize == tuple(f['shape'])[:2], f"Scan size entered does not match to the shape of the hdf5 file: {f['shape'][:2]}"
        else:
            scanSize = tuple(f['shape'])[:2]
        # if lazy: # this doesnt work as the file is closed at the end
        lines = 4096 / scanSize[0]
        chunks=(lines, scanSize[0], det_shape, det_shape)
        chunks1D = np.prod(chunks)
        s = da.from_array(f['4D'], chunks=chunks1D) # only 1D hdf5 files
        if len(s.shape) == 1: # for 1D array
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                s = s.reshape(scanSize[0], scanSize[1], det_shape, det_shape)
            # s = s.rechunk(32,32,32,32) #TODO it might be slow; should be tested
        s = hs.signals.Signal2D(s)
        s = s.as_lazy()
        navImg = s.sum(axis=(2,3)).compute()
    return navImg

def calculate_nav_signal_hs(fn, scanSize=None):
    dtype = os.path.splitext(fn)[1]
    s_temp = load_signal(fn, dtype)
    nav_img = calculate_nav_image(s_temp)
    # nav_img = nav_img.T #TODO check
    return nav_img

def calculate_nav_signal(path, dtype=None, scanSize=None, dwellTime=1):
    if dtype is None:
        dtype = os.path.splitext(path)[-1]
        
    if dtype == '.mib':
        nav_img = calculate_nav_signal_mib(path, scanSize)
    elif dtype == '.tpx3':
        nav_img = calculate_nav_signal_tpx3(path, scanSize, dwellTime)
    elif dtype == '.hdf5':
        nav_img = calculate_nav_signal_hdf5(path, scanSize)
    elif dtype in ['.zspy', '.hspy']:
        nav_img = calculate_nav_signal_hs(path, scanSize)
    elif dtype in ['.dm2', '.dm3', '.tif', '.tiff']:
        nav_img = hs.load(path).data
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

# def enhance_navigation_contrast(s, method='log'): #TODO
    

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

def create_frames(pathSave, s):
    # pathSave = os.path.join(path, 'frames')
    if os.path.isdir(pathSave):
        [os.remove(os.path.join(pathSave, fn)) for fn in os.listdir(pathSave)]
    else:
        os.mkdir(pathSave)
    for i, fr in enumerate(tqdm(s.data)):
        tifffile.imwrite(os.path.join(pathSave, f'{i+1:04d}.tif'), fr)

def create_clip_dp(fn, s, scale=None, dpi=300, fps=5, vmin=None, vmax=None, cmap='inferno'):
    plt.ioff()
    fig, ax = plt.subplots()
    img = ax.imshow(s.data[0], cmap=cmap, norm=mcolors.SymLogNorm(
        vmin=vmin, vmax=vmax, linthresh=1))
    if scale is not None:
        scalebar = ScaleBar(scale*10, '1/nm', dimension='si-length-reciprocal', location='lower left',
                            scale_formatter=lambda value, unit:  f'{value / 10}'r' $\AA^{-1}$', fixed_value=5)

        ax.add_artist(scalebar)
    def update_frame(fr_no):
        img.set_data(s.data[fr_no])
        return (img, )
    ani = FuncAnimation(fig, update_frame, frames=range(s.data.shape[0]), blit=True)
    ani.save(fn + '.gif', fps=fps)
    try:
        ani.save(fn + '.mp4', writer='ffmpeg', fps=5)
    except:
        print('No ffmpeg was found to make mp4 clip!')
    plt.ion()
        
def create_clip_tracking(fn, imgs, rois=None, scale=None, dpi=300, fps=5, cmap='viridis'):
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
    
    ani = FuncAnimation(fig, update_frame, frames=range(imgs.shape[0]), blit=True)
    path, fn_raw = os.path.split(fn)
    fn = fn_raw + '.gif'
    fn = os.path.join(path, fn)
    ani.save(fn, fps=fps)
    try:
        fn = fn_raw + '.mp4'
        fn = os.path.join(path, fn)
        ani.save(fn, writer='ffmpeg', fps=5)
    except:
        print('No ffmpeg was found to make mp4 clip!')
    plt.ion()

def create_clip_tracking_with_mask(fn, imgs, masks, obj_id=1, scale=None, 
                                   dpi=300, fps=5, cmap='viridis'):
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
        show_mask(masks[fr_no], obj_id, disp_ax=img_mask)
        return (img, )
    
    plt.ioff()
    fig, ax = plt.subplots()
    img = ax.imshow(imgs[0], cmap=cmap)
    img_0 = np.zeros(imgs[0].shape, dtype=bool)
    img_mask = ax.imshow(img_0)
    show_mask(masks[0], obj_id, img_mask)
    if scale is not None:
        scalebar = ScaleBar(scale, 'nm', dimension='si-length', location='lower left')
        ax.add_artist(scalebar)
    ani = FuncAnimation(fig, update_frame, frames=range(imgs.shape[0]), blit=True)
    ani.save(fn + '.gif', fps=fps)
    ani.save(fn + '.mp4', writer='ffmpeg', fps=fps)
    plt.ion()
# =============================================================================
# if __name__ == '__main__':
#     pass
# =============================================================================
