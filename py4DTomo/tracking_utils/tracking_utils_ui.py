# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:58:43 2024

@author: SGholam
"""

import numpy as np
import pyxem as px
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import hyperspy.api as hs
from copy import deepcopy
import os
# import sys
# sys.path.append(r'E:\OneDrive - Universiteit Antwerpen\GitHub\Others\4D Tomo\4DSTEM Tomography')
# import pyLiveProcessing as pyLP
import io_utils_ui as io
from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_mean
from py4DTomo.io_utils import create_array_from_dissimilar_imgs
from dask import config
# import matplotlib.patches as patches
from dask.diagnostics import ProgressBar
import dask.array as da
import time
config.set(scheduler='single-threaded')
# from dask import config as da_config
# da_config.set(scheduler='processes')
#%%
def select_roi(img):
    # cv2.namedWindow('ROI Selection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('ROI Selection', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('ROI Selection', img.shape[1]*4, img.shape[0]*4)
    # img = ((img - img.min()) / (img.max() - img.min())) * 255.0
    # img = img.astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    cv2.imshow('ROI Selection', img)
    box = cv2.selectROI('ROI Selection', img)
    print(f'box coordinations: {box}')
    return box

def select_rois(img):
    rois = []
    img = deepcopy(img)
    img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    # Create a window with a specific name
    cv2.namedWindow('ROI Selection', cv2.WINDOW_NORMAL)

    # Resize the window to the image size
    cv2.resizeWindow('ROI Selection', 1024, 1024)

    # font for writing the rectangular number
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (255, 255, 255)  # BGR color format (blue, green, red)
    count = 1
    while True:
        # Select ROI
        # roi = cv2.selectROI('Frame', self.img)
        roi = cv2.selectROI('ROI Selection', img)

        # Check if a valid ROI was selected
        if roi[2] > 0 and roi[3] > 0:
            # Append the ROI to the list
            rois.append(roi)

            # Draw the selected ROI on the image
            cv2.rectangle(img, (int(roi[0]), int(roi[1])),
                          (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (0, 255, 0), 1)
            cv2.putText(img, str(count), (roi[0]-5, roi[1]-5), font, font_scale, font_color, font_thickness)

            # Update the displayed image
            # self.ax.imshow(self.img)
            # self.canvas.draw()
            count += 1
        else:
            break
    print("Selected ROIs:", rois)
    return rois

def select_rois_manual(s):
    imgs = s.data
    rois = []
    tracked_imgs = []
    for i, img in enumerate(imgs):
        roisTemp = []
        img = deepcopy(img)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        # Create a window with a specific name
        cv2.namedWindow('ROI Selection', cv2.WINDOW_NORMAL)
        
        # Resize the window to have a large window
        imgSize = img.shape
        scaler = 1
        s = imgSize[0]
        while s < 750:
            scaler *= 2
            s *= scaler
        cv2.resizeWindow('ROI Selection', imgSize[0]*scaler, imgSize[1]*scaler)
        
        # font for writing the rectangular number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        font_color = (255, 255, 255)  # BGR color format (blue, green, red)
        count = 1
        while True:
            # Select ROI
            # roi = cv2.selectROI('Frame', self.img)
            roi = cv2.selectROI('ROI Selection', img)
        
            # Check if a valid ROI was selected
            if roi[2] > 0 and roi[3] > 0:
                # Append the ROI to the list
                roisTemp.append(roi)
        
                # Draw the selected ROI on the image
                cv2.rectangle(img, (int(roi[0]), int(roi[1])),
                              (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (0, 255, 0), 1)
                cv2.putText(img, str(count), (roi[0]-5, roi[1]-5), font, font_scale, font_color, font_thickness)
        
                # Update the displayed image
                # self.ax.imshow(self.img)
                # self.canvas.draw()
                count += 1
            else:
                break
        # print("Selected ROIs:", rois)
        rois.append(roisTemp)
        tracked_imgs.append(img)
    rois = np.swapaxes(rois, 0, 1) # swap so: [roiNo, imgNo, roiCoords]
    tracked_imgs = hs.signals.Signal2D(tracked_imgs)
    return rois, tracked_imgs



def track_roi_cv2(imgs, rois, init=0, tracking_method='csrt'):
    path_origin = os.getcwd()
    path_file = os.path.abspath(__file__)
    path_file = os.path.split(path_file)[0]
    path_trackerModels = os.path.join(path_file, 'opencv_models')
    os.chdir(path_trackerModels)
    if tracking_method == 'csrt':
        tracker = cv2.TrackerCSRT_create()
    elif tracking_method == 'mil':
        tracker = cv2.TrackerMIL_create()
    elif tracking_method == 'nano':
        tracker = cv2.TrackerNano_create()
    elif tracking_method == 'dasiamrpn':
        tracker = cv2.TrackerDaSiamRPN_create()
    else:
        os.chdir(path_origin)
        raise NotImplementedError('The tracker used is not available')
    os.chdir(path_origin)
    
    flag_3ch_cvt = False # flag for converting to 3 channel images
    if tracking_method in ['nano', 'dasiamrpn']:
        flag_3ch_cvt = True
    
    init.append(len(imgs))
    tracked_rois = []
    for i_c, _ in enumerate(init[:-1]):
        imgs_temp = imgs[init[i_c]:init[i_c+1]]
        # roi = rois[init[i_c]]
        roi = rois[i_c]
        x,y,w,h = roi
        # y = imgs_temp[0].shape[1] - y - h # origin is top left in cv2 and bottom left in mpl
        # roi = convert_roi_to_int((x,y,w,h))
        img_0 = imgs_temp[0]
        if flag_3ch_cvt:
            img_0 = cv2.cvtColor(img_0, cv2.COLOR_GRAY2BGR)
        # print(roi)
        tracker.init(img_0, roi)
        tracked_rois.append(roi)
        for img in imgs_temp[1:]:
            if flag_3ch_cvt:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            (success, box) = tracker.update(img)
            if success:
                box = [int(a) for a in box]
                x, y, w, h = box
                # roi might be out of image
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if (x+w) > img.shape[0]:
                    w = img.shape[0] - x
                if (y+h) > img.shape[1]:
                    h = img.shape[1] - y
                # y = img.shape[1] - y - h # origin is top left in cv2 and bottom left in mpl
                # TODO check box is correct
                box = (x,y,w,h)
                tracked_rois.append(box)
    cv2.destroyAllWindows() #TODO not sure if it is needed
    return tracked_rois

def convert_roi_to_int(roi):
    x,y,w,h = roi
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    return (x,y,w,h)

def track_roi(box, imgs, label=None, method='csrt'):
    if method == 'csrt':
        tracker = cv2.TrackerCSRT_create()
    elif method == 'mil':
        tracker = cv2.TrackerMIL_create()
    elif method == 'nano':
        tracker = cv2.TrackerNano_create()
    elif method == 'dasiamrpn':
        tracker = cv2.TrackerDaSiamRPN_create()
    else:
        raise NotImplementedError('The tracker used is not available')
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.25
    font_thickness = 1
    font_color = (255, 255, 255)  # BGR color format (blue, green, red)
    
    img = deepcopy(imgs[0])
    img_0 = deepcopy(img)
    tracked_imgs = []
    tracked_rois = []
    x,y,w,h = box
    cv2.rectangle(img_0, (x,y), (x+w,y+h), (255,255,255), 1)
    cv2.putText(img, str(label), (x-5, y-5), font, font_scale, font_color, font_thickness)
    tracked_imgs.append(img_0)
    tracked_rois.append(box)
    if method in ['nano', 'dasiamrpn']:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print(img.shape)
    tracker.init(img, box)
    for i, img in enumerate(imgs[1:]):
        img = deepcopy(img)
        if method in ['nano', 'dasiamrpn']:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        (success, box) = tracker.update(img)
        if success:
            box = [int(a) for a in box]
            x, y, w, h = box
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if (x+w) > img.shape[0]:
                w = img.shape[0] - x
            if (y+h) > img.shape[1]:
                h = img.shape[1] - y
            box = [x,y,w,h]
            tracked_rois.append(box)
            (x,y,w,h) = box
            if method in ['nano', 'dasiamrpn']:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 1)
            cv2.putText(img, str(label), (x-3, y-3), font, font_scale, font_color, font_thickness)
        else:
            tracked_rois.append([0,0,0,0])
            print(f'frame #{i} failed')
        tracked_imgs.append(img)
    cv2.destroyAllWindows()
    tracked_imgs = hs.signals.Signal2D(tracked_imgs)
    tracked_rois = np.array(tracked_rois)
    return tracked_rois, tracked_imgs
#%% funcs for extracting 3DED
def subtract_image_background(img, sub_method, sub_scale=1):
    if sub_method == 'otsu':
        thresh = threshold_otsu(img)
    elif sub_method == 'yen':
        thresh = threshold_yen(img)
    elif sub_method == 'li':
        # thresh = threshold_li(img)
        raise('Li threshold gives error! Try something else.')
    elif sub_method == 'mean':
        thresh = threshold_mean(img)
    thresh *= sub_scale
    binary_img = img >= thresh
    # binary_img = binary_img.compute(show_progressbar=False)
    return binary_img

def extract_3ded(nav_signal, fns_4d, rois, i_roi, dtype, sig_shape, scanSize, ext_method='sum',
                 sub_method=None, sub_scale=1, sub_center_mask=False, centeringRep=False, init_roi=False):
    s_3ded = np.zeros((len(fns_4d), sig_shape, sig_shape), dtype=np.uint32)
    if sub_method:
        sub_imgs = []
    # for i, fn in tqdm(enumerate(fns_4d), total=len(fns_4d), desc=f"ROI #{i_roi}", position=i_roi): #as pbar
    for i, fn in tqdm(enumerate(fns_4d), total=len(fns_4d), desc=f"ROI #{i_roi}"): #as pbar
    # for i, fn in enumerate(fns_4d):
        if np.any(init_roi):
            x0, y0, w0, h0 = init_roi[0][i]
            x, y, w, h = rois[i]
            r = [x0+x, y0+y, w, h]
            # s_temp = io.load_signal(fn, dtype, scanSize, roi=init_roi[0][i]) #only works for 1 initial roi
            # s_temp = s_temp.inav[x:x+w, y:y+h]
            s_temp = io.load_signal(fn, dtype, scanSize, roi=r) #only works for 1 initial roi
        else:            
            x, y, w, h = rois[i]
            s_temp = io.load_signal(fn, dtype, scanSize, roi=rois[i])
        # img = (img / np.max(img) * 255).astype(np.uint8) #not sure whether it is necessary
        if ext_method == 'sum':
            if sub_method:
                img = nav_signal.inav[i].isig[x:x+w, y:y+h]
                img = img.data
                # plt.imshow(img)
                binary_img = subtract_image_background(img, sub_method, sub_scale)
                binary_img_true = np.where(binary_img==1)
                sub_imgs.append(binary_img)
                
                if sub_center_mask:
                    if sub_center_mask == 'auto':
                        dp = s_temp.inav[0,0].data.compute(show_progressbar=False)
                        center = np.unravel_index(np.argmax(dp, axis=None), dp.shape)
                        sub_center_mask = (center[0], center[1], 10) #TODO check
                    s_temp.center_direct_beam(method='center_of_mass', mask=sub_center_mask, lazy_output=True)
                    if centeringRep:
                        if s_temp.data.shape[-1] == 256:
                            s_temp.center_direct_beam(method='center_of_mass', mask=(128,128,5), lazy_output=True)
                        elif s_temp.data.shape[-1] == 512:
                            s_temp.center_direct_beam(method='center_of_mass', mask=(256,256,5), lazy_output=True)
                    
                    # s_temp.center_direct_beam(method='center_of_mass', mask=sub_center_mask)
                try:
                    s_temp.compute(show_progressbar=False)
                except AttributeError:
                    pass
                for j, _ in enumerate(binary_img_true[0]):
                    y, x = binary_img_true[0][j], binary_img_true[1][j] # binary_img_true is a tuple; x,y are inverted in np and hs
                    # dp = s_temp.inav[x, y].data.compute(show_progressbar=False)
                    try:
                        dp = s_temp.inav[x, y].data
                    except:
                        pass
                    try:
                        s_3ded[i] += dp
                    except:
                        s_3ded[i] += dp.astype('uint32')
                
            else:
                s_temp = s_temp.sum(0).sum(0).data.compute(show_progressbar=False)
                # s_temp.compute(show_progressbar=False)
                s_3ded[i] += s_temp
                
        elif ext_method == 'brightest':
            img = s_temp.sum(-1).sum(-1).data
            br_pixel = np.unravel_index(np.argmax(img, axis=None), img.shape)
            x, y = br_pixel # x, y are reversed between hyperspy and numpy
            dp = s_temp.data[x,y].compute(show_progressbar=False)
            try:
                s_3ded[i] = dp
            except:
                s_3ded[i] = dp.astype('uint32')
                
    s_3ded = hs.signals.Signal2D(s_3ded)
    if 'sub_imgs' in locals():
        try:
            sub_imgs = px.signals.ElectronDiffraction2D(sub_imgs) # wont work for rois with different shapes
        except:
            sub_imgs = create_array_from_dissimilar_imgs(sub_imgs)
            # pass
        return s_3ded, sub_imgs, i_roi
    else:
        return s_3ded, i_roi

# =============================================================================
# def extract_3ded_mask(fns_4d, masks, make_signal=False): 
#     shape_x, shape_y, shape_dx, shape_dy = hs.load(fns_4d[0], lazy=True).data.shape #TODO x,y are not reversed?
#     s_3ded = np.zeros((len(fns_4d), shape_dx, shape_dy), dtype='uint32')
#     # for i, mask in enumerate(tqdm(masks)):
#     for i, mask in enumerate(masks):
#         s_temp = hs.load(fns_4d[i], lazy=True)
#         with config.set(**{'array.slicing.split_large_chunks': False}):
#             arr_flat = s_temp.data.reshape(-1, *s_temp.data.shape[2:])
#         mask_flat = mask.flatten()
#         
#         # Apply the 1D mask
#         sliced_flat = arr_flat[mask_flat, :, :]
#         
#         s_3ded[i] = sliced_flat.sum(axis=(0)).compute()
#     if make_signal:
#         s_3ded = hs.signals.Signal2D(s_3ded)
#     return s_3ded
# =============================================================================

def extract_3ded_mask_single_frame(fn, mask, dtype=None, scanSize=None, 
                                   roi=None):
    time.sleep(0.1)
    if roi is not None:
        x,y,w,h = roi
        # mask = mask[x:x+w, y:y+h]
        mask = mask[y:y+h, x:x+w] #TODO check
    dtype = os.path.splitext(fn)[1]
    s = io.load_signal(fn, dtype=dtype, scanSize=scanSize, 
                       roi=roi, lazy=True)
    if type(s) == tuple: # hdf5 sends the file handler
        s, f = s
    
    with config.set(**{'array.slicing.split_large_chunks': False}):
        arr_flat = s.data.reshape(-1, *s.data.shape[2:])
    # Apply the 1D mask
# =============================================================================
#     mask_flat = mask.flatten()
#     sliced_flat = arr_flat[mask_flat, :, :]
# =============================================================================
    mask_flat = np.where(mask.flatten()==1)[0]
    sliced_flat = arr_flat[mask_flat]
    
    dp = sliced_flat.sum(axis=(0))
    if hasattr(dp, 'compute'):
        if isinstance(dp, da.Array):
            with ProgressBar():
                dp = dp.compute()
        else:
            dp.compute()
    return dp

def check_threshold(img, dev=0.1, step=0.05):
    rng = np.arange(1-dev, 1+dev, step)
    fig, ax = plt.subplots(3, len(rng))
    threshes = [t(img) for t in [threshold_otsu, threshold_yen, threshold_mean]]
    lbls = ['otsu', 'yen', 'mean']
    for i_t, t in enumerate(threshes):
        for i_r, r in enumerate(rng):
            img_b = img > (t * r)
            ax[i_t, i_r].imshow(img_b)
            # ax[i_t, i_r].set_axis_off()
            ax[i_t, i_r].set_xticks([])
            ax[i_t, i_r].set_yticks([])
            ax[i_t, i_r].set_title(f'{r:.02f}')
        ax[i_t,0].set_ylabel(lbls[i_t])

def create_masks(navImgs, rois, thresh_method='otsu', thresh_offset=0, blur_kernel=1):
    threshold_methods = {
    'otsu': threshold_otsu,
    'li': threshold_li,
    'yen': threshold_yen,
    'mean': threshold_mean}
    threshold_func = threshold_methods[thresh_method]
    
    masks = np.zeros(navImgs.shape, dtype=navImgs.dtype)
    for i, img in enumerate(navImgs):
        y,x,h,w = rois[i]
        if blur_kernel != 1:
            img = io.convert_img_to_8bit(img)
            img = io.gaussian_blur(img, blur_kernel)
        masks[i][x:x+w, y:y+h] = img[x:x+w, y:y+h]
        th = threshold_func(masks[i])
        th *= thresh_offset
        
        masks[i] = masks[i] > th
    masks = masks.astype('bool')
    return masks
    
def cut_imgs_by_roi(imgs, rois):
    imgs_cut = []
    for i_r, r in enumerate(rois):
        y,x,h,w = r
        cut_img = imgs[i_r][x:x+w, y:y+h]
        imgs_cut.append(cut_img)
    imgs_cut = io.create_array_from_dissimilar_imgs(imgs_cut, mode='edge', 
                                                        signal=False)
    return imgs_cut

def translate_roiInRoi(rois_1, rois_2, fwd=True):
    rois_new = []
    for i, roi in rois_1:
        x,y,w,h = roi
        x0,y0,w0,h0 = rois_2[i]
        if fwd:
            if x+w > x0+w0:
                w = x-(x0+w0)
            if y+h > y0+h0:
                h = y-(y0+h0)
            rois_new.append((x-x0, y-y0, w, h))
        else:
            rois_new.append((x+x0, y+y0, w, h))
    return rois_new

def make_tracking_signal(imgs, rois, border_value=65000):
    tracking_images = np.zeros(imgs.shape, dtype=imgs.dtype)
    for i, img in enumerate(imgs):
        tracking_images[i] = make_tracking_image(img, rois[i], border_value)
    s_tr = hs.signals.Signal2D(tracking_images)
    return s_tr, tracking_images
        
def make_tracking_image(img, roi, border_value=65000):
    x,y,w,h = roi
    img[y, x:x+w] = border_value
    # Bottom edge
    img[y+h-1, x:x+w] = border_value
    # Left edge
    img[y:y+h, x] = border_value
    # Right edge
    img[y:y+h, x+w-1] = border_value
    return img
    
# =============================================================================
# if __name__ == '__main__':
#     pass
# =============================================================================
