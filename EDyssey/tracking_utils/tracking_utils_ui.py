# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:58:43 2024

@author: SGholam
"""

import logging
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
import hyperspy.api as hs
from copy import deepcopy
import os
# import sys
# sys.path.append(r'E:\OneDrive - Universiteit Antwerpen\GitHub\Others\4D Tomo\4DSTEM Tomography')
# import pyLiveProcessing as pyLP
import io_utils_ui as io
from . import asset_fetch
from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_mean
from EDyssey.io_utils import create_array_from_dissimilar_imgs
from dask import config
# import matplotlib.patches as patches
from dask.diagnostics import ProgressBar
import dask.array as da
# from dask import config as da_config
# da_config.set(scheduler='processes')

_logger = logging.getLogger(__name__)
#%%
def select_roi(img):
    """Open an interactive OpenCV window for the user to draw a single ROI.

    Args:
        img: 2-D numpy array displayed as a VIRIDIS colourmap.

    Returns:
        Tuple (x, y, w, h) of the selected bounding box in pixel coordinates.
    """
    cv2.namedWindow('ROI Selection', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('ROI Selection', img.shape[1]*4, img.shape[0]*4)
    img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    cv2.imshow('ROI Selection', img)
    box = cv2.selectROI('ROI Selection', img)
    print(f'box coordinations: {box}')
    return box

def select_rois(img):
    """Open an interactive OpenCV window for the user to draw multiple ROIs on one image.

    Press Enter/Space after each ROI to confirm; draw a zero-size ROI (just press Esc or
    Enter without dragging) to finish.

    Args:
        img: 2-D numpy array displayed as a VIRIDIS colourmap.

    Returns:
        List of (x, y, w, h) tuples, one per selected ROI.
    """
    rois = []
    img = deepcopy(img)
    img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    cv2.namedWindow('ROI Selection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROI Selection', 1024, 1024)

    # font for writing the rectangular number
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (255, 255, 255)  # BGR color format (blue, green, red)
    count = 1
    while True:
        roi = cv2.selectROI('ROI Selection', img)

        if roi[2] > 0 and roi[3] > 0:
            rois.append(roi)

            cv2.rectangle(img, (int(roi[0]), int(roi[1])),
                          (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (0, 255, 0), 1)
            cv2.putText(img, str(count), (roi[0]-5, roi[1]-5), font, font_scale, font_color, font_thickness)

            count += 1
        else:
            break
    print("Selected ROIs:", rois)
    return rois

def select_rois_manual(s):
    """Manually select ROIs frame-by-frame for a HyperSpy signal stack.

    Opens an OpenCV window for each frame in `s` and lets the user draw one or more ROIs
    per frame. Useful when particles move too fast for automatic tracking initialisation.

    Args:
        s: HyperSpy Signal2D of shape (N, H, W).

    Returns:
        Tuple (rois, tracked_imgs) where:
            - rois is a numpy array of shape (n_rois, N, 4) with (x, y, w, h) per frame.
            - tracked_imgs is a HyperSpy Signal2D of shape (N, H, W) with drawn boxes.
    """
    imgs = s.data
    rois = []
    tracked_imgs = []
    for i, img in enumerate(imgs):
        roisTemp = []
        img = deepcopy(img)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
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
            roi = cv2.selectROI('ROI Selection', img)

            if roi[2] > 0 and roi[3] > 0:
                roisTemp.append(roi)

                cv2.rectangle(img, (int(roi[0]), int(roi[1])),
                              (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (0, 255, 0), 1)
                cv2.putText(img, str(count), (roi[0]-5, roi[1]-5), font, font_scale, font_color, font_thickness)

                count += 1
            else:
                break
        rois.append(roisTemp)
        tracked_imgs.append(img)
    rois = np.swapaxes(rois, 0, 1) # swap so: [roiNo, imgNo, roiCoords]
    tracked_imgs = hs.signals.Signal2D(tracked_imgs)
    return rois, tracked_imgs



_XCORR_METHODS = ('xcorr-phase', 'xcorr-template')


def track_roi_cv2(imgs, rois, init=[0], tracking_method='csrt'):
    """Track one ROI across a sequence of images using an OpenCV tracker.

    Supports multiple initialisation frames: `init` and `rois` define pairs of
    (start_frame, initial_box). Between consecutive init frames the tracker runs
    forward; a new tracker is created at each init point.

    Args:
        imgs: numpy.ndarray of shape (N, H, W) with uint8 frames.
        rois: List/array of (x, y, w, h) initial boxes, one per entry in `init`.
        init: List of frame indices where each box in `rois` is used to (re-)initialise
              the tracker. Default is `[0]` (single init at frame 0).
        tracking_method: OpenCV tracker name — `'csrt'`, `'mil'`, `'nano'`,
                         `'dasiamrpn'` — or a frame-to-frame cross-correlation
                         method, `'xcorr-phase'` (FFT phase correlation) or
                         `'xcorr-template'` (normalized template matching in a
                         local search window). The xcorr methods keep the ROI
                         size fixed and only estimate translation.

    Returns:
        numpy.ndarray of shape (N, 4) with (x, y, w, h) per frame.
    """
    tracker = None
    if tracking_method not in _XCORR_METHODS:
        # 'nano'/'dasiamrpn' need external .onnx weight files that aren't
        # bundled with the app (see THIRD_PARTY_NOTICES.md) - fetch them on
        # first use rather than shipping ~164MB in every installer. This
        # runs inside the background WorkerThread_General track_roi_cv2 is
        # called from, so it blocks that worker (not the GUI thread) while
        # downloading; a failure raises asset_fetch.AssetDownloadError,
        # which WorkerThread_General routes to the caller's connected
        # `.error` signal instead of crashing silently.
        asset_fetch.ensure_tracker_models(tracking_method)
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

    # the image numbers might not be sorted
    init = np.array(init)
    rnk = np.argsort(init)
    rois = np.array(rois)[rnk]
    init = init[rnk]
    init = np.append(init, len(imgs))

    tracked_rois = []
    for i_c, _ in enumerate(init[:-1]):
        imgs_temp = imgs[init[i_c]:init[i_c+1]]
        if len(imgs_temp) > 1:
            roi = rois[i_c]
            if tracking_method in _XCORR_METHODS:
                tracked_rois.extend(track_roi_xcorr(imgs_temp, roi, tracking_method))
                continue
            x,y,w,h = roi
            # y = imgs_temp[0].shape[1] - y - h # origin is top left in cv2 and bottom left in mpl
            img_0 = imgs_temp[0]
            if flag_3ch_cvt:
                img_0 = cv2.cvtColor(img_0, cv2.COLOR_GRAY2BGR)
            tracker.init(img_0, roi)
            tracked_rois.append(roi)
            for img in imgs_temp[1:]:
                if flag_3ch_cvt:
                    img = img.copy()
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
        else:
            box = rois[i_c]
            tracked_rois.append(box)
    cv2.destroyAllWindows() #TODO not sure if it is needed
    return np.array(tracked_rois)


def _xcorr_context_window(x, y, w, h, img_w, img_h, min_margin=32):
    """Padded (x0, y0, x1, y1) window around a (x, y, w, h) box, used as the
    correlation input in `track_roi_xcorr` instead of the bare box.

    Correlating just the tight ROI starves the band-pass prefilter of
    spatial support and gives phase/template correlation very little
    structure to lock onto. Padding out to real image context (clipped to
    the frame) is what made `other_scripts/cross correlation.py` - which
    correlates whole navigation frames - track more reliably than a
    tight-crop approach.
    """
    margin_x, margin_y = max(w, min_margin), max(h, min_margin)
    x0, y0 = max(0, x - margin_x), max(0, y - margin_y)
    x1, y1 = min(img_w, x + w + margin_x), min(img_h, y + h + margin_y)
    return x0, y0, x1, y1


def _xcorr_bandpass_normalize(img, low_sigma=8.0, high_sigma=1.5):
    """Suppress slow illumination drift and high-frequency noise before
    correlating: subtract the mean, band-pass with a difference-of-
    Gaussians, then normalize to unit standard deviation - the same
    preprocessing `other_scripts/cross correlation.py` uses
    (`preprocess_for_cross_correlation`), found to track more reliably on
    real navigation data than correlating raw pixel values, since a slow
    background gradient otherwise dominates the correlation peak.

    `low_sigma` is capped to the crop size so small crops (e.g. the tight,
    unpadded template used by 'xcorr-template') don't get flattened to
    near-zero by a kernel wider than the crop itself.
    """
    # cv2.matchTemplate only accepts CV_8U/CV_32F (not CV_64F), and numpy's
    # float64 .mean()/.std() scalars silently upcast an otherwise-float32
    # pipeline under NEP 50 promotion rules, so dtype is pinned to float32
    # throughout rather than left to fall out of the arithmetic.
    img = img.astype(np.float32) - np.float32(img.mean())
    low_sigma = min(low_sigma, max(min(img.shape) / 4.0, high_sigma + 0.5))
    img = gaussian_filter(img, sigma=high_sigma) - gaussian_filter(img, sigma=low_sigma)
    std = np.float32(img.std())
    if std > 1e-8:
        img = img / std
    return img.astype(np.float32)


def track_roi_xcorr(imgs, roi, tracking_method='xcorr-phase'):
    """Track a fixed-size ROI across a sequence of images via frame-to-frame
    cross-correlation, the way cross-correlation is classically used for
    image centering / drift correction.

    Unlike the appearance-based OpenCV trackers, this only estimates a
    translation (x, y) per frame; the ROI's (w, h) stays exactly as given in
    `roi`. Each frame's shift is estimated against the *previous* frame
    (rolling reference), so it adapts to slow appearance changes but can
    accumulate drift error over long sequences. Both methods correlate a
    band-pass-filtered, padded context window around the box (see
    `_xcorr_context_window`/`_xcorr_bandpass_normalize`) rather than the raw
    box pixels, for the same reasons `other_scripts/cross correlation.py`
    preprocesses before correlating.

    Args:
        imgs: numpy.ndarray of shape (N, H, W) with uint8 (or similar) frames.
        roi: Initial (x, y, w, h) bounding box on the first frame.
        tracking_method: `'xcorr-phase'` for FFT phase correlation
            (`cv2.phaseCorrelate`, sub-pixel) or `'xcorr-template'` for
            normalized cross-correlation template matching
            (`cv2.matchTemplate` of the tight box within the padded context
            window, integer-pixel, more robust to local contrast/intensity
            changes).

    Returns:
        List of (x, y, w, h) tuples, one per frame in `imgs`.
    """
    x, y, w, h = (int(v) for v in roi)
    img_h, img_w = imgs[0].shape[:2]
    w = min(w, img_w)
    h = min(h, img_h)
    x = min(max(x, 0), img_w - w)
    y = min(max(y, 0), img_h - h)

    tracked_rois = [(x, y, w, h)]

    if tracking_method == 'xcorr-phase':
        cx0, cy0, cx1, cy1 = _xcorr_context_window(x, y, w, h, img_w, img_h)
        prev_prep = _xcorr_bandpass_normalize(imgs[0][cy0:cy1, cx0:cx1])
        hann = cv2.createHanningWindow((prev_prep.shape[1], prev_prep.shape[0]), cv2.CV_32F)

        for img in imgs[1:]:
            cur_prep = _xcorr_bandpass_normalize(img[cy0:cy1, cx0:cx1])
            (dx, dy), _response = cv2.phaseCorrelate(prev_prep * hann, cur_prep * hann)

            x = min(max(int(round(x + dx)), 0), img_w - w)
            y = min(max(int(round(y + dy)), 0), img_h - h)
            tracked_rois.append((x, y, w, h))

            cx0, cy0, cx1, cy1 = _xcorr_context_window(x, y, w, h, img_w, img_h)
            prev_prep = _xcorr_bandpass_normalize(img[cy0:cy1, cx0:cx1])
            if prev_prep.shape != hann.shape:
                hann = cv2.createHanningWindow((prev_prep.shape[1], prev_prep.shape[0]), cv2.CV_32F)

    else: # xcorr-template
        prev_prep = _xcorr_bandpass_normalize(imgs[0][y:y+h, x:x+w])

        for img in imgs[1:]:
            sx0, sy0, sx1, sy1 = _xcorr_context_window(x, y, w, h, img_w, img_h)
            search = img[sy0:sy1, sx0:sx1]
            if search.shape[0] < h or search.shape[1] < w:
                dx, dy = 0.0, 0.0
            else:
                search_prep = _xcorr_bandpass_normalize(search)
                result = cv2.matchTemplate(search_prep, prev_prep, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(result)
                dx, dy = (sx0 + max_loc[0]) - x, (sy0 + max_loc[1]) - y

            x = min(max(int(round(x + dx)), 0), img_w - w)
            y = min(max(int(round(y + dy)), 0), img_h - h)
            tracked_rois.append((x, y, w, h))
            prev_prep = _xcorr_bandpass_normalize(img[y:y+h, x:x+w])

    return tracked_rois

def convert_roi_to_int(roi):
    """Cast an (x, y, w, h) ROI tuple to integer components.

    Args:
        roi: Tuple or array of four numeric values (x, y, w, h).

    Returns:
        Tuple (x, y, w, h) with all values converted to int.
    """
    x, y, w, h = roi
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    return (x,y,w,h)

def track_roi(box, imgs, label=None, method='csrt'):
    """Track a single ROI across all frames using an OpenCV tracker, returning images and boxes.

    Unlike `track_roi_cv2`, this function also returns annotated images and handles the
    label overlay for display purposes.

    Args:
        box: Initial (x, y, w, h) bounding box on the first frame.
        imgs: numpy.ndarray of shape (N, H, W) with uint8 frames.
        label: Text label drawn next to the ROI rectangle on each frame.
        method: OpenCV tracker name — `'csrt'`, `'mil'`, `'nano'`, or `'dasiamrpn'`.

    Returns:
        Tuple (tracked_rois, tracked_imgs) where:
            - tracked_rois is a numpy.ndarray of shape (N, 4).
            - tracked_imgs is a HyperSpy Signal2D of shape (N, H, W) with boxes drawn.
    """
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
    
    img = imgs[0].copy()
    img_0 = imgs[0].copy()
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
    """Threshold an image to produce a binary mask using the selected method.

    Args:
        img: 2-D numpy array.
        sub_method: Thresholding algorithm — `'otsu'`, `'yen'`, `'li'`, or `'mean'`.
        sub_scale: Multiplicative scale applied to the computed threshold. Default is 1.

    Returns:
        Boolean numpy.ndarray with True where the image exceeds the scaled threshold.
    """
    if sub_method == 'otsu':
        thresh = threshold_otsu(img)
    elif sub_method == 'yen':
        thresh = threshold_yen(img)
    elif sub_method == 'li':
        # thresh = threshold_li(img)
        raise ValueError('Li threshold gives error! Try something else.')
    elif sub_method == 'mean':
        thresh = threshold_mean(img)
    thresh *= sub_scale
    binary_img = img >= thresh
    # binary_img = binary_img.compute(show_progressbar=False)
    return binary_img

def extract_3ded(nav_signal, fns_4d, rois, i_roi, dtype, sig_shape, scanSize, ext_method='sum',
                 sub_method=None, sub_scale=1, sub_center_mask=False, centeringRep=False, init_roi=False):
    """Extract a 3D electron diffraction (3DED) pattern stack from a series of 4D-STEM files.

    For each file in `fns_4d`, loads the ROI-cropped signal and accumulates diffraction
    patterns using the chosen extraction method.

    Args:
        nav_signal: HyperSpy Signal2D navigation image stack (one frame per file).
        fns_4d: List of 4D-STEM file paths in acquisition order.
        rois: Array of shape (N, 4) with per-frame (x, y, w, h) ROI coordinates.
        i_roi: Index of this ROI (used only for tqdm label).
        dtype: File extension string (e.g. `.hdf5`).
        sig_shape: Detector size in pixels (square assumed).
        scanSize: (nx, ny) scan dimensions.
        ext_method: `'sum'` to sum all ROI pixels; `'brightest'` to use the brightest pixel.
        sub_method: Background subtraction threshold method (`'otsu'`, `'yen'`, etc.) or None.
        sub_scale: Scale factor for the subtraction threshold.
        sub_center_mask: Mask for direct-beam centering. `'auto'` to detect automatically.
        centeringRep: If True, apply a second centering pass at the detector centre.
        init_roi: Optional outer ROI array for roi-in-roi extraction.

    Returns:
        If `sub_method` is set: (s_3ded, sub_imgs, i_roi).
        Otherwise: (s_3ded, i_roi), where s_3ded is a HyperSpy Signal2D of shape (N, det, det).
    """
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
                    except Exception:
                        _logger.warning(
                            'Failed to read pixel (%d, %d) at frame %d; reusing the '
                            'previous pixel\'s diffraction pattern instead.', x, y, i,
                            exc_info=True)
                    try:
                        s_3ded[i] += dp
                    except Exception:
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
            except Exception:
                s_3ded[i] = dp.astype('uint32')
                
    s_3ded = hs.signals.Signal2D(s_3ded)
    if 'sub_imgs' in locals():
        try:
            # Imported here rather than at module level: pyxem pulls in a
            # large, slow (~5s) dependency chain (orix, diffsims, sympy)
            # that would otherwise be paid on every app startup even though
            # this is the only place it's used, guarded by a fallback below.
            import pyxem as px
            sub_imgs = px.signals.ElectronDiffraction2D(sub_imgs) # wont work for rois with different shapes
        except Exception:
            sub_imgs = create_array_from_dissimilar_imgs(sub_imgs)
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

def extract_3ded_mask_single_frame(fn, mask, dtype=None, scanSize=None, roi=None):
    """Extract a single 3DED diffraction pattern from one 4D-STEM frame using a binary mask.

    Loads the file lazily, flattens the scan dimensions, selects pixels where `mask == 1`,
    and sums those diffraction patterns.

    Args:
        fn: Path to the 4D-STEM file.
        mask: 2-D boolean array matching the scan dimensions; True pixels are summed.
        dtype: File extension. Inferred from `fn` if None.
        scanSize: (nx, ny) scan dimensions used for reshaping flat .hdf5 data.
        roi: Optional (x, y, w, h) crop applied to both the signal and the mask.

    Returns:
        numpy.ndarray of shape (det_y, det_x) with the summed diffraction pattern.
    """
    # load_signal/load_tpx3/load_hs/load_hdf5 no longer return HyperSpy
    # Signal2D objects (nor a (signal, file_handle) tuple for hdf5 - the
    # file is now closed internally): .tpx3 returns the raw eventem.Roi
    # object itself, everything else a plain numpy/dask array.
    if dtype is None:
        dtype = os.path.splitext(fn)[1]

    if dtype == '.tpx3':
        # eventem's mask-based ROI applies over the full scan grid directly
        # (see loaders.load_tpx3) - it doesn't compose with a separate `roi`
        # crop, so `roi` (if given) is ignored here and the *uncropped*
        # mask is used as-is.
        roi_obj = io.load_tpx3(fn, scanSize=scanSize, mask=mask, get_4d=False)
        # Timepix3's detector is fixed at 512x512 - the same default used
        # everywhere else in this app for .tpx3.
        return np.array(roi_obj.Roi_diffraction_pattern).reshape(512, 512)

    if roi is not None:
        x, y, w, h = roi
        mask = mask[y:y+h, x:x+w]
    # .hdf5's lazy=True path returns a dask array built from a dataset whose
    # file is already closed by the time load_signal returns (see
    # loaders.load_hdf5) - not safe to compute later here, so read it
    # eagerly instead; .hspy/.zspy/.mib's lazy loading is HyperSpy-managed
    # and stays safely readable across the call boundary.
    s = io.load_signal(fn, dtype=dtype, scanSize=scanSize,
                       roi=roi, lazy=(dtype != '.hdf5'))

    with config.set(**{'array.slicing.split_large_chunks': False}):
        arr_flat = s.reshape(-1, *s.shape[2:])
    mask_flat = np.where(mask.flatten()==1)[0]
    sliced_flat = arr_flat[mask_flat]

    dp = sliced_flat.sum(axis=(0))
    if hasattr(dp, 'compute'):
        if isinstance(dp, da.Array):
            with ProgressBar():
                dp = dp.compute()
        else:
            dp = dp.compute()
    return dp

def check_threshold(img, dev=0.1, step=0.05):
    """Display a grid comparing three thresholding methods at multiple scale factors.

    Creates a matplotlib figure with rows for otsu/yen/mean and columns sweeping the
    threshold scale in the range [1-dev, 1+dev]. Useful for picking `thresh_offset`.

    Args:
        img: 2-D numpy array to threshold.
        dev: Half-range around scale=1 to sweep. Default is 0.1.
        step: Step size between scale values. Default is 0.05.
    """
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
    """Create per-frame binary segmentation masks from navigation images and ROIs.

    For each frame, crops the image to the ROI, optionally blurs it, applies a threshold,
    and returns a boolean mask over the full navigation image.

    Args:
        navImgs: numpy.ndarray of shape (N, H, W) with navigation images.
        rois: Array of shape (N, 4) with per-frame (y, x, h, w) ROI coordinates.
        thresh_method: Thresholding algorithm — `'otsu'`, `'li'`, `'yen'`, or `'mean'`.
        thresh_offset: Multiplicative scale applied to the computed threshold.
        blur_kernel: Gaussian blur kernel size. 1 means no blur.

    Returns:
        Boolean numpy.ndarray of shape (N, H, W).
    """
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
        th = io.threshold_ignore_zero(threshold_func, img[x:x+w, y:y+h])
        th *= thresh_offset
        
        masks[i] = masks[i] >= th
    masks = masks.astype('bool')
    return masks
    
def cut_imgs_by_roi(imgs, rois):
    """Crop each image in a stack to its corresponding ROI and pad to a common size.

    Args:
        imgs: numpy.ndarray of shape (N, H, W).
        rois: Array of shape (N, 4) with per-frame (y, x, h, w) ROI coordinates.

    Returns:
        numpy.ndarray of shape (N, H_max, W_max) with edge-padded crops.
    """
    imgs_cut = []
    for i_r, r in enumerate(rois):
        y,x,h,w = r
        cut_img = imgs[i_r][x:x+w, y:y+h]
        imgs_cut.append(cut_img)
    imgs_cut = io.create_array_from_dissimilar_imgs(imgs_cut, mode='edge', 
                                                        signal=False)
    return imgs_cut

def translate_roiInRoi(rois_1, rois_2, fwd=True):
    """Convert inner ROI coordinates between global scan space and outer-ROI-relative space.

    Forward (`fwd=True`): subtracts the outer ROI origin, giving coordinates relative to
    the top-left corner of the outer crop. Reverse (`fwd=False`): adds the outer origin
    back to convert back to global coordinates.

    Args:
        rois_1: Array of shape (N, 4) — inner ROI coordinates (x, y, w, h) to translate.
        rois_2: Array of shape (N, 4) — outer ROI coordinates (x0, y0, w0, h0).
        fwd: If True, go from global → relative. If False, go from relative → global.

    Returns:
        List of (x, y, w, h) tuples in the target coordinate system.
    """
    rois_new = []
    for i, roi in enumerate(rois_1):
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
    """Draw ROI border lines on every frame of an image stack.

    Args:
        imgs: numpy.ndarray of shape (N, H, W).
        rois: Array of shape (N, 4) with per-frame (x, y, w, h) bounding boxes.
        border_value: Pixel intensity used for the drawn border. Default is 65000.

    Returns:
        Tuple (s_tr, tracking_images) where s_tr is a HyperSpy Signal2D and
        tracking_images is a numpy.ndarray of shape (N, H, W) with borders drawn.
    """
    tracking_images = np.zeros(imgs.shape, dtype=imgs.dtype)
    for i, img in enumerate(imgs):
        tracking_images[i] = make_tracking_image(img, rois[i], border_value)
    s_tr = hs.signals.Signal2D(tracking_images)
    return s_tr, tracking_images
        
def make_tracking_image(img, roi, border_value=65000):
    """Draw a rectangular border on a single 2-D image in-place.

    Args:
        img: 2-D numpy array to modify.
        roi: (x, y, w, h) bounding box in pixel coordinates.
        border_value: Pixel intensity for the four border edges. Default is 65000.

    Returns:
        The modified `img` array (same object, mutated in place).
    """
    x, y, w, h = roi
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
