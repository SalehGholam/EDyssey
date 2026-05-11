# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:12:44 2026

@author: sgholam
"""

from glob import glob
import tifffile
import hyperspy.api as hs
import os
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from tqdm import tqdm
import dask.array as da
from dask.diagnostics import ProgressBar
from skimage.draw import disk
from scipy.ndimage import center_of_mass
from copy import deepcopy

# path_main = r'Z:\merlin\Saleh_merlin\Titan 1\251202\S2\2025-12-02_13-23-15'
# path_main = r'Z:\merlin\Saleh_merlin\Titan 1\251202\S3\2025-12-02_14-50-27'
# path_main = r'Z:\merlin\Saleh_merlin\Titan 1\251202\S4\2025-12-02_16-41-05'
path_main = r'Z:\merlin\Saleh_merlin\Titan 1\251112\s2\2025-11-12_17-47-00'

#%% funcs
def get_scan_size(path_main):
    with open(os.path.join(path_main, 'comment.txt'), 'r') as f:
        ls = f.readlines()
        shape_x = 0
        shape_y = 0
        for item in ls:
            if 'scan size x' in item:
                shape_x = int(item.split(':')[1])
            elif 'scan size y' in item:
                shape_y = int(item.split(':')[1])
            if (shape_x != 0) & (shape_y!=0):
                break
        return (shape_x, shape_y)

def create_virtual_detector(dp_sum, center, r_out, r_in=0):
    cx, cy = center
    mask_out = disk((cy,cx), r_out, shape=dp_sum.shape)
    mask_arr = np.zeros(dp_sum.shape)
    mask_arr[mask_out] = 1
    if r_in > 0:
        mask_in = disk((cy,cx), r_in)
        mask_arr[mask_in] = 0
    return mask_arr

def get_haadf_zeros(img):
    values, counts = np.unique(img, return_counts=True)
    mode = values[np.argmax(counts)]
    return int(mode)

def get_virtual_image(path_sig, path_pat, det_mask=None, scan_size=(512,512)):
    pattern = np.loadtxt(path_pat).astype('int')
    s = hs.load(path_pat, lazy=True)
    if det_mask is None: # Dose image
        img = np.zeros((scan_size[0] * scan_size[1]), dtype='uint64')
        img[pattern] = s.data.sum(axis=(1,2))
        img = img.reshape(scan_size)
        
    elif det_mask is not None: # virtual detector
        img_temp = deepcopy(s.data)
        img_temp = img_temp.reshape(img_temp.shape[0],-1)
        det_mask_idx = np.where(det_mask.flatten()==1)[0]
        img_temp = img_temp[:, det_mask_idx].sum(axis=-1)
        with ProgressBar():
            img_temp = img_temp.compute()
        img = np.zeros(scan_size[0]*scan_size[1], dtype='uint64')
        img[pattern] = img_temp
        img = img.reshape(scan_size)
    return img
#%% load file names
scan_size = get_scan_size(path_main)
fns_mib = [fn for fn in os.listdir(path_main) if '.mib' in fn]
fns_mib.sort()
fns_pat = [fn for fn in os.listdir(path_main) if '.txt' in fn]
fns_pat.sort()
fns_pat = fns_pat[:-2] # removing comment and pattern.txt (not sure what it is)

fns_haadf = [fn for fn in os.listdir(path_main) if '.tiff' in fn]
fns_haadf.sort()
fns_haadf_det = [fn for fn in fns_haadf if len(fn.split('_'))==4]
fns_haadf_acq = [fn for fn in fns_haadf if len(fn.split('_'))==5]

alpha_det = [float(item.split('_')[-1].split('.tiff')[0]) for item in fns_haadf_det]
alpha_acq = [float(item.split('_')[-2]) for item in fns_haadf_acq]
alpha_mib = [float(item.split('_')[-1][:-4].replace(',','.')) for item in fns_mib]
alpha_pat = [float(item.split('_')[1][5:]) for item in fns_pat]
# =============================================================================
# #%% plot alpha values to check files
# fig, ax = plt.subplots()
# ax.plot(alpha_det, 'o', label='HAADF FAST', alpha=0.5)
# ax.plot(alpha_det, 'o', label='HAADF SLOW', alpha=0.5)
# ax.plot(alpha_mib, 'o', label='MIB', alpha=0.5)
# ax.plot(alpha_pat, 'o', label='PATTERNS', alpha=0.5)
# ax.set(ylabel='Alpha angle ($^o$)', alpha=0.5)
# ax.legend()
# =============================================================================
#%% make haadf signal
shape = tifffile.imread(os.path.join(path_main, fns_haadf_det[0])).shape
# fast scans
s_ha_det = np.zeros((len(fns_haadf_det), *shape), dtype='int64')
for i, fn in enumerate(tqdm(fns_haadf_det)):
    s_ha_det[i] = tifffile.imread(os.path.join(path_main, fn))
s_ha_det = hs.signals.Signal2D(s_ha_det)

# slow scans
s_ha_acq = np.zeros((len(fns_haadf_acq), *shape), dtype='int64')
for i, fn in enumerate(tqdm(fns_haadf_acq)):
    s_ha_acq[i] = tifffile.imread(os.path.join(path_main, fn))
s_ha_acq = hs.signals.Signal2D(s_ha_acq)
#%% plot haadf
s_ha_det.plot()
s_ha_acq.plot()
#%% normalzie and correct haadf acquisitions
# correct the normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((0, 1000))
s_ha_acq_norm = np.zeros_like(s_ha_acq, dtype='float')
for i, im in enumerate(tqdm(s_ha_acq.data)):
    count = get_haadf_zeros(im)
    mask = (im != count)
    
    # mask first and last pixel (probably caused by a bug)
    last_pixel = np.where(mask)[0][-1], np.where(mask)[1][-1]
    mask[last_pixel] = 0
    first_pixel = np.where(mask)[0][0], np.where(mask)[1][0]
    mask[first_pixel] = 0
    
    vals = im[mask].reshape(-1, 1)
    norm_vals = scaler.fit_transform(vals).ravel()
    im_norm = np.zeros_like(im, dtype=float)
    im_norm[mask] = norm_vals
    s_ha_acq_norm[i] = im_norm
s_ha_acq_norm = hs.signals.Signal2D(s_ha_acq_norm)
s_ha_acq_norm.plot()
#%%% save
path_save = r'Z:\merlin\Saleh_merlin\Titan 1\251112\s2\Analysis'
s_ha_acq_norm.save(os.path.join(path_save, 'haadf_det_normalized.hspy'))
#%% calculate one navigation image
# find files related to an alpha
alpha = 0
i_hAcq = alpha_det.index(alpha)
i_hDet = alpha_acq.index(alpha)
i_p = alpha_pat.index(alpha)
i_m = alpha_mib.index(alpha)
# check if files are correct
print(fns_pat[i_p])
print(fns_mib[i_m])
print(fns_haadf_det[i_hAcq])
print(fns_haadf_acq[i_hDet])
#%%% create virtual detector
s = hs.load(os.path.join(path_main, fns_mib[i_m]), lazy=True)
dp_sum = s.sum(axis=0).data
cy, cx = center_of_mass(dp_sum)
# det_vdf = create_virtual_detector(dp_sum, center=(cx,cy), r_in=20, r_out=175)
det_vdf = create_virtual_detector(dp_sum, center=(cx,cy), r_in=20, r_out=170)
det_vbf = create_virtual_detector(dp_sum, center=(cx,cy), r_in=0, r_out=15)

# check virtual detectors
fig, ax = plt.subplots(1,2)
for i in [0,1]:
    ax[i].imshow(s.data.sum(axis=0), norm=SymLogNorm(linthresh=0.1))
# ax.scatter(cx,cy, color='r')
ax[0].imshow(det_vbf, alpha=0.3, cmap='inferno')
ax[1].imshow(det_vdf, alpha=0.3, cmap='inferno')
ax[0].set_title('VBF')
ax[1].set_title('VDF')
#%%% load and create all imgs
# haadf images
img_hDet = tifffile.imread(os.path.join(path_main, fns_haadf_det[i_hDet]))
# img_hAcq = tifffile.imread(os.path.join(path_main, fns_haadf_acq[i_hAcq]))
img_hAcq = s_ha_acq_norm.data[i_hAcq]

# navigation image
pattern = np.loadtxt(os.path.join(path_main, fns_pat[i_p])).astype('int')
img_nav = np.zeros((scan_size[0] * scan_size[1]), dtype='uint64')
s = hs.load(os.path.join(path_main, fns_mib[i_m]), lazy=True)
img_nav[pattern] = s.data.sum(axis=(1,2))#[:-1]
img_nav = img_nav.reshape(scan_size)
# vdf
img = deepcopy(s.data)
img = img.reshape(img.shape[0],-1)
mask_vdf = np.where(det_vdf.flatten()==1)[0]
img = img[:, mask_vdf].sum(axis=-1)
with ProgressBar():
    img = img.compute()
img_vdf = np.zeros(scan_size[0]*scan_size[1], dtype='uint64')
img_vdf[pattern] = img
img_vdf = img_vdf.reshape(scan_size)
# vbf
img = deepcopy(s.data)
img = img.reshape(img.shape[0],-1)
mask_vbf = np.where(det_vbf.flatten()==1)[0]
img = img[:, mask_vbf].sum(axis=-1)
with ProgressBar():
    img = img.compute()
img_vbf = np.zeros(scan_size[0]*scan_size[1], dtype='uint64')
img_vbf[pattern] = img
img_vbf = img_vbf.reshape(scan_size)
#%%% plot all images
fig, ax = plt.subplots(1, 5, figsize=(15,6),
                       constrained_layout=True, sharex=True, sharey=True)
# fig, ax = plt.subplots(1, 5, constrained_layout=True)
ax[0].imshow(img_hDet)
masked = np.ma.masked_equal(img_hAcq, get_haadf_zeros(img_hAcq))
ax[1].imshow(masked)
masked = np.ma.masked_equal(img_nav, 0)
# ax[2].imshow(img_nav, vmin=img_nav[img_nav!=0].min())
ax[2].imshow(masked)
masked = np.ma.masked_equal(img_vbf, 0)
ax[3].imshow(masked)
# ax[3].imshow(img_vbf, vmin=img_vbf[img_vbf!=0].min())
masked = np.ma.masked_equal(img_vdf, 0)
ax[4].imshow(masked)

for i, t in enumerate(['HAADF Detection', 'HAADF Acquisition', 
                       'Dose', 'VBF', 'VDF']):
    ax[i].set_title(t)
#%% recreate 4D file
# vmin = img_nav[img_nav>0].min()
det_shape = (512,512)
dask_arr = da.zeros(((scan_size[0] * scan_size[1],) + det_shape), 
                    dtype='uint16') # chunksize=(4,4)+det_shape, 
dask_arr[pattern] = s.data
dask_arr = dask_arr.reshape(scan_size + det_shape)
s_4d = hs.signals.Signal2D(dask_arr)
s_4d.set_signal_type('electron_diffraction')
s_4d = s_4d.as_lazy()
# s_4d.set_diffraction_calibration()
# s_4d.set_scan_calibration(3.86*2)
s_4d.plot(cmap='inferno', norm='symlog')
# s_4d.plot(navigator=hs.signals.Signal2D(img_vdf), cmap='inferno', norm='symlog', vmin=1, 
#           ) # plot with vdf as navigator
#%% segmentation using opencv
import cv2
def segment_image(
    image,
    blur_kernel=5,
    morph_kernel=3,
    dist_thresh=0.4,
    invert=False
):
    """
    Segment a 2D image using thresholding + watershed.

    Parameters
    ----------
    image : np.ndarray
        Input image, grayscale or BGR.
    blur_kernel : int
        Kernel size for Gaussian blur. Must be odd.
    morph_kernel : int
        Kernel size for morphological operations.
    dist_thresh : float
        Fraction of max distance transform used to define sure foreground.
        Typical range: 0.2 to 0.6
    invert : bool
        If True, inverts thresholding logic.

    Returns
    -------
    results : dict
        Dictionary containing:
        - gray
        - binary
        - opening
        - sure_fg
        - sure_bg
        - unknown
        - markers
        - labels
        - segmented_overlay
    """

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original = image.copy()
    else:
        gray = image.copy()
        original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Optional denoising
    gray_blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Otsu thresholding
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(
        gray_blur, 0, 255, thresh_type + cv2.THRESH_OTSU
    )

    # Remove small noise
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground from distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform,
        dist_thresh * dist_transform.max(),
        255,
        0
    )
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # Watershed requires background != 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers_ws = cv2.watershed(original, markers.copy())

    # Create label image: boundaries are -1
    labels = markers_ws.copy()

    # Overlay result
    overlay = original.copy()
    overlay[labels == -1] = [0, 0, 255]  # watershed boundaries in red

    return {
        "gray": gray,
        "binary": binary,
        "opening": opening,
        "sure_fg": sure_fg,
        "sure_bg": sure_bg,
        "unknown": unknown,
        "markers": markers_ws,
        "labels": labels,
        "segmented_overlay": overlay
    }

def to_uint8(img):
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)

    min_val = img.min()
    max_val = img.max()

    if max_val == min_val:
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - min_val) / (max_val - min_val)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

# image = s_ha_acq_norm.data[30]
image = img_vdf
image = to_uint8(image)

results = segment_image(
    image,
    blur_kernel=5,
    morph_kernel=3,
    dist_thresh=0.4,
    invert=False
)

# Save results
# cv2.imwrite("binary.png", results["binary"])
# cv2.imwrite("sure_fg.png", results["sure_fg"])
# cv2.imwrite("segmented_overlay.png", results["segmented_overlay"])

# Show results
cv2.imshow("Gray", results["gray"])
cv2.imshow("Binary", results["binary"])
cv2.imshow("Sure Foreground", results["sure_fg"])
cv2.imshow("Segmented Overlay", results["segmented_overlay"])
cv2.waitKey(0)
cv2.destroyAllWindows()
#%% segment within previous object
import cv2
import numpy as np
def normalize_to_uint8(img):
    img = np.asarray(img, dtype=np.float32)
    mn, mx = img.min(), img.max()
    if mx <= mn:
        return np.zeros(img.shape, dtype=np.uint8)
    img = (img - mn) / (mx - mn)
    return (img * 255).astype(np.uint8)


def segment_components_inside_object(image, object_mask, n_classes=4, min_area=20, blur=3):
    """
    Segment connected components inside one already-detected object.

    Parameters
    ----------
    image : 2D ndarray
        Original grayscale image
    object_mask : 2D ndarray
        Binary mask of the single object (nonzero = object)
    n_classes : int
        Number of intensity classes inside the object
    min_area : int
        Remove tiny segments
    blur : int
        Gaussian blur kernel size

    Returns
    -------
    labels : 2D int32 array
        0 = background, 1..N = internal segments
    class_map : 2D uint8 array
        Intensity-class map inside the object
    masked_img : 2D uint8 array
        Normalized image used for segmentation
    """

    img = normalize_to_uint8(image)
    mask = (object_mask > 0).astype(np.uint8)

    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        img = cv2.GaussianBlur(img, (blur, blur), 0)

    # Keep only object pixels
    masked_img = img.copy()
    masked_img[mask == 0] = 0

    # Extract object pixels for clustering
    pixels = img[mask > 0].reshape(-1, 1).astype(np.float32)

    # K-means on intensities inside the object only
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, km_labels, centers = cv2.kmeans(
        pixels, n_classes, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    centers = centers.flatten()
    order = np.argsort(centers)

    # remap classes so 0=darkest, ..., n_classes-1=brightest
    remap = np.zeros(n_classes, dtype=np.uint8)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id

    km_labels = remap[km_labels.flatten()]

    # Put class labels back into image
    class_map = np.full(img.shape, 255, dtype=np.uint8)  # 255 outside object
    class_map[mask > 0] = km_labels

    # Split each class into connected components
    labels = np.zeros(img.shape, dtype=np.int32)
    next_id = 1

    for c in range(n_classes):
        class_mask = np.zeros_like(mask, dtype=np.uint8)
        class_mask[(class_map == c) & (mask > 0)] = 255

        num_cc, cc = cv2.connectedComponents(class_mask)

        for i in range(1, num_cc):
            region = (cc == i)
            if region.sum() >= min_area:
                labels[region] = next_id
                next_id += 1

    return labels, class_map, masked_img
#%%
import matplotlib.pyplot as plt
import numpy as np


def plot_segmentation_results(image, object_mask, class_map, labels):
    """
    Visualize segmentation pipeline results.

    Parameters
    ----------
    image : 2D ndarray
    object_mask : binary mask
    class_map : output from k-means (inside object)
    labels : final connected component labels
    """

    # Masked image
    masked = image.copy()
    # masked[object_mask == 0] = np.nan
    masked = image.astype(float).copy()
    masked[object_mask == 0] = np.nan

    # Random color map for labels
    rng = np.random.default_rng(0)
    colors = rng.random((labels.max() + 1, 3))
    label_rgb = colors[labels]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=True)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(masked, cmap="gray")
    axes[1].set_title("Masked Object")
    axes[1].axis("off")

    im2 = axes[2].imshow(class_map, cmap="viridis")
    axes[2].set_title("Intensity Classes")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[3].imshow(label_rgb)
    axes[3].set_title(f"Segments (N={labels.max()})")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

labels, class_map, masked_img = segment_components_inside_object(
    image,
    results['binary'],
    n_classes=3,
    min_area=10,
    blur=3
)

plot_segmentation_results(
    image=image,
    object_mask=results['binary'],
    class_map=class_map,
    labels=labels
)
#%% SAM2
import numpy as np
import cv2


def _normalize_float01(image):
    image = np.asarray(image, dtype=np.float32)
    mn = np.nanmin(image)
    mx = np.nanmax(image)
    if mx <= mn:
        return np.zeros_like(image, dtype=np.float32)
    return (image - mn) / (mx - mn)


def _to_uint8_gray(image):
    x = _normalize_float01(image)
    return (255 * x).clip(0, 255).astype(np.uint8)


def _to_rgb_uint8(image):
    arr = np.asarray(image)

    if arr.ndim == 2:
        g = _to_uint8_gray(arr)
        return np.stack([g, g, g], axis=-1)

    if arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype == np.uint8:
            return arr
        x = _normalize_float01(arr)
        return (255 * x).clip(0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[..., :3]
        if rgb.dtype == np.uint8:
            return rgb
        x = _normalize_float01(rgb)
        return (255 * x).clip(0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _bbox_from_mask(mask, pad=5):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("The binary mask is empty.")

    x0 = max(xs.min() - pad, 0)
    x1 = xs.max() + pad + 1
    y0 = max(ys.min() - pad, 0)
    y1 = ys.max() + pad + 1
    return x0, y0, x1, y1


def _masks_to_label_image_filtered(masks, roi_mask, min_overlap_with_roi=0.5, sort_key="area"):
    """
    Convert SAM2 masks to a label image, keeping only masks that sufficiently
    overlap the supplied binary ROI mask.

    Parameters
    ----------
    masks : list of dict
        Output from SAM2AutomaticMaskGenerator.generate(...)
    roi_mask : 2D bool/uint8
        Binary mask in the cropped ROI coordinates
    min_overlap_with_roi : float
        Keep a SAM2 mask only if:
            intersection(mask, roi_mask) / area(mask) >= threshold
    """
    roi_mask = roi_mask.astype(bool)
    labels = np.zeros(roi_mask.shape, dtype=np.int32)

    if not masks:
        return labels, []

    if sort_key == "area":
        masks = sorted(masks, key=lambda m: m.get("area", 0), reverse=True)
    elif sort_key == "predicted_iou":
        masks = sorted(masks, key=lambda m: m.get("predicted_iou", 0), reverse=True)

    kept = []
    next_id = 1

    for m in masks:
        seg = m["segmentation"].astype(bool)
        area = seg.sum()
        if area == 0:
            continue

        overlap = np.logical_and(seg, roi_mask).sum()
        frac = overlap / area

        if frac >= min_overlap_with_roi:
            labels[seg] = next_id
            kept.append(m)
            next_id += 1

    # hard-clip labels to the OpenCV ROI mask if desired
    labels[~roi_mask] = 0

    return labels, kept


def run_sam2_on_opencv_mask(
    image,
    binary_mask,
    checkpoint,
    model_cfg,
    device="cuda",
    pad=10,
    points_per_side=32,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    min_mask_region_area=20,
    min_overlap_with_roi=0.5,
    sort_key="area",
):
    """
    Run SAM2 automatic mask generation only inside an OpenCV binary mask region.

    Parameters
    ----------
    image : ndarray
        2D image or RGB image.
    binary_mask : ndarray
        OpenCV-style binary mask. Nonzero pixels define the region of interest.
    checkpoint : str
        Path to SAM2 checkpoint.
    model_cfg : str
        SAM2 config path.
    device : str
        'cuda' or 'cpu'
    pad : int
        Padding around the mask bounding box before cropping.
    min_overlap_with_roi : float
        Reject SAM2 masks that do not sufficiently lie inside the OpenCV mask.

    Returns
    -------
    dict with:
        labels_full
        labels_crop
        kept_masks
        all_masks
        crop_bbox
        image_crop
        mask_crop
    """
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    mask = (binary_mask > 0).astype(np.uint8)
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {image.shape}")

    x0, y0, x1, y1 = _bbox_from_mask(mask, pad=pad)

    img_rgb = _to_rgb_uint8(image)
    img_crop = img_rgb[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]

    # Optional: suppress outside-mask pixels inside the crop
    # This makes SAM2 focus more on the masked structure.
    img_crop_masked = img_crop.copy()
    img_crop_masked[mask_crop == 0] = 0

    sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    if device == "cuda" and torch.cuda.is_available():
        context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        context = torch.no_grad()

    with torch.inference_mode():
        with context:
            all_masks = mask_generator.generate(img_crop_masked)

    labels_crop, kept_masks = _masks_to_label_image_filtered(
        all_masks,
        roi_mask=mask_crop,
        min_overlap_with_roi=min_overlap_with_roi,
        sort_key=sort_key,
    )

    labels_full = np.zeros(mask.shape, dtype=np.int32)
    labels_full[y0:y1, x0:x1] = labels_crop

    return {
        "labels_full": labels_full,
        "labels_crop": labels_crop,
        "kept_masks": kept_masks,
        "all_masks": all_masks,
        "crop_bbox": (x0, y0, x1, y1),
        "image_crop": img_crop_masked,
        "mask_crop": mask_crop,
    }

sam2_masked_result = run_sam2_on_opencv_mask(
    image=image,
    binary_mask=results['binary'],
    checkpoint=r"C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\py5DED\py4DTomo\tracking_utils\SAM2_checkpoints\sam2.1_hiera_large.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    device="cuda",
    pad=15,
    points_per_side=32, # 16 coarse, 32 standard, 64 detailed, 128 very dense
    pred_iou_thresh=0.5, # 0.9  → very strict (few masks), 0.7  → balanced, 0.5  → permissive
    stability_score_thresh=0.92, # 0.92: 0.95 → very strict, 0.9  → default-ish, 0.8  → more segments, 0.7  → aggressive (many fragments)
    min_mask_region_area=5,
    min_overlap_with_roi=0.6, # 0.8 → very strict, 0.6 → balanced, 0.3 → permissive
)

labels = sam2_masked_result["labels_full"]
print("Number of SAM2 segments:", labels.max())


import matplotlib.pyplot as plt
import numpy as np


def plot_sam2_masked_result(image, binary_mask, sam2_result):
    labels = sam2_result["labels_full"]

    img = np.asarray(image)
    if img.ndim == 3:
        img2d = np.mean(img[..., :3], axis=2)
    else:
        img2d = img

    overlay = np.stack([_normalize_float01(img2d)] * 3, axis=-1)

    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundaries[labels == 0] = False

    overlay[boundaries] = [1, 0, 0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(img2d, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(binary_mask, cmap="gray")
    axes[1].set_title("OpenCV mask")
    axes[1].axis("off")

    axes[2].imshow(labels, cmap="nipy_spectral")
    axes[2].set_title(f"SAM2 labels (N={labels.max()})")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Boundaries on image")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
plot_sam2_masked_result(image, results['binary'], sam2_masked_result)
#%%
import os

base = os.path.join(os.path.expanduser("~"), ".keras", "models", "StarDist2D")
print("Base:", base)

for name in ["2D_versatile_fluo", "2D_versatile_fluo_extracted"]:
    path = os.path.join(base, name)
    print(path, "exists:", os.path.isdir(path))
    if os.path.isdir(path):
        print("contents:", os.listdir(path)[:10])
#%% find peaks
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def find_diffraction_peaks(
    dp,
    min_distance=10,
    threshold_rel=0.1,
    sigma=1.0,
    exclude_center_radius=20,
    refine_radius=3,
    max_peaks=None,
    center=None
):
    """
    Find diffracted beam / peak positions in a 2D diffraction pattern.

    Parameters
    ----------
    dp : 2D numpy.ndarray
        Input diffraction pattern.
    min_distance : int, optional
        Minimum allowed distance between peaks in pixels.
    threshold_rel : float, optional
        Relative threshold with respect to the max intensity
        after preprocessing. Example: 0.1 means keep peaks above
        10% of max intensity.
    sigma : float, optional
        Gaussian smoothing sigma. Use 0 for no smoothing.
    exclude_center_radius : float, optional
        Radius around the image center to ignore, useful for removing
        the central direct beam.
    refine_radius : int, optional
        Radius of the local window used for subpixel centroid refinement.
    max_peaks : int or None, optional
        Maximum number of peaks to return, sorted by intensity descending.

    Returns
    -------
    peaks : numpy.ndarray, shape (N, 2)
        Peak positions as (y, x), possibly subpixel-refined.
    intensities : numpy.ndarray, shape (N,)
        Peak intensities from the preprocessed image.
    """

    if dp.ndim != 2:
        raise ValueError("Input diffraction pattern must be a 2D array.")

    img = np.asarray(dp, dtype=np.float32)

    # Basic cleanup
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img[img < 0] = 0

    # Optional smoothing
    if sigma and sigma > 0:
        img_smooth = gaussian_filter(img, sigma=sigma)
    else:
        img_smooth = img.copy()

    # Exclude central beam region
    if exclude_center_radius and exclude_center_radius > 0:
        if center is None:
            h, w = img_smooth.shape
            cy, cx = h / 2.0, w / 2.0
        else:
            cy, cx = center
        yy, xx = np.indices(img_smooth.shape)
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        img_smooth = img_smooth.copy()
        img_smooth[rr < exclude_center_radius] = 0

    # Local maxima detection
    footprint_size = 2 * min_distance + 1
    local_max = maximum_filter(img_smooth, size=footprint_size) == img_smooth

    # Thresholding
    threshold = threshold_rel * img_smooth.max()
    candidates = local_max & (img_smooth > threshold)

    peak_indices = np.argwhere(candidates)
    if peak_indices.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    # Collect intensities
    peak_intensities = img_smooth[peak_indices[:, 0], peak_indices[:, 1]]

    # Sort by descending intensity
    order = np.argsort(peak_intensities)[::-1]
    peak_indices = peak_indices[order]
    peak_intensities = peak_intensities[order]

    # Keep only max_peaks if requested
    if max_peaks is not None:
        peak_indices = peak_indices[:max_peaks]
        peak_intensities = peak_intensities[:max_peaks]

    # Subpixel refinement by intensity-weighted centroid
    refined_peaks = []
    refined_intensities = []

    h, w = img_smooth.shape
    for (py, px), inten in zip(peak_indices, peak_intensities):
        y0 = max(0, py - refine_radius)
        y1 = min(h, py + refine_radius + 1)
        x0 = max(0, px - refine_radius)
        x1 = min(w, px + refine_radius + 1)

        patch = img_smooth[y0:y1, x0:x1]
        if patch.size == 0 or np.sum(patch) <= 0:
            refined_peaks.append((float(py), float(px)))
            refined_intensities.append(float(inten))
            continue

        yy, xx = np.indices(patch.shape)
        yy = yy + y0
        xx = xx + x0

        total = np.sum(patch)
        y_ref = np.sum(yy * patch) / total
        x_ref = np.sum(xx * patch) / total

        refined_peaks.append((y_ref, x_ref))
        refined_intensities.append(float(inten))

    return np.array(refined_peaks), np.array(refined_intensities)


from skimage.filters import threshold_otsu
s_sum = s.sum(axis=(1,2)).data.compute()
thresh = threshold_otsu(s_sum)
s_strong = s.data[s_sum > thresh]
s_strong[s_strong <=4] = 0
dp = s_strong.sum(axis=(0))

peaks, intensities = find_diffraction_peaks(
    s.inav[42].data, # dp,
    min_distance=5,
    threshold_rel=0.05,
    sigma=1,
    exclude_center_radius=30,
    refine_radius=1,
    center=(cy,cx)
)

print("Peak positions (y, x):")
print(peaks)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(dp, cmap="gray", norm=SymLogNorm(linthresh=0.1))
# plt.imshow(dp_sum, cmap="gray", norm=SymLogNorm(linthresh=0.1))
plt.scatter(peaks[:, 1], peaks[:, 0], s=80, facecolors='none', edgecolors='r')
plt.show()
#%%
from matplotlib.widgets import Slider, Button
from scipy.ndimage import gaussian_filter, maximum_filter

def peaks_to_mask(shape, peak_coords, radius):
    """
    Create a boolean mask for diffraction peaks.

    Parameters
    ----------
    shape : tuple
        Shape of the diffraction pattern as (height, width).
    peak_coords : array-like of shape (N, 2)
        Peak coordinates as [(y1, x1), (y2, x2), ...].
        Subpixel coordinates are allowed.
    radius : float
        Radius around each peak to set to True.

    Returns
    -------
    mask : 2D numpy.ndarray of bool
        Boolean mask with True inside the disk around each peak.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)

    if len(peak_coords) == 0:
        return mask

    yy, xx = np.indices((h, w))
    r2 = radius ** 2

    for y, x in peak_coords:
        mask |= (yy - y) ** 2 + (xx - x) ** 2 <= r2

    return mask

mask = peaks_to_mask(shape=dp.shape, peak_coords=peaks, radius=6)
plt.figure(figsize=(6, 6))
plt.imshow(mask, cmap="gray")
plt.show()


img = deepcopy(s.data)
# img[img<=2] = 0
img = img.reshape(img.shape[0],-1)
mask_arr = np.where(mask.flatten()==1)[0]
img = img[:, mask_arr].sum(axis=-1)
with ProgressBar():
    img = img.compute()
img_mvdf = np.zeros(scan_size[0]*scan_size[1], dtype='uint64')
img_mvdf[pattern] = img
img_mvdf = img_mvdf.reshape(scan_size)


fig, ax = plt.subplots(constrained_layout=True)
masked = np.ma.masked_equal(img_mvdf, 0)
ax.imshow(masked)
# ax[3].imshow(img_vbf, vmin=img_vbf[img_vbf!=0].min())


def find_diffraction_peaks(
    dp,
    min_distance=10,
    threshold_rel=0.1,
    sigma=1.0,
    exclude_center_radius=20,
    center=(256,256)
):
    img = np.asarray(dp, dtype=np.float32)
    img = np.nan_to_num(img)
    img[img < 0] = 0

    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)

    if exclude_center_radius > 0:
        h, w = img.shape
        cy, cx = center
        yy, xx = np.indices((h, w))
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        img[rr < exclude_center_radius] = 0

    size = int(2 * min_distance + 1)
    local_max = maximum_filter(img, size=size) == img

    threshold = threshold_rel * img.max()
    peaks = np.argwhere(local_max & (img > threshold))

    return peaks


def peak_finder_widget(dp):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)

    im = ax.imshow(dp, cmap="gray", norm=SymLogNorm(linthresh=0.1))
    scatter = ax.scatter([], [], facecolors='none', edgecolors='r')

    ax.set_title("Diffraction Peak Finder")

    # Sliders
    ax_sigma = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_thresh = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_dist = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_center = plt.axes([0.25, 0.10, 0.65, 0.03])

    s_sigma = Slider(ax_sigma, 'sigma', 0.0, 5.0, valinit=1.0)
    s_thresh = Slider(ax_thresh, 'threshold', 0.01, 0.5, valinit=0.1)
    s_dist = Slider(ax_dist, 'min_dist', 1, 30, valinit=10, valstep=1)
    s_center = Slider(ax_center, 'center_excl', 0, 100, valinit=20)

    def update(val=None):
        peaks = find_diffraction_peaks(
            dp,
            min_distance=int(s_dist.val),
            threshold_rel=s_thresh.val,
            sigma=s_sigma.val,
            exclude_center_radius=s_center.val,
            center=(cy,cx)
        )

        if len(peaks) > 0:
            scatter.set_offsets(peaks[:, ::-1])  # (y,x) → (x,y)
        else:
            scatter.set_offsets([])

        fig.canvas.draw_idle()

    s_sigma.on_changed(update)
    s_thresh.on_changed(update)
    s_dist.on_changed(update)
    s_center.on_changed(update)

    # Initial run
    update()

    plt.show()

dp = s.inav[42].data
dp[dp<=2] = 0
peak_finder_widget(dp)