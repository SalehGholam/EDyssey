# -*- coding: utf-8 -*-
"""8-bit contrast normalization for navigation-signal display and
downstream processing (tracking/SAM2), plus a small general-purpose blur
helper used by the same tracking-prep code paths.
"""
import numpy as np
import cv2

CONTRAST_METHODS = ('percentile', 'minmax', 'std')

def _contrast_bounds(data, axis, method='percentile', plow=1.0, phigh=99.0, n_std=3.0):
    """Compute the (lo, hi) value bounds contrast is stretched between, for
    the given method - shared by convert_to_8bit/convert_img_to_8bit.

    Args:
        data: float array.
        axis: axis (or tuple of axes) reduced over - the trailing two axes
            for a per-frame stack, or None for a single 2-D image.
        method: One of CONTRAST_METHODS.
            'percentile' (default): stretch between the plow/phigh
                percentiles - robust to the handful of hot/dead pixels
                common on electron-counting detectors, which would
                otherwise dominate a raw min/max range and wash out
                everything else (a real cause of SAM2 producing inaccurate
                masks on effectively-flat-looking 8-bit input).
            'minmax': stretch between the raw min and max - simple, but
                exactly as vulnerable to those outlier pixels as the name
                implies.
            'std': stretch between mean ± n_std standard deviations -
                another common way to reject outliers, tunable via n_std.
        plow, phigh: 'percentile' method's low/high percentiles (0-100).
        n_std: 'std' method's clip half-width, in standard deviations.

    Returns:
        (lo, hi), each broadcastable against `data`.
    """
    if method == 'minmax':
        lo = data.min(axis=axis, keepdims=True)
        hi = data.max(axis=axis, keepdims=True)
    elif method == 'std':
        mean = data.mean(axis=axis, keepdims=True)
        std = data.std(axis=axis, keepdims=True)
        lo = mean - n_std * std
        hi = mean + n_std * std
    elif method == 'percentile':
        lo = np.percentile(data, plow, axis=axis, keepdims=True)
        hi = np.percentile(data, phigh, axis=axis, keepdims=True)
    else:
        raise ValueError(f'Unknown contrast method {method!r}; expected one of {CONTRAST_METHODS}')
    return lo, hi

def convert_to_8bit(s, method='percentile', plow=1.0, phigh=99.0, n_std=3.0,
                     clip_low=None, clip_high=None):
    """Normalise each frame of a HyperSpy signal to 8-bit (0-255).

    See `_contrast_bounds` for the available methods and their tunable
    parameters.

    Args:
        s: HyperSpy Signal2D with arbitrary numeric dtype.
        method: Contrast method - one of CONTRAST_METHODS.
        plow, phigh: 'percentile' method's low/high percentiles (per frame).
        n_std: 'std' method's clip half-width, in standard deviations.
        clip_low, clip_high: Optional raw-value thresholds - values below
            clip_low/above clip_high are clamped to that threshold *before*
            the method's own stretch is computed, e.g. to knock a saturated
            beam stop or a dead-pixel border out of consideration. None
            (default) skips this pre-clip entirely.

    Returns:
        A deep copy of `s` with data converted to uint8, each frame independently normalised.
    """
    data = s.data.astype(np.float32)
    if clip_low is not None or clip_high is not None:
        data = np.clip(data, clip_low, clip_high)
    # Normalise per-frame using broadcasting (vectorised, no Python loop)
    axis = tuple(range(data.ndim - 2, data.ndim))
    los, his = _contrast_bounds(data, axis, method, plow, phigh, n_std)
    data = np.clip((data - los) / (his - los + 1e-8), 0.0, 1.0)
    data = (data * 255.0).astype(np.uint8)
    s_8bit = s._deepcopy_with_new_data(data)
    return s_8bit

def convert_img_to_8bit(img, method='percentile', plow=1.0, phigh=99.0, n_std=3.0,
                         clip_low=None, clip_high=None):
    """Normalise a single 2-D numpy array to uint8 - see `convert_to_8bit`/
    `_contrast_bounds` for the available methods and their tunable parameters.

    Args:
        img: 2-D numpy array of any numeric dtype.
        method: Contrast method - one of CONTRAST_METHODS.
        plow, phigh: 'percentile' method's low/high percentiles.
        n_std: 'std' method's clip half-width, in standard deviations.
        clip_low, clip_high: Optional raw-value thresholds - see
            `convert_to_8bit`. None (default) skips this pre-clip entirely.

    Returns:
        numpy.ndarray of dtype uint8.
    """
    img = np.asarray(img, dtype=np.float32)
    if clip_low is not None or clip_high is not None:
        img = np.clip(img, clip_low, clip_high)
    lo, hi = _contrast_bounds(img, None, method, plow, phigh, n_std)
    img_8bit = np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0) * 255.0
    return img_8bit.astype(np.uint8)

def gaussian_blur(img, kernel_size=3):
    """Apply a square Gaussian blur kernel to a 2-D image.

    Args:
        img: 2-D numpy array.
        kernel_size: Side length of the kernel (must be odd). Default is 3.

    Returns:
        Blurred numpy.ndarray of the same shape and dtype as `img`.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def threshold_ignore_zero(threshold_func, img):
    """Apply a skimage threshold function (threshold_otsu/threshold_li/...)
    considering only pixels with a value above 0.

    True-zero pixels are background/never-visited scan positions (always the
    case for unvisited positions in a reconstructed smart-scan array - see
    `loaders._reconstruct_smart_scan` - and common at the frame border on
    dense scans too), not real signal - letting them dominate the histogram
    skews the computed threshold low. Falls back to thresholding the whole
    image if there are no nonzero pixels at all (an all-zero `img` has
    nothing meaningful to threshold either way).

    Args:
        threshold_func: A skimage.filters threshold function (e.g.
            `threshold_otsu`), called with a 1-D array of the nonzero pixels.
        img: 2-D (or N-D) numpy array.

    Returns:
        The scalar threshold value.
    """
    nonzero = img[img > 0]
    if nonzero.size == 0:
        return threshold_func(img)
    return threshold_func(nonzero)

def _directional_kernel(length, angle_deg):
    """Build a one-sided ray-shaped erosion kernel: a `length`-pixel line
    from the kernel's anchor point out to angle `angle_deg` (0 = +x/right,
    increasing clockwise to match image row-down/col-right pixel axes).

    Used by `erode_mask_edge` for directional erosion: `cv2.erode` with this
    kernel and anchor keeps a pixel only if the mask extends the full ray
    length from it in that direction, so pixels within `length` of the
    boundary *as seen looking along `angle_deg`* get eroded away - i.e. the
    boundary crossed by walking in that direction (its "far side"). A
    kernel_size square (the isotropic default) is just every angle's ray
    combined, hence eroding uniformly from all sides.

    Returns:
        (kernel, anchor) - kernel is a uint8 numpy.ndarray, anchor an
        (x, y) tuple for cv2.erode's own `anchor` argument.
    """
    length = max(1, int(round(length)))
    size = 2 * length + 1
    anchor = (length, length)
    kernel = np.zeros((size, size), np.uint8)
    theta = np.deg2rad(angle_deg)
    x1 = anchor[0] + length * np.cos(theta)
    y1 = anchor[1] + length * np.sin(theta)
    cv2.line(kernel, anchor, (int(round(x1)), int(round(y1))), 1, thickness=1)
    kernel[anchor[1], anchor[0]] = 1
    return kernel, anchor

def erode_mask_edge(mask, kernel_size=3, direction=None, revert=False):
    """Reduce a binary mask to just its edge/outline, `kernel_size` pixels wide.

    Erodes the mask, then subtracts the eroded mask from the original -
    what's left is the boundary band the erosion ate away. Used as an
    optional post-processing step on segmentation/threshold masks (ROI on
    4D, ROI Tracker, SAM2 Tracker) so extraction/display can be restricted
    to a particle's edge instead of its whole area.

    Args:
        mask: 2-D array, truthy where the mask is set (any dtype).
        kernel_size: Erosion kernel size, in pixels (the square kernel's
            side when `direction` is None, the ray length otherwise).
            Larger values eat further into the mask before subtracting,
            producing a wider edge band. Must be >= 1.
        direction: None (default) erodes uniformly from every side with a
            square kernel, keeping the mask's whole outline - matches prior
            behaviour. A float angle in degrees (0 = +x/right, increasing
            clockwise) instead erodes only along that one direction (see
            `_directional_kernel`), keeping just the boundary band facing
            that angle - e.g. 0 keeps the mask's right-hand edge, 90 its
            bottom edge, 180 its left edge, 270 its top edge.
        revert: If False (default), returns just the edge band (prior
            behaviour). If True, returns its complement *within the
            original mask* instead - i.e. the mask's interior (and, with
            `direction` set, its other sides too), with only the one
            detected edge band cut out.

    Returns:
        numpy.ndarray of dtype bool, same shape as `mask` - True on the
        mask's edge band (or, if `revert`, everywhere in the original mask
        except that edge band).
    """
    if direction is None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        anchor = (-1, -1)  # cv2's own "center" sentinel
    else:
        kernel, anchor = _directional_kernel(kernel_size, direction)
    mask_u8 = mask.astype('uint8')
    mask_eroded = cv2.erode(mask_u8, kernel, anchor=anchor, iterations=1)
    edge_mask = (mask_u8 & ~mask_eroded).astype(bool)
    if revert:
        return mask_u8.astype(bool) & ~edge_mask
    return edge_mask

def shift_mask_edge(mask, direction, grow=True):
    """Grow or shrink a binary mask by one pixel along one direction -
    manual, directional single-pixel mask fine-tuning (see mask_edit_dialog.py).

    Reuses `_directional_kernel`'s one-sided ray kernel (see `erode_mask_edge`
    for the angle convention: 0 = +x/right, 90 = +y/down, 180 = left,
    270 = up), dilating to grow the mask outward in that direction or
    eroding to pull its edge inward from that side.

    Args:
        mask: 2-D array, truthy where the mask is set (any dtype).
        direction: Angle in degrees (0/90/180/270 for right/bottom/left/top).
        grow: If True, dilate (extend outward); if False, erode (pull inward).

    Returns:
        numpy.ndarray of dtype bool, same shape as `mask`.
    """
    mask_u8 = mask.astype('uint8')
    if grow:
        # Dilating with the *same* kernel/anchor erosion uses to shrink from
        # `direction` actually extends the mask the opposite way (erosion's
        # "keep only if supported looking towards `direction`" becomes, for
        # dilation's max-over-neighbourhood, "pull in support from
        # `direction`" instead of "push out towards it") - the 180-degree
        # rotated kernel is what actually grows the mask towards `direction`.
        kernel, anchor = _directional_kernel(1, direction + 180)
        return cv2.dilate(mask_u8, kernel, anchor=anchor, iterations=1).astype(bool)
    kernel, anchor = _directional_kernel(1, direction)
    return cv2.erode(mask_u8, kernel, anchor=anchor, iterations=1).astype(bool)
