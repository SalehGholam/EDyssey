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
