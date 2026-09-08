# -*- coding: utf-8 -*-
"""Conventional (non deep-learning) image denoising methods, shared by the
"Adjust Contrast" box's "Denoise" control (ROI on 4D / ROI Tracker / SAM2
Tracker) and its "Check Methods" comparison window - see
ui_tabs/denoise_widget.py. Ported from the other_scripts/
denoise_tomogram_comparison.ipynb prototype, where these same methods were
first tried out and compared on real tomogram data.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
# scipy.signal (convolve2d below, in estimate_noise_sigma) is deferred to
# first use - it's a slow import (~0.8s, mostly its own array-API-backend
# setup) that would otherwise run at app startup even for sessions that
# never denoise anything.
from skimage.restoration import (
    denoise_bilateral, denoise_tv_chambolle, denoise_wavelet, denoise_nl_means,
)

# 'None' first (no-op, the default/disabled state) - every other key is one
# denoise_image() branch, and (save for 'Wavelet', which has none) a
# DENOISE_PARAM_SPECS entry describing its one adjustable parameter.
DENOISE_METHODS = ('None', 'Gaussian Blur', 'Median Filter', 'Bilateral',
                   'Non-Local Means', 'Total Variation (Chambolle)', 'Wavelet')

# Each method's single user-adjustable parameter (label + spinbox range/step/
# decimals to build a QDoubleSpinBox from, and the default value denoise_image
# falls back to when `param` is None) - None for methods with nothing to
# adjust ('None', 'Wavelet'). Kept here (not in ui_tabs) so the "Check
# Methods" comparison window and the live "Denoise" control both build their
# spinboxes from, and run denoise_image() against, exactly the same values.
DENOISE_PARAM_SPECS = {
    'None': None,
    'Gaussian Blur': {'label': 'Sigma', 'default': 1.5,
                      'min': 0.1, 'max': 20.0, 'step': 0.1, 'decimals': 1},
    'Median Filter': {'label': 'Size (px)', 'default': 3,
                      'min': 1, 'max': 51, 'step': 2, 'decimals': 0},
    'Bilateral': {'label': 'Sigma Color', 'default': 0.05,
                 'min': 0.01, 'max': 1.0, 'step': 0.01, 'decimals': 2},
    'Non-Local Means': {'label': 'h (x noise sigma)', 'default': 1.15,
                        'min': 0.1, 'max': 5.0, 'step': 0.05, 'decimals': 2},
    'Total Variation (Chambolle)': {'label': 'Weight', 'default': 0.05,
                                    'min': 0.001, 'max': 1.0, 'step': 0.01, 'decimals': 3},
    'Wavelet': None,
}


def estimate_noise_sigma(img):
    """Fast, dependency-free noise-sigma estimate (Immerkjaer, 1996):
    convolving with a discrete Laplacian isolates high-frequency noise from
    the (locally smooth) underlying signal; the mean absolute response,
    rescaled, approximates the Gaussian noise standard deviation. Used as a
    starting point for the Non-Local Means kernel width in denoise_image
    below - skimage's own estimate_sigma needs PyWavelets even for methods
    that otherwise don't, this doesn't need any of that."""
    from scipy.signal import convolve2d
    laplacian_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)
    conv = convolve2d(img, laplacian_kernel, mode='same', boundary='symm')
    h, w = img.shape
    return np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))


def denoise_image(img, method, param=None):
    """Apply `method` (one of DENOISE_METHODS) to a single 2-D float image.
    Best given an image already normalized to roughly [0, 1] (e.g. via
    convert_img_to_8bit(...)/255, or a percentile-clip normalization) - every
    method's own default parameter assumes that range.

    Args:
        img: 2-D float array.
        method: One of DENOISE_METHODS. 'None' is a no-op (returns `img`
            unchanged) - lets a caller use one consistent code path whether
            or not denoising is actually enabled.
        param: That method's one adjustable parameter (see
            DENOISE_PARAM_SPECS) - falls back to that spec's own 'default'
            when None. Ignored for 'None'/'Wavelet' (neither has one).

    Returns:
        2-D float array, same shape as `img`.
    """
    if method is None or method == 'None':
        return img
    spec = DENOISE_PARAM_SPECS.get(method)
    if param is None and spec is not None:
        param = spec['default']

    if method == 'Gaussian Blur':
        return gaussian_filter(img, sigma=float(param))
    if method == 'Median Filter':
        return median_filter(img, size=max(1, int(round(param))))
    if method == 'Bilateral':
        return denoise_bilateral(img, sigma_color=float(param), sigma_spatial=3, channel_axis=None)
    if method == 'Non-Local Means':
        sigma_est = estimate_noise_sigma(img)
        return denoise_nl_means(img, h=float(param) * sigma_est, fast_mode=True,
                                patch_size=5, patch_distance=6, channel_axis=None)
    if method == 'Total Variation (Chambolle)':
        return denoise_tv_chambolle(img, weight=float(param), channel_axis=None)
    if method == 'Wavelet':
        return denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    raise ValueError(f'Unknown denoising method {method!r}')
