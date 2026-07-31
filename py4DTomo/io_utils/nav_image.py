# -*- coding: utf-8 -*-
"""Navigation-image and diffraction-pattern computation: virtual-detector
masks, beam-center finding, and the format-dispatching nav-image/summed-DP
functions used throughout the app (eventem fast paths for .tpx3, dask-based
fallbacks for everything else).
"""
import numpy as np
import os
import h5py
import eventem
import hyperspy.api as hs
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.draw import disk
from scipy.ndimage import center_of_mass, gaussian_filter

from .progress import redirect_console_to_logger, LoggingProgressBar
from .loaders import load_signal, load_hdf5

def create_nav_signal_from_haadf(fns, dtype):  #TODO fix
    """Stack a list of HAADF/navigation images into a HyperSpy Signal2D.

    Args:
        fns: List of file paths to individual image frames.
        dtype: File format string (currently unused; kept for API consistency).

    Returns:
        HyperSpy Signal2D with shape (N, H, W).
    """
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

def calculate_nav_img_tpx3(fn, scanSize, dwellTime=None, r_in=0, r_out=512,
                           offset=None, repetitions=1, fn_pattern=None, logger=None,
                           n_threads=None):
    """Compute a virtual STEM (vSTEM) navigation image from a .tpx3 file using an annular detector.

    Args:
        fn: Path to the .tpx3 file.
        scanSize: (nx, ny) scan dimensions in pixels.
        dwellTime: Dwell time per pixel in microseconds (multiplied by 1000 internally).
        r_in: Inner radius of the annular detector in pixels.
        r_out: Outer radius of the annular detector in pixels.
        offset: (x, y) shift of the virtual detector's center relative to the
            detector's own center, in pixels. None = centered.
        repetitions: Number of scan repetitions encoded in the file.
        fn_pattern: Optional path to a pattern/calibration file (passed through
            to eventem, when the acquisition needs one).
        logger: Optional logger the eventem progress output is redirected
            into, so it shows up in the Qt log console.
        n_threads: Optional override for eventem's own internal thread pool
            (see load_tpx3's docstring for why this matters for concurrent
            callers).

    Returns:
        numpy.ndarray of shape (ny, nx) with float values.
    """
    vstem = eventem.vSTEM(repetitions) #arg is number of reperitions in the scan
    if n_threads is not None:
        vstem.n_threads = n_threads
    vstem.b_cumulative = True
    vstem.set_file(fn)
    vstem.nx = scanSize[0]
    vstem.ny = scanSize[1]
    vstem.inner_radia = [r_in]
    vstem.outer_radia = [r_out]
    vstem.set_dwell_time(dwellTime*1000)
    if fn_pattern:
        vstem.set_pattern_file(fn_pattern)
    if offset:
        # eventem's vSTEM offsets are in (row, col) order, like the rest of
        # its C++/pybind11 API - `offset` itself is kept as (x, y) at this
        # function's own boundary (matching every other coordinate in this
        # module), so it's swapped only right here, at the eventem call site.
        vstem.set_offsets([[offset[1], offset[0]]])
    with redirect_console_to_logger(logger, 'Loading tpx3'):
        vstem.run()
    nav_image = np.asarray(vstem.vSTEM_image).reshape(
        scanSize[1],scanSize[0])
    return nav_image

def get_roi(fn_tpx3, dwellTime, roi=None, bitDepth=8,
           scanSize=(512,512), fn_pattern=None, fourd=False, logger=None,
           n_threads=None):
    """Run an eventem ROI extraction over a .tpx3 file (optionally 4D).

    Args:
        fn_tpx3: Path to the .tpx3 file.
        dwellTime: Dwell time per pixel in microseconds (multiplied by 1000 internally).
        roi: Optional (x, y, w, h) scan-space crop; None extracts the full scan.
        bitDepth: Bit depth eventem should accumulate counts at.
        scanSize: (nx, ny) scan dimensions in pixels.
        fn_pattern: Optional path to a pattern/calibration file.
        fourd: If True, also extract the full 4D array (`s.get_4D()`).
        logger: Optional logger the eventem progress output is redirected
            into, so it shows up in the Qt log console.
        n_threads: Optional override for eventem's own internal thread pool
            (see load_tpx3's docstring for why this matters for concurrent
            callers).

    Returns:
        The eventem.Roi instance (already run) - use `.get_4D()`,
        `.Roi_scan_image`, `.Roi_diffraction_pattern` etc. to read results.
    """
    s = eventem.Roi(repetitions=1, extract_4D=fourd)
    if n_threads is not None:
        s.n_threads = n_threads
    s.set_bitdepth(bitDepth)
    s.nx = scanSize[0]
    s.ny = scanSize[1]
    s.set_file(fn_tpx3)
    if fn_pattern:
        s.set_pattern_file(fn_pattern)
    if roi:
        x,y,w,h = roi
        s.set_roi(x=x, y=y, width=w, height=h)
    s.set_dwell_time(dwellTime*1000)
    with redirect_console_to_logger(logger, 'Loading tpx3'):
        s.run()
    return s

def get_dp(fn, scanSize, fn_pattern=None, roi=None, dwellTime=1, repetitions=1,
          bitDepth=16, logger=None, n_threads=None):
    """Compute a summed diffraction pattern from a .tpx3 file via eventem,
    without materialising the full 4D array.

    An ROI (roi=(x, y, w, h)) is summed via `get_roi`/eventem.Roi - purpose-
    built for scan-space cropping, unlike eventem.Pacbed, which is designed
    for cumulative position-averaged summing over the *whole* scan and is
    used here only when roi is None.

    Args:
        fn: Path to the .tpx3 file.
        scanSize: (nx, ny) scan dimensions in pixels.
        fn_pattern: Optional path to a pattern/calibration file.
        roi: Optional (x, y, w, h) scan-space crop; None sums the whole scan.
        dwellTime: Dwell time per pixel in microseconds (multiplied by 1000 internally).
        repetitions: Number of scan repetitions encoded in the file (whole-scan path only).
        bitDepth: Bit depth eventem should accumulate counts at (ROI path only).
        logger: Optional logger the eventem progress output is redirected into.
        n_threads: Optional override for eventem's own internal thread pool
            (see load_tpx3's docstring for why this matters for concurrent
            callers).

    Returns:
        numpy.ndarray of shape scanSize (detector-shaped) with the summed pattern.
    """
    if roi is not None:
        s_roi = get_roi(fn, dwellTime, roi=roi, bitDepth=bitDepth, scanSize=scanSize,
                        fn_pattern=fn_pattern, fourd=False, logger=logger,
                        n_threads=n_threads)
        return np.asarray(s_roi.Roi_diffraction_pattern).reshape(scanSize)

    s_dp_sum = eventem.Pacbed(repetitions) # repetitions; eventem.Pacbed is a third-party class name
    if n_threads is not None:
        s_dp_sum.n_threads = n_threads

    s_dp_sum.set_file(fn)
    s_dp_sum.b_cumulative = True
    s_dp_sum.set_dwell_time(int(dwellTime*1000))
    s_dp_sum.nx = scanSize[0]
    s_dp_sum.ny = scanSize[1]

    if fn_pattern:
        s_dp_sum.set_pattern_file(fn_pattern)

    with redirect_console_to_logger(logger, 'Loading tpx3'):
        s_dp_sum.run()
    im_2 = np.asarray(s_dp_sum.Pacbed_image).reshape(scanSize)
    return im_2

def create_virtual_detector(shape, center, r_out, r_in=0):
    """Build a boolean-like annular (or full-disk, if r_in=0) detector mask.

    Args:
        shape: (det_y, det_x) shape of the mask/detector.
        center: (x, y) center of the detector in pixel coordinates.
        r_out: Outer radius in pixels.
        r_in: Inner radius in pixels (0 = a filled disk, no hole).

    Returns:
        numpy.ndarray of dtype int8, shape `shape`, 1 inside the annulus/disk
        and 0 elsewhere.
    """
    cx, cy = center
    mask_out = disk((cy,cx), r_out, shape=shape)
    mask_arr = np.zeros(shape, dtype=np.int8)
    mask_arr[mask_out] = 1
    if r_in > 0:
        mask_in = disk((cy,cx), r_in, shape=shape)
        mask_arr[mask_in] = 0
    return mask_arr

def find_dp_center(dp, r_mask=50, det_shape=(512,512), plot=False):
    """Estimate the direct-beam center of a diffraction pattern.

    Finds the brightest pixel as a first guess, then refines it via the
    center of mass within a circular mask of radius `r_mask` around that
    guess (more robust to hot pixels / noise than the brightest-pixel alone).

    Args:
        dp: 2-D diffraction pattern (e.g. a summed DP from `get_dp`/`get_sum_dp`).
        r_mask: Radius (pixels) of the circular mask used for the center-of-mass refinement.
        det_shape: Shape of `dp` (used to build the temporary mask).
        plot: If True, show a debug figure comparing the masked pattern and the found center.

    Returns:
        (y, x) center coordinates (row, column), matching `dp`'s own
        row/column axis order.
    """
    # max value as center of the mask
    idx = np.argmax(dp)# index in flattened array
    cx, cy = np.unravel_index(idx, dp.shape)
    mask_temp = create_virtual_detector(det_shape, (cy,cx), r_mask)

    # center of mass
    cx, cy = center_of_mass(dp*mask_temp)

    if plot:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(dp*mask_temp, norm=mcolors.SymLogNorm(linthresh=1))
        cx, cy = center_of_mass(dp * mask_temp)
        ax[0].set_title('Masked Pattern for\nFinding the Center')

        ax[1].imshow(dp, norm=mcolors.SymLogNorm(linthresh=1))
        ax[1].scatter(cy, cx, color='r')
        ax[1].set_title('Found Center')
    return (cx,cy)

def find_dp_center_blurred(dp, sigma=15):
    """Estimate the direct-beam center of a diffraction pattern by heavily
    blurring it first, via HyperSpy's `Signal2D.map`, then taking the
    location of the blurred maximum.

    A large-sigma Gaussian blur washes out single hot/dead pixels and other
    per-pixel noise (common on electron-counting detectors) while leaving
    the (much larger) direct-beam spot as the dominant broad maximum - more
    robust than searching the raw pattern directly, and cheap enough to
    re-run on every redraw for automatic re-centering.

    Args:
        dp: 2-D diffraction pattern.
        sigma: Gaussian blur standard deviation, in pixels. Deliberately
            large (default 15) so the blur genuinely suppresses single-pixel
            outliers rather than just smoothing them.

    Returns:
        (x, y) center in pixel coordinates.
    """
    s = hs.signals.Signal2D(np.asarray(dp, dtype=float))
    s.map(gaussian_filter, sigma=sigma, show_progressbar=False)
    y, x = np.unravel_index(np.argmax(s.data), s.data.shape)
    return (float(x), float(y))

def get_sum_dp(fn, dtype=None, scanSize=None, dwellTime=1, roi=None, logger=None,
               n_threads=None, fn_pattern=None):
    """Return a summed diffraction pattern (position-averaged over the
    scan), summed either over the whole scan (roi=None) or over a
    scan-space rectangle (roi=(x, y, w, h)) - used both as the reference
    image for `find_dp_center`/`find_dp_center_blurred` and for the
    "Summed DP from ROI" feature.

    For .tpx3 this uses eventem's fast Roi (when roi is given) or Pacbed
    (whole-scan) algorithm via `get_dp`, without materialising the full 4D
    array. For other formats the signal is loaded lazily and summed over
    its navigation axes via dask.

    Args:
        fn: Path to the 4D-STEM file.
        dtype: File extension (e.g. `.hdf5`, `.tpx3`, `.hspy`). Inferred from `fn` if None.
        scanSize: (nx, ny) scan dimensions, if required by the format.
        dwellTime: Dwell time in microseconds; only used for `.tpx3` files.
        roi: Optional (x, y, w, h) scan-space crop; None sums the whole scan.
        logger: Optional logger to report progress through (dask compute, or
            eventem's own console progress for `.tpx3`).
        n_threads: Optional override for eventem's own internal thread pool
            (`.tpx3` only) - see `load_tpx3`'s docstring for why this
            matters for concurrent callers.
        fn_pattern: Optional path to a smart-scan pattern file (see
            `loaders.load_tpx3`/`loaders._load_mib_smart_scan`) - `.tpx3`
            and `.mib` only.

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    if dtype is None:
        dtype = os.path.splitext(fn)[-1]
    if dtype == '.tpx3':
        return get_dp(fn, scanSize, fn_pattern=fn_pattern, roi=roi, dwellTime=dwellTime,
                      logger=logger, n_threads=n_threads)

    s = load_signal(fn, dtype=dtype, scanSize=scanSize, roi=roi, lazy=True, logger=logger,
                    fn_pattern=fn_pattern)
    f = None
    if isinstance(s, tuple):  # lazy .hdf5 load returns (signal, open file handle)
        s, f = s
    try:
        sum_dp = s.data.sum(axis=(0, 1))
        if hasattr(sum_dp, 'compute'):
            with LoggingProgressBar(logger, 'Summed DP'):
                sum_dp = sum_dp.compute()
    finally:
        # Without this, a failed .compute() (or anything else that raises
        # here) leaves the HDF5 file handle open - on Windows that can then
        # make every subsequent attempt on the same file fail too, since the
        # file is still exclusively locked by this now-abandoned handle.
        if f is not None:
            f.close()
    return np.asarray(sum_dp)

def calculate_nav_img_masked(fn, dtype=None, scanSize=None, dwellTime=1,
                             r_in=0, r_out=None, center=None, logger=None,
                             n_threads=None, fn_pattern=None):
    """Compute a navigation image by summing each diffraction pattern's
    intensity within an annular virtual-detector mask (inner/outer radius
    around a user-defined center), instead of the whole detector.

    For .tpx3 this uses eventem's vSTEM algorithm directly (no full 4D array
    materialised). For other formats the full 4D signal is loaded lazily and
    the mask applied and summed over its detector axes via dask.

    Args:
        fn: Path to the 4D-STEM file.
        dtype: File extension. Inferred from `fn` if None.
        scanSize: (nx, ny) scan dimensions, if required by the format.
        dwellTime: Dwell time in microseconds; only used for `.tpx3` files.
        r_in: Inner radius of the virtual detector, in detector pixels.
        r_out: Outer radius of the virtual detector, in detector pixels.
            None defaults to half the (smaller) detector dimension.
        center: (x, y) center of the virtual detector, in detector pixels.
            None defaults to the detector's geometric center.
        logger: Optional logger to report progress through (dask compute, or
            eventem's own console progress for `.tpx3`).
        n_threads: Optional override for eventem's own internal thread pool
            (`.tpx3` only) - see `load_tpx3`'s docstring for why this
            matters for concurrent callers.

    Returns:
        numpy.ndarray of shape (ny, nx).
    """
    if dtype is None:
        dtype = os.path.splitext(fn)[-1]
    if dtype == '.tpx3':
        if center is not None:
            center = (center[1], center[0])
            return calculate_nav_img_tpx3(fn, scanSize, dwellTime,
                                          r_in=r_in, r_out=r_out or 512, offset=center,
                                          fn_pattern=fn_pattern,
                                          logger=logger, n_threads=n_threads)

    s = load_signal(fn, dtype=dtype, scanSize=scanSize, lazy=True, logger=logger,
                    fn_pattern=fn_pattern)
    f = None
    if isinstance(s, tuple):  # lazy .hdf5 load returns (signal, open file handle)
        s, f = s
    try:
        det_shape = s.data.shape[-2:]
        if center is None:
            center = (det_shape[1] / 2, det_shape[0] / 2)
        if r_out is None:
            r_out = min(det_shape) / 2
        mask = create_virtual_detector(det_shape, center, r_out, r_in)
        nav_img = (s.data * mask).sum(axis=(-2, -1))
        if hasattr(nav_img, 'compute'):
            with LoggingProgressBar(logger, 'Masked navigation image'):
                nav_img = nav_img.compute()
    finally:
        # Without this, a failure partway through leaves the HDF5 file
        # handle open - on Windows that can then make every subsequent
        # attempt on the same file fail too, since it's still locked.
        if f is not None:
            f.close()
    return np.asarray(nav_img)

def calculate_nav_img_hdf5(fn, scanSize=None, logger=None):
    """Return a navigation image from an .hdf5 file.

    Uses the pre-computed `dose_image` dataset if present; otherwise sums all
    diffraction patterns via `load_hdf5(..., sum_dp=True)`.

    Args:
        fn: Path to the .hdf5 file.
        scanSize: (nx, ny) scan dimensions. Required when `dose_image` is absent and
                  the 4D array is stored flat.

    Returns:
        numpy.ndarray of shape (ny, nx).
    """
    with h5py.File(fn, 'r') as f:
        if 'dose_image' in f.keys():
            nav_img = f['dose_image'][:]
            return nav_img
        # print('Navigation images is pre-calculated')
    from .loaders import load_hdf5
    nav_img = load_hdf5(fn, scanSize=scanSize, sum_dp=True, logger=logger)
    return nav_img

def calculate_nav_img(fn, dtype=None, scanSize=None, dwellTime=1, logger=None,
                      n_threads=None, fn_pattern=None):
    """Dispatch navigation image computation to the format-specific function.

    Args:
        fn: Path to the 4D-STEM file.
        dtype: File extension (e.g. `.hdf5`, `.tpx3`, `.hspy`). Inferred from `fn` if None.
        scanSize: (nx, ny) scan dimensions. Required for formats that don't store it internally.
        dwellTime: Dwell time in microseconds; only used for `.tpx3` files.
        logger: Optional logger to report dask compute progress through.
        n_threads: Optional override for eventem's own internal thread pool
            (`.tpx3` only) - see `load_tpx3`'s docstring for why this
            matters for concurrent callers.

    Returns:
        numpy.ndarray of shape (ny, nx) representing the navigation image.
    """
    if dtype is None:
        dtype = os.path.splitext(fn)[-1]

    if dtype == '.mib':
        nav_img = load_signal(fn, dtype=dtype, scanSize=scanSize, sum_dp=True, logger=logger,
                              fn_pattern=fn_pattern)
    elif dtype == '.tpx3':
        nav_img = calculate_nav_img_tpx3(fn, scanSize, dwellTime, fn_pattern=fn_pattern,
                                         logger=logger, n_threads=n_threads)
    elif dtype == '.hdf5':
        nav_img = calculate_nav_img_hdf5(fn, scanSize, logger=logger)
    elif dtype in ['.zspy', '.hspy']:
        nav_img = load_signal(fn, dtype=dtype, scanSize=scanSize, sum_dp=True, logger=logger)
    elif dtype in ['.dm2', '.dm3', '.tif', '.tiff']:
        nav_img = hs.load(fn).data
    return nav_img
