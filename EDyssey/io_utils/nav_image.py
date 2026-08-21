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
from dask.diagnostics import ProgressBar
import dask.array as da
from .progress import redirect_console_to_logger, LoggingProgressBar
from .loaders import get_scan_size, get_det_size

# "Cover the whole detector" sentinel for an annular virtual detector's outer
# radius, deliberately far larger than any real detector - used as a default/
# fallback in place of a literal like 512 (which silently under-covers a
# larger real detector, or is simply meaningless clipped noise on a smaller
# one) wherever the caller wants "the full frame" without needing to know the
# actual detector size. eventem clips the annulus to the frame's real extent
# internally, so an oversized radius is fine in general - but eventem's vSTEM
# squares this radius internally using a 32-bit int, so it must stay below
# sqrt(2**31) (~46340) or that squaring silently overflows/wraps and the
# "inside the disk" test becomes always-false, zeroing the whole image (this
# was the actual cause of .tpx3 nav-image "Test File" coming back completely
# empty whenever "Use Virtual Mask" was off - confirmed by bisecting r_out
# against real data: 46340 works, 46341 already returns an all-zero image).
# 1 << 15 keeps a wide safety margin under that limit while still being far
# larger than any real detector.
FULL_DETECTOR_RADIUS = 1 << 15
#%% get navigation image
def create_nav_signal_from_haadf(fns):
    """Stack a list of HAADF/navigation images into a HyperSpy Signal2D.

    Args:
        fns: List of file paths to individual image frames.

    Returns:
        HyperSpy Signal2D with shape (N, H, W).
    """
    s = hs.load(fns, stack=True)
    return s

def calculate_nav_img_tpx3(fn, scanSize, dwellTime=None, r_in=0, r_out=FULL_DETECTOR_RADIUS,
                           offset=None, repetitions=1, fn_pattern=None, logger=None,
                           n_threads=None, det_shape=(512, 512)):
    """Compute a virtual STEM (vSTEM) navigation image from a .tpx3 file using
    one or several annular detectors.

    Args:
        fn: Path to the .tpx3 file.
        scanSize: (nx, ny) scan dimensions in pixels.
        dwellTime: Dwell time per pixel in microseconds (multiplied by 1000 internally).
        r_in: Inner radius of the annular detector in pixels, or a list of
            inner radii for several virtual detectors at once (paired by
            position with r_out/offset).
        r_out: Outer radius, or a matching list of outer radii.
        offset: (x, y) absolute center of the virtual detector, in detector
            pixels, or a list of centers matching r_in/r_out. None = a
            single detector covering the whole frame. 
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
    vstem = eventem.vSTEM(repetitions)
    if n_threads is not None:
        vstem.n_threads = n_threads
    vstem.b_cumulative = True
    vstem.set_file(fn)
    vstem.nx = scanSize[0]
    vstem.ny = scanSize[1]
    # Only actually applied when it differs from what eventem itself already
    # reports (reflecting the real file's own hardware layout) - a genuine
    # mismatch segfaults .run() instead of gracefully reshaping/cropping.
    if det_shape != (vstem.detector_size_x, vstem.detector_size_y):
        vstem.detector_size_x, vstem.detector_size_y = det_shape
    vstem.inner_radia = list(r_in) if isinstance(r_in, (list, tuple)) else [r_in]
    vstem.outer_radia = list(r_out) if isinstance(r_out, (list, tuple)) else [r_out]
    vstem.set_dwell_time(dwellTime*1000)
    if fn_pattern:
        vstem.set_pattern_file(fn_pattern)
    if offset:
        offsets = offset if isinstance(offset[0], (list, tuple)) else [offset]
        # eventem's set_offsets takes (x, y) pairs directly, same order as
        # this function's own `offset` docstring - no axis flip. Confirmed
        # wrong against real hardware (the virtual detector's actual center
        # was swapped relative to the requested Center X/Y) - previously
        # flipped to [y, x] here, inherited unverified from an earlier
        # rewrite (see git history), never actually confirmed correct.
        vstem.set_offsets([[o[0], o[1]] for o in offsets])
    with redirect_console_to_logger(logger, 'Loading tpx3'):
        vstem.run()
    nav_image = vstem.get_image()
    return nav_image

def calculate_nav_img_variance_tpx3(fn, scanSize, dwellTime=None, r_in=0, r_out=FULL_DETECTOR_RADIUS,
                                    offset=None, repetitions=1, fn_pattern=None, logger=None,
                                    n_threads=None, det_shape=(512, 512)):
    """Compute a per-scan-position variance image from a .tpx3 file using
    eventem's Var processor - the variance-mode counterpart of
    calculate_nav_img_tpx3's vSTEM-based sum.

    Unlike vSTEM, eventem.Var only supports a single annular (or full-disk)
    region per run - its inner_radius/outer_radius/offset are scalars/one
    (x, y) pair, not the lists vSTEM's inner_radia/outer_radia/set_offsets
    take for combining several virtual detectors into one pass. That's not
    just an API gap: variance over the union of several disjoint regions
    isn't the sum of their individual variances the way a vSTEM signal is,
    so there's no single well-defined "combined" result to compute here
    even in principle - callers combining several detectors for variance
    mode need to call this once per detector instead.

    Args mirror calculate_nav_img_tpx3 (see its docstring), except r_in/
    r_out/offset here are a single scalar/scalar/(x, y) pair rather than
    lists.

    Returns:
        numpy.ndarray of shape (ny, nx) with float values.
    """
    var = eventem.Var(repetitions)
    if n_threads is not None:
        var.n_threads = n_threads
    var.b_cumulative = True
    var.set_file(fn)
    var.nx = scanSize[0]
    var.ny = scanSize[1]
    # See calculate_nav_img_tpx3's identical guard - only applied when it
    # differs from what eventem itself already reports, since a genuine
    # mismatch segfaults .run() instead of gracefully reshaping/cropping.
    if det_shape != (var.detector_size_x, var.detector_size_y):
        var.detector_size_x, var.detector_size_y = det_shape
    var.inner_radius = r_in
    var.outer_radius = r_out
    var.set_dwell_time(dwellTime*1000)
    if fn_pattern:
        var.set_pattern_file(fn_pattern)
    if offset:
        # (x, y) passed through as-is - see calculate_nav_img_tpx3's
        # identical fix; confirmed against real hardware that eventem wants
        # (x, y) directly here too, not a [y, x] flip.
        var.set_offset([offset[0], offset[1]])
    with redirect_console_to_logger(logger, 'Loading tpx3'):
        var.run()
    # Unlike vSTEM.get_image() (a wrapper method - already returns a
    # properly-shaped (ny, nx) array), Var_image is a raw property with no
    # such wrapper and comes back flat (confirmed against real hardware
    # data: a (512, 512) scan returned shape (262144,), i.e. nx*ny with no
    # reshape applied internally) - reshaped here to match every other
    # nav-image function's documented (ny, nx) return shape.
    var_image = np.asarray(var.Var_image)
    if var_image.ndim == 1:
        var_image = var_image.reshape(scanSize[1], scanSize[0])
    return var_image

def calculate_nav_img_hdf5(fn, scanSize, det_mask=None, logger=None, fn_pattern=None, mode='sum'):
    """Return a navigation image from an .hdf5 file.

    Uses the pre-computed `dose_image` dataset if present; otherwise sums all
    diffraction patterns directly (optionally through `det_mask`).

    Args:
        fn: Path to the .hdf5 file.
        scanSize: (nx, ny) scan dimensions. Required when `dose_image` is absent and
                  the 4D array is stored flat.
        mode: 'sum' (default) - per-scan-position sum of frame intensities
            (or masked-region intensities, when `det_mask` is given). 'variance'
            computes the per-scan-position variance instead - a virtual image that
            highlights local structural variation (e.g. amorphous vs. crystalline
            regions) rather than total scattered dose. The precomputed `dose_image`
            dataset is always a sum, so it's only used when mode == 'sum'.

    Returns:
        numpy.ndarray of shape (ny, nx).
    """
    if mode not in ('sum', 'variance'):
        raise ValueError(f"mode must be 'sum' or 'variance', got {mode!r}")
    with h5py.File(fn, 'r') as f:
        s = da.from_array(f['4D'], chunks=f['4D'].chunks)
        if (det_mask is None):
            if mode == 'sum' and 'dose_image' in f.keys() and fn_pattern is None:
                nav_img = f['dose_image'][:]
                return nav_img
            elif fn_pattern is None:
                nav_img = s.sum(axis=(-1,-2)) if mode == 'sum' else s.var(axis=(-1,-2))
        else:
            det_mask = det_mask.ravel().astype(bool)
            s = s.reshape(*scanSize,-1)
            if mode == 'sum':
                nav_img = (s*det_mask).sum(axis=-1)
            else:
                # Variance over only the masked pixels - multiplying by the
                # mask first (as the sum path does) would pull in a majority
                # of zeroed-out pixels from outside the mask and bias the
                # variance low, so the masked-out pixels are dropped via
                # boolean indexing instead of zeroed.
                nav_img = s[..., det_mask].var(axis=-1)

        if fn_pattern is not None:
            raise NotImplementedError('Smart Scanned is not yet implemented for converted data from tpx3!')

        with LoggingProgressBar(logger, 'Loading 4D signal'):
            nav_img = nav_img.compute()
        return nav_img

def calculate_nav_img_hs(fn, scanSize, det_mask=None, fn_pattern=None, logger=None, mode='sum'):
    if mode not in ('sum', 'variance'):
        raise ValueError(f"mode must be 'sum' or 'variance', got {mode!r}")
    s = hs.load(fn, lazy=True)
    if (det_mask is None):
        nav_img = (s.sum(axis=(-1,-2)) if mode == 'sum' else s.var(axis=(-1,-2))).data
    else:
        det_mask = det_mask.ravel().astype(bool)
        arr = s.data.reshape(*scanSize,-1)
        if mode == 'sum':
            nav_img = (arr*det_mask).sum(axis=-1)
        else:
            # See calculate_nav_img_hdf5's masked variance branch - only the
            # masked pixels themselves should enter the variance.
            nav_img = arr[..., det_mask].var(axis=-1)

    if fn_pattern is not None:
        pattern = np.loadtxt(fn_pattern).astype('int')
        nav_img_2d = da.zeros(shape=np.prod(scanSize), dtype='uint32')
        nav_img_2d[pattern] = nav_img
        with ProgressBar():
            nav_img_2d = nav_img_2d.compute()
        return nav_img_2d.reshape(scanSize)
    else:
        if len(nav_img.shape) == 1: # mib files might be loaded as 1D
            nav_img = nav_img.reshape(*scanSize)
        with ProgressBar():
            nav_img = nav_img.compute()
        return nav_img

def calculate_nav_img(fn, dtype=None, scanSize=None, dwellTime=1, logger=None,
                      n_threads=None, fn_pattern=None, det_shape=(512, 512), mode='sum'):
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
        det_shape: (det_x, det_y) detector pixel dimensions - `.tpx3` only
            (every other format reads its own detector shape from the file).
        mode: 'sum' (default) or 'variance' - see calculate_nav_img_hdf5's
            docstring. `.tpx3`'s variance path runs eventem's Var processor
            (see calculate_nav_img_variance_tpx3) over the whole detector.
            `.dm2`/`.dm3`/`.tif`/`.tiff` are already single 2D images with
            no per-position frame to take a variance over - raises
            ValueError for mode='variance' there.

    Returns:
        numpy.ndarray of shape (ny, nx) representing the navigation image.
    """
    if mode not in ('sum', 'variance'):
        raise ValueError(f"mode must be 'sum' or 'variance', got {mode!r}")
    if dtype is None:
        dtype = os.path.splitext(fn)[-1]
    if scanSize is None:
        scanSize = get_scan_size(fn, dtype)

    if dtype in ['.zspy', '.hspy', '.mib']:
        nav_img = calculate_nav_img_hs(fn, scanSize, fn_pattern=fn_pattern, logger=logger, mode=mode)
    elif dtype == '.tpx3':
        if mode == 'variance':
            nav_img = calculate_nav_img_variance_tpx3(fn, scanSize, dwellTime, fn_pattern=fn_pattern,
                                                       logger=logger, n_threads=n_threads,
                                                       det_shape=det_shape)
        else:
            nav_img = calculate_nav_img_tpx3(fn, scanSize, dwellTime, fn_pattern=fn_pattern,
                                             logger=logger, n_threads=n_threads,
                                             det_shape=det_shape)
    elif dtype == '.hdf5':
        nav_img = calculate_nav_img_hdf5(fn, scanSize, fn_pattern=fn_pattern, logger=logger, mode=mode)
    elif dtype in ['.dm2', '.dm3', '.tif', '.tiff']:
        if mode == 'variance':
            raise ValueError(
                f"Variance-mode virtual imaging does not apply to {dtype} files - "
                "these are already single 2D images, not a per-position frame stack.")
        nav_img = hs.load(fn).data
    else:
        raise ValueError(f"Unsupported file type '{dtype}' for calculate_nav_img() - "
                          "expected one of .zspy/.hspy/.mib/.tpx3/.hdf5/.dm2/.dm3/.tif/.tiff.")
    return nav_img
#%% virtual mask
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

def create_virtual_detector_multi(shape, detectors):
    """Union of several annular/disk virtual detectors into one mask - the
    "combine several detectors into a single mask" step for every format
    except .tpx3 (which instead sums its several detectors natively inside
    eventem's vSTEM - see calculate_nav_img_tpx3/calculate_nav_img_masked).

    Args:
        shape: (det_y, det_x) shape of the mask/detector.
        detectors: List of {'center': (x, y), 'r_in', 'r_out'} dicts (as
            built by Tab_Create_NavSignal.get_active_detectors()).

    Returns:
        numpy.ndarray of dtype int8, shape `shape`, 1 wherever any detector
        covers that pixel, 0 elsewhere.
    """
    mask = np.zeros(shape, dtype=np.int8)
    for det in detectors:
        mask |= create_virtual_detector(shape, det['center'], det['r_out'], det.get('r_in', 0))
    return mask

def calculate_nav_img_masked(fn, dtype=None, scanSize=None, dwellTime=1, detectors=None,
                             logger=None, n_threads=None, fn_pattern=None,
                             det_shape=(512, 512), mode='sum'):
    """Compute a navigation image through one or several virtual detectors -
    the masked counterpart of `calculate_nav_img`.

    Args:
        fn: Path to the 4D-STEM file.
        dtype: File extension (e.g. `.hdf5`, `.tpx3`, `.hspy`). Inferred from `fn` if None.
        scanSize: (nx, ny) scan dimensions. Auto-detected from the file if None.
        dwellTime: Dwell time in microseconds; only used for `.tpx3` files.
        detectors: List of {'center': (x, y), 'r_in', 'r_out'} dicts, as
            built by Tab_Create_NavSignal.get_active_detectors() - at least
            one entry is required.
        logger: Optional logger to report progress through (dask compute, or
            eventem's own console progress for `.tpx3`).
        n_threads: Optional override for eventem's own internal thread pool
            (`.tpx3` only) - see `loaders.load_tpx3`'s docstring for why this
            matters for concurrent callers.
        fn_pattern: Optional path to a smart-scan pattern file.
        det_shape: (det_x, det_y) detector pixel dimensions - `.tpx3` only
            (every other format reads its own detector shape from the file).
        mode: 'sum' (default) or 'variance' - see calculate_nav_img_hdf5's
            docstring. For `.tpx3` in variance mode, exactly one detector is
            required (see calculate_nav_img_variance_tpx3 - eventem's Var
            processor, unlike vSTEM, has no way to combine several regions
            into one pass); raises ValueError for more than one.

    Returns:
        numpy.ndarray of shape (ny, nx).
    """
    if mode not in ('sum', 'variance'):
        raise ValueError(f"mode must be 'sum' or 'variance', got {mode!r}")
    if dtype is None:
        dtype = os.path.splitext(fn)[-1]
    if scanSize is None:
        scanSize = get_scan_size(fn, dtype)

    if dtype == '.tpx3':
        if mode == 'variance':
            if len(detectors) != 1:
                raise ValueError(
                    f"Variance-mode virtual imaging on .tpx3 files supports exactly one "
                    f"virtual detector at a time (got {len(detectors)}) - eventem's Var "
                    "processor has no equivalent of vSTEM's multi-detector combining, and "
                    "variance over the union of several regions isn't the sum of their "
                    "individual variances anyway. Remove all but one detector, or switch "
                    "Mode back to Sum.")
            det = detectors[0]
            return calculate_nav_img_variance_tpx3(
                fn, scanSize, dwellTime, r_in=det.get('r_in', 0), r_out=det['r_out'],
                offset=det['center'], fn_pattern=fn_pattern, logger=logger,
                n_threads=n_threads, det_shape=det_shape)
        # eventem's vSTEM sums several detectors natively - no mask array
        # needed, just per-detector radius/center lists (see
        # calculate_nav_img_tpx3's docstring for why `offset` here is
        # already the absolute (x, y) center `detectors` uses).
        r_in = [d.get('r_in', 0) for d in detectors]
        r_out = [d['r_out'] for d in detectors]
        offset = [d['center'] for d in detectors]
        return calculate_nav_img_tpx3(fn, scanSize, dwellTime, r_in=r_in, r_out=r_out,
                                      offset=offset, fn_pattern=fn_pattern, logger=logger,
                                      n_threads=n_threads, det_shape=det_shape)

    if dtype in ['.zspy', '.hspy', '.mib', '.hdf5']:
        det_x, det_y = get_det_size(fn, dtype)
        det_mask = create_virtual_detector_multi((det_y, det_x), detectors)
        if dtype == '.hdf5':
            return calculate_nav_img_hdf5(fn, scanSize, det_mask=det_mask,
                                          fn_pattern=fn_pattern, logger=logger, mode=mode)
        return calculate_nav_img_hs(fn, scanSize, det_mask=det_mask,
                                    fn_pattern=fn_pattern, logger=logger, mode=mode)

    raise ValueError(
        f"calculate_nav_img_masked does not support {dtype!r} files - virtual "
        "detector masks only apply to 4D-STEM data.")