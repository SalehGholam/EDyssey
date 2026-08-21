# -*- coding: utf-8 -*-
"""4D-STEM file format loaders (.tpx3/.hdf5/.hspy/.zspy/.mib) and their
scan/detector-size header readers.
"""
import numpy as np
import os
from glob import glob
import hyperspy.api as hs
import dask.array as da
import h5py
import dask
import eventem
from scipy.ndimage import gaussian_filter

from .progress import redirect_console_to_logger, LoggingProgressBar, _log_or_print

def get_4d_files(path_4d_datasets, dtype):
    """Return sorted list of 4D-STEM files matching *dtype* in *path_4d_datasets* (recursive)."""
    fns_4d = glob(os.path.join(path_4d_datasets, f'*.{dtype}'))
    if len(fns_4d) == 0: # search in sub-directories
        fns_4d = glob(os.path.join(path_4d_datasets, '**', f'*.{dtype}'), recursive=True)
    assert len(fns_4d) != 0, "4D Datasets with correct format are not found!"
    return fns_4d

def load_signal(fn, **kwargs):
    """Dispatch-load a 4D-STEM file to the appropriate loader based on file extension."""
    dtype = kwargs.get('dtype')
    if dtype is None:
        dtype = os.path.splitext(fn)[1]
    if dtype == '.tpx3':
        result = load_tpx3(fn, **kwargs)
    elif dtype == '.hdf5':
        result = load_hdf5(fn, **kwargs)
    elif dtype in ['.zspy', '.hspy', '.mib']:
        result = load_hs(fn, **kwargs)
    else:
        _log_or_print(kwargs.get('logger'), f'The data type {dtype} is not implemented for the analysis!')
        return None
    return result

def load_tpx3(fn, roi=None, scanSize=(512,512), dwellTime=1, bitDepth=16,
              repetitions=1, logger=None, n_threads=None, fn_pattern=None,
              get_4d=False, mask=None, det_shape=(512, 512), **kwargs):
    """Load a .tpx3 4D-STEM file via eventem; returns a HyperSpy Signal2D or summed nav image.

    `det_shape`: (det_x, det_y) detector pixel dimensions. Only actually
    applied (via `.detector_size_x`/`.detector_size_y`) when it differs from
    what eventem itself already reports right after `set_file()` (which
    reflects the real file's own hardware layout) - eventem does NOT gracefully
    reshape/crop when told a detector size that doesn't match the real data:
    it segfaults the whole process on `.run()`. Only override this for a
    genuinely different real detector configuration, never as a guess.

    `logger`: optional logger the eventem progress output (printed straight
    to the console by the compiled extension) is redirected into, so it
    shows up in the Qt log console instead of only the terminal.

    `n_threads`: optional override for eventem's own internal thread pool
    (auto-sized to the whole machine by default, per instance). Left at
    eventem's default (None) for a single in-process call, but should be
    set to a small number (e.g. 1) by a caller that's already one of
    several *concurrent processes* doing the same thing - otherwise each
    process additionally claims the whole machine's threads for itself,
    causing severe oversubscription once more than one runs at a time.

    `fn_pattern`: optional path to a smart-scan pattern file (a text file of
    flat scan-pixel indices, one per acquired frame, in acquisition order) -
    required to correctly reshape a smart-scanned (sparsely acquired)
    acquisition into its full (ny, nx, det, det) grid. None (default) treats
    the file as a normal dense raster, matching prior behaviour.
    """
    # Deliberately not named `roi` here - that name is this function's own
    # `roi=(x, y, w, h)` crop parameter above, and reusing it for the
    # eventem object clobbered that parameter before it could ever be read
    # (every call ended up treating the crop as "full frame", the `roi is
    # None` branch, since `roi` was always this object, never None, by the
    # time it was checked).
    roi_obj = eventem.Roi(repetitions=repetitions, extract_4D=get_4d)
    if n_threads is not None:
        roi_obj.n_threads = n_threads
    roi_obj.set_bitdepth(bitDepth)
    roi_obj.nx = scanSize[0]
    roi_obj.ny = scanSize[1]
    roi_obj.set_file(fn)
    if det_shape != (roi_obj.detector_size_x, roi_obj.detector_size_y):
        roi_obj.detector_size_x, roi_obj.detector_size_y = det_shape
    if fn_pattern is not None:
        roi_obj.set_pattern_file(fn_pattern)

    if mask is None:
        if roi is None:
            x, y, w, h = 0, 0, scanSize[0], scanSize[1]
        else:
            x, y, w, h = roi
        roi_obj.set_roi(x=x, y=y, width=w, height=h)
    else:
        roi_obj.set_roi_mask([mask.flatten()])

    roi_obj.set_dwell_time(dwellTime*1000)
    with redirect_console_to_logger(logger, 'Loading tpx3'):
        roi_obj.run()
    # s = np.asarray(roi_obj.get_4D())
    # s = hs.signals.Signal2D(s)
    return roi_obj

def load_hdf5(fn, roi=None, scanSize=None, lazy=False, max_eager_frames=10000,
              logger=None, **kwargs): #TODO change to normal load
    """Load a .hdf5 4D-STEM file; supports lazy loading, ROI crop, and DP summation.

    Either path (lazy=True or lazy=False) ends up returning a fully
    materialized numpy array - the difference is HOW the read happens.
    lazy=False (the default) reads the ROI directly off the h5py dataset in
    one shot: fast, but its peak memory use scales with the whole requested
    ROI at once (loads every diffraction pattern in it into RAM
    simultaneously). lazy=True instead reads it through dask (chunked):
    noticeably slower, but far more memory-considerate.

    `max_eager_frames`: safety cap on the fast (lazy=False) path - above
    this many diffraction patterns (the ROI's width*height, or scanSize's
    if roi is None), loading automatically falls back to the dask path
    regardless of `lazy`, so a large ROI can't silently blow up RAM just
    because lazy=False (the default) was left unset. Only ever overrides an
    *unset/default* lazy=False - an explicit lazy=True request is always
    honored as-is.
    """
    with h5py.File(fn, 'r') as f:
        if roi is None:
            roi = [0, 0, scanSize[1], scanSize[0]]
        x, y, w, h = roi  # roi format: [x, y, w, h] — x=col, y=row
        if lazy or w * h > max_eager_frames:
            # Must be .compute()d before this `with` block exits, same as
            # get_dp()'s '.hdf5' branch (see its comment) - a dask array
            # built from a dataset inside a `with h5py.File(...)` block
            # stops being readable the moment that block exits, so it can
            # never actually be handed back to the caller still lazy.
            s = da.from_array(f['4D'])
            s = s[y:y+h, x:x+w]
            s = s.compute()
        else:
            s = f['4D'][y:y+h, x:x+w]  # numpy row-major: row (y) axis first
    return s

def load_hs(fn, roi=None, lazy=True, logger=None,
           fn_pattern=None, scanSize=None, **kwargs):
    """Load a .hspy/.zspy HyperSpy signal; supports lazy loading, ROI crop, and DP summation.

    Args:
        fn_pattern: Optional path to a smart-scan pattern file - treats `fn`
            as a flat, un-reshaped stream of acquired frames (the same
            situation as a smart-scanned .mib - both are loaded via
            `hs.load` identically, see `_reconstruct_smart_scan`) rather
            than an already-dense 4D signal. Requires `scanSize`. None
            (default) loads `fn` as an already-dense signal, matching prior
            behaviour.
        scanSize: (nx, ny) scan dimensions - only used when `fn_pattern` is given.
    """
    if fn_pattern:
        s = _reconstruct_smart_scan(fn, fn_pattern, scanSize, logger=logger)
    else:
        if scanSize is None and os.path.splitext(fn)[1] == '.mib': # Get scan size for mib
            fn_hdr = _resolve_mib_hdr(fn)
            if fn_hdr is not None:
                scanSize = get_scan_size_mib_hdr(fn_hdr)
            else:
                _log_or_print(logger, f'Failed to find a .hdr file for {fn}')

        # s = hs.load(fn, lazy=True, navigation_shape=scanSize)
        s = hs.load(fn, lazy=True)

    if roi is not None:
        x,y,w,h = roi
        s = s.inav[x:x+w, y:y+h]

    if not lazy:
        with LoggingProgressBar(logger, 'Loading 4D signal'):
            s.compute()
    return s.data

def _reconstruct_smart_scan(fn, fn_pattern, scanSize, logger=None):
    """Reconstruct a smart-scanned acquisition into its full dense
    (ny, nx, det, det) grid. Used for .mib/.hspy/.zspy - not for .tpx3
    """
    if scanSize is None:
        _log_or_print(logger, "scanSize is required to reconstruct a smart-scanned acquisition")
        raise ValueError(
            "scanSize is required to reconstruct a smart-scanned acquisition")

    pattern = np.loadtxt(fn_pattern).astype('uint64')
    s_flat = hs.load(fn, lazy=True)  # (n_frames, det, det), acquisition order
    det_shape = s_flat.data.shape[-2:]
    dask_arr = da.zeros((scanSize[0] * scanSize[1], *det_shape),
                        dtype=s_flat.data.dtype)
    dask_arr[pattern] = s_flat.data
    dask_arr = dask_arr.reshape(scanSize[1], scanSize[0], *det_shape)
    return dask_arr
#%% get scan size
def get_scan_size_mib_hdr(fn_hdr):
    """Parse scan dimensions from a Merlin .hdr header file.

    Args:
        fn_hdr: Path to the `.hdr` file (typically `default.hdr` in the .mib folder).

    Returns:
        Tuple (nx, ny) where nx = frames per trigger and ny = total frames / nx.
    """
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

def _resolve_mib_hdr(fn):
    """Resolve the .hdr header path for a .mib file.

    Prefers a per-file header matching this exact .mib's own basename (e.g.
    "default_0119_-50,00.mib" -> "...-50,00.hdr" - the convention a
    smart-scan acquisition writes, one .hdr per .mib), falling back to a
    shared "default.hdr" in the same folder (the convention a plain
    full-scan acquisition writes once, for every .mib in the folder).

    Returns:
        str, or None if neither convention matches an existing file.
    """
    fn_hdr = os.path.splitext(fn)[0] + '.hdr'
    if os.path.isfile(fn_hdr):
        return fn_hdr
    fn_hdr = os.path.join(os.path.dirname(fn), 'default.hdr')
    if os.path.isfile(fn_hdr):
        return fn_hdr
    return None

def get_scan_size(fn, dtype):
    if dtype in ['.hspy', '.zspy']:
        scanSize = hs.load(fn, lazy=True).shape[:2]
    elif dtype == '.mib':
        fn_hdr = _resolve_mib_hdr(fn)
        if fn_hdr is None:
            raise FileNotFoundError(
                f"No .hdr file found for {fn!r} (tried a per-file .hdr and a "
                "shared default.hdr in the same folder)")
        scanSize = get_scan_size_mib_hdr(fn_hdr)
    elif dtype == '.hdf5':
        with h5py.File(fn, 'r') as f:
            # f['shape'] is written as [nx, ny, det_x, det_y] - already in
            # this app's own scanSize=(nx, ny) convention (see
            # worker_nav_img._read_scansize_hdf5, whose docstring/return
            # confirms the same first-two-elements convention, and the
            # original pre-redesign load_hdf5, which compared this directly
            # against a caller-supplied (nx, ny) scanSize with no reversal).
            scanSize = tuple(int(x) for x in f['shape'][:2])
    return scanSize

def get_det_size(fn, dtype=None):
    """Return (det_x, det_y) detector pixel dimensions, read cheaply (a lazy
    load / header peek, no full read) from the file.

    Not supported for .tpx3 - detector size isn't stored anywhere in a
    .tpx3 file itself; callers fall back to a manual "Detector Size"
    setting for that format instead (see e.g. tab_sam2.get_detector_shape).
    """
    if dtype is None:
        dtype = os.path.splitext(fn)[1]
    if dtype == '.hdf5':
        with h5py.File(fn, 'r') as f:
            # f['shape'][-2:] == (det_x, det_y) - same [nx, ny, det_x, det_y]
            # convention as get_scan_size above, no axis swap needed here.
            det_shape = tuple(int(x) for x in f['shape'][-2:])
    else:
        s = load_hs(fn, lazy=True)
        # A dense array's own axis order is (..., det_y, det_x) - the last
        # two axes are swapped here to match this function's (det_x, det_y)
        # return contract (mirrors the removed pre-redesign get_det_size).
        det_shape = (s.shape[-1], s.shape[-2])
    return det_shape
#%% dp related
def get_dp(fn, dtype=None, roi=None, scanSize=None, fn_pattern=None,
           logger=None, mask=None, dwellTime=1, det_shape=(512, 512)):
    if dtype is None:
        dtype = os.path.splitext(fn)[1]
    if dtype not in ('.tpx3', '.hspy', '.zspy', '.mib', '.hdf5'):
        raise ValueError(f"Unsupported file type '{dtype}' for get_dp() - "
                          "expected one of .tpx3/.hspy/.zspy/.mib/.hdf5.")
    if scanSize is None:
        scanSize = get_scan_size(fn, dtype)
    if dtype == '.tpx3':
        if mask is None and roi is None:
            dp = get_dp_tpx3_full(fn, scanSize=scanSize, fn_pattern=fn_pattern,
                                  det_shape=det_shape)
        elif mask is None:
            dp = load_tpx3(fn, roi=roi, scanSize=scanSize, dwellTime=dwellTime,
                           fn_pattern=fn_pattern, logger=logger, get_4d=False,
                           det_shape=det_shape)
            dp = np.array(dp.Roi_diffraction_pattern).reshape(det_shape[1], det_shape[0])
        else:
            if mask.shape != scanSize:
                mask_temp = np.zeros(scanSize, dtype='uint8')
                x,y,w,h = roi
                mask_temp[y:y+h, x:x+w] = mask  # row-major: y (row) axis first
                mask = mask_temp
            dp = load_tpx3(fn, roi=None, scanSize=scanSize, dwellTime=dwellTime,
                           fn_pattern=fn_pattern, logger=logger, get_4d=False,
                           mask=mask, det_shape=det_shape)
            dp = np.array(dp.Roi_diffraction_pattern).reshape(det_shape[1], det_shape[0])
    
    if dtype in ['.hspy', '.zspy', '.mib']:
        s = load_hs(fn, roi=roi, logger=logger, fn_pattern=fn_pattern, 
                    scanSize=scanSize)
        dp = s.sum(axis=(0,1))
        with LoggingProgressBar(logger, 'Loading 4D signal'):
            dp = dp.compute()
    
    if dtype == '.hdf5':
        # Built and computed directly here (not via load_hdf5(lazy=True)) -
        # a dask array built from a dataset inside a `with h5py.File(...)`
        # block stops being readable the moment that block exits, so a lazy
        # array can only safely be handed back to a caller already computed,
        # same as calculate_nav_img_hdf5 does.
        with h5py.File(fn, 'r') as f:
            s = da.from_array(f['4D'], chunks=f['4D'].chunks)
            if roi is not None:
                x, y, w, h = roi
                s = s[y:y+h, x:x+w]
            dp = s.sum(axis=(0,1))
            with LoggingProgressBar(logger, 'Loading 4D signal'):
                dp = dp.compute()
    return dp

def get_dp_tpx3_full(fn_tpx3, scanSize, dwellTime=1, fn_pattern=None,
                     repititions=1, logger=None, det_shape=(512, 512)):
    dp = eventem.Pacbed(repetitions=repititions)
    dp.set_file(fn_tpx3)
    dp.nx = scanSize[0]
    dp.ny = scanSize[1]
    # Only actually applied when it differs from what eventem itself already
    # reports (reflecting the real file's own hardware layout) - see
    # load_tpx3's docstring for why a genuine mismatch segfaults .run().
    if det_shape != (dp.detector_size_x, dp.detector_size_y):
        dp.detector_size_x, dp.detector_size_y = det_shape
    dp.set_dwell_time(dwellTime*1000)
    if fn_pattern is not None:
        dp.set_pattern_file(fn_pattern)

    with redirect_console_to_logger(logger, 'Loading tpx3'):
        dp.run()
    dp = np.array(dp.Pacbed_image).reshape(det_shape[1], det_shape[0])
    return dp

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