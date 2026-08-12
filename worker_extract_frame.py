# -*- coding: utf-8 -*-
"""
Created on Tue May  6 22:47:49 2025

@author: sgholam
"""

import sys
import dask.array as da
import h5py
from dask.distributed import Client, LocalCluster
import os
from dask import config
import numpy as np
from dask.diagnostics import ProgressBar
import base64
import pickle
file_path = os.path.abspath(__file__)
main_path = os.path.dirname(file_path)
eventem_path = os.path.join(main_path, 'py4DTomo', 'io_utils')
sys.path.append(eventem_path)
# os.chdir()
import eventem
from hyperspy.api import signals, load
# from py4DTomo.io_utils import load_signal
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="distributed")
#%%
#TODO add cupy if possible
def extract_3ded_mask_single_frame(fn, roi, mask_path, dtype, scanSize, i_c, fn_pattern=None,
                                   det_shape=None):
    """Extract a single 3DED diffraction pattern from one 4D-STEM frame using a binary mask.

    Subprocess entry point called by `tab_sam2.py`. Deserialises CLI arguments, loads the
    per-frame mask from `mask_path`, sums diffraction patterns at mask-True pixels, and
    prints the result as a base64+pickle-encoded `(dp, i_c)` tuple.

    Args:
        fn: Path to the 4D-STEM file.
        roi: Scan-space crop as a string representation of a numpy array, e.g. `'[x y w h]'`.
        mask_path: Path to a `.npy` file containing the binary 2-D mask.
        dtype: File extension string (e.g. `.tpx3`, `.hdf5`).
        scanSize: Scan dimensions as `'(nx, ny)'` string.
        i_c: Frame index within the extraction batch, as a string.
        fn_pattern: Optional path to a smart-scan pattern file for this frame
            (empty string/None for a normal dense frame) - see `load_tpx3`/`load_mib` below.
        det_shape: Optional `'(det_x, det_y)'` string - `.tpx3` only, see
            `load_tpx3`'s docstring (None/'None' falls back to (512, 512)).
    """
    try:
        # Parse input args
        scanSize = tuple(map(int, scanSize.strip("()").split(",")))
        # roi = tuple(map(int, roi.strip("()").split(",")))
        roi = [int(a) for a in str(roi)[1:-1].split()] # read as str of numpy array
        # roi = eval(roi) # read as str of list
        mask = np.load(mask_path)
        # mask = mask_path
        fn_pattern = fn_pattern or None
        det_shape = (tuple(map(int, det_shape.strip("()").split(",")))
                    if det_shape not in (None, '', 'None') else None)

        dp = load_dp(fn, roi=roi, mask=mask, scanSize=scanSize, fn_pattern=fn_pattern,
                    det_shape=det_shape)
        # Serialize result to base64 string and print it to stdout
        serialized = base64.b64encode(pickle.dumps((dp, i_c))).decode('utf-8')
        print(serialized)  # <- this goes to QProcess output
        
        sys.stdout.flush()  # Ensure it's pushed
        sys.exit(0)
        
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)  # fail code
        
def load_dp(fn, **kwargs):
    """Dispatch diffraction pattern loading to the format-specific function.

    Args:
        fn: Path to the 4D-STEM file.
        **kwargs: Passed through to the format-specific loader (roi, mask, scanSize, etc.).

    Returns:
        numpy.ndarray of shape (det_y, det_x) with the summed diffraction pattern.
    """
    dtype = kwargs.get('dtype')
    if dtype is None:
        dtype = os.path.splitext(fn)[1]
    
    if dtype == '.tpx3':
        result = load_tpx3(fn, **kwargs)        
    elif dtype == '.hdf5':
        result = load_hdf5(fn, **kwargs)        
    elif dtype in ['.zspy', '.hspy']:
        result = load_hs(fn, **kwargs)
    elif dtype == '.mib':
        result = load_mib(fn, **kwargs)
    return result

def load_tpx3(fn, mask, scanSize, roi, dwellTime=1, fn_pattern=None, det_shape=None, **kwargs):
    """Load a .tpx3 file, apply an ROI crop and mask, and return the summed diffraction pattern.

    Args:
        fn: Path to the .tpx3 file.
        mask: 2-D boolean array matching the scan dimensions.
        scanSize: (nx, ny) scan dimensions.
        roi: (x, y, w, h) scan-space crop or None for the full scan.
        dwellTime: Dwell time in microseconds.
        fn_pattern: Optional path to a smart-scan pattern file for `fn` -
            reshapes/selects scan positions correctly for a smart-scanned
            (sparsely acquired) frame, same as `loaders.load_tpx3`.
        det_shape: (det_x, det_y) detector pixel dimensions ((512, 512) when
            this is None). Only actually applied to the eventem object when
            it differs from what eventem itself already reports after
            set_file() (which reflects the real file's own hardware layout)
            - eventem does NOT gracefully reshape/crop for a mismatched
            value, it segfaults the whole process on .run(). Also used to
            reshape the result, instead of `scanSize` (a real acquisition's
            detector and scan dimensions usually differ).

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    if det_shape is None:
        det_shape = (512, 512)
    repetitions = 1
    bitDepth=16
# =============================================================================
#     if roi is None:
#         x=0
#         y=0 
#         w=scanSize[0]
#         h=scanSize[1]
#     else:
#         # y, x, h, w = roi
#         x, y, w, h = roi
# =============================================================================
        
# =============================================================================
#     s_roi = eventem.Roi(repetitions=repetitions, extract_4D=True)
#     s_roi.set_bitdepth(bitDepth)
#     s_roi.nx = scanSize[0]
#     s_roi.ny = scanSize[1]
#     s_roi.set_file(fn)
#     s_roi.set_roi(x=x, y=y, width=w, height=h)
#     s_roi.set_dwell_time(dwellTime*1000)
#     s_roi.run()
#     # ROI_scan_image = np.asarray(roi.Roi_scan_image)
#     # ROI_diffp = np.asarray(roi.Roi_diffraction_pattern).reshape(512, 512)
#     s = np.asarray(s_roi.get_4D())
# 
#     mask_crop = mask[y:y+h, x:x+w]
#     s = s.reshape(-1, *s.shape[-2:])
#     dp = s[np.where(mask_crop.flatten() == 1)[0]].sum(axis=0)
# =============================================================================
    
    roi = eventem.Roi(repetitions=repetitions, extract_4D=False)
    # eventem auto-sizes its own internal thread pool to the whole machine
    # per instance unless told otherwise - this script always runs as one
    # of several concurrent per-ROI/per-frame worker subprocesses, so
    # leaving that unset would oversubscribe the machine (N processes x
    # full-core-count threads each) exactly like the same bug in
    # worker_nav_img.py's use of io_utils_ui.py's eventem call sites.
    roi.n_threads = 1
    # roi.b_cumulative = True
    roi.set_file(fn)
    roi.set_dwell_time(dwellTime * 1000)
    roi.set_bitdepth(bitDepth)
    roi.nx = scanSize[0]
    roi.ny = scanSize[1]
    # Only actually applied when it differs from what eventem itself already
    # reports (reflecting the real file's own hardware layout) - a genuine
    # mismatch segfaults .run() instead of gracefully reshaping/cropping.
    if det_shape != (roi.detector_size_x, roi.detector_size_y):
        roi.detector_size_x, roi.detector_size_y = det_shape
    if fn_pattern:
        roi.set_pattern_file(fn_pattern)
    # roi.set_roi_mask([np.where(mask.flatten())[0]])
    # roi.set_roi_mask([np.where(mask.flatten())[0]])
    roi.set_roi_mask([mask.flatten()])
    roi.run()
    dp = roi.Roi_diffraction_pattern
    dp = np.array(dp).reshape(det_shape[1], det_shape[0])
    roi.close_socket()
    return dp

def load_hdf5(fn, roi, mask, scanSize=None, chunks=(8, 512, 512, 512), **kwargs):
    """Load an .hdf5 file and sum diffraction patterns at mask-True scan pixels.

    Args:
        fn: Path to the .hdf5 file.
        roi: (x, y, w, h) scan-space crop (currently unused in this loader).
        mask: 2-D boolean array matching the full scan dimensions.
        scanSize: (nx, ny) scan dimensions for reshaping flat storage.
        chunks: Dask chunk shape.

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    with h5py.File(fn, 'r') as f:
        shape = tuple(f['shape'][:])
        if len(f['4D'].shape) == 4:
            s = da.from_array(f['4D'], chunks=chunks)
        elif len(f['4D'].shape) == 1:
            s = da.from_array(f['4D'], chunks=np.prod(chunks))
            s = s.reshape(shape)
        s = s.reshape(-1, *s.shape[2:])
        mask_idx = np.where(mask.flatten() == 1)[0]
        dp = s[mask_idx].sum(axis=0).compute()
    return dp
    
def load_hs(fn, roi, mask, fn_pattern=None, **kwargs):
    """Load a .hspy/.zspy file and sum diffraction patterns at mask-True scan pixels.

    Args:
        fn: Path to the HyperSpy signal file.
        roi: (x, y, w, h) scan-space crop, applied via HyperSpy `inav`. Ignored
            when `fn_pattern` is given (see below).
        mask: 2-D boolean array matching the full scan dimensions.
        fn_pattern: Optional path to a smart-scan pattern file - `.hspy`/
            `.zspy` are loaded via HyperSpy exactly like `.mib` (see
            `load_mib`'s docstring above for the full explanation); when
            given, only the raw frames whose scan position falls inside
            `mask` are read and summed directly (no dense-grid
            reconstruction - cheaper for the single/few-frame case this
            worker is called for).

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    if fn_pattern:
        pattern = np.loadtxt(fn_pattern).astype('int64')
        coords = set(np.where(mask.flatten())[0].tolist())
        idx = [i for i, val in enumerate(pattern) if val in coords]
        s_flat = load(fn, lazy=True).data  # (n_frames, det, det), acquisition order
        if not idx:
            return np.zeros(s_flat.shape[-2:], dtype='uint32')
        dp = s_flat[idx].sum(axis=0).compute()
        return dp

    # Left at its native on-disk chunking - no rechunk() call - rather than
    # an explicit shape that (via a HyperSpy rechunk() argument-binding
    # quirk) was actually fragmenting every diffraction pattern into small
    # sub-chunks instead of keeping each one whole.
    s = load(fn, lazy=True)
    x, y, w, h = roi
    s = s.inav[x:x+w, y:y+h].data
    # `mask` covers the full scan, but `s` was just cropped to the ROI
    # window above, so the mask must be cropped to that same window before
    # it's used to index s. Mirrors the equivalent crop in load_tpx3/load_mib.
    mask_crop = mask[y:y+h, x:x+w]
    s = s.reshape(-1, *s.shape[-2:])
    dp = s[np.where(mask_crop.flatten() == 1)[0]].sum(axis=0)
    dp = dp.compute()
    return dp

def load_mib(fn, roi, mask, scanSize=None, fn_pattern=None, **kwargs):
    """Load a .mib file and sum diffraction patterns at mask-True scan pixels.

    Args:
        fn: Path to the .mib file.
        roi: (x, y, w, h) scan-space crop applied via HyperSpy `inav`. Ignored
            when `fn_pattern` is given (see below).
        mask: 2-D boolean array.
        scanSize: (nx, ny) scan dimensions. Reads from `default.hdr` if None.
        fn_pattern: Optional path to a smart-scan pattern file - a text file
            of one flat scan-pixel index per frame actually stored in `fn`,
            in acquisition order (see
            "other_scripts/smart scanning guide/smart_scanning_analysis_mib.py").
            When given, only the raw frames whose scan position falls inside
            `mask` are read and summed directly (no dense-grid
            reconstruction, unlike `loaders._load_mib_smart_scan` - cheaper
            for the single/few-frame case this worker is called for).

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    if fn_pattern:
        pattern = np.loadtxt(fn_pattern).astype('int64')
        coords = set(np.where(mask.flatten())[0].tolist())
        idx = [i for i, val in enumerate(pattern) if val in coords]
        s_flat = load(fn, lazy=True).data  # (n_frames, det, det), acquisition order
        if not idx:
            return np.zeros(s_flat.shape[-2:], dtype='uint32')
        dp = s_flat[idx].sum(axis=0).compute()
        return dp

    s = load(fn, lazy=True)
    if len(s.data.shape) == 3:
        det_shape = s.data.shape[-1]
        if scanSize is None:
            # Prefer a per-file .hdr matching this exact .mib's own basename
            # (e.g. "default_0119_-50,00.mib" -> "...-50,00.hdr" - the
            # smart-scan convention, one .hdr per .mib), falling back to a
            # shared "default.hdr" in the same folder (the plain full-scan
            # convention, one .hdr for every .mib in the folder).
            fn_hdr = os.path.splitext(fn)[0] + '.hdr'
            if not os.path.isfile(fn_hdr):
                fn_hdr = os.path.join(os.path.split(fn)[0], 'default.hdr')
            scanSize = get_scan_size_mib_hdr(fn_hdr)
        s = s.reshape(scanSize[0], scanSize[1], det_shape, det_shape)
    # Left at its native on-disk chunking - no rechunk() call - rather than
    # an explicit shape that (via a HyperSpy rechunk() argument-binding
    # quirk) was actually fragmenting every diffraction pattern into small
    # sub-chunks instead of keeping each one whole.
    x,y,w,h = roi
    s = s.inav[x:x+w, y:y+h].data
    # `mask` covers the full scan, but `s` was just cropped to the ROI
    # window above, so the mask must be cropped to that same window before
    # it's used to index s - otherwise the flattened mask indices refer to
    # scan positions outside the (smaller) cropped array, either raising an
    # out-of-bounds error or, worse, silently selecting the wrong scan
    # positions. Mirrors the equivalent crop in load_tpx3() above.
    mask_crop = mask[y:y+h, x:x+w]
    s = s.reshape(-1, *s.shape[2:])
    dp = s[np.where(mask_crop.flatten() == 1)[0]].sum(axis=0)
    dp = dp.compute()
    return dp

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
#%%
if __name__ == "__main__":
    extract_3ded_mask_single_frame(*sys.argv[1:])
    
# =============================================================================
#     # for debugging
#     path_mask = r'D:\0_5ded test\Ayush\5DED Analysis\2026-03-26__17-11-25\roi No 1\output_mask.npy'
#     mask = np.load(path_mask)[-1]
#     args = ['D:/0_5ded test/Ayush/4d signals\\raw_0652_-37.75_000000.tpx3', 
#              '[314 117  23  23]', 
#              mask, 
#              '.tpx3', 
#              '(512, 512)', 
#              '(1, 9)']
#     extract_3ded_mask_single_frame(*args)
# =============================================================================
