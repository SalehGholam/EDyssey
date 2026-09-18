# -*- coding: utf-8 -*-
"""
Created on Tue May  6 22:47:49 2025

@author: sgholam
"""

import sys
import json
import h5py
from dask.distributed import Client, LocalCluster
import os
from dask import config
import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar
from scipy import ndimage
import base64
import pickle
file_path = os.path.abspath(__file__)
main_path = os.path.dirname(file_path)  # workers/
eventem_path = os.path.join(os.path.dirname(main_path), 'EDyssey', 'io_utils')
sys.path.append(eventem_path)
# os.chdir()
import hdf5_eventem_layout
import io_utils_ui as io  # re-exports EDyssey.io_utils.eventem_backend as io.eb - see io_utils_ui's docstring
from hyperspy.api import signals, load
# from EDyssey.io_utils import load_signal
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="distributed")
#%%
#TODO add cupy if possible
def extract_3ded_mask_core(task):
    """Extract a single 3DED diffraction pattern from one 4D-STEM frame
    using a binary mask - the pure-computation half of
    extract_3ded_mask_single_frame (below), factored out so
    worker_extract_frame_batch.py's pooled batch driver can call it
    directly per (ROI, frame) task instead of via a fresh CLI-parsing
    subprocess each time. extract_3ded_mask_single_frame is now a thin
    wrapper around this.

    Args:
        task: dict with keys 'fn' (path to the 4D-STEM file), 'roi'
            ([x, y, w, h]/(x, y, w, h) scan-space crop), 'mask_path' (path
            to a .npy file containing the binary 2-D mask), 'dtype' (file
            extension, e.g. '.tpx3'/'.hdf5_eventem'), 'scanSize' ([nx, ny]/(nx, ny)
            or None), 'fn_pattern' (optional smart-scan pattern-file path
            for this frame), 'det_shape' (optional [det_x, det_y]/
            (det_x, det_y) - .tpx3 only, see load_tpx3's docstring; None
            falls back to (512, 512)), 'backend' (optional
            eventem_backend.BACKENDS value - .tpx3 only, None = old
            eventem), 'decluster_cfg' (optional eventem_backend
            decluster_cfg dict - .tpx3 only, None = declustering off).

    Returns:
        numpy.ndarray - the masked/summed diffraction pattern.
    """
    fn = task['fn']
    roi = tuple(int(v) for v in task['roi'])
    mask = np.load(task['mask_path'])
    dtype = task['dtype']
    scanSize = task.get('scanSize')
    scanSize = tuple(scanSize) if scanSize is not None else None
    fn_pattern = task.get('fn_pattern') or None
    det_shape = task.get('det_shape')
    det_shape = tuple(det_shape) if det_shape is not None else None
    # n_threads: defaults to 1 (not passed through from AnalysisBackendSettings)
    # since this always runs as one of several concurrent pool workers - see
    # load_tpx3's docstring / worker_nav_img.py's identical convention.
    return load_dp(fn, roi=roi, mask=mask, dtype=dtype, scanSize=scanSize,
                   fn_pattern=fn_pattern, det_shape=det_shape,
                   backend=task.get('backend'), decluster_cfg=task.get('decluster_cfg'),
                   n_threads=task.get('n_threads', 1))


def extract_3ded_mask_single_frame(fn, roi, mask_path, dtype, scanSize, i_c, fn_pattern=None,
                                   det_shape=None, backend=None, decluster_cfg_json=None,
                                   n_threads=None):
    """Extract a single 3DED diffraction pattern from one 4D-STEM frame using a binary mask.

    Single-task CLI entry point - no longer used by ROI Tracker/SAM2's own
    3DED extraction, which now goes through the pooled batch driver
    (worker_extract_frame_batch.py, one driver process per tracked object,
    running extract_3ded_mask_core directly for every one of that object's
    frames via its own internal process pool) instead of one QProcess per
    (ROI, frame) task. Kept as a standalone single-task entry point.

    Deserialises CLI arguments, loads the per-frame mask from `mask_path`,
    sums diffraction patterns at mask-True pixels via extract_3ded_mask_core,
    and prints the result as a base64+pickle-encoded `(dp, i_c)` tuple.

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
        backend: Optional eventem_backend.BACKENDS value as a string (None/
            'None'/'' = old eventem).
        decluster_cfg_json: Optional JSON-encoded eventem_backend
            decluster_cfg dict, as a string (None/'None'/'' = declustering off).
        n_threads: Optional CPU-core-count override, as a string (None/
            'None'/'' = pin to 1, the prior/default behavior for this
            single-task, not-part-of-a-pool entry point).
    """
    try:
        scanSize_t = tuple(map(int, scanSize.strip("()").split(",")))
        roi_t = [int(a) for a in str(roi)[1:-1].split()]  # read as str of numpy array
        fn_pattern = fn_pattern or None
        det_shape_t = (tuple(map(int, det_shape.strip("()").split(",")))
                      if det_shape not in (None, '', 'None') else None)
        backend = None if backend in (None, '', 'None') else backend
        decluster_cfg = (json.loads(decluster_cfg_json)
                         if decluster_cfg_json not in (None, '', 'None') else None)
        n_threads_t = None if n_threads in (None, '', 'None') else int(n_threads)

        task = {'fn': fn, 'roi': roi_t, 'mask_path': mask_path, 'dtype': dtype,
               'scanSize': scanSize_t, 'fn_pattern': fn_pattern, 'det_shape': det_shape_t,
               'backend': backend, 'decluster_cfg': decluster_cfg}
        if n_threads_t is not None:
            task['n_threads'] = n_threads_t
        dp = extract_3ded_mask_core(task)
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
        **kwargs: Passed through to the format-specific loader (roi, mask,
            scanSize, fn_pattern, etc.). `patch_mode=True` (.tpx3 only - see
            load_tpx3_patches) only actually matters for a *smart-scanned*
            frame (`fn_pattern` set) with no single small bounding box of
            its own (e.g. "Summed DP from Threshold"); a normal dense-raster
            frame always takes the direct eventem-mask path regardless of
            patch_mode, since it doesn't need chopping down at all - see
            load_tpx3.

    Returns:
        numpy.ndarray of shape (det_y, det_x) with the summed diffraction pattern.
    """
    dtype = kwargs.get('dtype')
    if dtype is None:
        dtype = os.path.splitext(fn)[1]
    patch_mode = kwargs.pop('patch_mode', False)

    if dtype == '.tpx3':
        if patch_mode and kwargs.get('fn_pattern'):
            kwargs.pop('roi', None)  # patches derive their own ROI per component
            result = load_tpx3_patches(fn, **kwargs)
        else:
            result = load_tpx3(fn, **kwargs)
    elif dtype == '.hdf5_eventem':
        result = load_hdf5_eventem(fn, **kwargs)
    elif dtype == '.hdf5':
        result = load_hdf5_generic(fn, **kwargs)
    elif dtype in ['.zspy', '.hspy', '.blo']:
        result = load_hs(fn, **kwargs)
    elif dtype == '.mib':
        result = load_mib(fn, **kwargs)
    return result

def load_tpx3(fn, mask, scanSize, roi=None, dwellTime=1, fn_pattern=None, det_shape=None,
              backend=None, decluster_cfg=None, n_threads=None, **kwargs):
    """Load a .tpx3 file and return the diffraction pattern summed over
    mask-True scan pixels.

    Two very different paths depending on whether `fn` is smart-scanned:

    - Normal dense-raster scan (`fn_pattern` is None): `mask` is handed
      straight to eventem via `set_roi_mask()`, which applies it correctly
      over the *whole* scan in one call - no small ROI/bounding box needed,
      `roi` is ignored.
    - Smart-scanned (sparsely acquired) frame (`fn_pattern` given):
      eventem.Roi.set_roi_mask() is not actually implemented on the eventem
      side yet for smart-scanned .tpx3 (silently produces a wrong/unmasked
      result instead of erroring) - see the workaround comment below. `roi`
      is required here.

    Args:
        fn: Path to the .tpx3 file.
        mask: 2-D boolean array matching the full scan dimensions - never
            pre-cropped by the caller, even when `roi` is also given.
        scanSize: (nx, ny) scan dimensions.
        roi: (x, y, w, h) scan-space crop - required, and must be a small
            box (a tracked ROI, a SAM2 segmentation's own bounding box, or
            one connected patch from load_tpx3_patches below), only on the
            smart-scanned path; unused otherwise.
        dwellTime: Dwell time in microseconds.
        fn_pattern: Optional path to a smart-scan pattern file for `fn` -
            None (the default) means a normal dense-raster scan.
        det_shape: (det_x, det_y) detector pixel dimensions ((512, 512) when
            this is None). Only actually applied to the eventem object when
            it differs from what eventem itself already reports after
            set_file() (which reflects the real file's own hardware layout)
            - eventem does NOT gracefully reshape/crop for a mismatched
            value, it segfaults the whole process on .run(). Also used to
            reshape the result, instead of `scanSize` (a real acquisition's
            detector and scan dimensions usually differ).
        backend: one of eventem_backend.BACKENDS (None = old eventem,
            preserving this function's prior behavior exactly for any
            caller that doesn't pass this).
        decluster_cfg: optional eventem_backend decluster_cfg dict. None =
            declustering off.
        n_threads: optional override for eventem/pyeventem's own internal
            thread pool - defaults to 1 (not None) here, unlike
            eventem_backend.run_*'s own default, since this worker always
            runs as one of several concurrent pool workers (see
            worker_extract_frame_batch.py) - leaving it unset would
            oversubscribe the machine exactly like the equivalent
            worker_nav_img.py comment explains.

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    if det_shape is None:
        det_shape = (512, 512)
    if n_threads is None:
        n_threads = 1
    backend = backend or io.eb.BACKEND_OLD

    if fn_pattern is None:
        # Normal dense-raster scan: eventem/pyeventem's own mask-based ROI
        # works correctly here, so apply `mask` directly, server-side, in
        # one call - no small ROI or Python-side masking needed at all.
        result = io.eb.run_roi_masked(
            fn, scanSize, mask, dwell_time_ns=dwellTime * 1000, det_shape=det_shape,
            backend=backend, decluster_cfg=decluster_cfg, n_threads=n_threads, bitdepth=16,
        )
        return np.asarray(result.Roi_diffraction_pattern).reshape(det_shape[1], det_shape[0])

    # TEMPORARY WORKAROUND (smart-scanned only): eventem.Roi.set_roi_mask()
    # is not actually implemented on the eventem side yet for smart-scanned
    # .tpx3 (silently produces a wrong/unmasked result instead of erroring).
    # Until eventem implements it for real, extract the ROI as a full 4D
    # block instead (extract_4D=True) and apply the mask ourselves in
    # Python below. This materializes the whole ROI's frames at once, so
    # `roi` must stay a small box - callers with no natural small ROI (a
    # scattered/large threshold mask) must split it into patches first, see
    # load_tpx3_patches, rather than passing its full bounding box here
    # (that was tried and crashed/OOM'd on smart-scanned data). Swap this
    # back to set_roi_mask() once that's genuinely supported upstream for
    # smart-scanned data too.
    x, y, w, h = roi
    result = io.eb.run_roi(
        fn, scanSize, roi_rect=(x, y, w, h), dwell_time_ns=dwellTime * 1000, det_shape=det_shape,
        fn_pattern=fn_pattern, get_4d=True, backend=backend, decluster_cfg=decluster_cfg,
        n_threads=n_threads,
    )
    s = np.asarray(result.get_4D())

    mask_crop = mask[y:y+h, x:x+w]
    s = s.reshape(-1, *s.shape[-2:])
    dp = s[np.where(mask_crop.flatten() == 1)[0]].sum(axis=0)
    return dp

def load_tpx3_patches(fn, mask, scanSize, dwellTime=1, fn_pattern=None, det_shape=None,
                      backend=None, decluster_cfg=None, n_threads=None, **kwargs):
    """Sum diffraction patterns over an arbitrary (possibly large/scattered)
    .tpx3 mask, for a *smart-scanned* caller with no small ROI of their own
    to begin with - "Summed DP from Threshold" is the only one today, since
    a real-space threshold can select scan positions anywhere in the scan,
    unlike a tracked ROI or a SAM2 segmentation (both already a single small
    box, loaded directly via load_tpx3 instead - see load_dp's patch_mode
    note). Only reached at all when `fn_pattern` is set - a normal
    dense-raster scan never needs patching, load_tpx3 applies `mask`
    directly via eventem's own set_roi_mask() in one call regardless of the
    mask's shape/extent.

    Splits `mask` into its connected components (8-connectivity, so
    diagonally-touching pixels count as one patch) and calls load_tpx3 once
    per component with that component's own small bounding box, accumulating
    the result - the multi-patch equivalent of load_tpx3's own
    extract_4D=True workaround, needed because a *single* extraction
    spanning the mask's full bounding box would materialize the whole
    (potentially near-whole-scan) block at once and crash/OOM on
    smart-scanned data. Temporary, like load_tpx3's own workaround: drop
    this once eventem's set_roi_mask() is genuinely supported for
    smart-scanned data too.

    Args:
        fn: Path to the .tpx3 file.
        mask: 2-D boolean array matching the scan dimensions.
        scanSize: (nx, ny) scan dimensions.
        dwellTime: Dwell time in microseconds.
        fn_pattern: Path to the smart-scan pattern file for `fn`.
        det_shape: (det_x, det_y) detector pixel dimensions - see load_tpx3.

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    if det_shape is None:
        det_shape = (512, 512)
    dp_total = np.zeros((det_shape[1], det_shape[0]), dtype='float64')
    labeled, n_patches = ndimage.label(mask, structure=np.ones((3, 3)))
    for patch_id, sl in enumerate(ndimage.find_objects(labeled), start=1):
        if sl is None:
            continue
        y_sl, x_sl = sl
        patch_roi = (int(x_sl.start), int(y_sl.start),
                    int(x_sl.stop - x_sl.start), int(y_sl.stop - y_sl.start))
        patch_mask = labeled == patch_id
        dp_total += load_tpx3(fn, mask=patch_mask, scanSize=scanSize, roi=patch_roi,
                              dwellTime=dwellTime, fn_pattern=fn_pattern, det_shape=det_shape,
                              backend=backend, decluster_cfg=decluster_cfg, n_threads=n_threads)
    return dp_total

def load_hdf5_eventem(fn, roi, mask, scanSize=None, max_eager_frames=10000, **kwargs):
    """Load an eventem-format '.hdf5_eventem' file and sum diffraction
    patterns at mask-True scan pixels.

    Handles both the current native-4D storage layout and the older flat/
    1-D layout some pre-4D-native eventem exports still use (see
    hdf5_eventem_layout's module docstring). For the flat layout, reads the
    masked positions directly via h5py (no dask) when their count is within
    `max_eager_frames` - noticeably faster than always building and
    computing a dask graph, which is what this used to do unconditionally -
    falling back to dask above that, same as loaders.load_hdf5_eventem's
    own max_eager_frames.

    Args:
        fn: Path to the .hdf5_eventem file.
        roi: (x, y, w, h) scan-space crop (currently unused in this loader).
        mask: 2-D boolean array matching the full scan dimensions.
        scanSize: Unused - the file's own `f['shape']` is authoritative
            (see hdf5_eventem_layout.get_shape). Kept for call-signature
            compatibility with the other load_* functions load_dp dispatches to.
        max_eager_frames: See above.

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    # max_workers=1: this script always runs as one of several concurrent
    # pool workers (worker_extract_frame_batch.py's own ProcessPoolExecutor)
    # - same reasoning as load_tpx3's n_threads=1 above. Without this,
    # sum_masked_positions's own dynamically-sized thread pool would spin up
    # inside every one of those worker processes too, multiplying concurrent
    # ~1GB chunk buffers by the pool's own worker count.
    with h5py.File(fn, 'r') as f:
        dp = hdf5_eventem_layout.sum_masked_positions(f, mask, max_eager_frames=max_eager_frames, max_workers=1)
    return dp

def load_hdf5_generic(fn, roi, mask, fn_pattern=None, **kwargs):
    """Load a conventional/third-party '.hdf5' file and sum diffraction
    patterns at mask-True scan pixels - mirrors load_hs below, but via a
    plain h5py tree walk + dask instead of HyperSpy's own `load()`, which
    can't parse this format's arbitrary internal layout (see
    EDyssey.io_utils.loaders' module docstring - same one-4D-dataset
    convention as its own load_hdf5_generic).

    fn_pattern (smart-scan) isn't supported here, unlike load_hs's own
    .hspy/.zspy path - no real acquisition needing this loader has used a
    smart-scan pattern file in practice.

    Args:
        fn: Path to the .hdf5 file.
        roi: (x, y, w, h) scan-space crop.
        mask: 2-D boolean array matching the full scan dimensions.

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    with h5py.File(fn, 'r') as f:
        datasets_4d = []
        def _visit(name, obj):
            # Any numeric kind or plain boolean (a thresholded/binary
            # detector read is still meaningful data to sum) - see
            # EDyssey.io_utils.loaders._is_dp_dtype.
            if (isinstance(obj, h5py.Dataset) and obj.ndim == 4
                    and (np.issubdtype(obj.dtype, np.number) or np.issubdtype(obj.dtype, np.bool_))):
                datasets_4d.append(name)
        f.visititems(_visit)
        if len(datasets_4d) != 1:
            raise ValueError(
                f"Expected exactly one numerical 4D dataset in {fn!r}, found "
                f"{len(datasets_4d)}.")
        dset = f[datasets_4d[0]]
        data = da.from_array(dset, chunks=dset.chunks if dset.chunks is not None else 'auto')
        x, y, w, h = roi
        data = data[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]
        data = data.reshape(-1, *data.shape[-2:])
        dp = data[np.where(mask_crop.flatten() == 1)[0]].sum(axis=0)
        dp = dp.compute()
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
#     path_mask = r'D:\0_edyssey test\Ayush\EDyssey Analysis\2026-03-26__17-11-25\roi No 1\output_mask.npy'
#     mask = np.load(path_mask)[-1]
#     args = ['D:/0_edyssey test/Ayush/4d signals\\raw_0652_-37.75_000000.tpx3', 
#              '[314 117  23  23]', 
#              mask, 
#              '.tpx3', 
#              '(512, 512)', 
#              '(1, 9)']
#     extract_3ded_mask_single_frame(*args)
# =============================================================================
