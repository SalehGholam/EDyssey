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

from .progress import redirect_console_to_logger, LoggingProgressBar

def get_4d_files(path_4d_datasets, dtype):
    """Return sorted list of 4D-STEM files matching *dtype* in *path_4d_datasets* (recursive)."""
    fns_4d = glob(os.path.join(path_4d_datasets, f'*.{dtype}'))
    if len(fns_4d) == 0: # search in sub-directories
        fns_4d = glob(os.path.join(path_4d_datasets, '**', f'*.{dtype}'), recursive=True)
    assert len(fns_4d) != 0, "4D Datasets with correct format are not found!"
    return fns_4d

def load_signal(fn, **kwargs):
    """Dispatch-load a 4D-STEM file to the appropriate loader based on file extension."""
    try:
        dtype = kwargs.get('dtype')
    except:
        dtype = None
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

def load_tpx3(fn, roi=None, scanSize=(512,512), dwellTime=1, bitDepth=16,
              repetitions=1, sum_dp=False, logger=None,
              n_threads=None, fn_pattern=None, **kwargs):
    """Load a .tpx3 4D-STEM file via eventem; returns a HyperSpy Signal2D or summed nav image.

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
    if roi is None:
        x, y, w, h = 0, 0, scanSize[0], scanSize[1]
    else:
        x, y, w, h = roi

    roi = eventem.Roi(repetitions=repetitions, extract_4D=True)
    if n_threads is not None:
        roi.n_threads = n_threads
    roi.set_bitdepth(bitDepth)
    roi.nx = scanSize[0]
    roi.ny = scanSize[1]
    roi.set_file(fn)
    if fn_pattern:
        roi.set_pattern_file(fn_pattern)
    roi.set_roi(x=x, y=y, width=w, height=h)
    roi.set_dwell_time(dwellTime*1000)
    with redirect_console_to_logger(logger, 'Loading tpx3'):
        roi.run()
    # ROI_scan_image = np.asarray(roi.Roi_scan_image)
    # ROI_diffp = np.asarray(roi.Roi_diffraction_pattern).reshape(512, 512)
    s = np.asarray(roi.get_4D())
    if sum_dp:
        s = s.sum(axis=(-1,-2))
        return s
    s = hs.signals.Signal2D(s)
    return s

def load_hdf5(fn, roi=None, scanSize=None, chunks=None, lazy=False,
              sum_dp=False, logger=None, **kwargs):
    """Load a .hdf5 4D-STEM file using dask; supports lazy loading, ROI crop, and DP summation."""
    # with h5py.File(fn, 'r') as f:
    f = h5py.File(fn, 'r')
    arr_dim = len(f['4D'].shape) # data might be flattened
    if arr_dim == 1: # TODO do it only once before running tomo
        det_shape = f['shape'][:][-1] # works on only scquare detectors
        scanSize_written = tuple(f['shape'][:][:2])
        if scanSize is None:
            scanSize = scanSize_written
        else:
            assert scanSize == scanSize_written, f"Scan size entered does not match to the shape of the hdf5 file: {scanSize} vs {scanSize_written}"
    else:
        det_shape = f['4D'].shape[-1] # works on only square detectors

    if chunks is None:
        # Read using the dataset's own on-disk chunk shape whenever it's
        # actually chunked storage - any other chunk grid means every dask
        # chunk straddles multiple real HDF5 chunks, multiplying disk I/O
        # for no benefit. rechunk() afterwards doesn't avoid this either:
        # dask still has to read the (misaligned) native chunks first
        # before it can regroup them into the new shape.
        native_chunks = f['4D'].chunks
        if native_chunks is not None:
            chunks = native_chunks
        elif roi is not None:
            y,x,h,w = roi
            chunks = (h, w, det_shape, det_shape)
            if arr_dim == 1:
                chunks = np.prod(chunks)
        elif det_shape == 512:
            chunks = (16, 16, det_shape, det_shape)
            if arr_dim == 1:
                chunks = np.prod(chunks)
        else:
            chunks = (32, 32, det_shape, det_shape)
            if arr_dim == 1:
                chunks = np.prod(chunks)

    if arr_dim == 1:
        s = da.from_array(f['4D'], chunks=chunks)
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            s = s.reshape(scanSize[0], scanSize[1], det_shape, det_shape)
        # s = s.map_blocks(cp.asarray)
    else:
        s = da.from_array(f['4D'], chunks=chunks)
        # s = s.map_blocks(cp.asarray)

    if roi is not None and np.any(roi):
        x, y, w, h = roi  # roi format: [x, y, w, h] — x=col, y=row
        s = s[y:y+h, x:x+w]  # numpy row-major: row (y) axis first

    if sum_dp:
        dp = s.sum(axis=(2,3))
        with LoggingProgressBar(logger, 'Summing diffraction patterns'):
            dp_res = dp.compute()
        # dp_res = cp.asnump(dp_res)
        f.close()
        # cp.get_default_memory_pool().free_all_blocks()
        # del s, dp
        # cp.get_default_memory_pool().free_all_blocks()
        return dp_res

    if not lazy:
        with LoggingProgressBar(logger, 'Loading 4D signal'):
            s_get = s.compute()
        # s_get = cp.asnumpy(s_get) # turning it to numpy from cupy
        # cp.get_default_memory_pool().free_all_blocks()
        # del s
        f.close()
        s_get = hs.signals.Signal2D(s_get)
        return s_get

    else:
        s = hs.signals.Signal2D(s)
        s = s.as_lazy()
        # cp.get_default_memory_pool().free_all_blocks()
        return s, f

def load_hs(fn, roi=None, chunks=None, lazy=False, sum_dp=False, logger=None,
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
    #TODO add cupy if possible
    if fn_pattern:
        s = _reconstruct_smart_scan(fn, fn_pattern, scanSize, chunks)
    else:
        s = hs.load(fn, lazy=True)
        det_shape = s.data.shape[-1]
        # Unlike raw acquisition .hdf5 (whose native chunk is already a
        # reasonably-sized block, so it's read as-is - see load_hdf5), 4D-STEM
        # .hspy/.zspy files are typically written with a per-navigation-pixel
        # chunk (e.g. (1,1,det,det)) to make single-frame random access fast
        # during acquisition - exactly the wrong shape for a full-scan
        # reduction, so re-chunking to a coarser nav grid first pays for itself.
        if chunks is None:
            nav_chunk = 16 if det_shape == 512 else 32
            s.rechunk(nav_chunks=(nav_chunk, nav_chunk), sig_chunks=(det_shape, det_shape))
        else:
            s.rechunk(nav_chunks=chunks[:2], sig_chunks=chunks[2:])

    if np.any(roi):
        x,y,w,h = roi
        s = s.inav[x:x+w, y:y+h]

    if sum_dp:
        dp = s.sum(axis=(2,3)).data
        with LoggingProgressBar(logger, 'Summing diffraction patterns'):
            return dp.compute()

    if not lazy:
        with LoggingProgressBar(logger, 'Loading 4D signal'):
            s.compute()
    return s

def _reconstruct_smart_scan(fn, fn_pattern, scanSize, chunks=None):
    """Reconstruct a smart-scanned acquisition into its full dense
    (ny, nx, det, det) grid.

    Used for .mib/.hspy/.zspy - unlike .tpx3 (whose eventem reader
    understands a pattern file natively, see `load_tpx3`), these are all
    loaded via `hs.load` (just dispatching to a different reader plugin per
    extension) as a flat stream of only the frames that were actually
    acquired, in acquisition order - `fn_pattern` (a text file of one flat
    scan-pixel index per stored frame) says where each one belongs. Every
    scan position not visited is left as zero. Mirrors `recreate_4d` in
    "other_scripts/smart scanning guide/smart_scanning_analysis_mib.py", but
    reshapes to (ny, nx, det, det) - matching this app's own convention for
    a dense .mib load (see the comment in `load_mib` below) rather than that
    script's (nx, ny) - so `roi`/`.inav` cropping right after this call
    behaves identically to the non-smart-scan path.
    """
    if scanSize is None:
        raise ValueError(
            "scanSize is required to reconstruct a smart-scanned acquisition (fn_pattern was "
            f"given for {fn!r}, but scanSize was None) - a per-file header, when one exists, "
            "reflects the number of frames actually written to disk for this sparse "
            "acquisition, not the full scan grid, so it can't be auto-detected from the file "
            "itself. Pass the scan size explicitly (e.g. from comment.txt).")
    pattern = np.loadtxt(fn_pattern).astype('int64')
    s_flat = hs.load(fn, lazy=True)  # (n_frames, det, det), acquisition order
    det_shape = s_flat.data.shape[-2:]
    dask_arr = da.zeros((scanSize[0] * scanSize[1], *det_shape),
                        dtype=s_flat.data.dtype, chunks=(256, *det_shape))
    dask_arr[pattern] = s_flat.data
    dask_arr = dask_arr.reshape(scanSize[1], scanSize[0], *det_shape)
    s = hs.signals.Signal2D(dask_arr).as_lazy()

    if chunks is None:
        nav_chunk = 16 if det_shape[0] == 512 else 32
        s.rechunk(nav_chunks=(nav_chunk, nav_chunk), sig_chunks=det_shape)
    else:
        s.rechunk(nav_chunks=chunks[:2], sig_chunks=chunks[2:])
    return s

def load_mib(fn, roi=None, scanSize=None, chunks=None, lazy=False, sum_dp=False,
            logger=None, fn_pattern=None, **kwargs):
    """Load a .mib file, rechunk, and optionally crop/sum.

    Args:
        fn: Path to the .mib file.
        roi: Optional (x, y, w, h) crop region in scan coordinates.
        scanSize: (nx, ny) scan dimensions. If None, reads from `default.hdr` in the same folder.
        chunks: Explicit dask chunk shape. If None, auto-selected by detector size.
        lazy: Return a lazy (dask-backed) signal without computing.
        sum_dp: Return summed diffraction patterns (2D navigation image) instead of 4D signal.
        logger: Optional logger to report dask compute progress through.
        fn_pattern: Optional path to a smart-scan pattern file - see
            `_reconstruct_smart_scan`. None (default) loads `fn` as a normal
            dense raster, matching prior behaviour.

    Returns:
        numpy.ndarray if `sum_dp` is True; otherwise a HyperSpy Signal2D (lazy or computed).
    """
    if fn_pattern:
        # Never auto-resolved from a .hdr here (unlike the dense branch
        # below) - a smart-scanned acquisition's own .hdr "Frames in
        # Acquisition" reflects the sparse frame count actually written to
        # disk, not the full scan grid, so it would silently produce the
        # wrong shape instead of the clear error _reconstruct_smart_scan
        # raises when scanSize is still None.
        s = _reconstruct_smart_scan(fn, fn_pattern, scanSize, chunks)
    else:
        if scanSize is None:
            fld = os.path.split(fn)[0]
            # Prefer a per-file .hdr matching this exact .mib's own
            # basename (e.g. "default_0119_-50,00.mib" -> "...-50,00.hdr")
            # over the generic "default.hdr" - tomography acquisitions like
            # the smart-scan ones write one .hdr per .mib, not one shared
            # one, so the generic name never matches them.
            fn_hdr = os.path.splitext(fn)[0] + '.hdr'
            if not os.path.isfile(fn_hdr):
                fn_hdr = os.path.join(fld, 'default.hdr')
            if os.path.isfile(fn_hdr):
                scanSize = get_scan_size_mib_hdr(fn_hdr)
        # Passing navigation_shape=(nx, ny) lets the mib reader itself
        # reshape the raw frame stream into a proper 4D array (it reverses
        # the pair internally to (ny, nx, det, det) before returning it) -
        # manually reshaping a flat (N, det, det) stack afterwards, as this
        # used to do, silently assumed the same (nx, ny) axis order without
        # that reversal and could produce a transposed navigation image.
        s = hs.load(fn, lazy=True, navigation_shape=scanSize)
        det_shape = s.data.shape[-1]

        if chunks is None:
            if det_shape == 512:
                s.rechunk(nav_chunks=(16,16), sig_chunks=(det_shape,det_shape))
            else:
                s.rechunk(nav_chunks=(32,32), sig_chunks=(det_shape,det_shape))
        else:
            s.rechunk(nav_chunks=chunks[:2], sig_chunks=chunks[2:])

    if roi is not None:
        x,y,w,h = roi
        s = s.inav[x:x+w, y:y+h]

    if sum_dp:
        dp = s.sum(axis=(2,3)).data
        with LoggingProgressBar(logger, 'Summing diffraction patterns'):
            dp = dp.compute()
        return dp

    if not lazy:
        with LoggingProgressBar(logger, 'Loading 4D signal'):
            s.compute()
    return s

def get_scan_size(fn):
    """Return (nx, ny) scan dimensions read from the file header."""
    dtype = os.path.splitext(fn)[-1]
    if dtype == '.hdf5':
        with h5py.File(fn, 'r') as f:
            scanSize = tuple(f['shape'])[:2]
        return scanSize
    else:
        s = load_signal(fn, lazy=True)
        return (s.data.shape[1], s.data.shape[0]) # TODO return y and x?

def get_det_size(fn):
    """Return (det_x, det_y) detector pixel dimensions read from the file."""
    s = load_signal(fn, lazy=True)
    if type(s) == tuple: # for hdf5
        s, f = s
        f.close()
    det_shape = (s.data.shape[3], s.data.shape[2])
    return det_shape

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
