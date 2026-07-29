# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:58:43 2024

@author: SGholam
"""

import numpy as np
# import pyxem as px
import os
import sys
import threading
import contextlib
import uuid
from glob import glob
from tqdm import tqdm
import hyperspy.api as hs
import tifffile
import dask.array as da
import h5py
import cv2
import dask
from dask.diagnostics import ProgressBar
import eventem
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
from matplotlib_scalebar.scalebar import ScaleBar
from warnings import warn
import shutil
# import cupy as cp
# from moviepy.editor import VideoClip
# from moviepy.video.io.bindings import mplfig_to_npimage
# from moviepy import VideoClip
# from moviepy.video.io.ffmpeg_reader import ffmpeg_read_image
from matplotlib.animation import FuncAnimation
from typing import Literal
import subprocess
from skimage.draw import disk
from scipy.ndimage import center_of_mass, gaussian_filter
_ffmpeg = shutil.which('ffmpeg')
if _ffmpeg:
    plt.rcParams['animation.ffmpeg_path'] = _ffmpeg
#%% dask progress reporting
class _NullWriter:
    """Discards ProgressBar's own raw stdout writes - LoggingProgressBar
    reports through a logger instead (see below), so the base class's
    carriage-return text would otherwise still land on stdout."""
    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass


class LoggingProgressBar(ProgressBar):
    """dask ProgressBar that reports through a Python logger instead of
    writing raw carriage-return text to stdout - meaningless once mixed
    into a Qt log widget/rotating log file. Only logs when the integer
    percentage actually changes, so this doesn't spam a line every `dt`.
    A no-op (does nothing) when `logger` is None, so call sites can always
    wrap a `.compute()` in this without checking first."""
    def __init__(self, logger=None, label='', minimum=1.0, dt=0.5):
        super().__init__(minimum=minimum, dt=dt, out=_NullWriter())
        self.logger = logger
        self.label = label
        self._last_percent = -1

    def _draw_bar(self, frac, elapsed):
        if self.logger is None:
            return
        percent = int(100 * frac)
        if percent == self._last_percent:
            return
        self._last_percent = percent
        from dask.utils import format_time
        prefix = f'{self.label}: ' if self.label else ''
        self.logger.info('%s%d%% complete (%s elapsed)', prefix, percent, format_time(elapsed))

def _log_or_print(logger, msg):
    """Report a status message through `logger` if given, otherwise print it
    - lets shared helpers (e.g. the create_clip_* functions) report through
    a tab's Qt log console when called from the GUI, while still working
    standalone (no logger) from a plain script/console session."""
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

#%% console progress -> Qt logger bridge
# RLock (not Lock): redirect_console_to_logger nests two _capture_fd calls
# (fd 1 and fd 2) from the same thread - a plain Lock would self-deadlock.
_fd_redirect_lock = threading.RLock()

@contextlib.contextmanager
def _capture_fd(fd, on_line):
    """Tee OS file descriptor `fd` (1=stdout, 2=stderr) through a pipe for
    the duration of the block: everything written to it still reaches the
    real fd exactly as before (so a console/terminal keeps showing it live,
    same as before this existed), while also calling `on_line(text)` for
    each printed line - or each carriage-return-updated segment of an
    in-place progress bar - so it can additionally be forwarded elsewhere
    (e.g. a Qt log console). Needed for progress from compiled extensions
    (e.g. eventem's tpx3 loading/ROI-extraction progress) that write
    straight to the OS-level fd and are invisible to `sys.stdout`/
    `sys.stderr` overrides.

    A given fd is process-wide, not per-thread, so a lock serializes
    overlapping uses of *any* fd captured this way - two concurrent
    redirects would otherwise corrupt each other's captured output (or, if
    they target different fds, still race on the shared dup()'d-fd
    bookkeeping below).
    """
    with _fd_redirect_lock:
        sys.stdout.flush()
        sys.stderr.flush()
        original_fd = os.dup(fd)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)

        last_line = [None]

        def _pump():
            buf = b''
            try:
                while True:
                    chunk = os.read(read_fd, 65536)
                    if not chunk:
                        break
                    # Tee: the real console/terminal still sees this exactly
                    # as if it hadn't been intercepted at all.
                    os.write(original_fd, chunk)
                    buf += chunk
                    while True:
                        idx_n = buf.find(b'\n')
                        idx_r = buf.find(b'\r')
                        candidates = [i for i in (idx_n, idx_r) if i != -1]
                        if not candidates:
                            break
                        idx = min(candidates)
                        line = buf[:idx].decode('utf-8', errors='replace').strip()
                        buf = buf[idx + 1:]
                        if line and line != last_line[0]:
                            last_line[0] = line
                            on_line(line)
                tail = buf.decode('utf-8', errors='replace').strip()
                if tail and tail != last_line[0]:
                    on_line(tail)
            finally:
                os.close(read_fd)

        reader_thread = threading.Thread(target=_pump, daemon=True)
        reader_thread.start()
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(original_fd, fd)
            # Unbounded: the pipe's write end is deterministically closed by
            # the dup2 above (the only other reference, write_fd, was closed
            # right after the initial dup2), so EOF - and thus _pump's exit
            # - is guaranteed, not a hang risk. A bounded join here used to
            # leave the reader thread running concurrently with whatever the
            # caller did right after this context manager exited; do not
            # reintroduce a timeout without re-checking every call site for
            # that race.
            reader_thread.join()
            # Only close original_fd once the reader thread is fully done -
            # it tees every chunk through to original_fd (see _pump above),
            # so closing it earlier risks a write racing the close, or (once
            # the OS recycles the fd number) landing in a wrong,
            # unrelated file entirely.
            os.close(original_fd)


@contextlib.contextmanager
def redirect_console_to_logger(logger, label=''):
    """Capture both real stdout and stderr (OS fds 1 and 2) and forward
    everything written to either, line by line, to a Python logger instead
    of the terminal - eventem's native progress output has been observed on
    both, depending on version/platform, so both are captured to be safe. A
    no-op when `logger` is None, so call sites can always wrap a blocking
    call in this without checking first.

    All lines from one call (i.e. one load/compute operation) share a single
    `progress_key`, so consecutive progress updates collapse onto one
    updating line in the Qt log console (see LogConsole._append_log) instead
    of accumulating as one appended line per update.
    """
    if logger is None:
        yield
        return

    progress_key = f'{label or "console"}-{uuid.uuid4().hex[:8]}'

    def _on_line(line):
        logger.info('%s%s', f'{label}: ' if label else '', line,
                    extra={'progress_key': progress_key})

    with _capture_fd(1, _on_line), _capture_fd(2, _on_line):
        yield

#%% scale bar readability
class ReadableScaleBar(ScaleBar):
    """A matplotlib_scalebar ScaleBar with its background fully removed
    (box_alpha=0) and its label given a stroked outline so it stays legible
    over arbitrary image content underneath, regardless of `color`.

    matplotlib_scalebar rebuilds its Text artist from scratch inside
    draw() on every redraw and doesn't expose a path_effects hook, so this
    patches the module-level `TextArea` name it calls for the duration of
    our own draw() call, restoring it immediately after.
    """
    _OUTLINE = [patheffects.withStroke(linewidth=2.5, foreground='black')]

    def draw(self, renderer, *args, **kwargs):
        import matplotlib_scalebar.scalebar as _sb_mod
        original_text_area = _sb_mod.TextArea
        outline = self._OUTLINE

        def _outlined_text_area(s, textprops=None, **kw):
            textprops = dict(textprops or {})
            textprops['path_effects'] = outline
            return original_text_area(s, textprops=textprops, **kw)

        _sb_mod.TextArea = _outlined_text_area
        try:
            super().draw(renderer, *args, **kwargs)
        finally:
            _sb_mod.TextArea = original_text_area


def add_readable_scalebar(ax, dx, units='m', **kwargs):
    """Remove any existing ScaleBar on `ax` and add a `ReadableScaleBar` -
    transparent background, outlined text - in its place. `kwargs` are
    forwarded to ReadableScaleBar (e.g. `dimension`, `location`); `color`
    defaults to white, which reads well (with the outline) against this
    app's dark colormaps (viridis/inferno/gray).
    """
    kwargs.setdefault('dimension', 'si-length')
    kwargs.setdefault('location', 'lower left')
    kwargs.setdefault('color', 'w')
    kwargs.setdefault('box_alpha', 0)
    for artist in list(ax.artists):
        if isinstance(artist, ScaleBar):
            artist.remove()
    scalebar = ReadableScaleBar(dx, units, **kwargs)
    ax.add_artist(scalebar)
    return scalebar

#%%
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
              det_shape=512, repetitions=1, sum_dp=False, logger=None,
              n_threads=None, **kwargs):
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
    
def load_hs(fn, roi=None, chunks=None, lazy=False, sum_dp=False, logger=None, **kwargs):
    """Load a .hspy/.zspy HyperSpy signal; supports lazy loading, ROI crop, and DP summation."""
    #TODO add cupy if possible
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

def load_mib(fn, roi=None, scanSize=None, chunks=None, lazy=False, sum_dp=False,
            logger=None, **kwargs):
    """Load a .mib file, rechunk, and optionally crop/sum.

    Args:
        fn: Path to the .mib file.
        roi: Optional (x, y, w, h) crop region in scan coordinates.
        scanSize: (nx, ny) scan dimensions. If None, reads from `default.hdr` in the same folder.
        chunks: Explicit dask chunk shape. If None, auto-selected by detector size.
        lazy: Return a lazy (dask-backed) signal without computing.
        sum_dp: Return summed diffraction patterns (2D navigation image) instead of 4D signal.
        logger: Optional logger to report dask compute progress through.

    Returns:
        numpy.ndarray if `sum_dp` is True; otherwise a HyperSpy Signal2D (lazy or computed).
    """
    if scanSize is None:
        fld = os.path.split(fn)[0]
        fn_hdr = os.path.join(fld, 'default.hdr')
        if os.path.isfile(fn_hdr):
            scanSize = get_scan_size_mib_hdr(fn_hdr)
    # Passing navigation_shape=(nx, ny) lets the mib reader itself reshape
    # the raw frame stream into a proper 4D array (it reverses the pair
    # internally to (ny, nx, det, det) before returning it) - manually
    # reshaping a flat (N, det, det) stack afterwards, as this used to do,
    # silently assumed the same (nx, ny) axis order without that reversal
    # and could produce a transposed navigation image.
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

def _resolve_metadata_fn(path_main):
    """`path_main` is either a comment.txt path itself, or a directory/file
    next to which a comment.txt lives."""
    if os.path.isfile(path_main):
        return path_main if path_main.endswith('.txt') else os.path.join(
            os.path.dirname(path_main), 'comment.txt')
    return os.path.join(path_main, 'comment.txt')

def get_metadata(path_main, count=0):
    """Parse the (0-indexed) `count`-th measurement block from a tpx3
    acquisition's comment.txt. `path_main` may be the comment.txt file
    itself, or the directory/4D-signal-file next to which it sits. Blocks
    are separated by a blank line, or by two consecutive blank lines when
    the same block repeats back-to-back."""
    with open(_resolve_metadata_fn(path_main), 'r') as f:
        ls = f.readlines()
    metadata = {}
    blank = [i for i, x in enumerate(ls) if x == '\n']
    end = blank[count]
    while (count != 0) and (end == blank[count-1]+1):  # 2 empty lines
        end = blank[count+1]
    for item in ls[:end]:
        try:
            key, val = item.split(':')
            if ' microseconds' in val:
                val = val.rstrip(' microseconds\n')
            metadata[key] = float(val)
        except Exception:
            pass
    return metadata

def get_metadata_block_count(path_main):
    """Number of measurement blocks logged in a tpx3 comment.txt (see
    `get_metadata`), i.e. how many separate acquisitions it records - blocks
    are separated by one or more consecutive blank lines."""
    with open(_resolve_metadata_fn(path_main), 'r') as f:
        ls = f.readlines()
    n_blocks = 0
    in_block = False
    for line in ls:
        if line.strip():
            if not in_block:
                n_blocks += 1
                in_block = True
        else:
            in_block = False
    return n_blocks

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

def draw_reciprocal_scale_circles(ax, scale_recip, shape, center=None, old_artists=None):
    """Draw concentric dashed circles marking whole-1/A radii on a
    diffraction-pattern axis, in place of a conventional scale bar (which
    doesn't read naturally on a radially-symmetric reciprocal-space image).
    Circles are spaced every 1 1/A out to whatever fits in the image; if
    even the first one wouldn't fit (coarse reciprocal-space resolution), a
    single 0.5 1/A circle is drawn instead, in a visually distinct (dotted)
    style so it can't be mistaken for a whole-1/A ring. Each ring is also
    labeled with its 1/A value near its top-right edge.

    Args:
        ax: matplotlib Axes the diffraction-pattern image is displayed on.
        scale_recip: Reciprocal-space calibration in 1/Angstrom per pixel.
            No circles are drawn if this is None, 0, or unparseable.
        shape: (height, width) of the displayed image, in pixels.
        center: (x, y) center to draw circles around, in pixel coordinates.
            Defaults to the image's own geometric center.
        old_artists: Circle/label artists returned by a previous call,
            removed before drawing the new ones - so repeated calls (e.g. on
            every redraw) don't accumulate stale circles.

    Returns:
        List of the newly-added circle/label artists; pass this back in as
        `old_artists` on the next call.
    """
    # A mistaken/misunderstood-units entry (e.g. 1 instead of 0.01 1/A per
    # pixel) can make max_r_recip huge - without a hard cap this would try
    # to create thousands of patches and one or more full-figure redraws for
    # them on every keystroke, which is what actually froze the app rather
    # than the value itself being "wrong": the cap keeps the ring spacing
    # concept intact (still one ring per 1/A) while bounding the worst case.
    MAX_RINGS = 20

    if old_artists:
        for artist in old_artists:
            try:
                artist.remove()
            except Exception:
                pass
    new_artists = []
    try:
        scale_recip = float(scale_recip)
    except (TypeError, ValueError):
        return new_artists
    if not scale_recip or scale_recip <= 0:
        return new_artists

    h, w = shape[:2]
    cx, cy = center if center is not None else (w / 2, h / 2)
    max_r_px = min(cx, cy, w - cx, h - cy)
    max_r_recip = max_r_px * scale_recip

    # Top-right point of each ring (45 deg up from horizontal) - y grows
    # downward in image/pixel coordinates, so "up" is a negative y offset.
    label_dx, label_dy = np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))
    label_outline = [patheffects.withStroke(linewidth=2, foreground='black')]

    def _add_ring_label(r_px, value, color):
        label = ax.text(cx + r_px * label_dx, cy - r_px * label_dy,
                        f'{value:g}' r' $\AA^{-1}$',
                        color=color, fontsize=7, ha='center', va='center',
                        path_effects=label_outline)
        new_artists.append(label)

    n = 1
    while n <= max_r_recip and len(new_artists) < MAX_RINGS:
        r_px = n / scale_recip
        circle = patches.Circle((cx, cy), r_px, fill=False, edgecolor='cyan',
                                linestyle='--', linewidth=1.2, alpha=0.6)
        ax.add_patch(circle)
        new_artists.append(circle)
        _add_ring_label(r_px, n, 'cyan')
        n += 1

    if not new_artists and max_r_recip > 0:
        # Not even the first 1/A ring fits - fall back to a finer 0.5/A
        # ring, in a different (dotted, white) style so it reads as
        # distinct from the (absent) whole-1/A rings.
        r_px = 0.5 / scale_recip
        if r_px <= max_r_px:
            circle = patches.Circle((cx, cy), r_px, fill=False, edgecolor='white',
                                    linestyle=':', linewidth=1.4, alpha=0.6)
            ax.add_patch(circle)
            new_artists.append(circle)
            _add_ring_label(r_px, 0.5, 'white')

    return new_artists

def get_sum_dp(fn, dtype=None, scanSize=None, dwellTime=1, roi=None, logger=None,
               n_threads=None):
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

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    if dtype is None:
        dtype = os.path.splitext(fn)[-1]
    if dtype == '.tpx3':
        return get_dp(fn, scanSize, roi=roi, dwellTime=dwellTime, logger=logger,
                      n_threads=n_threads)

    s = load_signal(fn, dtype=dtype, scanSize=scanSize, roi=roi, lazy=True, logger=logger)
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
                             n_threads=None):
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
                                          logger=logger, n_threads=n_threads)

    s = load_signal(fn, dtype=dtype, scanSize=scanSize, lazy=True, logger=logger)
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
    nav_img = load_hdf5(fn, scanSize=scanSize, sum_dp=True, logger=logger)
    return nav_img

def calculate_nav_img(fn, dtype=None, scanSize=None, dwellTime=1, logger=None,
                      n_threads=None):
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
        nav_img = load_signal(fn, dtype=dtype, scanSize=scanSize, sum_dp=True, logger=logger)
    elif dtype == '.tpx3':
        nav_img = calculate_nav_img_tpx3(fn, scanSize, dwellTime, logger=logger,
                                         n_threads=n_threads)
    elif dtype == '.hdf5':
        nav_img = calculate_nav_img_hdf5(fn, scanSize, logger=logger)
    elif dtype in ['.zspy', '.hspy']:
        nav_img = load_signal(fn, dtype=dtype, scanSize=scanSize, sum_dp=True, logger=logger)
    elif dtype in ['.dm2', '.dm3', '.tif', '.tiff']:
        nav_img = hs.load(fn).data
    return nav_img
    
def convert_to_8bit(s, plow=1.0, phigh=99.0):
    """Normalise each frame of a HyperSpy signal to 8-bit (0-255).

    Contrast is stretched between the `plow`/`phigh` percentiles of each
    frame, not its raw min/max - a handful of hot/dead pixels (common on
    electron-counting detectors) would otherwise dominate a min/max range
    and wash out everything else, which is a real cause of SAM2 producing
    inaccurate masks on effectively-flat-looking 8-bit input.

    Args:
        s: HyperSpy Signal2D with arbitrary numeric dtype.
        plow: Lower percentile (per frame) mapped to 0.
        phigh: Upper percentile (per frame) mapped to 255.

    Returns:
        A deep copy of `s` with data converted to uint8, each frame independently normalised.
    """
    data = s.data.astype(np.float32)
    # Normalise per-frame using broadcasting (vectorised, no Python loop)
    axis = tuple(range(data.ndim - 2, data.ndim))
    los = np.percentile(data, plow, axis=axis, keepdims=True)
    his = np.percentile(data, phigh, axis=axis, keepdims=True)
    data = np.clip((data - los) / (his - los + 1e-8), 0.0, 1.0)
    data = (data * 255.0).astype(np.uint8)
    s_8bit = s._deepcopy_with_new_data(data)
    return s_8bit

def convert_img_to_8bit(img, plow=1.0, phigh=99.0):
    """Normalise a single 2-D numpy array to uint8, contrast-stretched
    between the `plow`/`phigh` percentiles rather than the raw min/max -
    see `convert_to_8bit` for why (outlier pixels otherwise dominate the
    range and starve SAM2's input image of real contrast).

    Args:
        img: 2-D numpy array of any numeric dtype.
        plow: Lower percentile mapped to 0.
        phigh: Upper percentile mapped to 255.

    Returns:
        numpy.ndarray of dtype uint8.
    """
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [plow, phigh])
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

def convert_to_rgb(imgs):
    """Convert a stack of grayscale images to an RGBA/RGB array suitable for video encoding.

    Args:
        imgs: numpy.ndarray of shape (N, W, H).

    Returns:
        numpy.ndarray of shape (W, H, 3, N) with the same dtype as `imgs`.
    """
    l, w, h = imgs.shape
    # imgs_8bit = convert_to_8bit(imgs)
    imgs_rgb = np.zeros((w,h,3,l), dtype=imgs.dtype)
    for i, img in enumerate(imgs):
        imgs_rgb[:,:,:,i] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return imgs_rgb

def create_tracking_result(s, rois):
    """Overlay ROI bounding boxes and index labels on every frame of a signal.

    Args:
        s: HyperSpy Signal2D (N, H, W) serving as the background images.
        rois: List of ROI arrays, each of shape (N, 4) with columns (x, y, w, h).

    Returns:
        HyperSpy Signal2D with white rectangles and numeric labels drawn on each frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.25
    font_thickness = 1
    font_color = (255, 255, 255)  # BGR color format (blue, green, red)
    s_roi = np.zeros(s.data.shape, dtype=np.uint16)
    for i_img, img in enumerate(tqdm(s.data)):
        img = img.copy()  # lightweight copy instead of deepcopy
        for i_r, roi in enumerate(rois):
            (x,y,w,h) = roi[i_img]
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 1)
            cv2.putText(img, str(i_r+1), (x-3, y-3), font, font_scale, font_color, font_thickness)
        s_roi[i_img] = img
    s_roi = hs.signals.Signals2D(s_roi)
    return s_roi

def create_array_from_dissimilar_imgs(imgs, mode='constant', signal=True):
    """Pad a list of differently-shaped 2-D images into a uniform 3-D array.

    Args:
        imgs: List of 2-D numpy arrays with potentially different (H, W) shapes.
        mode: `numpy.pad` mode for fill values. Default is `'constant'` (zero-padding).
        signal: If True, wrap the result in a HyperSpy Signal2D. Default is True.

    Returns:
        numpy.ndarray or HyperSpy Signal2D of shape (N, H_max, W_max) with dtype uint8.
    """
    shapes = [[img.shape[0], img.shape[1]] for img in imgs]
    shapes = np.array(shapes)
    w = shapes[:,0].max()
    h = shapes[:,1].max()
    new_arr = np.zeros((len(imgs), w, h), dtype=np.uint8)
    for i, img in enumerate(imgs):
        w2 = w - img.shape[0]
        h2 = h - img.shape[1]
        new_img = np.pad(img, pad_width=[(0,w2), (0,h2)], mode=mode)
        new_arr[i] = new_img
        # new_arr[i][:img.shape[0], :img.shape[1]] = img
    if signal:
        new_arr = hs.signals.Signal2D(new_arr)
    return new_arr

def create_frames(pathSave, frames):
    """Save each frame in `frames` as a numbered TIFF file under `pathSave`."""
    if frames is None:
        return
    if os.path.isdir(pathSave):
        [os.remove(os.path.join(pathSave, fn)) for fn in os.listdir(pathSave)]
    else:
        os.mkdir(pathSave)
    for i, fr in enumerate(tqdm(frames)):
        tifffile.imwrite(os.path.join(pathSave, f'{i+1:04d}.tif'), fr)

def create_clip_dp(fn, s, scale=None, dpi=150, fps=5, vmin=None, vmax=None, cmap='inferno',
                   logger=None):
    """Save an animated video of diffraction patterns with optional reciprocal-space scale bar.

    Pipes raw RGB frames directly to an FFmpeg subprocess for fast encoding.
    Falls back to a GIF via FuncAnimation when FFmpeg is not on PATH.

    Args:
        fn: Output file path without extension (`.mp4` appended, `.gif` as fallback).
        s: Array-like of shape (N, H, W) containing diffraction pattern frames.
        scale: Reciprocal-space pixel size in 1/nm. If None, no scale bar is drawn.
        dpi: Dots per inch for the matplotlib figure (controls output resolution).
        fps: Frames per second. Auto-set to max(1, N // 20) when None.
        vmin: Minimum value for the colour normalisation.
        vmax: Maximum value for the colour normalisation.
        cmap: Matplotlib colormap name.
        logger: Optional logger to report status through instead of stdout.
    """
    _log_or_print(logger, 'Making DP clip...')
    plt.ioff()
    fig, ax = plt.subplots(dpi=dpi)
    img_plot = ax.imshow(s[0], cmap=cmap, norm=mcolors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=1))
    fig.tight_layout()
    if scale is not None:
        scalebar = ReadableScaleBar(scale * 10, '1/nm', dimension='si-length-reciprocal',
                            location='lower left', box_alpha=0, color='w',
                            scale_formatter=lambda value, unit: f'{value / 10}' r' $\AA^{-1}$',
                            fixed_value=5)
        ax.add_artist(scalebar)
    if fps is None:
        fps = max(1, len(s) // 20)
    path_ffmpeg = shutil.which('ffmpeg')
    if path_ffmpeg:
        fig.canvas.draw()
        W, H = fig.canvas.get_width_height()
        cmd = [
            path_ffmpeg, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
            '-i', '-', '-an', '-vcodec', 'libx264',
            '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
            '-pix_fmt', 'yuv420p', '-crf', '18', fn + '.mp4',
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr_no in tqdm(range(len(s))):
            img_plot.set_data(s[fr_no])
            img_plot.set_clim(vmin=s[fr_no].min(), vmax=s[fr_no].max())
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            proc.stdin.write(buf[..., :3].tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        def _update_dp(fr_no):
            img_plot.set_data(s[fr_no])
            img_plot.set_clim(vmin=s[fr_no].min(), vmax=s[fr_no].max())
            return (img_plot,)
        ani = FuncAnimation(fig, _update_dp, frames=range(s.shape[0]), blit=True)
        ani.save(fn + '.gif', fps=fps)
        _log_or_print(logger, 'No ffmpeg found, saved as GIF')
    plt.close(fig)
    plt.ion()
    _log_or_print(logger, 'DP clip is created!')

def create_clip_tracking(fn, imgs, rois=None, scale=None, dpi=150,
                         fps=5, duration=None, cmap='viridis', logger=None):
    """Save an animated video of navigation images with an optional tracking ROI overlay.

    Pipes raw RGB frames directly to an FFmpeg subprocess for fast encoding.
    Falls back to a GIF via FuncAnimation when FFmpeg is not on PATH.

    Args:
        fn: Output file path without extension (`.mp4` appended, `.gif` as fallback).
        imgs: numpy.ndarray of shape (N, H, W) with navigation image frames.
        rois: Optional array/list of shape (N, 4) with (x, y, w, h) ROI per frame.
        scale: Real-space pixel size in nm. If None, no scale bar is drawn.
        dpi: Dots per inch for the matplotlib figure (controls output resolution).
        fps: Frames per second. Auto-set from `duration` or max(1, N // 20) when None.
        duration: Target clip duration in seconds used to derive `fps` when provided.
        cmap: Matplotlib colormap name.
        logger: Optional logger to report status through instead of stdout.
    """
    _log_or_print(logger, 'Making tracking clip...')
    plt.ioff()
    fig, ax = plt.subplots(dpi=dpi)
    img_plot = ax.imshow(imgs[0], cmap=cmap)
    if scale is not None:
        scalebar = ReadableScaleBar(scale, 'nm', dimension='si-length',
                            location='lower left', box_alpha=0, color='w')
        ax.add_artist(scalebar)
    active_rect = [None]
    if rois is not None:
        x, y, w, h = rois[0]
        active_rect[0] = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(active_rect[0])
    fig.tight_layout()
    if fps is None:
        denom = duration if duration else 20
        fps = max(1, len(imgs) // denom)

    def _update_tr(fr_no):
        img_plot.set_data(imgs[fr_no])
        img_plot.set_clim(vmin=imgs[fr_no].min(), vmax=imgs[fr_no].max())
        if rois is not None:
            if active_rect[0] is not None:
                active_rect[0].remove()
            x, y, w, h = rois[fr_no]
            active_rect[0] = patches.Rectangle((x, y), w, h, linewidth=1,
                                               edgecolor='r', facecolor='none')
            ax.add_patch(active_rect[0])

    path_ffmpeg = shutil.which('ffmpeg')
    if path_ffmpeg:
        fig.canvas.draw()
        W, H = fig.canvas.get_width_height()
        cmd = [
            path_ffmpeg, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
            '-i', '-', '-an', '-vcodec', 'libx264',
            '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
            '-pix_fmt', 'yuv420p', '-crf', '18', fn + '.mp4',
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr_no in tqdm(range(len(imgs))):
            _update_tr(fr_no)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            proc.stdin.write(buf[..., :3].tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        def _anim_tr(fr_no):
            _update_tr(fr_no)
            return (img_plot,)
        ani = FuncAnimation(fig, _anim_tr, frames=range(imgs.shape[0]), blit=False)
        ani.save(fn + '.gif', fps=fps)
        _log_or_print(logger, 'No ffmpeg found, saved as GIF')
    plt.close(fig)
    plt.ion()
    _log_or_print(logger, 'Tracking clip is created!')

def create_clip_tracking_with_mask(fn, imgs, masks, obj_id=1, scale=None,
                                   dpi=150, fps=5, cmap='viridis', logger=None):
    """Save an animated video of navigation images with a SAM2 segmentation mask overlay.

    Pipes raw RGB frames directly to an FFmpeg subprocess for fast encoding.
    Falls back to a GIF via FuncAnimation when FFmpeg is not on PATH.

    Args:
        fn: Output file path without extension (`.mp4` appended, `.gif` as fallback).
        imgs: numpy.ndarray of shape (N, H, W) with navigation image frames.
        masks: Boolean array of shape (N, H, W) with the segmentation mask per frame.
        obj_id: Object ID used to select the overlay colour from the `tab10` colormap.
        scale: Real-space pixel size in nm. If None, no scale bar is drawn.
        dpi: Dots per inch for the matplotlib figure (controls output resolution).
        fps: Frames per second. Auto-set to max(1, N // 20) when None.
        cmap: Matplotlib colormap name for the background image.
        logger: Optional logger to report status through instead of stdout.
    """
    _log_or_print(logger, 'Making tracking clip...')
    plt.ioff()
    fig, ax = plt.subplots(dpi=dpi)
    img_plot = ax.imshow(imgs[0], cmap=cmap)
    img_mask_plot = ax.imshow(np.zeros((*imgs[0].shape, 4)))
    if scale is not None:
        scalebar = ReadableScaleBar(scale, 'nm', dimension='si-length',
                            location='lower left', box_alpha=0, color='w')
        ax.add_artist(scalebar)
    fig.tight_layout()
    cmap_tab10 = plt.get_cmap('tab10')
    color = np.array([*cmap_tab10(obj_id)[:3], 0.6])

    def _make_mask_rgba(mask):
        h, w = mask.shape[-2:]
        return mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    def _update_mask(fr_no):
        img_plot.set_data(imgs[fr_no])
        img_plot.set_clim(vmin=imgs[fr_no].min(), vmax=imgs[fr_no].max())
        img_mask_plot.set_data(_make_mask_rgba(masks[fr_no]))

    img_mask_plot.set_data(_make_mask_rgba(masks[0]))
    if fps is None:
        fps = max(1, len(imgs) // 20)
    path_ffmpeg = shutil.which('ffmpeg')
    if path_ffmpeg:
        fig.canvas.draw()
        W, H = fig.canvas.get_width_height()
        cmd = [
            path_ffmpeg, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
            '-i', '-', '-an', '-vcodec', 'libx264',
            '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
            '-pix_fmt', 'yuv420p', '-crf', '18', fn + '.mp4',
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr_no in tqdm(range(len(imgs))):
            _update_mask(fr_no)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            proc.stdin.write(buf[..., :3].tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        def _anim_mask(fr_no):
            _update_mask(fr_no)
            return (img_plot,)
        ani = FuncAnimation(fig, _anim_mask, frames=range(imgs.shape[0]), blit=True)
        ani.save(fn + '.gif', fps=fps)
        _log_or_print(logger, 'No ffmpeg found, saved as GIF')
    plt.close(fig)
    plt.ion()
    _log_or_print(logger, 'Tracking clip is created!')
# =============================================================================
# if __name__ == '__main__':
#     pass
# =============================================================================
