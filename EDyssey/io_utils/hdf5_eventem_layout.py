# -*- coding: utf-8 -*-
"""Shared low-level read helpers for eventem's raw '.hdf5_eventem' `f['4D']`
dataset, handling both the current native-4D storage layout and the older
flat/1-D layout that some pre-4D-native eventem exports still use (where
`f['4D']` is a plain 1-D dataset and the true (ny, nx, det_y, det_x) shape
lives only in `f['shape']`).

Used by loaders.py/nav_image.py (the live app's load/get_dp/nav-image code
path) and workers/worker_extract_frame.py (ROI Tracker/SAM2's per-frame mask
extraction) so all call sites share one implementation of the 1-D-vs-4-D
detection and the chunk-aware fast-read logic, instead of each re-deriving
it on its own (as loaders.py/nav_image.py didn't do at all until now, and
worker_extract_frame.py did ad hoc).

`f['shape']` (written as a 4-element [d0, d1, det_y, det_x] array) is
present and authoritative for both storage layouts - it's what every
function below trusts, never `f['4D'].shape` directly, since that's only
meaningful for the native-4D layout. d0/d1 are the scan's row/column axes
(matching the app's own `roi=(x, y, w, h)` convention elsewhere: d0 <-> y,
d1 <-> x) - confirmed against real files by summing a reshaped row of
frames and comparing it to that file's own precomputed `dose_image`.

Every real '.hdf5_eventem' file seen so far (flat and native-4D alike)
stores `f['4D']` gzip-compressed in ~1GB chunks that each span many scan
positions but the *whole* detector frame and the *whole* scan-row width -
i.e. chunked only along the leading scan-row axis. That makes reading even
a small ROI/mask selection expensive two different ways, both addressed
below: (1) HDF5 must fully decompress a touched chunk to read any part of
it, so several separate reads touching the same chunk each pay its full
decompression cost again unless requested positions are grouped by chunk
first (`_flat_chunk_frames`/`_native_chunk_frames`); (2) HDF5's own gzip
filter uses stock zlib, which measured ~6-7x slower than decompressing the
same chunk's raw bytes directly via libdeflate (`_read_chunk_libdeflate_flat`/
`_read_chunk_libdeflate_native`, via `dataset.id.read_direct_chunk`) - e.g.
0.27-0.35s vs ~2s per real 1GB chunk. Combined, a 45x33-position ROI that
took ~92s the naive way (h5py, one read per row) dropped to ~2.9s.

On top of that, `full_scan_reduce`/`_read_frames_grouped` can *optionally*
decompress+reduce several chunks concurrently (`_run_processes`) - each
worker is a separate OS *process* (`_worker_reduce_chunk`/
`_worker_select_chunk`, via `ProcessPoolExecutor`), not a thread, each
opening its own independent `h5py.File` handle on the same path. This
was deliberate, not the first thing tried: a `ThreadPoolExecutor`-based
version (workers sharing the caller's already-open file handle, only the
GIL-releasing decompression/reduction actually running concurrently) was
built and benchmarked first - and measured real parallelism (~4.3x at 8
workers on a 16-core machine) - but concurrent
`dataset.id.read_direct_chunk()` calls from multiple *threads* were then
found, on real data, to corrupt this dataset's HDF5-level state in a way
that silently produced *wrong* results on a later call, not just crashes -
reproduced directly: a first parallel call completed and "looked" fine,
but a second call on the same still-open handle returned different
numbers, and even a brand-new `h5py.File()` open in the same *process*
afterward was also affected. Separate processes don't share that state at
all (each has its own independent copy of the HDF5 library with nothing
in common to corrupt), which is why this module uses them instead, at the
cost of some process-startup and IPC overhead the thread version didn't
have (paid once per call, not per chunk - see `_run_processes`) and of
returning a small pre-*reduced* result per chunk from each worker rather
than the whole decompressed block (crossing a process boundary, unlike a
thread, means anything returned has to be pickled).

How many worker processes run at once is resolved per call
(`_resolve_max_workers`) from the host's actual CPU count and *currently
available* RAM (each worker holds roughly one chunk's decompressed size),
not a fixed number - this app has no global cap on how many independent
file-processing batches can be running across its tabs at once (each tab
is independent, and tabs can be duplicated - see
EDyssey_MainWindow.py's duplicate_current_tab), so a live probe is what
lets this adapt to whatever else is already running instead of assuming
it has the machine to itself. `max_workers=1` is forced at any call site
that's already inside its own process pool (see workers/worker_nav_img.py,
workers/worker_extract_frame.py) - otherwise each of that pool's worker
processes would also spin up its own inner process pool here, multiplying
concurrent chunk buffers by the pool's own size (and nesting process pools
is its own source of fragility, especially on Windows' spawn-based
multiprocessing, so this is avoided entirely rather than relied upon to
behave).
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import dask.array as da
import h5py

try:
    import deflate as _libdeflate  # python bindings for libdeflate
except ImportError:
    _libdeflate = None

# Fraction of *currently available* RAM (measured fresh per call, not this
# machine's total) a dynamically-sized worker count is allowed to claim for
# concurrent ~1-chunk-sized buffers - conservative because this app has no
# global cap on how many independent batches/tabs can be running at once
# (see _resolve_max_workers), so headroom needs to be left for whatever else
# is already running, not just for this one call.
_RAM_SAFETY_FRACTION = 0.5


def _resolve_max_workers(max_workers, chunk_bytes):
    """How many on-disk chunks to decompress+reduce concurrently, as
    separate worker *processes* - see the module docstring for why
    processes, not threads.

    An explicit `max_workers` is always honored as-is - this is also how a
    caller already running inside its own process pool forces this down to
    1 (see workers/worker_nav_img.py, workers/worker_extract_frame.py):
    without that, each of that pool's own worker processes would spin up
    its own *inner* process pool here too, multiplying concurrent chunk
    buffers by the pool's own worker count.

    Otherwise sized per-machine: capped by `os.cpu_count() - 1` (always
    leave one core free) and by how many `chunk_bytes`-sized buffers fit in
    `_RAM_SAFETY_FRACTION` of *currently* available RAM (via psutil, read
    fresh on every call, falling back to the CPU-only cap if psutil isn't
    available). A live measurement is the right tool here specifically
    because this app has no global concurrency governor - several
    independent tabs (including user-duplicated ones) can each be running
    their own file-processing batch at once with no coordination between
    them, so a fixed worker count would have no way to react to memory
    another tab's batch is already using, while a fresh probe does.
    """
    if max_workers is not None:
        return max(1, int(max_workers))
    cpu_cap = max(1, (os.cpu_count() or 2) - 1)
    try:
        import psutil
        available = psutil.virtual_memory().available
        mem_cap = max(1, int(available * _RAM_SAFETY_FRACTION) // max(1, chunk_bytes))
    except Exception:
        mem_cap = cpu_cap  # psutil unavailable - fall back to the CPU-only cap
    return max(1, min(cpu_cap, mem_cap))


def _read_chunk(dset, flat_storage, shape, chunk_id):
    """One on-disk chunk of `dset` (whose true shape is `shape`, from
    `get_shape` - see the module docstring for why never `dset.shape`
    directly), as a (n, det_y, det_x) array - the libdeflate fast path if
    usable for this chunk, a plain h5py slice read otherwise (a short/
    partial edge chunk, libdeflate unavailable, wrong compression, or an
    unexpected per-chunk filter). Safe to call this way (fast-then-
    fallback, no special handling) from any single thread of execution - a
    worker *process* (see the module docstring) counts as one, so this is
    what both the sequential path and each parallel worker use."""
    if flat_storage:
        det_shape = shape[-2:]
        block = _read_chunk_libdeflate_flat(dset, det_shape, chunk_id)
        if block is None:
            block = _read_chunk_plain_flat(dset, det_shape, chunk_id)
    else:
        block = _read_chunk_libdeflate_native(dset, shape, chunk_id)
        if block is None:
            block = _read_chunk_plain_native(dset, shape, chunk_id)
    return block


def _worker_reduce_chunk(fn, flat_storage, chunk_id, mode, det_mask):
    """`full_scan_reduce`'s process-pool worker: opens its own `h5py.File`
    handle on `fn` (required - see the module docstring), reads one chunk,
    and returns its per-scan-position reduction - small enough to pickle
    back to the caller cheaply, unlike the whole decompressed chunk. Must
    stay a module-level function (picklable) for `ProcessPoolExecutor`.
    """
    with h5py.File(fn, 'r') as f:
        shape = get_shape(f)
        dset = f['4D']
        block = _read_chunk(dset, flat_storage, shape, chunk_id)
    flat_block = block.reshape(block.shape[0], -1)
    return _reduce_frames(flat_block, mode, det_mask=det_mask)


def _worker_select_chunk(fn, flat_storage, chunk_id, chunk_frames, idx, reduce):
    """`_read_frames_grouped`'s process-pool worker: opens its own
    `h5py.File` handle on `fn`, reads one chunk, and returns the
    scan positions in `idx` that fall in it - summed (`reduce='sum'`) or
    gathered (`reduce='none'`) - see `_worker_reduce_chunk`'s docstring for
    why a separate handle per worker, and `_select_and_reduce` (in
    `_read_frames_grouped`) for the equivalent sequential-path logic this
    mirrors."""
    with h5py.File(fn, 'r') as f:
        shape = get_shape(f)
        dset = f['4D']
        block = _read_chunk(dset, flat_storage, shape, chunk_id)
    c_start = chunk_id * chunk_frames
    idx = np.asarray(idx)
    sel = np.flatnonzero((idx // chunk_frames) == chunk_id)
    local = idx[sel] - c_start
    return sel, (block[local].sum(axis=0) if reduce == 'sum' else block[local])


def _run_processes(worker_fn, tasks, max_workers):
    """Run `worker_fn(*args)` for each `(key, args)` pair in `tasks`,
    yielding `(key, result)` pairs as they become available - `key` lets
    the caller identify which task a result belongs to, since results
    arrive in completion order, not task order.

    `max_workers <= 1` (or a single task) runs strictly sequentially in the
    calling process - no pool-creation overhead (measured ~0.6s to spin up
    a fresh multi-worker pool on Windows, dominated by each new process
    importing numpy/h5py/deflate - trivial next to the seconds of real work
    a worthwhile parallel call does, but not worth paying for a call with
    only one or two chunks). Otherwise uses a `ProcessPoolExecutor` - see
    the module docstring for why processes, not threads.

    On a worker exception, cancels the remaining not-yet-started futures
    before re-raising, so a doomed call doesn't keep paying for chunks
    already known to be discarded.
    """
    tasks = list(tasks)
    if max_workers <= 1 or len(tasks) <= 1:
        for key, args in tasks:
            yield key, worker_fn(*args)
        return
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker_fn, *args): key for key, args in tasks}
        try:
            for fut in as_completed(futures):
                yield futures[fut], fut.result()
        except Exception:
            for fut in futures:
                fut.cancel()
            raise


def get_shape(f):
    """True (d0, d1, det_y, det_x) shape of `f['4D']`, from `f['shape']`."""
    return tuple(int(x) for x in f['shape'][:])


def is_flat(f):
    """True if `f['4D']` is the older flat/1-D storage layout, False if
    it's the current native-4D layout."""
    return f['4D'].ndim == 1


def frame_size(shape):
    """Number of elements per diffraction pattern (det_y * det_x)."""
    return int(shape[-2] * shape[-1])


def as_dask(f):
    """Dask array view of `f['4D']`, correctly shaped as (d0, d1, det_y,
    det_x) regardless of whether the file uses the native-4D or the flat/
    1-D storage layout - the shape-aware counterpart of unconditionally
    doing `da.from_array(f['4D'], chunks=f['4D'].chunks)`, which only works
    for the native-4D layout."""
    dset = f['4D']
    s = da.from_array(dset, chunks=dset.chunks)
    if is_flat(f):
        s = s.reshape(get_shape(f))
    return s


def _contiguous_runs(sorted_unique_indices):
    """Group sorted unique integer indices into (start, stop) runs of
    consecutive values (stop exclusive) - so a selection of scan positions
    can be read as a handful of contiguous hyperslabs instead of one h5py
    read per individual position."""
    idx = np.asarray(sorted_unique_indices)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) != 1)[0] + 1
    return [(int(run[0]), int(run[-1]) + 1) for run in np.split(idx, breaks)]


def _decompress_chunk(dset, offset, n_elems):
    """Fetch one on-disk HDF5 chunk of `dset` via libdeflate instead of
    HDF5's own (stock zlib) gzip pipeline, as a flat 1-D array of
    `dset.dtype`, `n_elems` long - or None if that isn't safely possible
    right now (libdeflate not installed, `dset` not gzip-compressed, or
    HDF5 applied some other/additional filter to this particular chunk,
    e.g. an incompressible chunk stored raw) - callers fall back to a plain
    h5py read in that case.

    `offset`: the chunk's starting coordinate tuple (rank matching `dset`),
    as `dataset.id.read_direct_chunk` expects - must be chunk-aligned.
    `n_elems`: the chunk's exact element count (`dset.chunks` product) -
    only correct for a *full* chunk, so callers must only call this for one
    (a short/partial edge chunk needs the normal h5py path instead, since
    this fast path needs to know the decompressed size exactly upfront).
    """
    if _libdeflate is None or dset.compression != 'gzip':
        return None
    try:
        filter_mask, raw = dset.id.read_direct_chunk(offset)
        if filter_mask != 0:
            return None
        nbytes = n_elems * dset.dtype.itemsize
        buf = _libdeflate.zlib_decompress(raw, nbytes)
        return np.frombuffer(buf, dtype=dset.dtype)
    except Exception:
        return None  # anything unexpected about this chunk - safe fallback


def _flat_chunk_frames(dset, fs):
    """Number of whole scan positions per on-disk HDF5 chunk of a flat/1-D
    `dset`, or None if that's not a well-defined/usable number (the dataset
    isn't chunked, or its chunk size isn't an exact multiple of one
    frame)."""
    if dset.chunks is None:
        return None
    chunk_elems = dset.chunks[0]
    if chunk_elems % fs != 0:
        return None
    return chunk_elems // fs


def _native_chunk_frames(dset, shape):
    """Number of whole scan positions per on-disk HDF5 chunk of a native-4D
    `dset`, or None if the dataset isn't chunked *only* along its leading
    (scan-row) axis - i.e. every other axis's chunk size doesn't cover that
    axis's full extent (d1, det_y, det_x), in which case a chunk doesn't
    correspond to a whole number of complete scan rows and isn't handled by
    the fast path below (every real file seen so far chunks this way, but
    it isn't assumed in general)."""
    if dset.chunks is None or len(dset.chunks) != 4:
        return None
    if tuple(dset.chunks[1:]) != tuple(shape[1:]):
        return None
    return dset.chunks[0] * shape[1]


def _read_chunk_libdeflate_flat(dset, det_shape, chunk_id):
    """One on-disk chunk of a flat/1-D `dset` via *only* the libdeflate
    fast path (`_decompress_chunk`), as a (n, det_y, det_x) array - or None
    if that path isn't usable for this chunk (not a full chunk, libdeflate
    unavailable, wrong compression, or an unexpected filter on this
    particular chunk) - callers fall back to `_read_chunk_plain_flat` in
    that case (see `_read_chunk`, which does exactly that)."""
    chunk_elems = dset.chunks[0]
    c_start = chunk_id * chunk_elems
    c_stop = min(c_start + chunk_elems, dset.shape[0])
    if c_stop - c_start != chunk_elems:
        return None
    flat = _decompress_chunk(dset, (c_start,), chunk_elems)
    return None if flat is None else flat.reshape(-1, *det_shape)


def _read_chunk_plain_flat(dset, det_shape, chunk_id):
    """Plain h5py slice read of one on-disk chunk of a flat/1-D `dset` -
    the fallback `_read_chunk` uses when `_read_chunk_libdeflate_flat`
    can't get this particular chunk."""
    chunk_elems = dset.chunks[0]
    c_start = chunk_id * chunk_elems
    c_stop = min(c_start + chunk_elems, dset.shape[0])
    return dset[c_start:c_stop].reshape(-1, *det_shape)


def _read_chunk_libdeflate_native(dset, shape, chunk_id):
    """One on-disk chunk (`dset.chunks[0]` full scan rows) of a native-4D
    `dset` via *only* the libdeflate fast path - or None if unusable for
    this chunk. See `_read_chunk_libdeflate_flat`'s docstring - the same
    fast/fallback split applies here."""
    c0 = dset.chunks[0]
    row_start = chunk_id * c0
    row_stop = min(row_start + c0, shape[0])
    if row_stop - row_start != c0:
        return None
    flat = _decompress_chunk(dset, (row_start, 0, 0, 0), int(np.prod(dset.chunks)))
    return None if flat is None else flat.reshape(-1, *shape[-2:])


def _read_chunk_plain_native(dset, shape, chunk_id):
    """Plain h5py slice read of one on-disk chunk of a native-4D `dset` -
    the fallback `_read_chunk` uses when `_read_chunk_libdeflate_native`
    can't get this particular chunk."""
    c0 = dset.chunks[0]
    row_start = chunk_id * c0
    row_stop = min(row_start + c0, shape[0])
    return dset[row_start:row_stop].reshape(-1, *shape[-2:])


def _chunk_reader(dset, shape, flat_storage):
    """Return (chunk_frames, read_chunk) for `dset` - `chunk_frames` is the
    number of whole scan positions per on-disk chunk (None if chunked
    reading isn't usable here, e.g. uncompressed or irregularly chunked),
    and `read_chunk(chunk_id)` returns that chunk as a (n, det_y, det_x)
    array (the libdeflate fast path, falling back to a plain h5py slice
    read when needed - see `_read_chunk`). Picks the flat-storage or
    native-4D variant."""
    if dset.compression is None:
        return None, None
    if flat_storage:
        chunk_frames = _flat_chunk_frames(dset, frame_size(shape))
        if chunk_frames is None:
            return None, None
        return chunk_frames, lambda cid: _read_chunk(dset, True, shape, cid)
    chunk_frames = _native_chunk_frames(dset, shape)
    if chunk_frames is None:
        return None, None
    return chunk_frames, lambda cid: _read_chunk(dset, False, shape, cid)


def _read_frames_grouped(shape, frame_indices, reduce, chunk_frames, read_chunk, dtype,
                          fn, flat_storage, max_workers=1):
    """Sum (`reduce='sum'`) or gather in order (`reduce='none'`) the scan
    positions in `frame_indices` (flat indices into the (d0*d1)-length
    leading axis, row-major), reading each on-disk chunk of `chunk_frames`
    positions exactly once regardless of how many requested positions fall
    in it - this is what keeps a scattered or ROI selection fast on a
    large-chunked compressed dataset (see the module docstring): several
    separate per-position/per-row reads that each touch the same chunk
    would otherwise each pay that chunk's full decompression cost again.

    `max_workers` (already resolved to a plain int by the caller - see
    `_resolve_max_workers`) > 1 processes that many chunks concurrently as
    separate worker processes (`_worker_select_chunk`, via
    `_run_processes`) - each opening its own `h5py.File(fn)` handle, hence
    needing `fn`/`flat_storage` passed through here even though the
    sequential path (`read_chunk`, using the caller's already-open handle)
    doesn't need them.
    """
    det_shape = shape[-2:]
    idx = np.asarray(frame_indices)
    if reduce == 'sum':
        out = np.zeros(det_shape, dtype=np.float64)
    else:
        out = np.empty((idx.size, *det_shape), dtype=dtype)

    chunk_id_per_idx = idx // chunk_frames
    unique_chunk_ids = np.unique(chunk_id_per_idx)

    def _select_and_reduce(chunk_id, block):
        c_start = int(chunk_id) * chunk_frames
        sel = np.flatnonzero(chunk_id_per_idx == chunk_id)
        local = idx[sel] - c_start
        return sel, (block[local].sum(axis=0) if reduce == 'sum' else block[local])

    if max_workers <= 1 or len(unique_chunk_ids) <= 1:
        for chunk_id in unique_chunk_ids:
            sel, partial = _select_and_reduce(chunk_id, read_chunk(int(chunk_id)))
            if reduce == 'sum':
                out += partial
            else:
                out[sel] = partial
        return out

    tasks = [(cid, (fn, flat_storage, int(cid), chunk_frames, idx, reduce)) for cid in unique_chunk_ids]
    for _chunk_id, (sel, partial) in _run_processes(_worker_select_chunk, tasks, max_workers):
        if reduce == 'sum':
            out += partial
        else:
            out[sel] = partial
    return out


def _read_frames_ungrouped(dset, shape, frame_indices, reduce):
    """Plain contiguous-run h5py read (no chunk grouping, no dask) of
    specific scan positions from a flat/1-D `dset` - used when chunked fast
    reading isn't applicable (uncompressed dataset). Each scan position
    occupies `frame_size(shape)` contiguous elements in the flat dataset,
    so consecutive selected positions are merged into one read each
    (`_contiguous_runs`); this is the minimal possible I/O when there's no
    compression cost to economize on.
    """
    fs = frame_size(shape)
    det_shape = shape[-2:]
    idx = np.asarray(frame_indices)
    if reduce == 'sum':
        out = np.zeros(det_shape, dtype=np.float64)
    else:
        out = np.empty((idx.size, *det_shape), dtype=dset.dtype)
    for start, stop in _contiguous_runs(np.unique(idx)):
        block = dset[start * fs:stop * fs].reshape(stop - start, *det_shape)
        if reduce == 'sum':
            out += block.sum(axis=0)
        else:
            sel = np.flatnonzero((idx >= start) & (idx < stop))
            out[sel] = block[idx[sel] - start]
    return out


def read_positions_eager(dset, shape, position_indices, flat_storage=True, max_workers=None):
    """Direct h5py/libdeflate read (no dask) of specific scan positions,
    summed into one (det_y, det_x) array - handles both storage layouts,
    see the module docstring for how this stays fast on a compressed
    dataset. `flat_storage`: whether `dset` is the flat/1-D layout (True)
    or native-4D (False). `max_workers`: see `_resolve_max_workers`.
    """
    idx = np.unique(np.asarray(position_indices))
    chunk_frames, read_chunk = _chunk_reader(dset, shape, flat_storage)
    if chunk_frames:
        chunk_bytes = chunk_frames * frame_size(shape) * dset.dtype.itemsize
        workers = _resolve_max_workers(max_workers, chunk_bytes)
        return _read_frames_grouped(shape, idx, 'sum', chunk_frames, read_chunk, dset.dtype,
                                     dset.file.filename, flat_storage, max_workers=workers)
    if flat_storage:
        return _read_frames_ungrouped(dset, shape, idx, 'sum')
    raise ValueError('read_positions_eager: native-4D dataset has no usable chunk grouping - '
                      'caller should use the dask fallback instead')


def read_roi_eager(dset, shape, x, y, w, h, flat_storage=True, max_workers=None):
    """Direct h5py/libdeflate read (no dask) of a rectangular (x, y, w, h)
    scan-space ROI, as a dense (h, w, det_y, det_x) array - handles both
    storage layouts, see the module docstring for how this stays fast on a
    compressed dataset. `flat_storage`: whether `dset` is the flat/1-D
    layout (True) or native-4D (False). `max_workers`: see
    `_resolve_max_workers`.
    """
    d1 = shape[1]
    rows = (y + np.arange(h))[:, None]
    cols = x + np.arange(w)[None, :]
    frame_indices = (rows * d1 + cols).ravel()  # row-major, matches (h, w) order
    chunk_frames, read_chunk = _chunk_reader(dset, shape, flat_storage)
    if chunk_frames:
        chunk_bytes = chunk_frames * frame_size(shape) * dset.dtype.itemsize
        workers = _resolve_max_workers(max_workers, chunk_bytes)
        frames = _read_frames_grouped(shape, frame_indices, 'none', chunk_frames, read_chunk, dset.dtype,
                                       dset.file.filename, flat_storage, max_workers=workers)
    elif flat_storage:
        frames = _read_frames_ungrouped(dset, shape, frame_indices, 'none')
    else:
        raise ValueError('read_roi_eager: native-4D dataset has no usable chunk grouping - '
                          'caller should use a plain h5py slice instead')
    return frames.reshape(h, w, *shape[-2:])


def read_roi(f, roi, max_eager_frames=10000, lazy=False, max_workers=None):
    """Read a rectangular (x, y, w, h) scan-space ROI from an already-open
    '.hdf5_eventem' `f`, as a dense (h, w, det_y, det_x) array - handles
    both the native-4D and flat/1-D storage layouts, and picks a direct
    h5py/libdeflate read (fast, higher peak memory) vs. a dask-based read
    (slower, memory-considerate) the same way loaders.load_hdf5_eventem's
    own `max_eager_frames` does: eager unless `lazy` or the ROI is larger
    than `max_eager_frames` positions. `max_workers`: see
    `_resolve_max_workers` - pass 1 explicitly when already running inside
    another process pool (see workers/worker_extract_frame.py).
    """
    x, y, w, h = roi
    dset = f['4D']
    shape = get_shape(f)
    flat_storage = is_flat(f)
    use_eager = not lazy and w * h <= max_eager_frames
    if use_eager:
        if flat_storage:
            return read_roi_eager(dset, shape, x, y, w, h, flat_storage=True, max_workers=max_workers)
        chunk_frames, _ = _chunk_reader(dset, shape, flat_storage=False)
        if chunk_frames:
            return read_roi_eager(dset, shape, x, y, w, h, flat_storage=False, max_workers=max_workers)
        return dset[y:y + h, x:x + w]  # not chunk-groupable - HDF5's own single-call read is already fine here
    s = as_dask(f)
    return s[y:y + h, x:x + w].compute()


def sum_masked_positions(f, mask, max_eager_frames=10000, max_workers=None):
    """Sum diffraction patterns at scan positions where `mask` (2-D
    boolean/0-1, shape (d0, d1)) is True, from an already-open
    '.hdf5_eventem' `f` - handles both the native-4D and flat/1-D storage
    layouts.

    Reads the selected positions directly via h5py/libdeflate (no dask)
    when their count is within `max_eager_frames` and the dataset's chunking
    supports it (see the module docstring); otherwise falls back to a
    dask-based fancy-index read. `max_workers`: see `_resolve_max_workers` -
    pass 1 explicitly when already running inside another process pool
    (see workers/worker_extract_frame.py).

    Returns a (det_y, det_x) array.
    """
    shape = get_shape(f)
    dset = f['4D']
    flat_storage = is_flat(f)
    idx = np.unique(np.flatnonzero(np.asarray(mask)))
    if idx.size <= max_eager_frames:
        chunk_frames, _ = _chunk_reader(dset, shape, flat_storage)
        if chunk_frames or flat_storage:
            return read_positions_eager(dset, shape, idx, flat_storage=flat_storage, max_workers=max_workers)
    s = as_dask(f).reshape(-1, *shape[-2:])
    return s[idx].sum(axis=0).compute()


def _reduce_frames(flat_block, mode, det_mask=None):
    """Reduce `flat_block` (n_frames, n_pixels) along axis=-1 to one value
    per frame: sum, or variance, optionally over just `det_mask`'s True
    pixels.

    Variance is computed via the single-pass `E[x^2] - E[x]^2` formula
    (`sum_x`/`sum_x2` below), not numpy's own `.var()` - `.var()` needs a
    same-shaped float64 temporary (`arr - mean`, then squared) to do its
    work, which for one full ~1GB decompressed chunk (e.g. 4096 x 512x512
    uint8 frames) is an ~8.6GB temporary. Measured directly: calling
    `.var()` on a whole such chunk at once drove this process to 9.6GB RSS
    and left it stuck/thrashing rather than completing. `sum_x` (a plain
    `.sum()`) and `sum_x2` (`np.einsum('ij,ij->i', ...)`, a fused multiply-
    reduce) are both computed by numpy via a small internally-buffered
    iterator instead, with no such blow-up - measured at a bounded ~1-2GB
    RSS for the same chunk, and ~2x faster than batching `.var()` calls to
    stay within a memory budget.

    `det_mask` is applied by *multiplying* it in (zeroing the excluded
    pixels, which contribute 0 to both the sum and the sum-of-squares),
    not by boolean-indexing `flat_block[:, det_mask]` first - that gather
    is a scattered/non-contiguous memory access pattern and measured ~2x
    slower than the multiply for a real annular detector mask, on top of
    needing its own same-shape-as-the-selection temporary.
    """
    if det_mask is not None:
        n_pixels = int(np.count_nonzero(det_mask))
        flat_block = flat_block * det_mask
    else:
        n_pixels = flat_block.shape[1]
    sum_x = flat_block.sum(axis=-1, dtype=np.float64)
    if mode == 'sum':
        return sum_x
    sum_x2 = np.einsum('ij,ij->i', flat_block, flat_block, dtype=np.float64)
    mean_x = sum_x / n_pixels
    return sum_x2 / n_pixels - mean_x ** 2


def full_scan_reduce(f, mode='sum', det_mask=None, max_workers=None):
    """Compute a (d0, d1) navigation image over the *whole* scan - every
    scan position summed (mode='sum') or its per-position variance
    (mode='variance') - over the whole detector (det_mask=None) or just its
    True pixels (det_mask: 1-D boolean/0-1, `det_y*det_x` long).

    Unlike `read_roi`/`sum_masked_positions`, every scan position has to be
    visited here no matter what (`det_mask` only reduces what's kept from
    each already-decompressed frame, not what has to be decompressed - see
    the module docstring), so there's no "touch only the needed chunks"
    saving to be had here the way there is for a small ROI/mask selection.
    The win is from decompressing each chunk faster (libdeflate over HDF5's
    stock-zlib gzip pipeline) and, when `max_workers` allows it, decompressing
    + reducing several chunks concurrently as separate worker *processes*
    (see the module docstring for why processes, not threads, and
    `_run_processes`) - reading the dataset one on-disk chunk at a time
    either way (bounding peak memory to roughly `max_workers` chunks, like
    the dask path this replaces bounds it to one), reducing each chunk's
    frames directly in memory as it goes - no dask graph involved.

    `max_workers`: see `_resolve_max_workers` - None (default) sizes
    dynamically per-machine (CPU cores and currently-available RAM); pass 1
    explicitly when already running inside another process pool (see
    workers/worker_nav_img.py).

    Falls back to the (slower, dask-based) `as_dask()` path when the
    dataset's chunking doesn't support the fast per-chunk reader
    (uncompressed, or chunked along more than the scan-row axis).
    """
    if mode not in ('sum', 'variance'):
        raise ValueError(f"mode must be 'sum' or 'variance', got {mode!r}")
    shape = get_shape(f)
    dset = f['4D']
    flat_storage = is_flat(f)
    chunk_frames, read_chunk = _chunk_reader(dset, shape, flat_storage)

    if chunk_frames is None:
        s = as_dask(f)
        if det_mask is None:
            reduced = s.sum(axis=(-1, -2)) if mode == 'sum' else s.var(axis=(-1, -2))
        else:
            s = s.reshape(shape[0], shape[1], -1)
            reduced = (s * det_mask).sum(axis=-1) if mode == 'sum' else s[..., det_mask].var(axis=-1)
        return reduced.compute()

    n_positions = shape[0] * shape[1]
    nav_flat = np.empty(n_positions, dtype=np.float64)
    n_chunks = -(-n_positions // chunk_frames)  # ceil division
    chunk_bytes = chunk_frames * frame_size(shape) * dset.dtype.itemsize
    workers = _resolve_max_workers(max_workers, chunk_bytes)

    def _reduce_block(block):
        flat_block = block.reshape(block.shape[0], -1)
        return _reduce_frames(flat_block, mode, det_mask=det_mask)

    if workers <= 1 or n_chunks <= 1:
        for chunk_id in range(n_chunks):
            reduced = _reduce_block(read_chunk(chunk_id))
            c_start = chunk_id * chunk_frames
            nav_flat[c_start:c_start + reduced.shape[0]] = reduced
        return nav_flat.reshape(shape[0], shape[1])

    fn = f.filename
    tasks = [(cid, (fn, flat_storage, cid, mode, det_mask)) for cid in range(n_chunks)]
    for chunk_id, reduced in _run_processes(_worker_reduce_chunk, tasks, workers):
        c_start = chunk_id * chunk_frames
        nav_flat[c_start:c_start + reduced.shape[0]] = reduced
    return nav_flat.reshape(shape[0], shape[1])
