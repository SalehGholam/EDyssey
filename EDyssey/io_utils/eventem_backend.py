# -*- coding: utf-8 -*-
"""Backend-agnostic facade for .tpx3 analysis: old eventem / new (bug-fixed)
eventem / pyeventem.

Every .tpx3 call site in this app (loaders.py, nav_image.py,
workers/worker_extract_frame*.py) should go through ``run_pacbed``/
``run_vstem``/``run_var``/``run_roi``/``run_roi_masked`` here instead of
constructing ``eventem.Pacbed``/``vSTEM``/``Var``/``Roi`` directly - this is
the one place that knows the (real, breaking) API differences between the
three backends, and the one place declustering gets wired in.

Deliberately Qt-free: ``workers/worker_extract_frame*.py`` run as bare
subprocesses (via ``runpy``, see ``worker_launch.py``) specifically to avoid
paying for PyQt/matplotlib imports, so nothing in this module may import
PyQt. Callers that DO have Qt available (the GUI thread) build a
``decluster_cfg`` dict from ``ui_tabs.analysis_backend_settings
.AnalysisBackendSettings`` and pass it in as a plain dict; subprocess
workers receive the same dict serialized through the existing CLI-args/
``tasks.json`` mechanism, so this module never needs to know which side of
that boundary it's being called from.

**Why three backends need distinct code paths, not just a different
``eventem`` import:**
- Old eventem (vendored in this directory, one ``.pyd`` per Python version)
  predates declustering entirely - no ``decluster``/``dtime``/``dspace``/
  ``cluster_range``/``tot_per_electron``/``electron_count_lut_file``
  attribute exists on any of its classes (confirmed directly). It also uses
  a since-renamed ``detector_size_x``/``detector_size_y`` pair where the
  current C++ source uses a single ``detector_size``.
- New eventem (``eventem_new/``, built from the current, bug-fixed C++
  source - see ``evenTem`` repo commits ``699b305``/``a612120``) has both
  declustering and the single ``detector_size`` property.
- pyeventem is a separate, pure-Python/numba package with its own source/
  sink object shapes entirely (``pyeventem.Tpx3Raster``/``Tpx3Pixeltrig`` +
  ``Pacbed``/``VSTEM``/``Var``/``Roi`` sinks, run via ``pyeventem.run``).
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np

logger = logging.getLogger('EDyssey.eventem_backend')

BACKEND_OLD = 'old_eventem'
BACKEND_NEW = 'new_eventem'
BACKEND_PYEVENTEM = 'pyeventem'
BACKENDS = (BACKEND_OLD, BACKEND_NEW, BACKEND_PYEVENTEM)
BACKEND_LABELS = {
    BACKEND_OLD: 'Old eventem',
    BACKEND_NEW: 'New eventem',
    BACKEND_PYEVENTEM: 'pyeventem',
}

#: pyeventem-only: how to split one file's decode across CPU workers. Old/new
#: eventem have no equivalent knob here - their own "CPU cores" setting
#: (n_threads) always means the C++'s own internal thread pool, regardless of
#: this. Measured directly (pyeventem/Examples/07_backend_performance.ipynb,
#: 2026-09-21): threads win in every configuration checked - multiprocessing
#: adds process-spawn/import overhead with nothing to show for it, since
#: pyeventem's own kernels already release the GIL - so EXEC_THREADS is both
#: the default and the recommended choice. EXEC_PROCESSES exists for
#: comparison/testing and for the rare machine/workload where re-measuring is
#: worth it, not because it's expected to win.
EXEC_THREADS = 'threads'
EXEC_PROCESSES = 'processes'
EXECUTION_STRATEGIES = (EXEC_THREADS, EXEC_PROCESSES)
EXECUTION_STRATEGY_LABELS = {
    EXEC_THREADS: 'Threads',
    EXEC_PROCESSES: 'Processes',
}

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVENTEM_NEW_DIR = os.path.join(_THIS_DIR, 'eventem_new')

_eventem_new_mod = None


def _old_eventem():
    import eventem
    return eventem


def _new_eventem():
    global _eventem_new_mod
    if _eventem_new_mod is None:
        # KNOWN CRASH (reproduced 100% of the time, isolated 2026-09-18):
        # importing eventem_new in a process that already has PyQt5 loaded
        # segfaults immediately, inside the import itself - before any
        # eventem_new code (Roi/Pacbed/etc.) ever runs, and regardless of
        # n_threads, declustering, or which .tpx3 file. Reproduced in a
        # minimal 3-line repro (import PyQt5.QtCore; import eventem_new)
        # with nothing else involved, and confirmed NOT caused by the
        # h5en.dll/hdf5_cpp.dll rename (swapping in old eventem's proven-
        # good h5ev.dll/hdf5_cpp.dll pair alongside the same eventem_new.pyd
        # still crashed) - the bug is in eventem_new.pyd's own compiled code
        # (something in its module-load-time initialization only manifests
        # once Qt's DLLs are also in the process), not in this Python
        # wiring. A real fix needs a debugger session against the evenTem
        # C++ source/build - not diagnosable further from here.
        #
        # Every run_*() function below checks _new_eventem_needs_subprocess()
        # BEFORE ever reaching this function, and delegates the whole call to
        # a real, separate, Qt-free subprocess instead (see
        # _run_new_eventem_via_subprocess) - so by the time this actually
        # runs, either PyQt5 genuinely isn't loaded in this process (a batch
        # worker, or the dedicated subprocess itself), or a caller reached
        # this directly without going through run_*() at all, which is a
        # caller bug, not something to paper over here.
        if _EVENTEM_NEW_DIR not in sys.path:
            sys.path.insert(0, _EVENTEM_NEW_DIR)
        import eventem_new
        _eventem_new_mod = eventem_new
    return _eventem_new_mod


def _to_jsonable(value):
    """Recursively convert tuples (anywhere - top level or nested inside a
    list of (x, y) pairs) to lists, and numpy scalars to plain Python ones,
    so a run_*() argument that's normally a tuple/list of tuples (r_in/
    r_out/offset) round-trips through json.dump cleanly. None and plain
    scalars pass through unchanged."""
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _new_eventem_needs_subprocess() -> bool:
    """True once PyQt5 is loaded in this process - see _new_eventem's own
    comment for why importing eventem_new directly is then guaranteed to
    segfault, and _run_new_eventem_via_subprocess for the actual fix."""
    return 'PyQt5.QtCore' in sys.modules


def _run_new_eventem_via_subprocess(func_name: str, kwargs: dict, mask_array=None, logger_=None):
    """Run one run_*() call for backend=BACKEND_NEW in a real, separate,
    Qt-free process (workers/worker_eventem_call.py) instead of importing
    eventem_new directly here - see _new_eventem_needs_subprocess/
    _new_eventem's own comments for why this is necessary, not optional.
    Every other backend/sink combination stays in-process; this is New
    eventem's own, otherwise-invisible, workaround.

    `kwargs` must already be JSON-serializable (tuples become lists automatically;
    None/bool/int/float/str all pass through as-is) - the caller builds it
    from the same arguments it would have passed to the in-process backend
    directly. `mask_array`, if given (run_roi_masked only), is written to
    its own temp .npy and referenced by path instead - a boolean array
    isn't JSON-serializable, and round-tripping it as nested lists would
    bloat the request file for no reason.

    Subprocess console output (eventem_new's own native progress bar, or
    the request itself failing) is piped back through `logger_` live via
    pipe_process_output_to_logger, the same way redirect_console_to_logger
    already does for genuinely in-process calls - so New eventem's progress
    shows up in the Qt log console exactly like every other backend's does,
    despite now running in a different process entirely.

    Assigned to a Windows Job Object with KILL_ON_JOB_CLOSE, the same
    mechanism worker_pool_utils.py's own batch drivers already use for
    "Cancel" - here, its purpose is different: since this call blocks the
    calling QThreadPool worker thread until the subprocess exits, there is
    no user-facing Cancel button for it at all yet, but if EDyssey's own
    process is closed/killed while this is running, Windows closes the job
    handle as part of tearing down the process that created it, which -
    because of KILL_ON_JOB_CLOSE - kills this subprocess too, instead of
    leaving it as an orphaned process still declustering in the background
    with no window left to show for it (confirmed reproducible before this:
    closing EDyssey mid-declustering left the child process running).
    """
    import json
    import subprocess
    import tempfile
    from ui_tabs.worker_launch import worker_command
    from .progress import pipe_process_output_to_logger
    import worker_pool_utils as wpu

    with tempfile.TemporaryDirectory(prefix='edyssey_eventem_new_') as tmp_dir:
        request_path = os.path.join(tmp_dir, 'request.json')
        result_path = os.path.join(tmp_dir, 'result.npz')
        request_kwargs = dict(kwargs)
        if mask_array is not None:
            mask_path = os.path.join(tmp_dir, 'mask.npy')
            np.save(mask_path, np.asarray(mask_array))
            request_kwargs['mask_path'] = mask_path
        with open(request_path, 'w', encoding='utf-8') as f:
            json.dump({'func': func_name, 'kwargs': request_kwargs}, f)

        program, arguments = worker_command('eventem_call', [request_path, result_path])
        proc = subprocess.Popen([program, *arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        job_handle = wpu.create_job_object()
        if job_handle is not None:
            wpu.assign_process_to_job(job_handle, proc.pid)
        try:
            returncode, tail_lines = pipe_process_output_to_logger(proc, logger_, 'Loading tpx3 (New eventem)')
        finally:
            wpu.kill_job(job_handle)
        if returncode != 0:
            detail = '\n'.join(tail_lines) or '(no output captured)'
            raise RuntimeError(
                f"New eventem's subprocess failed (exit code {returncode}):\n{detail}")

        # `with np.load(...)`, not a bare call - np.load() on a .npz lazily
        # keeps the file open (it's a zip archive read on demand), and
        # TemporaryDirectory's own cleanup below fails with WinError 32
        # ("used by another process") if that handle is still open when it
        # tries to delete the directory.
        with np.load(result_path) as data:
            if func_name in ('run_pacbed', 'run_vstem', 'run_var'):
                return np.array(data['array'])
            roi_4d = np.array(data['roi_4d']) if 'roi_4d' in data.files else None
            return RoiResult(np.array(data['scan_image']), np.array(data['diffraction_pattern']), roi_4d)


def _pyeventem():
    import pyeventem
    return pyeventem


# ---------------------------------------------------------------------------
# Declustering config: a plain, JSON-serializable dict everywhere (so it can
# travel through tasks.json to a subprocess worker unchanged).
# ---------------------------------------------------------------------------

def default_decluster_cfg() -> dict:
    return {
        'enabled': False,
        'dspace': 6,
        'dtime_ns': 100.0,
        'cluster_range': 256,
        'tot_per_electron': 100.0,
        'lut_file': '',
        # Per-sink gating - only consulted by callers that pass `sink_name`
        # through to `_decluster_active` below; a dict missing this key (an
        # older/hand-built cfg) is treated as "every sink enabled".
        'sinks': {'pacbed': True, 'roi': True, 'vstem': True, 'var': True},
    }


def load_electron_count_lut(path: str) -> np.ndarray:
    """Parse a LUT file in the same plain-text format the C++ (
    ``ClusterResolver.hpp::load_electron_count_lut_file``) reads: lines
    starting with '#' are comments, every other non-empty line is one row
    of whitespace-separated integers (row = cluster size, column = ToT
    sum). ``np.loadtxt`` already implements exactly this format."""
    arr = np.loadtxt(path, comments='#', dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _decluster_active(backend: str, decluster_cfg: dict | None, sink_name: str) -> bool:
    if not decluster_cfg or not decluster_cfg.get('enabled'):
        return False
    sinks = decluster_cfg.get('sinks')
    if sinks is not None and not sinks.get(sink_name, True):
        return False
    if backend == BACKEND_OLD:
        logger.warning(
            "Declustering requested for %s, but the 'Old eventem' backend has no "
            "declustering support at all - running without it. Switch to 'New eventem' "
            "or 'pyeventem' to use declustering.", sink_name)
        return False
    return True


def _apply_decluster_eventem(obj, decluster_cfg: dict) -> None:
    """Set the decluster/dtime/dspace/cluster_range/tot_per_electron (or
    LUT) fields shared by new-eventem's Pacbed/vSTEM/Var/Roi classes."""
    obj.decluster = True
    obj.dtime = int(round(decluster_cfg['dtime_ns'] / 1.5625))
    obj.dspace = int(decluster_cfg['dspace'])
    obj.cluster_range = int(decluster_cfg['cluster_range'])
    lut_file = decluster_cfg.get('lut_file') or ''
    if lut_file:
        obj.electron_count_lut_file = lut_file
    else:
        obj.tot_per_electron = float(decluster_cfg['tot_per_electron'])


def _make_declusterer(decluster_cfg: dict):
    pe = _pyeventem()
    lut_file = decluster_cfg.get('lut_file') or ''
    lut = load_electron_count_lut(lut_file) if lut_file else None
    return pe.Declusterer(
        dspace=int(decluster_cfg['dspace']),
        dtime_ns=float(decluster_cfg['dtime_ns']),
        cluster_range=int(decluster_cfg['cluster_range']),
        tot_per_electron=float(decluster_cfg['tot_per_electron']),
        lut=lut,
    )


def _set_detector_size(obj, backend: str, det_shape: tuple[int, int]) -> None:
    """Old eventem: detector_size_x/detector_size_y. New eventem: a single
    detector_size. Only actually written when it differs from what eventem
    already reports (reflecting the real file's own hardware layout) - a
    genuine mismatch segfaults .run() instead of gracefully reshaping."""
    if backend == BACKEND_OLD:
        if det_shape != (obj.detector_size_x, obj.detector_size_y):
            obj.detector_size_x, obj.detector_size_y = det_shape
    else:
        if det_shape[0] != det_shape[1]:
            raise ValueError(
                f'New eventem only supports a square detector (got {det_shape}) - '
                'the detector_size_x/detector_size_y split was removed upstream.')
        if det_shape[0] != obj.detector_size:
            obj.detector_size = det_shape[0]


def _pyeventem_source(pe, fn, scan_size, dwell_time_ns, fn_pattern, n_threads):
    if fn_pattern:
        return pe.Tpx3Pixeltrig(fn, fn_pattern, nx=scan_size[0], ny=scan_size[1])
    return pe.Tpx3Raster(fn, nx=scan_size[0], ny=scan_size[1], dwell_time_ns=dwell_time_ns or 0.0)


#: What "Auto" resolves to for pyeventem. Matches pyeventem's own
#: run_parallel default, capped by the machine so a 2-core box doesn't
#: start 8 workers. Measured on this 8-core/16-thread test machine, ROI
#: extraction: 5 workers 3.58s, 8 workers 3.32s, 16 workers 3.37s - it
#: flattens out well before the thread count, so there is nothing to gain
#: from going higher and real memory to lose.
_PYEVENTEM_AUTO_WORKERS = 8


def _pyeventem_workers(n_threads):
    """Worker count for pyeventem: an explicit setting wins, None/0
    ("Auto") resolves rather than silently meaning single-threaded."""
    if n_threads:
        return int(n_threads)
    return max(1, min(_PYEVENTEM_AUTO_WORKERS, os.cpu_count() or 1))


def _pyeventem_run(pe, source, sink, n_threads, decluster_on=False, declusterer=None, logger_=None):
    """``source`` here is always the RAW Tpx3Raster/Tpx3Pixeltrig - never
    pre-wrapped with ``pe.decluster_source()`` (unlike this function's
    pre-existing behavior) - declustering, when on, is applied here instead,
    via whichever of pyeventem's two decluster-capable paths matches
    ``n_threads``:

    - more than one worker: ``pe.run_parallel(..., declusterer=declusterer)``
      - bounded-memory, checkpoint-seeded parallel decode+decluster, each
      chunk declustered independently (no cross-chunk boundary correction -
      see pyeventem's own run_parallel/run_sinks_parallel docstrings for the
      trade-off, which matches what the reference C++ implementation's own
      ring-buffer chunks already accept, just with far fewer boundary
      seams).
    - ``n_threads == 1``: sequential ``pe.decluster_source()`` +
      ``pe.run(..., progress=True)`` - fully correct at every boundary,
      single-threaded.

    ``n_threads`` of None or 0 is the settings dialog's "Auto". That used to
    fall into the sequential branch, which is not what Auto means anywhere
    else: for old/new eventem, leaving n_threads unset lets the C++ start
    its own threads. It also became the single worst thing to pick, because
    the parallel path is where pyeventem restricts the decode to the word
    range holding the ROI's scan rows - a sequential run gets the in-kernel
    row gate but still reads the whole file. Measured on a 3.2 GB file, ROI
    extraction with the file already indexed: 3.49s on Auto-as-sequential,
    0.18s parallel. Auto now resolves to a real worker count.

    Every sink this module builds - Pacbed/VSTEM/Var/Roi - implements
    clone()/merge(), required by both the plain and the declustering
    parallel path alike.

    Both paths pass progress=True (a live hit-count/rate tqdm bar - see
    pyeventem's own pipeline.py/decode/parallel.py) - old/new eventem's
    console progress already goes through redirect_console_to_logger, and
    pyeventem had no equivalent at all, which made a multi-minute
    declustered run look hung."""
    from .progress import redirect_console_to_logger
    workers = _pyeventem_workers(n_threads)
    if workers > 1:
        with redirect_console_to_logger(logger_, 'Loading tpx3'):
            pe.run_parallel(source, [sink], n_workers=workers, declusterer=declusterer, progress=True)
    else:
        if decluster_on:
            source = pe.decluster_source(source, declusterer)
        with redirect_console_to_logger(logger_, 'Loading tpx3'):
            pe.run(source, [sink], progress=True)


# ---------------------------------------------------------------------------
# pyeventem execution strategy: threads (default) vs. processes.
#
# Every run_*() function below builds one of pyeventem's four sink types
# (Pacbed/VSTEM/Var/Roi) and hands it to _pyeventem_dispatch, which is the
# one place that decides threads vs. processes - see EXEC_THREADS/
# EXEC_PROCESSES above for which one is actually recommended and why.
# ---------------------------------------------------------------------------

#: sink_kind -> pyeventem constructor, shared by _build_pyeventem_sink and
#: the multiprocessing worker below - keeps both in one place rather than
#: duplicating the mapping.
_PYEVENTEM_SINK_CTORS = {
    'pacbed': 'Pacbed',
    'vstem': 'VSTEM',
    'var': 'Var',
    'roi': 'Roi',
}


def _build_pyeventem_sink(pe, sink_kind, sink_kwargs):
    return getattr(pe, _PYEVENTEM_SINK_CTORS[sink_kind])(**sink_kwargs)


def _pyeventem_sink_result(sink_kind, sink):
    """The plain numpy result(s) a finished sink holds - a bare ndarray for
    Pacbed/VSTEM/Var, or the (scan_image, diffraction_pattern, roi_4d) tuple
    run_roi/run_roi_masked's RoiResult wraps, for 'roi'."""
    if sink_kind == 'pacbed':
        return np.asarray(sink.image)
    if sink_kind == 'vstem':
        return np.asarray(sink.images[0])
    if sink_kind == 'var':
        return np.asarray(sink.image)
    roi_4d = sink.roi_4d if getattr(sink, 'extract_4d', False) else None
    return np.asarray(sink.scan_image), np.asarray(sink.diffraction_pattern), roi_4d


class _RowBandSink:
    """Restricts ``sink`` to scan rows ``[row0, row1)`` - the multiprocessing
    driver's own work-splitting mechanism (see _pyeventem_run_multiprocess):
    each process gets one disjoint row band of the *same* full-size sink,
    decodes only the file words that can hold it (pyeventem's own scan-row
    gate - see sinks/base.py's scan_rows_union), and every process's result
    is then simply summed elementwise, since disjoint bands never overlap.

    Composes with any restriction the sink already has of its own (a Roi
    rectangle, a masked Pacbed) by intersecting rather than replacing - a
    process assigned a row band outside a small ROI correctly decodes
    nothing at all rather than double-counting.
    """

    def __init__(self, sink, row0, row1):
        self._sink, self._row0, self._row1 = sink, row0, row1

    def scan_rows(self, ny):
        band = np.zeros(ny, dtype=bool)
        band[self._row0:self._row1] = True
        own_rows = getattr(self._sink, 'scan_rows', None)
        if own_rows is not None:
            got = own_rows(ny)
            if got is not None:
                return band & np.asarray(got, dtype=bool)
        return band

    def update(self, block):
        self._sink.update(block)

    def finalize(self):
        self._sink.finalize()

    def clone(self):
        return _RowBandSink(self._sink.clone(), self._row0, self._row1)

    def merge(self, other):
        self._sink.merge(other._sink)


def _pyeventem_process_worker(job):
    """Top-level (picklable) entry point for one _pyeventem_run_multiprocess
    task: decode+accumulate this process's own row band, independently of
    every other process. Must stay a plain module-level function (not a
    closure) - ProcessPoolExecutor pickles it by reference on Windows'
    spawn start method, which needs to be able to re-import it by name in
    the child process.
    """
    (sink_kind, sink_kwargs, fn, scan_size, dwell_time_ns, fn_pattern,
     row0, row1, decluster_cfg, decluster_on) = job
    import pyeventem as pe
    source = _pyeventem_source(pe, fn, scan_size, dwell_time_ns, fn_pattern, None)
    sink = _build_pyeventem_sink(pe, sink_kind, sink_kwargs)
    banded = _RowBandSink(sink, row0, row1)
    declusterer = _make_declusterer(decluster_cfg) if decluster_on else None
    # n_workers=1: this process's own share of the parallelism, not a nested
    # thread pool inside it - see EXEC_PROCESSES's docstring on why more
    # threads-per-process was deliberately not layered on top here. Still
    # run_parallel, not run(): only run_parallel's checkpoint-seeded decode
    # honours the row-band gate above by skipping the rest of the file: see
    # pyeventem/Examples/07_backend_performance.ipynb ("The part that
    # surprised me: run_parallel(n_workers=1)").
    pe.run_parallel(source, [banded], n_workers=1, declusterer=declusterer)
    return _pyeventem_sink_result(sink_kind, sink)


def _pyeventem_run_multiprocess(sink_kind, sink_kwargs, fn, scan_size, dwell_time_ns, fn_pattern,
                                 n_workers, decluster_cfg, decluster_on, row_lo, row_hi, logger_=None):
    """Split one file's decode across ``n_workers`` real OS processes
    instead of pyeventem's own thread pool - EXEC_PROCESSES's implementation.
    ``[row_lo, row_hi)`` is divided into ``n_workers`` disjoint scan-row
    bands (see _RowBandSink); every process's result is then summed
    elementwise, which is exact because the bands never overlap - validated
    the same way in pyeventem/Examples/07_backend_performance.ipynb (§4:
    every process-pool result there matched ``pyeventem.run()`` bit for
    bit).

    No live progress bar: that notebook's own §4 measurement is also why
    this exists mainly for comparison rather than as the recommended path
    (see EXEC_PROCESSES) - piping several processes' progress back to one
    live bar was not judged worth building for a strategy that's slower in
    every configuration measured so far. A single before/after log line
    goes to ``logger_`` instead, matching how ``_run_new_eventem_via_
    subprocess`` also can't offer a live in-app bar for its own, unrelated
    reason (a real separate program, not a worker pool).
    """
    from concurrent.futures import ProcessPoolExecutor
    if logger_ is not None:
        logger_.info('Loading tpx3 (pyeventem, %d processes)...', n_workers)
    edges = np.linspace(row_lo, row_hi, n_workers + 1).astype(int)
    jobs = [(sink_kind, sink_kwargs, fn, scan_size, dwell_time_ns, fn_pattern,
             int(edges[i]), int(edges[i + 1]), decluster_cfg, decluster_on)
            for i in range(n_workers)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_pyeventem_process_worker, jobs))
    if logger_ is not None:
        logger_.info('Loading tpx3 (pyeventem, %d processes): done', n_workers)

    if sink_kind == 'roi':
        scan_images, dps, cubes = zip(*parts)
        roi_4d = sum(cubes) if cubes[0] is not None else None
        return sum(scan_images), sum(dps), roi_4d
    return sum(parts)


def _pyeventem_dispatch(sink_kind, sink_kwargs, fn, scan_size, dwell_time_ns, fn_pattern,
                         n_threads, execution_strategy, decluster_cfg, decluster_on,
                         row_band=None, logger_=None):
    """Builds and runs one pyeventem sink (Pacbed/VSTEM/Var/Roi) and returns
    its plain result - the one place every run_*() function's pyeventem
    branch goes through, so EXEC_THREADS-vs-EXEC_PROCESSES only needs
    implementing once. See _pyeventem_sink_result for the return shape.

    ``row_band``: the ``(lo, hi)`` scan-row extent worth splitting across
    processes for this call - defaults to the whole scan (``0, ny``); a
    rectangular Roi passes its own ``(y, y+height)`` instead, since that's
    the only extent that can possibly hold a hit for it (see run_roi's call
    site) and splitting a tight extent balances the process pool's work
    better than splitting the whole scan would.
    """
    workers = _pyeventem_workers(n_threads)
    if execution_strategy == EXEC_PROCESSES and workers > 1:
        lo, hi = row_band if row_band is not None else (0, scan_size[1])
        return _pyeventem_run_multiprocess(
            sink_kind, sink_kwargs, fn, scan_size, dwell_time_ns, fn_pattern,
            n_workers=workers, decluster_cfg=decluster_cfg, decluster_on=decluster_on,
            row_lo=lo, row_hi=hi, logger_=logger_)

    pe = _pyeventem()
    source = _pyeventem_source(pe, fn, scan_size, dwell_time_ns, fn_pattern, n_threads)
    sink = _build_pyeventem_sink(pe, sink_kind, sink_kwargs)
    declusterer = _make_declusterer(decluster_cfg) if decluster_on else None
    _pyeventem_run(pe, source, sink, n_threads, decluster_on, declusterer=declusterer, logger_=logger_)
    return _pyeventem_sink_result(sink_kind, sink)


# ---------------------------------------------------------------------------
# Pacbed
# ---------------------------------------------------------------------------

def run_pacbed(fn, scan_size, dwell_time_ns=1000.0, det_shape=(512, 512),
               fn_pattern=None, repetitions=1, backend=BACKEND_OLD,
               decluster_cfg=None, logger_=None, n_threads=None,
               execution_strategy=EXEC_THREADS):
    """Full-frame summed diffraction pattern. Returns a (det_y, det_x) array.

    ``execution_strategy``: pyeventem only (see EXEC_THREADS/EXEC_PROCESSES) -
    ignored for old/new eventem, which have no equivalent choice."""
    decluster_cfg = decluster_cfg or default_decluster_cfg()
    decluster_on = _decluster_active(backend, decluster_cfg, 'pacbed')

    if backend == BACKEND_NEW and _new_eventem_needs_subprocess():
        return _run_new_eventem_via_subprocess('run_pacbed', dict(
            fn=fn, scan_size=list(scan_size), dwell_time_ns=dwell_time_ns, det_shape=list(det_shape),
            fn_pattern=fn_pattern, repetitions=repetitions, backend=backend,
            decluster_cfg=decluster_cfg, n_threads=n_threads,
        ), logger_=logger_)

    if backend in (BACKEND_OLD, BACKEND_NEW):
        mod = _old_eventem() if backend == BACKEND_OLD else _new_eventem()
        dp = mod.Pacbed(repetitions=repetitions)
        if n_threads is not None:
            dp.n_threads = n_threads
        dp.set_file(fn)
        dp.nx = scan_size[0]
        dp.ny = scan_size[1]
        _set_detector_size(dp, backend, det_shape)
        dp.set_dwell_time(int(round(dwell_time_ns)))
        if fn_pattern is not None:
            dp.set_pattern_file(fn_pattern)
        if decluster_on:
            _apply_decluster_eventem(dp, decluster_cfg)
        from .progress import redirect_console_to_logger
        # close_socket() (every LiveProcessor-derived class - Pacbed/vSTEM/
        # Var/Roi - has one, live-streaming or not) was dropped when this
        # facade was written, unlike the direct eventem.Roi construction
        # worker_extract_frame.py used to do before it was rewired through
        # here - restored, in a finally so it still runs if .run() itself
        # raises. Uncertain this alone explains every repeated-call hang
        # reported against this facade, but it's a real, previously-dropped
        # cleanup call, not optional.
        try:
            with redirect_console_to_logger(logger_, 'Loading tpx3'):
                dp.run()
            return np.array(dp.Pacbed_image).reshape(det_shape[1], det_shape[0])
        finally:
            dp.close_socket()

    return _pyeventem_dispatch(
        'pacbed', dict(detector_size=det_shape[0]), fn, scan_size, dwell_time_ns, fn_pattern,
        n_threads, execution_strategy, decluster_cfg, decluster_on, logger_=logger_)


# ---------------------------------------------------------------------------
# vSTEM
# ---------------------------------------------------------------------------

def run_vstem(fn, scan_size, dwell_time_ns=1000.0, r_in=0, r_out=1 << 15,
              offset=None, det_shape=(512, 512), fn_pattern=None, repetitions=1,
              backend=BACKEND_OLD, decluster_cfg=None, logger_=None, n_threads=None,
              execution_strategy=EXEC_THREADS):
    """Virtual-STEM navigation image, one or several annular detectors at
    once. Returns a (ny, nx) array (single detector) - see old-eventem's
    ``get_image()``/pyeventem's ``images[0]`` for the multi-detector case,
    not currently exposed through this facade (EDyssey only ever asks for
    one detector's worth per call today)."""
    decluster_cfg = decluster_cfg or default_decluster_cfg()
    decluster_on = _decluster_active(backend, decluster_cfg, 'vstem')
    inner = list(r_in) if isinstance(r_in, (list, tuple)) else [r_in]
    outer = list(r_out) if isinstance(r_out, (list, tuple)) else [r_out]

    if backend == BACKEND_NEW and _new_eventem_needs_subprocess():
        return _run_new_eventem_via_subprocess('run_vstem', dict(
            fn=fn, scan_size=list(scan_size), dwell_time_ns=dwell_time_ns,
            r_in=_to_jsonable(r_in), r_out=_to_jsonable(r_out), offset=_to_jsonable(offset),
            det_shape=list(det_shape), fn_pattern=fn_pattern, repetitions=repetitions,
            backend=backend, decluster_cfg=decluster_cfg, n_threads=n_threads,
        ), logger_=logger_)

    if backend in (BACKEND_OLD, BACKEND_NEW):
        mod = _old_eventem() if backend == BACKEND_OLD else _new_eventem()
        vstem = mod.vSTEM(repetitions)
        if n_threads is not None:
            vstem.n_threads = n_threads
        vstem.b_cumulative = True
        vstem.set_file(fn)
        vstem.nx = scan_size[0]
        vstem.ny = scan_size[1]
        _set_detector_size(vstem, backend, det_shape)
        vstem.inner_radia = inner
        vstem.outer_radia = outer
        vstem.set_dwell_time(int(round(dwell_time_ns)))
        if fn_pattern:
            vstem.set_pattern_file(fn_pattern)
        if offset:
            offsets = offset if isinstance(offset[0], (list, tuple)) else [offset]
            vstem.set_offsets([[o[0], o[1]] for o in offsets])
        if decluster_on:
            _apply_decluster_eventem(vstem, decluster_cfg)
        from .progress import redirect_console_to_logger
        try:
            with redirect_console_to_logger(logger_, 'Loading tpx3'):
                vstem.run()
            return vstem.get_image()
        finally:
            vstem.close_socket()

    centers = None
    if offset:
        offsets = offset if isinstance(offset[0], (list, tuple)) else [offset]
        centers = [(float(o[0]), float(o[1])) for o in offsets]
    return _pyeventem_dispatch(
        'vstem', dict(nx=scan_size[0], ny=scan_size[1], inner_radii=inner, outer_radii=outer,
                      centers=centers, detector_size=det_shape[0]),
        fn, scan_size, dwell_time_ns, fn_pattern,
        n_threads, execution_strategy, decluster_cfg, decluster_on, logger_=logger_)


# ---------------------------------------------------------------------------
# Var
# ---------------------------------------------------------------------------

def run_var(fn, scan_size, dwell_time_ns=1000.0, r_in=0, r_out=1 << 15,
            offset=None, det_shape=(512, 512), fn_pattern=None, repetitions=1,
            backend=BACKEND_OLD, decluster_cfg=None, logger_=None, n_threads=None,
            execution_strategy=EXEC_THREADS):
    """Per-scan-position variance image (single annular region only - see
    calculate_nav_img_variance_tpx3's docstring for why there's no
    multi-detector variant). Returns a (ny, nx) array."""
    decluster_cfg = decluster_cfg or default_decluster_cfg()
    decluster_on = _decluster_active(backend, decluster_cfg, 'var')

    if backend == BACKEND_NEW and _new_eventem_needs_subprocess():
        return _run_new_eventem_via_subprocess('run_var', dict(
            fn=fn, scan_size=list(scan_size), dwell_time_ns=dwell_time_ns,
            r_in=_to_jsonable(r_in), r_out=_to_jsonable(r_out), offset=_to_jsonable(offset),
            det_shape=list(det_shape), fn_pattern=fn_pattern, repetitions=repetitions,
            backend=backend, decluster_cfg=decluster_cfg, n_threads=n_threads,
        ), logger_=logger_)

    if backend in (BACKEND_OLD, BACKEND_NEW):
        mod = _old_eventem() if backend == BACKEND_OLD else _new_eventem()
        var = mod.Var(repetitions)
        if n_threads is not None:
            var.n_threads = n_threads
        var.b_cumulative = True
        var.set_file(fn)
        var.nx = scan_size[0]
        var.ny = scan_size[1]
        _set_detector_size(var, backend, det_shape)
        var.inner_radius = r_in
        var.outer_radius = r_out
        var.set_dwell_time(int(round(dwell_time_ns)))
        if fn_pattern:
            var.set_pattern_file(fn_pattern)
        if offset:
            # (x, y) passed through as-is, as a single [x, y] list -
            # set_offset takes one std::array<float,2>, not two args - see
            # calculate_nav_img_variance_tpx3's identical, already-verified call.
            var.set_offset([offset[0], offset[1]])
        if decluster_on:
            _apply_decluster_eventem(var, decluster_cfg)
        from .progress import redirect_console_to_logger
        try:
            with redirect_console_to_logger(logger_, 'Loading tpx3'):
                var.run()
            return np.array(var.Var_image).reshape(scan_size[1], scan_size[0])
        finally:
            var.close_socket()

    center = (float(offset[0]), float(offset[1])) if offset else None
    return _pyeventem_dispatch(
        'var', dict(nx=scan_size[0], ny=scan_size[1], inner_radius=float(r_in), outer_radius=float(r_out),
                   center=center, detector_size=det_shape[0]),
        fn, scan_size, dwell_time_ns, fn_pattern,
        n_threads, execution_strategy, decluster_cfg, decluster_on, logger_=logger_)


# ---------------------------------------------------------------------------
# Roi (rectangular and arbitrary-mask)
# ---------------------------------------------------------------------------

class RoiResult:
    """Duck-types the fields of old/new eventem's own Roi object
    (Roi_scan_image/Roi_diffraction_pattern/get_4D()) so every existing
    caller of load_tpx3 (loaders.py's own get_dp, tab_roi_4d.py,
    tracking_utils_ui.py) keeps working unchanged regardless of which
    backend actually produced the result - old/new eventem's Roi run_roi/
    run_roi_masked return this wrapping their own object's real attributes;
    pyeventem's plain numpy arrays are wrapped in one of these directly,
    since its Roi sink has no equivalent object of its own."""

    def __init__(self, scan_image, diffraction_pattern, roi_4d=None):
        self.Roi_scan_image = np.asarray(scan_image)
        self.Roi_diffraction_pattern = np.asarray(diffraction_pattern)
        self._roi_4d = None if roi_4d is None else np.asarray(roi_4d)

    def get_4D(self):
        if self._roi_4d is None:
            raise RuntimeError('This Roi result was not computed with get_4d=True')
        return self._roi_4d


def run_roi(fn, scan_size, roi_rect=None, dwell_time_ns=1000.0, det_shape=(512, 512),
            fn_pattern=None, repetitions=1, get_4d=False, backend=BACKEND_OLD,
            decluster_cfg=None, logger_=None, n_threads=None, bitdepth=None,
            execution_strategy=EXEC_THREADS):
    """Rectangular ROI extraction: scan image + diffraction pattern (and,
    if get_4d, a sub-cube). ``roi_rect`` is (x, y, width, height); None =
    the whole scan. Returns a :class:`RoiResult`, matching load_tpx3's
    existing contract regardless of backend (callers read
    ``.Roi_scan_image``/``.Roi_diffraction_pattern``/``.get_4D()``).
    ``bitdepth``: old/new eventem's accumulator bit depth (``set_bitdepth``);
    pyeventem's equivalent constructor arg (default 64/uint64 in both if
    left None)."""
    decluster_cfg = decluster_cfg or default_decluster_cfg()
    decluster_on = _decluster_active(backend, decluster_cfg, 'roi')
    if roi_rect is None:
        x, y, w, h = 0, 0, scan_size[0], scan_size[1]
    else:
        x, y, w, h = roi_rect

    if backend == BACKEND_NEW and _new_eventem_needs_subprocess():
        return _run_new_eventem_via_subprocess('run_roi', dict(
            fn=fn, scan_size=list(scan_size), roi_rect=list(roi_rect) if roi_rect is not None else None,
            dwell_time_ns=dwell_time_ns, det_shape=list(det_shape), fn_pattern=fn_pattern,
            repetitions=repetitions, get_4d=get_4d, backend=backend, decluster_cfg=decluster_cfg,
            n_threads=n_threads, bitdepth=bitdepth,
        ), logger_=logger_)

    if backend in (BACKEND_OLD, BACKEND_NEW):
        mod = _old_eventem() if backend == BACKEND_OLD else _new_eventem()
        roi_obj = mod.Roi(repetitions=repetitions, extract_4D=get_4d)
        if n_threads is not None:
            roi_obj.n_threads = n_threads
        if bitdepth is not None:
            roi_obj.set_bitdepth(bitdepth)
        roi_obj.nx = scan_size[0]
        roi_obj.ny = scan_size[1]
        _set_detector_size(roi_obj, backend, det_shape)
        roi_obj.set_file(fn)
        if fn_pattern is not None:
            roi_obj.set_pattern_file(fn_pattern)
        roi_obj.set_roi(x=x, y=y, width=w, height=h)
        roi_obj.set_dwell_time(int(round(dwell_time_ns)))
        if decluster_on:
            _apply_decluster_eventem(roi_obj, decluster_cfg)
        from .progress import redirect_console_to_logger
        try:
            with redirect_console_to_logger(logger_, 'Loading tpx3'):
                roi_obj.run()
            roi_4d = roi_obj.get_4D() if get_4d else None
            return RoiResult(roi_obj.Roi_scan_image, roi_obj.Roi_diffraction_pattern, roi_4d)
        finally:
            roi_obj.close_socket()

    roi_kwargs = {} if bitdepth is None else {'bitdepth': bitdepth}
    scan_image, diffraction_pattern, roi_4d = _pyeventem_dispatch(
        'roi', dict(nx=scan_size[0], ny=scan_size[1], x=x, y=y, width=w, height=h,
                   detector_size=det_shape[0], extract_4d=get_4d, **roi_kwargs),
        fn, scan_size, dwell_time_ns, fn_pattern,
        n_threads, execution_strategy, decluster_cfg, decluster_on,
        row_band=(y, y + h), logger_=logger_)
    return RoiResult(scan_image, diffraction_pattern, roi_4d)


def run_roi_masked(fn, scan_size, mask, dwell_time_ns=1000.0, det_shape=(512, 512),
                    fn_pattern=None, repetitions=1, backend=BACKEND_OLD,
                    decluster_cfg=None, logger_=None, n_threads=None, bitdepth=None,
                    execution_strategy=EXEC_THREADS):
    """Arbitrary-mask ROI extraction (ROI Tracker / SAM2's masked/patched
    3DED extraction). ``mask``: 2-D array, shape (ny, nx), truthy = included
    scan position. Returns a :class:`RoiResult` (``.Roi_scan_image`` here is
    effectively meaningless - one accumulated diffraction pattern for the
    whole mask, not a real per-pixel layout - callers of this path only
    ever use ``.Roi_diffraction_pattern``; included anyway for a uniform
    return type with run_roi). ``bitdepth``: see run_roi's docstring."""
    decluster_cfg = decluster_cfg or default_decluster_cfg()
    decluster_on = _decluster_active(backend, decluster_cfg, 'roi')
    mask = np.asarray(mask)

    if backend == BACKEND_NEW and _new_eventem_needs_subprocess():
        return _run_new_eventem_via_subprocess('run_roi_masked', dict(
            fn=fn, scan_size=list(scan_size), dwell_time_ns=dwell_time_ns, det_shape=list(det_shape),
            fn_pattern=fn_pattern, repetitions=repetitions, backend=backend,
            decluster_cfg=decluster_cfg, n_threads=n_threads, bitdepth=bitdepth,
        ), mask_array=mask, logger_=logger_)

    if backend in (BACKEND_OLD, BACKEND_NEW):
        mod = _old_eventem() if backend == BACKEND_OLD else _new_eventem()
        roi_obj = mod.Roi(repetitions=repetitions, extract_4D=False)
        if n_threads is not None:
            roi_obj.n_threads = n_threads
        if bitdepth is not None:
            roi_obj.set_bitdepth(bitdepth)
        roi_obj.nx = scan_size[0]
        roi_obj.ny = scan_size[1]
        _set_detector_size(roi_obj, backend, det_shape)
        roi_obj.set_file(fn)
        if fn_pattern is not None:
            roi_obj.set_pattern_file(fn_pattern)
        roi_obj.set_roi_mask([mask.flatten().astype(np.int32)])
        roi_obj.set_dwell_time(int(round(dwell_time_ns)))
        if decluster_on:
            _apply_decluster_eventem(roi_obj, decluster_cfg)
        from .progress import redirect_console_to_logger
        try:
            with redirect_console_to_logger(logger_, 'Loading tpx3'):
                roi_obj.run()
            return RoiResult(roi_obj.Roi_scan_image, roi_obj.Roi_diffraction_pattern)
        finally:
            roi_obj.close_socket()

    roi_kwargs = {} if bitdepth is None else {'bitdepth': bitdepth}
    scan_image, diffraction_pattern, _roi_4d = _pyeventem_dispatch(
        'roi', dict(nx=scan_size[0], ny=scan_size[1], detector_size=det_shape[0], mask=mask, **roi_kwargs),
        fn, scan_size, dwell_time_ns, fn_pattern,
        n_threads, execution_strategy, decluster_cfg, decluster_on, logger_=logger_)
    return RoiResult(scan_image, diffraction_pattern)
