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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVENTEM_NEW_DIR = os.path.join(_THIS_DIR, 'eventem_new')

_eventem_new_mod = None


def _old_eventem():
    import eventem
    return eventem


def _new_eventem():
    global _eventem_new_mod
    if _eventem_new_mod is None:
        if _EVENTEM_NEW_DIR not in sys.path:
            sys.path.insert(0, _EVENTEM_NEW_DIR)
        import eventem_new
        _eventem_new_mod = eventem_new
    return _eventem_new_mod


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


def _pyeventem_run(pe, source, sink, n_threads):
    """Sequential pe.run() - the checkpoint-seeded pe.run_parallel() path
    only supports Tpx3Raster/Tpx3Pixeltrig sources with clone()/merge()-
    capable sinks and its own n_workers knob, which doesn't map cleanly onto
    this app's existing `n_threads` concept; not wired in here (a real
    follow-up, not a correctness gap - see BENCHMARKS.md's own parallel-
    decode numbers for the win available)."""
    pe.run(source, [sink])


# ---------------------------------------------------------------------------
# Pacbed
# ---------------------------------------------------------------------------

def run_pacbed(fn, scan_size, dwell_time_ns=1000.0, det_shape=(512, 512),
               fn_pattern=None, repetitions=1, backend=BACKEND_OLD,
               decluster_cfg=None, logger_=None, n_threads=None):
    """Full-frame summed diffraction pattern. Returns a (det_y, det_x) array."""
    decluster_cfg = decluster_cfg or default_decluster_cfg()
    decluster_on = _decluster_active(backend, decluster_cfg, 'pacbed')

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
        with redirect_console_to_logger(logger_, 'Loading tpx3'):
            dp.run()
        return np.array(dp.Pacbed_image).reshape(det_shape[1], det_shape[0])

    pe = _pyeventem()
    source = _pyeventem_source(pe, fn, scan_size, dwell_time_ns, fn_pattern, n_threads)
    sink = pe.Pacbed(detector_size=det_shape[0])
    if decluster_on:
        source = pe.decluster_source(source, _make_declusterer(decluster_cfg))
    _pyeventem_run(pe, source, sink, n_threads)
    return np.asarray(sink.image)


# ---------------------------------------------------------------------------
# vSTEM
# ---------------------------------------------------------------------------

def run_vstem(fn, scan_size, dwell_time_ns=1000.0, r_in=0, r_out=1 << 15,
              offset=None, det_shape=(512, 512), fn_pattern=None, repetitions=1,
              backend=BACKEND_OLD, decluster_cfg=None, logger_=None, n_threads=None):
    """Virtual-STEM navigation image, one or several annular detectors at
    once. Returns a (ny, nx) array (single detector) - see old-eventem's
    ``get_image()``/pyeventem's ``images[0]`` for the multi-detector case,
    not currently exposed through this facade (EDyssey only ever asks for
    one detector's worth per call today)."""
    decluster_cfg = decluster_cfg or default_decluster_cfg()
    decluster_on = _decluster_active(backend, decluster_cfg, 'vstem')
    inner = list(r_in) if isinstance(r_in, (list, tuple)) else [r_in]
    outer = list(r_out) if isinstance(r_out, (list, tuple)) else [r_out]

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
        with redirect_console_to_logger(logger_, 'Loading tpx3'):
            vstem.run()
        return vstem.get_image()

    pe = _pyeventem()
    source = _pyeventem_source(pe, fn, scan_size, dwell_time_ns, fn_pattern, n_threads)
    centers = None
    if offset:
        offsets = offset if isinstance(offset[0], (list, tuple)) else [offset]
        centers = [(float(o[0]), float(o[1])) for o in offsets]
    sink = pe.VSTEM(nx=scan_size[0], ny=scan_size[1], inner_radii=inner, outer_radii=outer,
                    centers=centers, detector_size=det_shape[0])
    if decluster_on:
        source = pe.decluster_source(source, _make_declusterer(decluster_cfg))
    _pyeventem_run(pe, source, sink, n_threads)
    return np.asarray(sink.images[0])


# ---------------------------------------------------------------------------
# Var
# ---------------------------------------------------------------------------

def run_var(fn, scan_size, dwell_time_ns=1000.0, r_in=0, r_out=1 << 15,
            offset=None, det_shape=(512, 512), fn_pattern=None, repetitions=1,
            backend=BACKEND_OLD, decluster_cfg=None, logger_=None, n_threads=None):
    """Per-scan-position variance image (single annular region only - see
    calculate_nav_img_variance_tpx3's docstring for why there's no
    multi-detector variant). Returns a (ny, nx) array."""
    decluster_cfg = decluster_cfg or default_decluster_cfg()
    decluster_on = _decluster_active(backend, decluster_cfg, 'var')

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
        with redirect_console_to_logger(logger_, 'Loading tpx3'):
            var.run()
        return np.array(var.Var_image).reshape(scan_size[1], scan_size[0])

    pe = _pyeventem()
    source = _pyeventem_source(pe, fn, scan_size, dwell_time_ns, fn_pattern, n_threads)
    center = (float(offset[0]), float(offset[1])) if offset else None
    sink = pe.Var(nx=scan_size[0], ny=scan_size[1], inner_radius=float(r_in), outer_radius=float(r_out),
                  center=center, detector_size=det_shape[0])
    if decluster_on:
        source = pe.decluster_source(source, _make_declusterer(decluster_cfg))
    _pyeventem_run(pe, source, sink, n_threads)
    return np.asarray(sink.image)


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
            decluster_cfg=None, logger_=None, n_threads=None, bitdepth=None):
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
        with redirect_console_to_logger(logger_, 'Loading tpx3'):
            roi_obj.run()
        roi_4d = roi_obj.get_4D() if get_4d else None
        return RoiResult(roi_obj.Roi_scan_image, roi_obj.Roi_diffraction_pattern, roi_4d)

    pe = _pyeventem()
    source = _pyeventem_source(pe, fn, scan_size, dwell_time_ns, fn_pattern, n_threads)
    roi_kwargs = {} if bitdepth is None else {'bitdepth': bitdepth}
    sink = pe.Roi(nx=scan_size[0], ny=scan_size[1], x=x, y=y, width=w, height=h,
                  detector_size=det_shape[0], extract_4d=get_4d, **roi_kwargs)
    if decluster_on:
        source = pe.decluster_source(source, _make_declusterer(decluster_cfg))
    _pyeventem_run(pe, source, sink, n_threads)
    roi_4d = sink.roi_4d if get_4d else None
    return RoiResult(sink.scan_image, sink.diffraction_pattern, roi_4d)


def run_roi_masked(fn, scan_size, mask, dwell_time_ns=1000.0, det_shape=(512, 512),
                    fn_pattern=None, repetitions=1, backend=BACKEND_OLD,
                    decluster_cfg=None, logger_=None, n_threads=None):
    """Arbitrary-mask ROI extraction (ROI Tracker / SAM2's masked/patched
    3DED extraction). ``mask``: 2-D array, shape (ny, nx), truthy = included
    scan position. Returns a :class:`RoiResult` (``.Roi_scan_image`` here is
    effectively meaningless - one accumulated diffraction pattern for the
    whole mask, not a real per-pixel layout - callers of this path only
    ever use ``.Roi_diffraction_pattern``; included anyway for a uniform
    return type with run_roi)."""
    decluster_cfg = decluster_cfg or default_decluster_cfg()
    decluster_on = _decluster_active(backend, decluster_cfg, 'roi')
    mask = np.asarray(mask)

    if backend in (BACKEND_OLD, BACKEND_NEW):
        mod = _old_eventem() if backend == BACKEND_OLD else _new_eventem()
        roi_obj = mod.Roi(repetitions=repetitions, extract_4D=False)
        if n_threads is not None:
            roi_obj.n_threads = n_threads
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
        with redirect_console_to_logger(logger_, 'Loading tpx3'):
            roi_obj.run()
        return RoiResult(roi_obj.Roi_scan_image, roi_obj.Roi_diffraction_pattern)

    pe = _pyeventem()
    source = _pyeventem_source(pe, fn, scan_size, dwell_time_ns, fn_pattern, n_threads)
    sink = pe.Roi(nx=scan_size[0], ny=scan_size[1], detector_size=det_shape[0], mask=mask)
    if decluster_on:
        source = pe.decluster_source(source, _make_declusterer(decluster_cfg))
    _pyeventem_run(pe, source, sink, n_threads)
    return RoiResult(sink.scan_image, sink.diffraction_pattern)
