# -*- coding: utf-8 -*-
"""Tests for EDyssey.io_utils.eventem_backend - the parts that don't need a
real .tpx3 file or a compiled eventem/eventem_new .pyd (neither is checked
into this repo - see INSTALL.md section 3b - and no personal machine's real
acquisition data is used in tests, matching test_metadata.py's convention).

Covers: decluster_cfg construction/gating logic, the LUT text-file parser,
RoiResult's duck-typed interface, and _set_detector_size's old/new-eventem
branching - using tiny mock objects standing in for real eventem.Pacbed/
vSTEM/Var/Roi instances, plus pyeventem itself (a pure-Python/numba
dependency this repo already requires - see requirements.txt - so exercising
it directly, unlike old/new eventem, needs nothing extra)."""
import types

import numpy as np
import pytest

from EDyssey.io_utils import eventem_backend as eb


# ---------------------------------------------------------------------------
# default_decluster_cfg / load_electron_count_lut
# ---------------------------------------------------------------------------

def test_default_decluster_cfg_shape():
    cfg = eb.default_decluster_cfg()
    assert cfg['enabled'] is False
    assert set(cfg['sinks']) == {'pacbed', 'roi', 'vstem', 'var'}
    assert all(cfg['sinks'].values())


def test_load_electron_count_lut(tmp_path):
    fn = tmp_path / 'lut.txt'
    fn.write_text('# comment line\n1 2 3\n4 5 6\n', encoding='utf-8')
    lut = eb.load_electron_count_lut(str(fn))
    assert lut.dtype == np.int64
    assert lut.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_load_electron_count_lut_single_row(tmp_path):
    fn = tmp_path / 'lut.txt'
    fn.write_text('10 20 30\n', encoding='utf-8')
    lut = eb.load_electron_count_lut(str(fn))
    assert lut.shape == (1, 3)


# ---------------------------------------------------------------------------
# _decluster_active
# ---------------------------------------------------------------------------

def test_decluster_active_requires_enabled():
    cfg = eb.default_decluster_cfg()
    assert eb._decluster_active(eb.BACKEND_NEW, cfg, 'roi') is False  # enabled=False
    cfg['enabled'] = True
    assert eb._decluster_active(eb.BACKEND_NEW, cfg, 'roi') is True


def test_decluster_active_old_backend_never_active(caplog):
    cfg = eb.default_decluster_cfg()
    cfg['enabled'] = True
    assert eb._decluster_active(eb.BACKEND_OLD, cfg, 'pacbed') is False


def test_decluster_active_per_sink_gating():
    cfg = eb.default_decluster_cfg()
    cfg['enabled'] = True
    cfg['sinks']['roi'] = False
    assert eb._decluster_active(eb.BACKEND_NEW, cfg, 'roi') is False
    assert eb._decluster_active(eb.BACKEND_NEW, cfg, 'pacbed') is True


def test_decluster_active_missing_sinks_key_defaults_to_enabled():
    # An older/hand-built cfg with no 'sinks' key at all - see
    # default_decluster_cfg's own comment on this fallback.
    cfg = {'enabled': True}
    assert eb._decluster_active(eb.BACKEND_NEW, cfg, 'roi') is True


def test_decluster_active_none_cfg():
    assert eb._decluster_active(eb.BACKEND_NEW, None, 'roi') is False


# ---------------------------------------------------------------------------
# _apply_decluster_eventem
# ---------------------------------------------------------------------------

def test_apply_decluster_eventem_tot_per_electron():
    obj = types.SimpleNamespace()
    cfg = eb.default_decluster_cfg()
    cfg['dtime_ns'] = 100.0
    eb._apply_decluster_eventem(obj, cfg)
    assert obj.decluster is True
    assert obj.dtime == round(100.0 / 1.5625)
    assert obj.tot_per_electron == cfg['tot_per_electron']
    assert not hasattr(obj, 'electron_count_lut_file')


def test_apply_decluster_eventem_lut_takes_priority(tmp_path):
    fn = tmp_path / 'lut.txt'
    fn.write_text('1 2\n', encoding='utf-8')
    obj = types.SimpleNamespace()
    cfg = eb.default_decluster_cfg()
    cfg['lut_file'] = str(fn)
    eb._apply_decluster_eventem(obj, cfg)
    assert obj.electron_count_lut_file == str(fn)
    assert not hasattr(obj, 'tot_per_electron')


# ---------------------------------------------------------------------------
# _set_detector_size
# ---------------------------------------------------------------------------

def test_set_detector_size_old_backend_writes_x_y():
    obj = types.SimpleNamespace(detector_size_x=256, detector_size_y=256)
    eb._set_detector_size(obj, eb.BACKEND_OLD, (512, 512))
    assert (obj.detector_size_x, obj.detector_size_y) == (512, 512)


def test_set_detector_size_old_backend_noop_when_matching():
    obj = types.SimpleNamespace(detector_size_x=512, detector_size_y=512)
    eb._set_detector_size(obj, eb.BACKEND_OLD, (512, 512))
    assert (obj.detector_size_x, obj.detector_size_y) == (512, 512)


def test_set_detector_size_new_backend_requires_square():
    obj = types.SimpleNamespace(detector_size=512)
    with pytest.raises(ValueError):
        eb._set_detector_size(obj, eb.BACKEND_NEW, (512, 256))


def test_set_detector_size_new_backend_writes_single_value():
    obj = types.SimpleNamespace(detector_size=256)
    eb._set_detector_size(obj, eb.BACKEND_NEW, (512, 512))
    assert obj.detector_size == 512


# ---------------------------------------------------------------------------
# RoiResult
# ---------------------------------------------------------------------------

def test_roi_result_basic_fields():
    r = eb.RoiResult([1, 2, 3], [[4, 5], [6, 7]])
    assert np.array_equal(r.Roi_scan_image, [1, 2, 3])
    assert np.array_equal(r.Roi_diffraction_pattern, [[4, 5], [6, 7]])


def test_roi_result_get_4d_without_data_raises():
    r = eb.RoiResult([1], [[1]])
    with pytest.raises(RuntimeError):
        r.get_4D()


def test_roi_result_get_4d_with_data():
    cube = np.zeros((2, 2, 2, 2))
    r = eb.RoiResult([1], [[1]], roi_4d=cube)
    assert np.array_equal(r.get_4D(), cube)


# ---------------------------------------------------------------------------
# _pyeventem_run: parallel-vs-sequential dispatch (pyeventem itself is a
# required dependency - see requirements.txt - so this exercises the real
# pe.run/pe.run_parallel, not a mock, but with no real source/sink needed
# since we only check *which* function gets called).
# ---------------------------------------------------------------------------

pe = pytest.importorskip('pyeventem')


def test_pyeventem_run_parallel_when_n_threads_is_auto(monkeypatch):
    """"Auto" (None) used to dispatch to the sequential path. It now
    resolves to a worker count, matching what Auto means for the C++
    backends - and mattering much more than it used to, since only the
    parallel path restricts the decode to the ROI's word range."""
    monkeypatch.setattr(eb.os, 'cpu_count', lambda: 16)
    calls = []
    fake_pe = types.SimpleNamespace(
        run=lambda source, sinks, **kw: calls.append(('run', source, sinks)),
        run_parallel=lambda source, sinks, **kw: calls.append(('run_parallel', source, sinks, kw)),
        decluster_source=lambda source, declusterer: source,
    )
    eb._pyeventem_run(fake_pe, 'SRC', 'SINK', n_threads=None, decluster_on=False)
    assert calls == [('run_parallel', 'SRC', ['SINK'],
                      {'n_workers': eb._PYEVENTEM_AUTO_WORKERS,
                       'declusterer': None, 'progress': True})]


def test_pyeventem_run_sequential_when_n_threads_is_one():
    calls = []
    fake_pe = types.SimpleNamespace(
        run=lambda source, sinks, **kw: calls.append('run'),
        run_parallel=lambda source, sinks, **kw: calls.append('run_parallel'),
        decluster_source=lambda source, declusterer: source,
    )
    eb._pyeventem_run(fake_pe, 'SRC', 'SINK', n_threads=1, decluster_on=False)
    assert calls == ['run']


def test_pyeventem_run_parallel_when_multiple_threads_and_no_declustering():
    calls = []
    fake_pe = types.SimpleNamespace(
        run=lambda source, sinks, **kw: calls.append('run'),
        run_parallel=lambda source, sinks, **kw: calls.append(('run_parallel', kw)),
        decluster_source=lambda source, declusterer: source,
    )
    eb._pyeventem_run(fake_pe, 'SRC', 'SINK', n_threads=4, decluster_on=False)
    assert calls == [('run_parallel', {'n_workers': 4, 'declusterer': None, 'progress': True})]


def test_pyeventem_run_parallel_declustered_when_multiple_threads():
    # Bounded-memory parallel declustering: each of n_threads chunks
    # declustered independently via pe.run_parallel(declusterer=...), not
    # a decluster_source()-wrapped sequential pe.run() (see
    # eventem_backend._pyeventem_run's own docstring for the trade-off).
    calls = []
    fake_pe = types.SimpleNamespace(
        run=lambda source, sinks, **kw: calls.append('run'),
        run_parallel=lambda source, sinks, **kw: calls.append(('run_parallel', kw)),
        decluster_source=lambda source, declusterer: calls.append('decluster_source') or source,
    )
    eb._pyeventem_run(fake_pe, 'SRC', 'SINK', n_threads=8, decluster_on=True, declusterer='DECL')
    assert calls == [('run_parallel', {'n_workers': 8, 'declusterer': 'DECL', 'progress': True})]


def test_pyeventem_run_sequential_declustered_when_single_threaded():
    calls = []
    fake_pe = types.SimpleNamespace(
        run=lambda source, sinks, **kw: calls.append(('run', source, kw)),
        run_parallel=lambda source, sinks, **kw: calls.append('run_parallel'),
        decluster_source=lambda source, declusterer: f'DECLUSTERED({source},{declusterer})',
    )
    eb._pyeventem_run(fake_pe, 'SRC', 'SINK', n_threads=1, decluster_on=True, declusterer='DECL')
    assert calls == [('run', 'DECLUSTERED(SRC,DECL)', {'progress': True})]


# ---------------------------------------------------------------------------
# New eventem's subprocess delegation (importing eventem_new directly in a
# process that already has PyQt5 loaded segfaults - see _new_eventem's own
# comment) - _to_jsonable, the sys.modules-based detection, and
# _run_eventem_via_subprocess's request/response marshaling (mocked
# subprocess - these don't need a real eventem_new.pyd or PyQt5 installed).
# ---------------------------------------------------------------------------

def test_to_jsonable_tuples_become_lists():
    assert eb._to_jsonable((1, 2, 3)) == [1, 2, 3]
    assert eb._to_jsonable([(1, 2), (3, 4)]) == [[1, 2], [3, 4]]


def test_to_jsonable_passes_through_scalars_and_none():
    assert eb._to_jsonable(None) is None
    assert eb._to_jsonable(5) == 5
    assert eb._to_jsonable(5.5) == 5.5
    assert eb._to_jsonable('x') == 'x'


def test_to_jsonable_numpy_scalar():
    out = eb._to_jsonable(np.float64(3.5))
    assert out == 3.5
    assert isinstance(out, float)


def test_new_eventem_needs_subprocess_reflects_pyqt5_presence(monkeypatch):
    monkeypatch.delitem(eb.sys.modules, 'PyQt5.QtCore', raising=False)
    assert eb._new_eventem_needs_subprocess() is False
    eb.sys.modules['PyQt5.QtCore'] = object()
    try:
        assert eb._new_eventem_needs_subprocess() is True
    finally:
        del eb.sys.modules['PyQt5.QtCore']


# ---------------------------------------------------------------------------
# _needs_subprocess / _resident_backend: old eventem and pyeventem cannot
# coexist in the same OS process (confirmed directly - one corrupts the
# other's results, then segfaults, far faster than either alone) - the
# first backend to actually run in-process "owns" the process; every call
# for a *different* backend must be routed through a subprocess instead.
# _resident_backend is reset around each test (a plain module-level global,
# not per-test state) so these can't leak into each other or into whichever
# other test in this file happens to run first/after.
# ---------------------------------------------------------------------------

@pytest.fixture
def _reset_resident_backend(monkeypatch):
    """Only for the tests below - _resident_backend is a plain module-level
    global, not per-test state, so it (and, where relevant, whether
    PyQt5.QtCore looks "loaded") must be pinned to a known value around
    each of these rather than leaking from/into whichever other test in
    this file happens to run first/after."""
    monkeypatch.setattr(eb, '_resident_backend', None)
    monkeypatch.delitem(eb.sys.modules, 'PyQt5.QtCore', raising=False)


def test_first_backend_claims_residency_and_stays_in_process(_reset_resident_backend):
    assert eb._needs_subprocess(eb.BACKEND_OLD) is False
    assert eb._resident_backend == eb.BACKEND_OLD


def test_same_backend_repeated_stays_in_process(_reset_resident_backend):
    assert eb._needs_subprocess(eb.BACKEND_PYEVENTEM) is False
    assert eb._needs_subprocess(eb.BACKEND_PYEVENTEM) is False
    assert eb._needs_subprocess(eb.BACKEND_PYEVENTEM) is False


def test_switching_to_a_different_backend_needs_a_subprocess(_reset_resident_backend):
    assert eb._needs_subprocess(eb.BACKEND_OLD) is False  # old eventem claims residency
    assert eb._needs_subprocess(eb.BACKEND_PYEVENTEM) is True  # pyeventem must not run in-process
    # old eventem itself is still the resident backend - stays fast.
    assert eb._needs_subprocess(eb.BACKEND_OLD) is False


def test_switching_back_and_forth_repeatedly_never_lets_the_non_resident_one_in(_reset_resident_backend):
    eb._needs_subprocess(eb.BACKEND_PYEVENTEM)  # pyeventem claims residency first this time
    for _ in range(5):
        assert eb._needs_subprocess(eb.BACKEND_OLD) is True
        assert eb._needs_subprocess(eb.BACKEND_PYEVENTEM) is False


def test_new_eventem_with_pyqt5_loaded_always_needs_a_subprocess_regardless_of_residency(_reset_resident_backend):
    eb.sys.modules['PyQt5.QtCore'] = object()
    assert eb._needs_subprocess(eb.BACKEND_NEW) is True
    # And doesn't claim residency by going through that path - a
    # not-yet-resident old eventem/pyeventem call right after must still
    # get the fast, in-process path.
    assert eb._resident_backend is None
    assert eb._needs_subprocess(eb.BACKEND_OLD) is False


def test_new_eventem_without_pyqt5_can_claim_residency_like_any_other_backend(_reset_resident_backend):
    """The Qt-free case (a real --worker subprocess) - New eventem has no
    special exemption from the cross-backend rule once it's not being
    routed out for the PyQt5 reason."""
    assert eb._needs_subprocess(eb.BACKEND_NEW) is False
    assert eb._resident_backend == eb.BACKEND_NEW
    assert eb._needs_subprocess(eb.BACKEND_PYEVENTEM) is True


class _FakeCompletedProcess:
    """Stands in for subprocess.Popen - only .pid/.stdout are touched by
    _run_eventem_via_subprocess before pipe_process_output_to_logger
    (itself mocked out in these tests) takes over."""
    pid = 12345
    stdout = None


def _patch_subprocess_plumbing(monkeypatch, tmp_path, returncode=0, tail_lines=()):
    """Mocks every external dependency _run_eventem_via_subprocess
    reaches for (the worker_launch/worker_pool_utils modules it imports
    lazily, and subprocess.Popen itself), so these tests exercise only its
    own request-building/result-unpacking logic - never a real subprocess,
    real Qt, or real eventem_new."""
    import sys
    import types as _types

    fake_ui_tabs = _types.ModuleType('ui_tabs')
    fake_worker_launch = _types.ModuleType('ui_tabs.worker_launch')
    fake_worker_launch.worker_command = lambda name, args: ('python', ['worker_eventem_call.py', *args])
    fake_ui_tabs.worker_launch = fake_worker_launch
    monkeypatch.setitem(sys.modules, 'ui_tabs', fake_ui_tabs)
    monkeypatch.setitem(sys.modules, 'ui_tabs.worker_launch', fake_worker_launch)

    fake_wpu = _types.ModuleType('worker_pool_utils')
    fake_wpu.create_job_object = lambda: None
    fake_wpu.assign_process_to_job = lambda job_handle, pid: False
    fake_wpu.kill_job = lambda job_handle: None
    monkeypatch.setitem(sys.modules, 'worker_pool_utils', fake_wpu)

    import subprocess as _subprocess_module
    monkeypatch.setattr(_subprocess_module, 'Popen', lambda *a, **kw: _FakeCompletedProcess())

    monkeypatch.setattr(
        'EDyssey.io_utils.progress.pipe_process_output_to_logger',
        lambda proc, logger, label='': (returncode, list(tail_lines)),
    )


def test_run_eventem_via_subprocess_array_result(monkeypatch, tmp_path):
    _patch_subprocess_plumbing(monkeypatch, tmp_path)

    captured_request = {}

    # Intercept the request file write and supply a fake result.npz by
    # monkeypatching np.load instead of relying on a real subprocess to
    # produce one.
    def _fake_load(path, *a, **kw):
        class _NpzLike:
            files = ['array']

            def __getitem__(self, key):
                return np.array([1.0, 2.0, 3.0])

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _NpzLike()

    monkeypatch.setattr(np, 'load', _fake_load)

    import json as _json
    real_json_dump = _json.dump

    def _capturing_dump(obj, fp, *a, **kw):
        captured_request.update(obj)
        return real_json_dump(obj, fp, *a, **kw)

    monkeypatch.setattr(_json, 'dump', _capturing_dump)

    result = eb._run_eventem_via_subprocess('run_pacbed', {'fn': 'f.tpx3', 'scan_size': [512, 512]})
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    assert captured_request['func'] == 'run_pacbed'
    assert captured_request['kwargs']['fn'] == 'f.tpx3'


def test_run_eventem_via_subprocess_roi_result_with_mask(monkeypatch, tmp_path):
    _patch_subprocess_plumbing(monkeypatch, tmp_path)

    captured_request = {}

    def _fake_load(path, *a, **kw):
        class _NpzLike:
            files = ['scan_image', 'diffraction_pattern']

            def __getitem__(self, key):
                return np.zeros((2, 2))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _NpzLike()

    monkeypatch.setattr(np, 'load', _fake_load)

    import json as _json
    real_json_dump = _json.dump

    def _capturing_dump(obj, fp, *a, **kw):
        captured_request.update(obj)
        return real_json_dump(obj, fp, *a, **kw)

    monkeypatch.setattr(_json, 'dump', _capturing_dump)

    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    result = eb._run_eventem_via_subprocess(
        'run_roi_masked', {'fn': 'f.tpx3', 'scan_size': [4, 4]}, mask_array=mask)
    assert isinstance(result, eb.RoiResult)
    assert 'mask_path' in captured_request['kwargs']
    with pytest.raises(RuntimeError):
        result.get_4D()  # no roi_4d in this fake result - matches get_4d=False


def test_run_eventem_via_subprocess_raises_on_nonzero_exit(monkeypatch, tmp_path):
    _patch_subprocess_plumbing(monkeypatch, tmp_path, returncode=1, tail_lines=['Traceback...', 'RuntimeError: boom'])
    with pytest.raises(RuntimeError, match='boom'):
        eb._run_eventem_via_subprocess('run_pacbed', {'fn': 'f.tpx3', 'scan_size': [512, 512]})


def test_pyeventem_workers_resolves_auto(monkeypatch):
    """None/0 is the settings dialog's "Auto". It used to fall through to
    pyeventem's sequential path, which is not what Auto means for the C++
    backends (they start their own threads) and is now the slowest option
    by far - only the parallel path restricts the decode to the ROI's word
    range. An explicit 1 must still mean genuinely single-threaded."""
    monkeypatch.setattr(eb.os, 'cpu_count', lambda: 16)
    assert eb._pyeventem_workers(None) == eb._PYEVENTEM_AUTO_WORKERS
    assert eb._pyeventem_workers(0) == eb._PYEVENTEM_AUTO_WORKERS
    assert eb._pyeventem_workers(1) == 1
    assert eb._pyeventem_workers(5) == 5


def test_pyeventem_workers_auto_is_capped_by_the_machine(monkeypatch):
    monkeypatch.setattr(eb.os, 'cpu_count', lambda: 2)
    assert eb._pyeventem_workers(None) == 2
    # An explicit setting is the user's call, not ours to cap.
    assert eb._pyeventem_workers(12) == 12

    monkeypatch.setattr(eb.os, 'cpu_count', lambda: None)  # unknowable
    assert eb._pyeventem_workers(None) == 1


# ---------------------------------------------------------------------------
# New eventem's checkpoint-based multi-process Roi split
# (_new_eventem_build_worker_plan/_new_eventem_run_roi_multiprocess) -
# confirmed directly (against the real EvenTem C++ source and a real
# eventem_new.pyd, outside this test file - neither is checked into this
# repo, matching every other old/new-eventem test here) that
# n_threads/BoundedThreadPool never parallelizes New eventem's own .tpx3
# decode at all; find_checkpoints is the real mechanism. These tests mock
# out eventem_new/ProcessPoolExecutor entirely, so they exercise only this
# module's own orchestration logic (checkpoint -> worker plan -> job list ->
# sum), the same way test_run_eventem_via_subprocess_* mocks subprocess.Popen
# rather than needing a real child process.
# ---------------------------------------------------------------------------

def test_new_eventem_build_worker_plan_shape():
    # (byte_offset, start_line, dt, rise_t, rise_fall, line_count, chip_id)
    checkpoints = [
        (1000, 8, 111, [1, 2, 3, 4], [0, 0, 0, 0], [8, 8, 8, 8], 2),
        (2000, 16, 222, [5, 6, 7, 8], [0, 0, 0, 0], [16, 16, 16, 16], 1),
    ]
    plan = eb._new_eventem_build_worker_plan(checkpoints, 3)
    assert len(plan) == 3

    fbo0, lno0, stop0, seed0 = plan[0]
    assert (fbo0, lno0, seed0) == (0, 0, None)  # worker 0: true file start, no seed
    assert stop0 == 8  # stops where worker 1 starts

    fbo1, lno1, stop1, seed1 = plan[1]
    assert (fbo1, lno1) == (1000, 8)
    assert seed1 == {'dt': 111, 'rise_t': [1, 2, 3, 4], 'rise_fall': [0, 0, 0, 0],
                     'line_count': [8, 8, 8, 8], 'chip_id': 2}
    assert stop1 == 16

    fbo2, lno2, stop2, seed2 = plan[2]
    assert (fbo2, lno2) == (2000, 16)
    assert seed2['dt'] == 222
    assert stop2 == -1  # last worker reads to the true end of file


def test_new_eventem_build_worker_plan_single_worker():
    plan = eb._new_eventem_build_worker_plan([], 1)
    assert plan == [(0, 0, -1, None)]


class _FakeNewEventemRoi:
    """Stands in for eventem_new.Roi - only what
    _new_eventem_run_roi_multiprocess/_new_eventem_roi_worker touch."""
    _checkpoints = [(1000, 5, 50, [0] * 4, [0] * 4, [5] * 4, 0)]
    _find_checkpoints_error = None

    def __init__(self, repetitions=1, extract_4D=False):
        self.repetitions = repetitions
        self.extract_4D = extract_4D
        self.closed = False
        self.ran = False
        self.set_roi_calls = []
        self.set_roi_mask_calls = []
        self.detector_size = 0  # read by _set_detector_size(BACKEND_NEW) before it's ever set

    def set_file(self, fn):
        self.fn = fn

    def set_dwell_time(self, ns):
        self.dwell_time_ns = ns

    def set_bitdepth(self, bd):
        self.bitdepth = bd

    def set_pattern_file(self, fn_pattern):
        self.fn_pattern = fn_pattern

    def set_roi(self, x, y, width, height):
        self.set_roi_calls.append((x, y, width, height))

    def set_roi_mask(self, masks):
        self.set_roi_mask_calls.append(masks)

    def find_checkpoints(self, n_splits, allow_sidecar):
        if self._find_checkpoints_error is not None:
            raise self._find_checkpoints_error
        return self._checkpoints

    def run(self):
        self.ran = True

    def get_4D(self):
        return np.ones((2, 2))

    def close_socket(self):
        self.closed = True

    @property
    def Roi_scan_image(self):
        return np.ones((4, 4)) * (self.line_number_offset + 1)

    @property
    def Roi_diffraction_pattern(self):
        return np.ones((4, 4)) * (self.line_number_offset + 1)


@pytest.fixture
def fake_new_eventem_module(monkeypatch):
    """Injects a fake eventem_new module (via eb._new_eventem) so
    _new_eventem_run_roi_multiprocess/_new_eventem_roi_worker never touch a
    real .pyd - matches _set_detector_size's own SimpleNamespace-mock
    convention, just for a whole module instead of one object."""
    _FakeNewEventemRoi._find_checkpoints_error = None
    fake_module = types.SimpleNamespace(Roi=_FakeNewEventemRoi)
    monkeypatch.setattr(eb, '_new_eventem', lambda: fake_module)
    return fake_module


@pytest.fixture
def fake_process_pool(monkeypatch):
    """Runs ProcessPoolExecutor.map's jobs in-process (via a plain list
    comprehension) instead of spawning real child processes - the point of
    these tests is _new_eventem_run_roi_multiprocess's own orchestration,
    not real multiprocessing (already covered by manual, outside-pytest
    verification against a real eventem_new.pyd - see this module's own
    section docstring)."""
    class _FakeExecutor:
        def __init__(self, max_workers=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, func, jobs):
            return [func(job) for job in jobs]

    import concurrent.futures
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', _FakeExecutor)


def test_run_roi_multiprocess_returns_none_when_checkpointing_unsupported(fake_new_eventem_module):
    """A smart-scan pixel-trigger file (fn_pattern set) - find_checkpoints
    only supports plain-raster CHEETAH .tpx3 - must fall back to the
    caller's own single-process path, not raise."""
    result = eb._new_eventem_run_roi_multiprocess(
        'f.tpx3', (16, 16), (0, 0, 16, 16), None, 1000.0, (512, 512),
        'pattern.txt', 1, False, None, eb.default_decluster_cfg(), False, 4)
    assert result is None


def test_run_roi_multiprocess_returns_none_on_runtime_error(fake_new_eventem_module):
    _FakeNewEventemRoi._find_checkpoints_error = RuntimeError('not a CHEETAH file')
    result = eb._new_eventem_run_roi_multiprocess(
        'f.tpx3', (16, 16), (0, 0, 16, 16), None, 1000.0, (512, 512),
        None, 1, False, None, eb.default_decluster_cfg(), False, 4)
    assert result is None


def test_run_roi_multiprocess_sums_every_worker(fake_new_eventem_module, fake_process_pool):
    scan_image, dp, roi_4d = eb._new_eventem_run_roi_multiprocess(
        'f.tpx3', (16, 16), (0, 0, 16, 16), None, 1000.0, (512, 512),
        None, 1, False, None, eb.default_decluster_cfg(), False, 2)
    # 2 workers: worker 0 (line_number_offset=0) contributes 1s, worker 1
    # (line_number_offset=5, from the fake checkpoint) contributes 6s.
    assert np.array_equal(dp, np.full((4, 4), 7.0))
    assert np.array_equal(scan_image, np.full((4, 4), 7.0))
    assert roi_4d is None  # get_4d=False


def test_run_roi_multiprocess_extract_4d_sums_cubes(fake_new_eventem_module, fake_process_pool):
    _, _, roi_4d = eb._new_eventem_run_roi_multiprocess(
        'f.tpx3', (16, 16), (0, 0, 16, 16), None, 1000.0, (512, 512),
        None, 1, True, None, eb.default_decluster_cfg(), False, 2)
    assert roi_4d is not None
    assert roi_4d.shape == (2, 2)


def test_run_roi_multiprocess_uses_mask_not_rect(fake_new_eventem_module, fake_process_pool):
    mask_flat = np.array([1, 0, 1, 0], dtype=np.int32)
    eb._new_eventem_run_roi_multiprocess(
        'f.tpx3', (16, 16), None, mask_flat, 1000.0, (512, 512),
        None, 1, False, None, eb.default_decluster_cfg(), False, 2)
    # Every worker instance got set_roi_mask, never set_roi - can't inspect
    # the actual per-worker objects here (each is built fresh inside the
    # picklable worker function), but a mismatched-argument bug would have
    # raised (set_roi requires x/y/width/height, not a flat mask array) -
    # reaching this line at all is the assertion.


def test_run_roi_multiprocess_caps_workers_to_available_checkpoints(fake_new_eventem_module, monkeypatch):
    """find_checkpoints degrades gracefully on a small scan - only 1
    checkpoint means at most 2 real workers, however many were requested."""
    calls = []

    class _CountingExecutor:
        def __init__(self, max_workers=None):
            calls.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, func, jobs):
            return [func(job) for job in jobs]

    import concurrent.futures
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', _CountingExecutor)
    eb._new_eventem_run_roi_multiprocess(
        'f.tpx3', (16, 16), (0, 0, 16, 16), None, 1000.0, (512, 512),
        None, 1, False, None, eb.default_decluster_cfg(), False, 8)  # 8 requested, only 1 checkpoint available
    assert calls == [2]  # 1 checkpoint -> at most 2 workers, not 8
