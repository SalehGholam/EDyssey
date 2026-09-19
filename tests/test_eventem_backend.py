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
# _run_new_eventem_via_subprocess's request/response marshaling (mocked
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


class _FakeCompletedProcess:
    """Stands in for subprocess.Popen - only .pid/.stdout are touched by
    _run_new_eventem_via_subprocess before pipe_process_output_to_logger
    (itself mocked out in these tests) takes over."""
    pid = 12345
    stdout = None


def _patch_subprocess_plumbing(monkeypatch, tmp_path, returncode=0, tail_lines=()):
    """Mocks every external dependency _run_new_eventem_via_subprocess
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


def test_run_new_eventem_via_subprocess_array_result(monkeypatch, tmp_path):
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

    result = eb._run_new_eventem_via_subprocess('run_pacbed', {'fn': 'f.tpx3', 'scan_size': [512, 512]})
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    assert captured_request['func'] == 'run_pacbed'
    assert captured_request['kwargs']['fn'] == 'f.tpx3'


def test_run_new_eventem_via_subprocess_roi_result_with_mask(monkeypatch, tmp_path):
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
    result = eb._run_new_eventem_via_subprocess(
        'run_roi_masked', {'fn': 'f.tpx3', 'scan_size': [4, 4]}, mask_array=mask)
    assert isinstance(result, eb.RoiResult)
    assert 'mask_path' in captured_request['kwargs']
    with pytest.raises(RuntimeError):
        result.get_4D()  # no roi_4d in this fake result - matches get_4d=False


def test_run_new_eventem_via_subprocess_raises_on_nonzero_exit(monkeypatch, tmp_path):
    _patch_subprocess_plumbing(monkeypatch, tmp_path, returncode=1, tail_lines=['Traceback...', 'RuntimeError: boom'])
    with pytest.raises(RuntimeError, match='boom'):
        eb._run_new_eventem_via_subprocess('run_pacbed', {'fn': 'f.tpx3', 'scan_size': [512, 512]})


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
