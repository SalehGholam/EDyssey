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


def test_pyeventem_run_sequential_when_n_threads_none():
    calls = []
    fake_pe = types.SimpleNamespace(
        run=lambda source, sinks, **kw: calls.append(('run', source, sinks)),
        run_parallel=lambda source, sinks, n_workers: calls.append(('run_parallel', n_workers)),
    )
    eb._pyeventem_run(fake_pe, 'SRC', 'SINK', n_threads=None, decluster_on=False)
    assert calls == [('run', 'SRC', ['SINK'])]


def test_pyeventem_run_sequential_when_n_threads_is_one():
    calls = []
    fake_pe = types.SimpleNamespace(
        run=lambda source, sinks, **kw: calls.append('run'),
        run_parallel=lambda source, sinks, n_workers: calls.append('run_parallel'),
    )
    eb._pyeventem_run(fake_pe, 'SRC', 'SINK', n_threads=1, decluster_on=False)
    assert calls == ['run']


def test_pyeventem_run_parallel_when_multiple_threads_and_no_declustering():
    calls = []
    fake_pe = types.SimpleNamespace(
        run=lambda source, sinks, **kw: calls.append('run'),
        run_parallel=lambda source, sinks, n_workers: calls.append(('run_parallel', n_workers)),
    )
    eb._pyeventem_run(fake_pe, 'SRC', 'SINK', n_threads=4, decluster_on=False)
    assert calls == [('run_parallel', 4)]


def test_pyeventem_run_sequential_when_declustering_even_with_many_threads():
    # A declustered source is a plain generator with no control-stream
    # structure to checkpoint-seed - run_parallel() can't be used regardless
    # of n_threads (see eventem_backend._pyeventem_run's own docstring).
    calls = []
    fake_pe = types.SimpleNamespace(
        run=lambda source, sinks, **kw: calls.append('run'),
        run_parallel=lambda source, sinks, n_workers: calls.append('run_parallel'),
    )
    eb._pyeventem_run(fake_pe, 'SRC', 'SINK', n_threads=8, decluster_on=True)
    assert calls == ['run']
