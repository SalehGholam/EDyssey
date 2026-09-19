# -*- coding: utf-8 -*-
"""Regression tests for "DP by Threshold"'s two DP-computing workers
(Tab_ROI_on_4D's Worker_CalculateDP_Mask, Tab_Create_NavSignal's
_sum_dp_from_mask_worker).

Both had the same defect, found by comparing them against every *other*
DP-computing worker in the same files: they never passed `logger=` into
load_dp, so eventem/pyeventem's own progress never reached the Qt log
console for this specific button - the underlying computation still ran
and produced a correct result, but nothing in the app showed that (a
multi-minute declustered run looked hung). The navSignal tab's version had
a second, larger gap: it never read AnalysisBackendSettings at all, so
picking New eventem/pyeventem or turning declustering on had no effect on
this button specifically, unlike every neighboring one in the same tab.

`get_tab_logger`/the real Qt log handler are monkeypatched out rather than
exercised - they write to the real user data directory, which isn't
appropriate for a unit test and isn't what's being verified here anyway.
`worker.run()` is called directly (not via QThreadPool), and PyQt's direct
(same-thread) signal connection invokes connected slots synchronously with
no running event loop needed, so nothing here requires a QApplication.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

pytest.importorskip("PyQt5")

from ui_tabs import tab_roi_4d  # noqa: E402
from ui_tabs import tab_create_navSignal  # noqa: E402


class _FakeLogger:
    def __init__(self):
        self.calls = []

    def info(self, *a, **k):
        self.calls.append(('info', a, k))

    def warning(self, *a, **k):
        self.calls.append(('warning', a, k))

    def error(self, *a, **k):
        self.calls.append(('error', a, k))

    def exception(self, *a, **k):
        self.calls.append(('exception', a, k))


# --------------------------------------------------------------------
# Tab_ROI_on_4D: Worker_CalculateDP_Mask
# --------------------------------------------------------------------


def _make_mask_worker(monkeypatch, **overrides):
    fake_logger = _FakeLogger()
    monkeypatch.setattr(tab_roi_4d, 'get_tab_logger', lambda name: fake_logger)
    kwargs = dict(
        fn='dummy.tpx3', roi=(0, 0, 4, 4), mask=np.ones((4, 4), dtype=bool),
        dtype='.tpx3', scanSize=(4, 4), dwellTime=100.0, fn_pattern=None,
        det_shape=(512, 512), patch_mode=True, backend='pyeventem',
        decluster_cfg={'enabled': True}, n_threads=5,
    )
    kwargs.update(overrides)
    worker = tab_roi_4d.Worker_CalculateDP_Mask(**kwargs)
    return worker, fake_logger


def test_worker_calculate_dp_mask_passes_its_own_logger_to_load_dp(monkeypatch):
    worker, fake_logger = _make_mask_worker(monkeypatch)

    captured = {}

    def fake_load_dp(fn, **kw):
        captured.update(kw)
        return np.zeros((2, 2))

    monkeypatch.setattr(tab_roi_4d, 'load_dp', fake_load_dp)

    results = []
    worker.signals.result.connect(results.append)
    worker.run()

    assert 'logger' in captured, "Worker_CalculateDP_Mask must pass logger= through to load_dp"
    assert captured['logger'] is fake_logger
    assert len(results) == 1 and results[0].shape == (2, 2)


def test_worker_calculate_dp_mask_forwards_backend_settings(monkeypatch):
    """The other half of the same call - backend/decluster_cfg/n_threads
    already reached load_dp before this fix; pinned here so a future
    refactor of this call can't silently drop them alongside logger."""
    worker, _ = _make_mask_worker(monkeypatch, backend='pyeventem',
                                  decluster_cfg={'enabled': True, 'sinks': {'roi': True}},
                                  n_threads=7)
    captured = {}
    monkeypatch.setattr(tab_roi_4d, 'load_dp',
                        lambda fn, **kw: captured.update(kw) or np.zeros((2, 2)))
    worker.run()
    assert captured['backend'] == 'pyeventem'
    assert captured['decluster_cfg'] == {'enabled': True, 'sinks': {'roi': True}}
    assert captured['n_threads'] == 7


def test_worker_calculate_dp_mask_emits_error_not_silence_on_failure(monkeypatch):
    """A load_dp exception must still surface as signals.error, logger= not
    being in load_dp's signature (an older EDyssey build, or a caller
    passing an unexpected kwarg) included - the failure must be visible,
    not swallowed."""
    worker, fake_logger = _make_mask_worker(monkeypatch)

    def raising_load_dp(fn, **kw):
        raise RuntimeError('boom')

    monkeypatch.setattr(tab_roi_4d, 'load_dp', raising_load_dp)

    errors = []
    worker.signals.error.connect(errors.append)
    worker.run()

    assert len(errors) == 1 and 'boom' in errors[0]
    assert any(c[0] == 'exception' for c in fake_logger.calls)


# --------------------------------------------------------------------
# Tab_Create_NavSignal: _sum_dp_from_mask_worker
# --------------------------------------------------------------------


def test_sum_dp_from_mask_worker_forwards_logger_and_backend_settings(monkeypatch):
    fake_logger = _FakeLogger()
    captured = {}

    def fake_load_dp(fn, **kw):
        captured.update(kw)
        return np.zeros((2, 2))

    monkeypatch.setattr(tab_create_navSignal, 'load_dp', fake_load_dp)

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:6] = True
    dp = tab_create_navSignal.Tab_Create_NavSignal._sum_dp_from_mask_worker(
        'dummy.tpx3', '.tpx3', (10, 10), mask, fn_pattern=None, det_shape=(512, 512),
        logger=fake_logger, backend='pyeventem', decluster_cfg={'enabled': True}, n_threads=6,
    )

    assert dp.shape == (2, 2)
    assert captured.get('logger') is fake_logger, \
        "_sum_dp_from_mask_worker must pass logger= through to load_dp"
    assert captured.get('backend') == 'pyeventem', \
        "AnalysisBackendSettings' backend choice must reach load_dp, not just old eventem"
    assert captured.get('decluster_cfg') == {'enabled': True}
    assert captured.get('n_threads') == 6
    # Mask bounding box (rows 2-4, cols 3-5), matching the tab's own
    # convention: crop to the mask's extent, not the whole scan.
    assert captured.get('roi') == (3, 2, 3, 3)


def test_sum_dp_from_mask_worker_uses_patch_mode(monkeypatch):
    """A threshold mask can be scattered across the whole scan with no
    natural small bounding box - load_tpx3's own docstring warns that
    extracting its full bounding box as one 4D block can OOM/crash on
    smart-scanned data (the same reasoning tab_roi_4d.py's identical
    caller already applies). patch_mode=True routes a smart-scanned mask
    through per-connected-component extraction instead."""
    captured = {}
    monkeypatch.setattr(tab_create_navSignal, 'load_dp',
                        lambda fn, **kw: captured.update(kw) or np.zeros((2, 2)))
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1] = True
    tab_create_navSignal.Tab_Create_NavSignal._sum_dp_from_mask_worker(
        'dummy.tpx3', '.tpx3', (10, 10), mask)
    assert captured.get('patch_mode') is True


def test_sum_dp_from_mask_worker_defaults_match_prior_behavior(monkeypatch):
    """backend=None/decluster_cfg=None/n_threads=None must still reach
    load_dp as None (its own default of old eventem, no declustering) -
    the fix adds forwarding, it must not force a particular backend."""
    captured = {}
    monkeypatch.setattr(tab_create_navSignal, 'load_dp',
                        lambda fn, **kw: captured.update(kw) or np.zeros((2, 2)))
    mask = np.ones((4, 4), dtype=bool)
    tab_create_navSignal.Tab_Create_NavSignal._sum_dp_from_mask_worker(
        'dummy.tpx3', '.tpx3', (4, 4), mask)
    assert captured.get('backend') is None
    assert captured.get('decluster_cfg') is None
    assert captured.get('n_threads') is None
