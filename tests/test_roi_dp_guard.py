# -*- coding: utf-8 -*-
"""Regression test for the _roi_dp_running guard on Tab_ROI_on_4D's
rectangle-ROI DP computation.

**The bug this guards against, found from a real user's log** (not
inferred from reading code): drawing a rectangle ROI and pressing
"Recompute DP" (button_computeEdgeDp -> _refresh_edge_mask ->
_compute_roi_dp) had no guard at all against overlapping runs, unlike
every other DP-computing action in this file (compute_seg_dp/
compute_sum_dp_from_threshold both disable their own button). The real
session's log showed "calculating the dp..." firing three times in the
same second, repeating every few seconds, with no new "ROI: ..." line in
between - i.e. no new ROI was drawn; the SAME unchanged ROI was being
recomputed repeatedly. With no visual feedback that a computation was
already running, each impatient extra click (or extra ROI redraw) started
*another*, fully independent Worker_CalculateDP on top of the still-
running one, and the growing pile of concurrent, CPU-competing workers
made each one slower - which prompted more clicks. That compounding
slowdown is what looked exactly like "recompute dp stopped working",
though nothing was actually deadlocked.

The fix makes _compute_roi_dp() reject a call while one is already
in flight, matching the button-disabling pattern already used everywhere
else in this file, and makes Cancel free the button immediately rather
than leaving it stuck until a soon-to-be-discarded worker eventually
calls back.

These tests fake out the actual decode (Worker_CalculateDP.run() calls
EDyssey.io_utils.load_tpx3) with a controllable, event-gated stand-in, so
they run in milliseconds and deterministically - no real .tpx3 file, no
pyeventem, no timing race. What's under test is the guard/button-state
logic, not the decode itself (see test_eventem_backend.py and
tests/test_dp_threshold_worker.py for that).
"""
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import PyQt5.QtWidgets as qtw  # noqa: E402

import ui_tabs  # noqa: E402 - sets up sys.path for bare worker_*.py imports
from ui_tabs import tab_roi_4d  # noqa: E402
from ui_tabs.analysis_backend_settings import AnalysisBackendSettings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    # Qt allows exactly one QApplication per process - shared across every
    # test in this module rather than created per test.
    app = qtw.QApplication.instance() or qtw.QApplication(sys.argv)
    return app


class _GatedDecode:
    """Stands in for EDyssey.io_utils.load_tpx3: blocks on an Event until
    the test releases it, so a test can assert "still running" state
    deterministically instead of racing a real decode's timing. Counts
    calls so a test can assert exactly how many actually started.

    Returns arrays already shaped as Worker_CalculateDP.run() expects
    (det_shape for the DP, the ROI's own (h, w) for the nav crop) - a
    mismatched fake shape here previously made that reshape() raise,
    routing every one of these "success path" tests into the error path
    by accident (see test_a_failed_computation_also_clears_the_guard's
    own module docstring note below for why that path is separately
    guarded)."""

    def __init__(self, det_shape=(16, 16), roi_hw=(4, 4)):
        self.gate = threading.Event()
        self.calls = 0
        self.raise_instead = None
        self._det_shape = det_shape
        self._roi_hw = roi_hw

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.gate.wait(timeout=10)
        if self.raise_instead is not None:
            raise self.raise_instead
        dp = np.ones((self._det_shape[1], self._det_shape[0]))
        nav = np.ones(self._roi_hw)

        class _Result:
            Roi_diffraction_pattern = dp
            Roi_scan_image = nav
        return _Result()


@pytest.fixture
def tab(qapp, monkeypatch):
    settings = AnalysisBackendSettings.instance()
    settings.set_values(backend="old_eventem", n_threads=1, decluster_enabled=False)

    # A modal QMessageBox has no real window system to show/dismiss under
    # QT_QPA_PLATFORM=offscreen - each one is a native access violation
    # there (verified: found this way twice, once via a mismatched fake
    # shape accidentally routing into the error path, once via
    # cancel_running_work's own QMessageBox.information). Guarded for all
    # three here rather than only where a test intends to hit one, since
    # the crash is otherwise a confusing way to learn a test fixture took
    # an unexpected path.
    for kind in ("critical", "information", "warning"):
        monkeypatch.setattr(qtw.QMessageBox, kind, lambda *a, **k: None)

    t = tab_roi_4d.Tab_ROI_on_4D()
    t.fn = "dummy.tpx3"
    t.scanSize = (16, 16)
    t.dwellTime = 100.0
    t.checkbox_detectorSizeAuto.setChecked(True)
    t.checkbox_smartScan.setChecked(False)
    t.roi = (0, 0, 4, 4)
    yield t


def _install_gate(monkeypatch):
    gate = _GatedDecode(det_shape=(512, 512), roi_hw=(4, 4))
    monkeypatch.setattr(tab_roi_4d.io, "load_tpx3", gate)
    return gate


def _pump(qapp, predicate, timeout_s=10):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_repeated_clicks_while_running_do_not_stack_workers(tab, qapp, monkeypatch):
    gate = _install_gate(monkeypatch)

    tab._refresh_edge_mask()  # click 1: starts the (blocked) worker
    tab._refresh_edge_mask()  # click 2: must be rejected, not stacked
    tab._refresh_edge_mask()  # click 3: same
    qapp.processEvents()

    assert gate.calls == 1, "the guard must reject overlapping recompute clicks"
    assert tab._roi_dp_running is True

    gate.gate.set()  # let the one real worker finish
    assert _pump(qapp, lambda: not tab._roi_dp_running)


def test_button_states_track_the_running_computation(tab, qapp, monkeypatch):
    gate = _install_gate(monkeypatch)

    assert tab.button_computeEdgeDp.isEnabled()
    assert not tab.button_cancel.isEnabled()

    tab._refresh_edge_mask()
    qapp.processEvents()
    assert not tab.button_computeEdgeDp.isEnabled(), \
        "Recompute DP must disable itself while a computation is in flight, like every other DP button here"
    assert tab.button_cancel.isEnabled()

    gate.gate.set()
    assert _pump(qapp, lambda: tab.button_computeEdgeDp.isEnabled())
    assert not tab.button_cancel.isEnabled()


def test_a_second_computation_is_accepted_after_the_first_completes(tab, qapp, monkeypatch):
    gate = _install_gate(monkeypatch)

    tab._refresh_edge_mask()
    gate.gate.set()
    assert _pump(qapp, lambda: gate.calls == 1 and not tab._roi_dp_running)

    gate.gate.clear()
    tab._refresh_edge_mask()
    assert _pump(qapp, lambda: gate.calls == 2), \
        "a fresh computation must not be blocked once the previous one has finished"
    gate.gate.set()
    assert _pump(qapp, lambda: not tab._roi_dp_running)


def test_cancel_frees_the_button_without_waiting_for_the_worker(tab, qapp, monkeypatch):
    """The real-world failure mode: a click that's actually still running
    (possibly slow) must not leave the UI stuck - Cancel has to free things
    up immediately, before the background worker (which cannot be force-
    killed - see cancel_running_work's own docstring) actually returns."""
    gate = _install_gate(monkeypatch)

    tab._refresh_edge_mask()
    qapp.processEvents()
    assert tab._roi_dp_running is True
    assert not tab.button_computeEdgeDp.isEnabled()

    tab.cancel_running_work()  # the worker is still blocked on gate.gate here
    assert tab._roi_dp_running is False
    assert tab.button_computeEdgeDp.isEnabled()
    assert not tab.button_cancel.isEnabled()

    # A new computation must be startable right away, not blocked behind
    # the still-running (result-to-be-discarded) cancelled one.
    tab._refresh_edge_mask()
    qapp.processEvents()
    assert gate.calls == 2

    gate.gate.set()  # release both the cancelled and the new worker
    assert _pump(qapp, lambda: not tab._roi_dp_running)


def test_a_failed_computation_also_clears_the_guard(tab, qapp, monkeypatch):
    gate = _install_gate(monkeypatch)
    gate.raise_instead = RuntimeError("boom")

    tab._refresh_edge_mask()
    qapp.processEvents()
    assert tab._roi_dp_running is True

    gate.gate.set()
    assert _pump(qapp, lambda: not tab._roi_dp_running)
    assert tab.button_computeEdgeDp.isEnabled()
    assert not tab.button_cancel.isEnabled()

    # And the guard doesn't stay tripped after a failure.
    gate.raise_instead = None
    gate.gate.clear()
    tab._refresh_edge_mask()
    assert _pump(qapp, lambda: gate.calls == 2)
    gate.gate.set()
    assert _pump(qapp, lambda: not tab._roi_dp_running)


def test_drawing_a_new_roi_is_also_guarded(tab, qapp, monkeypatch):
    """on_release's own trigger (_compute_roi_dp directly, not through the
    Recompute DP button) must be covered by the same guard - it's the
    other of the two real callers, and the real log showed no "ROI: ..."
    line at all during the runaway, so this path needed checking too."""
    gate = _install_gate(monkeypatch)

    tab._compute_roi_dp()
    tab._compute_roi_dp()
    qapp.processEvents()
    assert gate.calls == 1

    gate.gate.set()
    assert _pump(qapp, lambda: not tab._roi_dp_running)
