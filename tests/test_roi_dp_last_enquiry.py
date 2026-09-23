# -*- coding: utf-8 -*-
"""Regression test for "Recompute DP" always re-running the last mask-based
extraction (SAM2/Edge Detection/"DP by Threshold") even after a plain
rectangle ROI was drawn afterward - found from a real report ("Recompute
dp is now always going back to the dp by threshold").

Root cause: _refresh_edge_mask (what "Recompute DP"/button_computeEdgeDp
actually runs) dispatches on self._mask_source/self.seg_mask, falling
back to the plain rectangle ROI (_compute_roi_dp) only when both are
None. Drawing a new ROI (on_release) never cleared them, so once any
mask-based extraction had run once, "Recompute DP" kept re-running that
same stale mask/threshold forever - regardless of how many new plain
ROIs were drawn afterward, since on_release's own direct call to
_compute_roi_dp() (for the immediate draw) masked the problem until the
next "Recompute DP" click.

Reuses tests/test_roi_dp_guard.py's own tab fixture/style rather than
duplicating it - see that file for why QMessageBox is patched out and
why worker.run() is called synchronously.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

pytest.importorskip("PyQt5")

import PyQt5.QtWidgets as qtw  # noqa: E402

import ui_tabs  # noqa: E402 - sets up sys.path for bare worker_*.py imports
from ui_tabs import tab_roi_4d  # noqa: E402
from ui_tabs.analysis_backend_settings import AnalysisBackendSettings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return qtw.QApplication.instance() or qtw.QApplication(sys.argv)


@pytest.fixture
def tab(qapp, monkeypatch):
    settings = AnalysisBackendSettings.instance()
    settings.set_values(backend="old_eventem", n_threads=1, decluster_enabled=False)
    for kind in ("critical", "information", "warning"):
        monkeypatch.setattr(qtw.QMessageBox, kind, lambda *a, **k: None)

    t = tab_roi_4d.Tab_ROI_on_4D()
    t.fn = "dummy.tpx3"
    t.scanSize = (16, 16)
    t.dwellTime = 100.0
    t.checkbox_detectorSizeAuto.setChecked(True)
    t.checkbox_smartScan.setChecked(False)
    yield t


def _release_event(xdata, ydata):
    return SimpleNamespace(xdata=xdata, ydata=ydata, inaxes=object())


def test_drawing_a_new_roi_clears_a_stale_mask_source(tab, monkeypatch):
    """The core fix: on_release must reset seg_mask/_mask_source so a fresh
    plain ROI isn't shadowed by whatever mask-based extraction ran before it."""
    monkeypatch.setattr(tab, '_compute_roi_dp', lambda: None)  # isolate on_release's own state change
    tab.seg_mask = np.ones((16, 16), dtype=bool)
    tab._mask_source = 'threshold'
    tab._threshold_method = 'otsu'

    tab.press = (1, 1)
    tab.on_release(_release_event(5, 5))

    assert tab.roi == (1, 1, 4, 4)
    assert tab.seg_mask is None
    assert tab._mask_source is None


def test_recompute_dp_uses_the_fresh_roi_not_the_stale_threshold_mask(tab, monkeypatch):
    """End-to-end: draw a ROI after a prior threshold-based extraction, then
    click "Recompute DP" (_refresh_edge_mask) - it must recompute the ROI,
    not silently re-run the old threshold mask."""
    calls = []
    monkeypatch.setattr(tab, '_compute_roi_dp', lambda: calls.append('roi'))
    monkeypatch.setattr(tab, 'compute_sum_dp_from_threshold', lambda mask, method: calls.append('threshold'))
    monkeypatch.setattr(tab, 'compute_seg_dp', lambda mask: calls.append('seg'))

    # Simulate a prior "DP by Threshold" run having left its mark.
    tab.seg_mask = np.ones((16, 16), dtype=bool)
    tab._mask_source = 'threshold'
    tab._threshold_method = 'otsu'

    tab.press = (1, 1)
    tab.on_release(_release_event(5, 5))
    calls.clear()  # only care about what "Recompute DP" itself does next

    tab._refresh_edge_mask()  # button_computeEdgeDp's own slot

    assert calls == ['roi'], \
        f"Recompute DP must use the freshly drawn ROI, not a stale mask - got {calls!r}"


def test_threshold_activate_still_remembered_as_the_new_roi_s_own_source(tab, monkeypatch):
    """The reset must not be so aggressive it breaks the existing, working
    case: drawing a ROI while the Threshold section's own Activate checkbox
    is on must still end up with _mask_source == 'roi_threshold' for that
    new ROI (see _on_threshold_control_changed)."""
    monkeypatch.setattr(tab, '_compute_roi_dp', lambda: None)
    monkeypatch.setattr(tab, 'navImg', np.ones((16, 16)) * 10.0, raising=False)
    tab.checkbox_thresholdActivate.setChecked(True)

    # Stale prior state from an unrelated earlier threshold-dialog use.
    tab._mask_source = 'threshold'
    tab._threshold_method = 'li'

    tab.press = (1, 1)
    tab.on_release(_release_event(5, 5))

    assert tab._mask_source == 'roi_threshold'
    assert tab.seg_mask is not None
