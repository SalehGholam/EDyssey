# -*- coding: utf-8 -*-
"""Virtual-detector radii of None ("no bound"): facade argument translation per
backend, the numpy mask logic, the label/validation helpers, and the UI state
round trip. No real .tpx3 file or compiled eventem is needed."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

from EDyssey.io_utils import eventem_backend as eb  # noqa: E402
from EDyssey.io_utils import nav_image  # noqa: E402

FULL = eb._EVENTEM_FULL_RADIUS


class _FakeSink:
    """Stands in for eventem's vSTEM/Var; records what the facade configured."""
    last = None

    def __init__(self, *a, **k):
        self.detector_size_x = self.detector_size_y = self.detector_size = 512
        self.centers = None
        _FakeSink.last = self

    def set_file(self, fn): pass
    def set_dwell_time(self, v): pass
    def set_pattern_file(self, v): pass
    def set_offsets(self, v): self.centers = v
    def set_offset(self, v): self.centers = [v]
    def run(self): pass
    def close_socket(self): pass
    def get_image(self): return np.zeros((2, 2))

    @property
    def Var_image(self):
        return np.zeros(4)


def _fake_module():
    return types.SimpleNamespace(vSTEM=_FakeSink, Var=_FakeSink)


@pytest.fixture
def backends(monkeypatch):
    monkeypatch.setattr(eb, '_needs_subprocess', lambda backend: False)
    monkeypatch.setattr(eb, '_old_eventem', _fake_module)
    monkeypatch.setattr(eb, '_new_eventem', _fake_module)
    _FakeSink.last = None


def _vstem(backend, r_in, r_out, offset=(256, 256)):
    eb.run_vstem('f.tpx3', (2, 2), r_in=r_in, r_out=r_out, offset=offset, backend=backend)
    return _FakeSink.last


def test_new_eventem_vstem_none_maps_to_zero_and_full(backends):
    sink = _vstem(eb.BACKEND_NEW, None, None)
    assert sink.inner_radia == [0] and sink.outer_radia == [FULL]
    assert sink.centers == [[256.0, 256.0]]


def test_new_eventem_vstem_numeric_unchanged(backends):
    sink = _vstem(eb.BACKEND_NEW, 15, 150)
    assert sink.inner_radia == [15] and sink.outer_radia == [150]


def test_old_eventem_vstem_inner_none_adds_inert_second_detector(backends):
    sink = _vstem(eb.BACKEND_OLD, None, 30)
    assert sink.inner_radia == [0, 5000] and sink.outer_radia == [30, 5001]
    assert sink.centers == [[256.0, 256.0], [256.0, 256.0]]


def test_old_eventem_vstem_inner_none_without_offset_sets_default_centre(backends):
    sink = _vstem(eb.BACKEND_OLD, None, None, offset=None)
    assert sink.centers == [[256.0, 256.0], [256.0, 256.0]]
    assert sink.outer_radia == [FULL, 5001]


def test_old_eventem_vstem_numeric_inner_has_no_dummy(backends):
    sink = _vstem(eb.BACKEND_OLD, 0, 30)
    assert sink.inner_radia == [0] and sink.outer_radia == [30]


def test_old_eventem_vstem_outer_none_only_has_no_dummy(backends):
    sink = _vstem(eb.BACKEND_OLD, 10, None)
    assert sink.inner_radia == [10] and sink.outer_radia == [FULL]


def test_old_eventem_multi_detector_none_is_per_element(backends):
    sink = _vstem(eb.BACKEND_OLD, [None, 15], [30, None], offset=[(256, 256), (271, 272)])
    assert sink.inner_radia == [0, 15] and sink.outer_radia == [30, FULL]
    assert len(sink.centers) == 2


@pytest.mark.parametrize('backend', [eb.BACKEND_OLD, eb.BACKEND_NEW])
def test_var_none_maps_to_zero_and_full_for_cpp_backends(backends, backend):
    eb.run_var('f.tpx3', (2, 2), r_in=None, r_out=None, offset=(8, 8), backend=backend)
    assert _FakeSink.last.inner_radius == 0 and _FakeSink.last.outer_radius == FULL


def test_old_eventem_var_numeric_inner_passed_unchanged(backends):
    # Old eventem's Var is exclusive at a numeric inner edge; no workaround exists.
    eb.run_var('f.tpx3', (2, 2), r_in=7, r_out=9, offset=(8, 8), backend=eb.BACKEND_OLD)
    assert _FakeSink.last.inner_radius == 7 and _FakeSink.last.outer_radius == 9


def test_pyeventem_passes_none_through(monkeypatch):
    monkeypatch.setattr(eb, '_needs_subprocess', lambda backend: False)
    captured = []
    monkeypatch.setattr(eb, '_pyeventem_dispatch', lambda kind, kw, *a, **k: captured.append((kind, kw)))
    eb.run_vstem('f.tpx3', (2, 2), r_in=[None, 5], r_out=[None, 9], offset=[(1, 2), (3, 4)],
                 backend=eb.BACKEND_PYEVENTEM)
    eb.run_var('f.tpx3', (2, 2), r_in=None, r_out=None, offset=(8, 8), backend=eb.BACKEND_PYEVENTEM)
    (_, vkw), (_, nkw) = captured
    assert vkw['inner_radii'] == [None, 5] and vkw['outer_radii'] == [None, 9]
    assert nkw['inner_radius'] is None and nkw['outer_radius'] is None


def test_subprocess_request_keeps_none_as_json_null(monkeypatch):
    monkeypatch.setattr(eb, '_needs_subprocess', lambda backend: True)
    seen = {}

    def fake_subprocess(func_name, kwargs, mask_array=None, logger_=None):
        seen.update(kwargs)
        return json.loads(json.dumps(kwargs))

    monkeypatch.setattr(eb, '_run_eventem_via_subprocess', fake_subprocess)
    eb.run_vstem('f.tpx3', (2, 2), r_in=[None], r_out=None, offset=(1, 2), backend=eb.BACKEND_OLD)
    assert seen['r_in'] == [None] and seen['r_out'] is None
    assert json.loads(json.dumps(seen))['r_out'] is None


def test_radii_helpers_directly():
    assert eb._var_radii_for_backend(eb.BACKEND_OLD, None, None) == (0, FULL)
    assert eb._var_radii_for_backend(eb.BACKEND_PYEVENTEM, None, None) == (None, None)
    inner, outer, centers = eb._vstem_radii_for_backend(
        eb.BACKEND_PYEVENTEM, [None], [None], None, (512, 512))
    assert (inner, outer, centers) == ([None], [None], None)


# ---------------------------------------------------------------------------
# numpy mask path
# ---------------------------------------------------------------------------

def test_virtual_detector_inner_none_includes_centre_pixel():
    mask = nav_image.create_virtual_detector((11, 11), (5, 5), 3, None)
    assert mask[5, 5] == 1
    assert np.array_equal(mask, nav_image.create_virtual_detector((11, 11), (5, 5), 3, 0))


def test_virtual_detector_outer_none_has_no_upper_bound():
    mask = nav_image.create_virtual_detector((11, 11), (5, 5), None, 2)
    assert mask[0, 0] == 1 and mask[10, 10] == 1
    assert mask[5, 5] == 0 and mask[5, 6] == 0  # still inside the inner hole


def test_virtual_detector_both_none_is_full_frame():
    assert nav_image.create_virtual_detector((6, 7), (3, 3), None, None).all()


def test_virtual_detector_numeric_unchanged():
    mask = nav_image.create_virtual_detector((11, 11), (5, 5), 4, 2)
    assert mask[5, 5] == 0 and mask[5, 8] == 1 and mask[5, 10] == 0


def test_virtual_detector_multi_union_with_none_and_legacy_dict():
    dets = [{'center': (2, 2), 'r_in': None, 'r_out': 1},
            {'center': (8, 8), 'r_out': 1}]  # no r_in key: legacy state
    mask = nav_image.create_virtual_detector_multi((11, 11), dets)
    assert mask[2, 2] == 1 and mask[8, 8] == 1 and mask[5, 5] == 0


def test_format_detector_and_error():
    assert nav_image.format_detector({'center': (256, 256), 'r_in': None, 'r_out': 30}) == \
        'center=(256, 256), r=none-30'
    assert nav_image.format_detector({'center': (1, 2), 'r_in': 5, 'r_out': None}) == \
        'center=(1, 2), r=5-none'
    assert nav_image.format_detector({'center': (1, 2)}) == 'center=(1, 2), r=0-none'
    assert nav_image.detector_radii_error(None, 5) is None
    assert nav_image.detector_radii_error(5, None) is None
    assert nav_image.detector_radii_error(5, 5) is not None


def test_masked_tpx3_passes_none_through_to_wrappers(monkeypatch):
    calls = {}
    monkeypatch.setattr(nav_image, 'calculate_nav_img_tpx3',
                        lambda *a, **k: calls.update(k) or np.zeros((2, 2)))
    nav_image.calculate_nav_img_masked(
        'f.tpx3', dtype='.tpx3', scanSize=(2, 2),
        detectors=[{'center': (1, 2), 'r_in': None, 'r_out': None}])
    assert calls['r_in'] == [None] and calls['r_out'] == [None]


# ---------------------------------------------------------------------------
# UI: checkboxes, labels, state round trip
# ---------------------------------------------------------------------------

qtw = pytest.importorskip("PyQt5.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    return qtw.QApplication.instance() or qtw.QApplication(sys.argv)


def test_navigator_tab_none_checkboxes_and_state_round_trip(qapp):
    import ui_tabs  # noqa: F401
    from ui_tabs import tab_create_navSignal
    tab = tab_create_navSignal.Tab_Create_NavSignal()
    tab.nav_imgs = np.zeros((1, 4, 4))
    assert tab.get_active_detectors()[0]['r_in'] == 0
    tab.checkbox_rInNone.setChecked(True)
    tab.checkbox_rOutNone.setChecked(True)
    det = tab.get_active_detectors()[0]
    assert det['r_in'] is None and det['r_out'] is None
    assert not tab.spinbox_rIn.isEnabled() and not tab.spinbox_rOut.isEnabled()
    assert tab.list_detectors.item(0).text().endswith('r=none-none')
    assert json.loads(json.dumps(tab.get_active_detectors()))[0]['r_out'] is None

    tab.add_extra_detector()  # valid: None bounds never conflict
    assert len(tab.get_active_detectors()) == 2

    state = tab.get_duplicate_state()
    tab2 = tab_create_navSignal.Tab_Create_NavSignal()
    tab2.apply_duplicate_state(state)
    assert [d['r_in'] for d in tab2.get_active_detectors()] == [None, None]
    assert tab2.checkbox_rInNone.isChecked() and tab2.checkbox_rOutNone.isChecked()

    tab2.checkbox_rInNone.setChecked(False)
    assert tab2.get_active_detectors()[tab2._active_detector_row]['r_in'] == tab2.spinbox_rIn.value()
    assert tab2.spinbox_rIn.isEnabled()


def test_navigator_tab_legacy_state_without_none_still_loads(qapp):
    from ui_tabs import tab_create_navSignal
    tab = tab_create_navSignal.Tab_Create_NavSignal()
    tab.nav_imgs = np.zeros((1, 4, 4))
    state = tab.get_duplicate_state()
    state['extra_detectors'] = [{'center': (10, 20), 'r_in': 5, 'r_out': 50}]
    state['active_detector_row'] = 0
    tab2 = tab_create_navSignal.Tab_Create_NavSignal()
    tab2.apply_duplicate_state(state)
    assert tab2.get_active_detectors() == [{'center': (10, 20), 'r_in': 5, 'r_out': 50}]
    assert not tab2.checkbox_rInNone.isChecked()


def test_roi_tab_none_checkboxes_and_state_round_trip(qapp):
    import ui_tabs  # noqa: F401
    from ui_tabs import tab_roi_4d
    tab = tab_roi_4d.Tab_ROI_on_4D()
    tab.navImg = np.zeros((4, 4))
    tab.checkbox_vi_rOutNone.setChecked(True)
    det = tab.get_active_virtual_detectors()[0]
    assert det['r_out'] is None and det['r_in'] == 0
    assert tab.list_detectors_vi.item(0).text().endswith('r=0-none')
    tab.checkbox_vi_rInNone.setChecked(True)
    state = tab.get_duplicate_state()
    tab2 = tab_roi_4d.Tab_ROI_on_4D()
    tab2.apply_duplicate_state(state)
    det2 = tab2.get_active_virtual_detectors()[0]
    assert det2['r_in'] is None and det2['r_out'] is None
    assert tab2.checkbox_vi_rInNone.isChecked() and not tab2.spinbox_vi_rOut.isEnabled()
