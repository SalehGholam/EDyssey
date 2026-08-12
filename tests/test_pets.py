# -*- coding: utf-8 -*-
"""Tests for EDyssey.io_utils.pets - the wavelength physics formula and the
.pts2 imagelist/parameter templating used by the "Make *.pts2" checkbox
added this session (see ui_tabs/pets2_dialog.py)."""
import pytest

from EDyssey.io_utils import pets


def test_electron_wavelength_200kv_matches_reference():
    # Standard TEM reference value at 200 kV, per PETS' own example project
    # files (lambda 0.02508000).
    assert pets.electron_wavelength_angstrom(200) == pytest.approx(0.02508, abs=1e-4)


def test_electron_wavelength_increases_as_voltage_decreases():
    assert pets.electron_wavelength_angstrom(100) > pets.electron_wavelength_angstrom(300)


def test_write_pts2_frame_count_and_filenames(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=3, aperpixel=0.006, voltage_kv=200,
                     geometry='continuous', omega=-2.5, phi=0, exposure_s=0.1,
                     detector='default', tilt_start_deg=0.0, tilt_step_deg=0.25,
                     frame_shape=(478, 478))
    content = fn.read_text()
    assert '"frames/0001.tif"' in content
    assert '"frames/0002.tif"' in content
    assert '"frames/0003.tif"' in content
    assert '"frames/0004.tif"' not in content


def test_write_pts2_alpha_ramp_matches_start_and_step(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=4, aperpixel=0.006, voltage_kv=200,
                     tilt_start_deg=1.0, tilt_step_deg=0.5, frame_shape=(100, 100))
    lines = [l for l in fn.read_text().splitlines() if l.startswith('"frames/')]
    alphas = [float(l.split()[1]) for l in lines]
    assert alphas == pytest.approx([1.0, 1.5, 2.0, 2.5])


def test_write_pts2_header_fields_present(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.00612, voltage_kv=300,
                     geometry='precession', omega=1.2345, phi=0.5,
                     exposure_s=0.25, detector='asi', frame_shape=(256, 256))
    content = fn.read_text()
    lam = pets.electron_wavelength_angstrom(300)
    assert f'lambda  {lam:.8f}' in content
    assert 'aperpixel  0.00612000' in content
    assert 'detector asi' in content
    assert 'geometry precession' in content
    assert 'phi  0.50000' in content
    assert 'exposureperframe    0.250000' in content


def test_write_pts2_imagelist_rows_are_imgname_and_alpha_only(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200,
                     tilt_start_deg=2.5, frame_shape=(200, 300))
    line = [l for l in fn.read_text().splitlines() if l.startswith('"frames/')][0]
    fields = line.split()
    assert fields == ['"frames/0001.tif"', '2.5000']


def test_write_pts2_notes_include_roi_id(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200,
                     roi_id=3, frame_shape=(100, 100))
    content = fn.read_text()
    assert 'notes\nROI No 3\nendnotes' in content


def test_write_pts2_notes_empty_when_no_roi_id(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200, frame_shape=(100, 100))
    content = fn.read_text()
    assert 'notes\n\nendnotes' in content


# ---------------------------------------------------------------------------
# center / autotask / keepautotasks / advanced_overrides
# ---------------------------------------------------------------------------

def test_write_pts2_center_line_defaults_to_auto(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200, frame_shape=(100, 100))
    assert 'center AUTO' in fn.read_text()


def test_write_pts2_center_line_uses_explicit_value(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200,
                     center=(123.5, 67.25), frame_shape=(100, 100))
    assert 'center 123.5000 67.2500' in fn.read_text()


def test_write_pts2_autotask_block_and_keepautotasks(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200,
                     autotask=['peak search', 'find cell'], keepautotasks='yes',
                     frame_shape=(100, 100))
    content = fn.read_text()
    assert 'autotask\npeak search\nfind cell\nendautotask' in content
    assert 'keepautotasks yes' in content


def test_write_pts2_autotask_defaults_to_empty(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200, frame_shape=(100, 100))
    content = fn.read_text()
    assert 'autotask\n\nendautotask' in content
    assert 'keepautotasks no' in content


def test_write_pts2_advanced_override_replaces_default(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200,
                     advanced_overrides={'rcshape': 'gaussian', 'i/sigma': '3.00   12.00'},
                     frame_shape=(100, 100))
    content = fn.read_text()
    assert 'rcshape gaussian' in content
    assert 'i/sigma    3.00   12.00' in content


def test_write_pts2_advanced_override_unknown_key_ignored(tmp_path):
    fn = tmp_path / 'v1.pts2'
    # Should not raise even if a stray/unknown keyword is passed in.
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200,
                     advanced_overrides={'not_a_real_keyword': 'x'}, frame_shape=(100, 100))
    assert fn.is_file()


def test_write_pts2_omega_flags_default_to_global_search_on(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200,
                     omega=1.5, frame_shape=(100, 100))
    assert 'omega   1.5000 1 1' in fn.read_text()


def test_write_pts2_omega_flags_can_be_disabled(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200,
                     omega=2.5, omega_global_search=0, omega_local_refine=0,
                     frame_shape=(100, 100))
    assert 'omega   2.5000 0 0' in fn.read_text()


def test_write_pts2_basic_fields_appear(tmp_path):
    fn = tmp_path / 'v1.pts2'
    pets.write_pts2(str(fn), n_frames=1, aperpixel=0.006, voltage_kv=200,
                     binning=2, reflectionsize=10.0, dstarmax=1.6, dstarmaxps=1.6,
                     dstarmin=0.02, frame_shape=(100, 100))
    content = fn.read_text()
    assert 'bin  2' in content
    assert 'reflectionsize  10.0' in content
    assert 'dstarmax  1.6' in content
    assert 'dstarmaxps  1.6' in content
    assert 'dstarmin  0.02' in content


# ---------------------------------------------------------------------------
# load_pets2_defaults / save_pets2_defaults
# ---------------------------------------------------------------------------

def test_pets2_defaults_roundtrip(tmp_path, monkeypatch):
    cache_file = tmp_path / 'pets2_last_params.json'
    monkeypatch.setattr(pets, '_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(pets, '_CACHE_FILE', str(cache_file))
    assert pets.load_pets2_defaults() == {}
    pets.save_pets2_defaults({'voltage_kv': 300.0, 'geometry': 'precession'})
    assert pets.load_pets2_defaults() == {'voltage_kv': 300.0, 'geometry': 'precession'}


def test_pets2_defaults_returns_empty_on_malformed_json(tmp_path, monkeypatch):
    cache_file = tmp_path / 'pets2_last_params.json'
    cache_file.write_text('{not valid json', encoding='utf-8')
    monkeypatch.setattr(pets, '_CACHE_FILE', str(cache_file))
    assert pets.load_pets2_defaults() == {}


# ---------------------------------------------------------------------------
# BASIC_FIELDS / ADVANCED_FIELDS / AUTOTASK_OPTIONS catalogue sanity
# ---------------------------------------------------------------------------

def test_advanced_fields_keywords_are_unique():
    keywords = [kw for kw, _ph, _default, _comment in pets.ADVANCED_FIELDS]
    assert len(keywords) == len(set(keywords))


def test_advanced_fields_placeholders_are_unique():
    placeholders = [ph for _kw, ph, _default, _comment in pets.ADVANCED_FIELDS]
    assert len(placeholders) == len(set(placeholders))


def test_autotask_options_contains_expected_tasks():
    assert 'peak search' in pets.AUTOTASK_OPTIONS
    assert 'process frames' in pets.AUTOTASK_OPTIONS
    assert 'none' in pets.AUTOTASK_OPTIONS
