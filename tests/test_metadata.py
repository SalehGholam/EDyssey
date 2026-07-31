# -*- coding: utf-8 -*-
"""Tests for py4DTomo.io_utils.metadata - all pure/deterministic (no Qt, no
compiled eventem/pacbed dependency), using synthetic fixtures rather than
any personal machine's real acquisition data.
"""
import json
import os
import pytest

from py4DTomo.io_utils import metadata as md


def _write(tmp_path, name, text):
    fn = tmp_path / name
    fn.write_text(text, encoding='utf-8')
    return str(fn)


# ---------------------------------------------------------------------------
# get_metadata / get_metadata_block_count
# ---------------------------------------------------------------------------

SINGLE_BLOCK = (
    'measurement metadata\n'
    'STEM magnification: 100000.0\n'
    'scan size x: 256\n'
    'scan size y: 256\n'
    'dwelltime: 20.0 microseconds\n'
    '\n'
)

TWO_BLOCKS_SINGLE_BLANK = (
    'measurement metadata\n'
    'scan size x: 100\n'
    'scan size y: 100\n'
    'dwelltime: 10 microseconds\n'
    '\n'
    'measurement metadata\n'
    'scan size x: 200\n'
    'scan size y: 200\n'
    'dwelltime: 30 microseconds\n'
    '\n'
)


def test_get_metadata_parses_single_block(tmp_path):
    _write(tmp_path, 'comment.txt', SINGLE_BLOCK)
    result = md.get_metadata(str(tmp_path), count=0)
    assert result['scan size x'] == 256.0
    assert result['scan size y'] == 256.0
    assert result['dwelltime'] == 20.0


def test_get_metadata_second_block_has_distinct_values(tmp_path):
    _write(tmp_path, 'comment.txt', TWO_BLOCKS_SINGLE_BLANK)
    block0 = md.get_metadata(str(tmp_path), count=0)
    block1 = md.get_metadata(str(tmp_path), count=1)
    assert block0['scan size x'] == 100.0
    assert block1['scan size x'] == 200.0


def test_get_metadata_accepts_direct_file_path(tmp_path):
    fn = _write(tmp_path, 'comment.txt', SINGLE_BLOCK)
    result = md.get_metadata(fn, count=0)
    assert result['scan size x'] == 256.0


def test_get_metadata_accepts_sibling_file_as_path_main(tmp_path):
    _write(tmp_path, 'comment.txt', SINGLE_BLOCK)
    fn_signal = _write(tmp_path, 'raw_0001.tpx3', '')
    result = md.get_metadata(fn_signal, count=0)
    assert result['scan size x'] == 256.0


def test_get_metadata_block_count_single(tmp_path):
    _write(tmp_path, 'comment.txt', SINGLE_BLOCK)
    assert md.get_metadata_block_count(str(tmp_path)) == 1


def test_get_metadata_block_count_multi(tmp_path):
    _write(tmp_path, 'comment.txt', TWO_BLOCKS_SINGLE_BLANK)
    assert md.get_metadata_block_count(str(tmp_path)) == 2


def test_get_metadata_skips_unparsable_lines(tmp_path):
    # "measurement metadata" (no colon) and "Scan strategy: Raster" (val
    # isn't a float) must both be silently skipped, not raise.
    text = (
        'measurement metadata\n'
        'scan size x: 512\n'
        'Scan strategy: Raster\n'
        '\n'
    )
    _write(tmp_path, 'comment.txt', text)
    result = md.get_metadata(str(tmp_path), count=0)
    assert result == {'scan size x': 512.0}


# ---------------------------------------------------------------------------
# default_analysis_save_dir
# ---------------------------------------------------------------------------

def test_default_analysis_save_dir_inside_navigator_structure():
    fn = os.path.join('D:', os.sep, 'data', 'mySample_5DED Analysis',
                       'navigator signal', '2026-07-29__12-00-00', 'navigation_signal.hspy')
    expected = os.path.join('D:', os.sep, 'data', 'mySample_5DED Analysis')
    assert md.default_analysis_save_dir(fn) == expected


def test_default_analysis_save_dir_standalone_file_falls_back():
    fn = os.path.join('D:', os.sep, 'data', 'some_old_signal.hspy')
    expected = os.path.join('D:', os.sep, 'data', '5DED Analysis')
    assert md.default_analysis_save_dir(fn) == expected


# ---------------------------------------------------------------------------
# load_analysis_metadata
# ---------------------------------------------------------------------------

def test_load_analysis_metadata_reads_sibling_json(tmp_path):
    payload = {'scan_size': [512, 512], 'dwell_time_us': 50}
    (tmp_path / 'metadata.json').write_text(json.dumps(payload), encoding='utf-8')
    fn_nav = str(tmp_path / 'navigation_signal.hspy')
    result = md.load_analysis_metadata(fn_nav)
    assert result == payload


def test_load_analysis_metadata_returns_none_when_absent(tmp_path):
    fn_nav = str(tmp_path / 'navigation_signal.hspy')
    assert md.load_analysis_metadata(fn_nav) is None


def test_load_analysis_metadata_returns_none_on_malformed_json(tmp_path):
    (tmp_path / 'metadata.json').write_text('{not valid json', encoding='utf-8')
    fn_nav = str(tmp_path / 'navigation_signal.hspy')
    assert md.load_analysis_metadata(fn_nav) is None


# ---------------------------------------------------------------------------
# save_analysis_info / load_analysis_info
# ---------------------------------------------------------------------------

def test_save_and_load_analysis_info_roundtrip(tmp_path):
    fn_source = str(tmp_path / 'somewhere' / 'nav.hspy')
    md.save_analysis_info(str(tmp_path), fn_source)
    result = md.load_analysis_info(str(tmp_path))
    assert result == {'nav_signal_source': fn_source, 'analysis_type': None}


def test_save_analysis_info_records_analysis_type(tmp_path):
    fn_source = str(tmp_path / 'somewhere' / 'nav.hspy')
    md.save_analysis_info(str(tmp_path), fn_source, analysis_type='sam2')
    result = md.load_analysis_info(str(tmp_path))
    assert result['analysis_type'] == 'sam2'


def test_save_analysis_info_noop_when_source_falsy(tmp_path):
    md.save_analysis_info(str(tmp_path), None)
    assert not (tmp_path / 'analysis_info.json').exists()
    assert md.load_analysis_info(str(tmp_path)) is None


def test_load_analysis_info_returns_none_when_absent(tmp_path):
    assert md.load_analysis_info(str(tmp_path)) is None


def test_load_analysis_info_returns_none_on_malformed_json(tmp_path):
    (tmp_path / 'analysis_info.json').write_text('{not valid json', encoding='utf-8')
    assert md.load_analysis_info(str(tmp_path)) is None
