# -*- coding: utf-8 -*-
"""numba's kernel cache must not default to the read-only install folder of a frozen build, where
probing it makes the first pyeventem import effectively hang (see runtime_env.py)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import runtime_env  # noqa: E402


def test_frozen_build_gets_a_per_user_numba_cache(tmp_path, monkeypatch):
    monkeypatch.delenv('NUMBA_CACHE_DIR', raising=False)
    got = runtime_env.configure_numba_cache(frozen=True, local_app_data=str(tmp_path))
    assert got == str(tmp_path / 'EDyssey' / 'numba_cache')
    assert os.environ['NUMBA_CACHE_DIR'] == got
    assert os.path.isdir(got)


def test_an_existing_setting_is_respected(tmp_path, monkeypatch):
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(tmp_path / 'mine'))
    got = runtime_env.configure_numba_cache(frozen=True, local_app_data=str(tmp_path))
    assert got == str(tmp_path / 'mine')
    assert os.environ['NUMBA_CACHE_DIR'] == str(tmp_path / 'mine')


def test_a_source_run_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.delenv('NUMBA_CACHE_DIR', raising=False)
    assert runtime_env.configure_numba_cache(frozen=False, local_app_data=str(tmp_path)) is None
    assert 'NUMBA_CACHE_DIR' not in os.environ


def test_unwritable_location_does_not_crash_startup(tmp_path, monkeypatch):
    monkeypatch.delenv('NUMBA_CACHE_DIR', raising=False)
    blocker = tmp_path / 'file_not_dir'
    blocker.write_text('x')
    assert runtime_env.configure_numba_cache(frozen=True, local_app_data=str(blocker)) is None
    assert 'NUMBA_CACHE_DIR' not in os.environ
