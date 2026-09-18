# -*- coding: utf-8 -*-
"""Tests for ui_tabs.analysis_backend_settings.AnalysisBackendSettings - the
JSON persistence round-trip and set_values()/reset() behavior, matching the
pattern of this app's other settings singletons (display_settings.py/
app_theme.py have no equivalent test file yet to mirror; this establishes
the pattern instead).

Each test gets its own fresh singleton and its own tmp_path settings file
(monkeypatched in) so tests never read/write the real user config or leak
state into each other via the class-level _instance cache.
"""
import pytest

from ui_tabs import analysis_backend_settings as abs_mod
from EDyssey.io_utils import eventem_backend as eb


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(abs_mod, 'SETTINGS_FILE', str(tmp_path / 'analysis_backend_settings.json'))
    abs_mod.AnalysisBackendSettings._instance = None
    yield abs_mod.AnalysisBackendSettings.instance()
    abs_mod.AnalysisBackendSettings._instance = None


def test_defaults(settings):
    assert settings.backend == eb.BACKEND_OLD
    assert settings.n_threads is None
    assert settings.decluster_enabled is False
    assert all(settings.sinks.values())


def test_instance_is_a_singleton(settings):
    assert abs_mod.AnalysisBackendSettings.instance() is settings


def test_set_values_updates_and_persists(settings, tmp_path):
    settings.set_values(backend=eb.BACKEND_PYEVENTEM, n_threads=4, decluster_enabled=True,
                        dspace=8, dtime_ns=50.0, cluster_range=128, tot_per_electron=200.0,
                        sinks={'roi': False})
    assert settings.backend == eb.BACKEND_PYEVENTEM
    assert settings.n_threads == 4
    assert settings.decluster_enabled is True
    assert settings.dspace == 8
    assert settings.sinks['roi'] is False
    assert settings.sinks['pacbed'] is True  # untouched keys stay as they were

    # Persisted: a *new* instance reading the same file sees the same state.
    abs_mod.AnalysisBackendSettings._instance = None
    reloaded = abs_mod.AnalysisBackendSettings.instance()
    assert reloaded.backend == eb.BACKEND_PYEVENTEM
    assert reloaded.n_threads == 4
    assert reloaded.sinks['roi'] is False


def test_set_values_n_threads_zero_means_auto(settings):
    settings.set_values(n_threads=4)
    assert settings.n_threads == 4
    settings.set_values(n_threads=0)
    assert settings.n_threads is None


def test_set_values_none_leaves_backend_unchanged(settings):
    settings.set_values(backend=eb.BACKEND_NEW)
    settings.set_values(decluster_enabled=True)  # backend arg omitted (defaults to None)
    assert settings.backend == eb.BACKEND_NEW


def test_set_values_invalid_backend_falls_back_on_reload(settings, tmp_path):
    # Simulate a config file from a version with a since-removed backend name.
    import json
    with open(abs_mod.SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'backend': 'no_longer_exists'}, f)
    abs_mod.AnalysisBackendSettings._instance = None
    reloaded = abs_mod.AnalysisBackendSettings.instance()
    assert reloaded.backend == eb.BACKEND_OLD


def test_changed_signal_emitted_on_set_values(settings):
    calls = []
    settings.changed.connect(lambda: calls.append(1))
    settings.set_values(backend=eb.BACKEND_NEW)
    assert calls == [1]


def test_reset_restores_defaults(settings):
    settings.set_values(backend=eb.BACKEND_PYEVENTEM, n_threads=8, decluster_enabled=True)
    settings.reset()
    assert settings.backend == eb.BACKEND_OLD
    assert settings.n_threads is None
    assert settings.decluster_enabled is False


def test_decluster_cfg_shape(settings):
    settings.set_values(decluster_enabled=True, dspace=10, dtime_ns=42.0,
                        cluster_range=99, tot_per_electron=150.0, lut_file='')
    cfg = settings.decluster_cfg()
    assert cfg == {
        'enabled': True, 'dspace': 10, 'dtime_ns': 42.0, 'cluster_range': 99,
        'tot_per_electron': 150.0, 'lut_file': '', 'sinks': settings.sinks,
    }


def test_analysis_kwargs_bundles_everything(settings):
    settings.set_values(backend=eb.BACKEND_NEW, n_threads=6)
    kwargs = settings.analysis_kwargs()
    assert kwargs['backend'] == eb.BACKEND_NEW
    assert kwargs['n_threads'] == 6
    assert kwargs['decluster_cfg'] == settings.decluster_cfg()
