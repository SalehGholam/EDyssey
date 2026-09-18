# -*- coding: utf-8 -*-
"""Which .tpx3 analysis backend (old eventem / new eventem / pyeventem) to
use, and declustering configuration - one shared singleton, changed from the
Edit menu's "Analysis Backend & Declustering..." dialog
(ui_tabs/tab_backend_settings.py) and read by EDyssey.io_utils.eventem_backend
via a plain dict (see AnalysisBackendSettings.decluster_cfg()) rather than
the QObject itself, since eventem_backend.py must stay importable from a
Qt-free subprocess worker.

Persisted the same way as display_settings.py/app_theme.py (this app's only
settings-persistence convention): a JSON file under writable_data_dir()/
config/, read back on next launch.
"""
import os
import json
import logging
from PyQt5.QtCore import QObject, pyqtSignal
from EDyssey.io_utils.app_dirs import writable_data_dir
from EDyssey.io_utils import eventem_backend as eb

logger = logging.getLogger('EDyssey.analysis_backend_settings')

_CONFIG_DIR = os.path.join(writable_data_dir(), 'config')
SETTINGS_FILE = os.path.join(_CONFIG_DIR, 'analysis_backend_settings.json')

# One entry per sink this app actually drives through eventem_backend.py
# (Pacbed/Roi in Tab_ROI_on_4D + workers/worker_extract_frame*.py, vSTEM/Var
# in Tab_Create_NavSignal) - the dialog's "Sinks" checkboxes and
# Select-All/Select-None buttons iterate this list, so adding a new sink
# elsewhere in the app later only means adding one entry here.
SINK_KEYS = ('pacbed', 'roi', 'vstem', 'var')
SINK_LABELS = {'pacbed': 'Pacbed', 'roi': 'Roi', 'vstem': 'vSTEM', 'var': 'Var'}

_UNSET = object()  # set_values()'s "argument not passed" marker - distinct from None, which n_threads uses to mean "Auto"


def _default_state():
    cfg = eb.default_decluster_cfg()
    return {
        'backend': eb.BACKEND_OLD,
        'n_threads': None,  # None = "Auto" - don't override the backend's own default
        'decluster_enabled': cfg['enabled'],
        'dspace': cfg['dspace'],
        'dtime_ns': cfg['dtime_ns'],
        'cluster_range': cfg['cluster_range'],
        'tot_per_electron': cfg['tot_per_electron'],
        'lut_file': cfg['lut_file'],
        'sinks': dict(cfg['sinks']),
    }


def _load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning('Could not read %s: %s', path, e)
        return None


def _save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.warning('Could not write %s: %s', path, e)


class AnalysisBackendSettings(QObject):
    """Singleton - use AnalysisBackendSettings.instance(), never construct
    directly (see DisplaySettings for why)."""
    changed = pyqtSignal()

    _instance = None

    def __init__(self):
        super().__init__()
        self._apply_state(_load_json(SETTINGS_FILE) or _default_state())

    def _apply_state(self, state):
        defaults = _default_state()
        self.backend = state.get('backend', defaults['backend'])
        if self.backend not in eb.BACKENDS:
            self.backend = defaults['backend']
        n_threads = state.get('n_threads', defaults['n_threads'])
        self.n_threads = int(n_threads) if n_threads else None
        self.decluster_enabled = bool(state.get('decluster_enabled', defaults['decluster_enabled']))
        self.dspace = state.get('dspace', defaults['dspace'])
        self.dtime_ns = state.get('dtime_ns', defaults['dtime_ns'])
        self.cluster_range = state.get('cluster_range', defaults['cluster_range'])
        self.tot_per_electron = state.get('tot_per_electron', defaults['tot_per_electron'])
        self.lut_file = state.get('lut_file', defaults['lut_file'])
        saved_sinks = state.get('sinks') or {}
        self.sinks = {key: bool(saved_sinks.get(key, True)) for key in SINK_KEYS}

    def _current_state(self):
        return {
            'backend': self.backend,
            'n_threads': self.n_threads,
            'decluster_enabled': self.decluster_enabled,
            'dspace': self.dspace,
            'dtime_ns': self.dtime_ns,
            'cluster_range': self.cluster_range,
            'tot_per_electron': self.tot_per_electron,
            'lut_file': self.lut_file,
            'sinks': dict(self.sinks),
        }

    def _persist(self):
        _save_json(SETTINGS_FILE, self._current_state())

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def decluster_cfg(self) -> dict:
        """The plain, JSON-serializable dict eventem_backend.run_*/tasks.json
        actually consume - see eventem_backend.default_decluster_cfg()'s
        shape."""
        return {
            'enabled': self.decluster_enabled,
            'dspace': self.dspace,
            'dtime_ns': self.dtime_ns,
            'cluster_range': self.cluster_range,
            'tot_per_electron': self.tot_per_electron,
            'lut_file': self.lut_file,
            'sinks': dict(self.sinks),
        }

    def analysis_kwargs(self) -> dict:
        """The full `backend`/`decluster_cfg`/`n_threads` triple every
        eventem_backend.run_*/loaders.py/nav_image.py call takes - one call
        site only needs `**AnalysisBackendSettings.instance().analysis_kwargs()`
        instead of reading each field separately. `n_threads` also travels
        through tasks.json unchanged to subprocess workers (plain int or
        None), same as decluster_cfg."""
        return {
            'backend': self.backend,
            'decluster_cfg': self.decluster_cfg(),
            'n_threads': self.n_threads,
        }

    def set_values(self, backend=None, n_threads=_UNSET, decluster_enabled=None, dspace=None, dtime_ns=None,
                   cluster_range=None, tot_per_electron=None, lut_file=None, sinks=None):
        """Update whichever values are given (None = leave unchanged), then
        emit `changed` once and persist - the dialog's Apply button calls
        this once with every control's current value. `n_threads` uses a
        private sentinel default (rather than None) since None is itself a
        valid, meaningful value here ("Auto")."""
        if backend is not None:
            self.backend = backend
        if n_threads is not _UNSET:
            self.n_threads = int(n_threads) if n_threads else None
        if decluster_enabled is not None:
            self.decluster_enabled = decluster_enabled
        if dspace is not None:
            self.dspace = dspace
        if dtime_ns is not None:
            self.dtime_ns = dtime_ns
        if cluster_range is not None:
            self.cluster_range = cluster_range
        if tot_per_electron is not None:
            self.tot_per_electron = tot_per_electron
        if lut_file is not None:
            self.lut_file = lut_file
        if sinks:
            self.sinks.update({k: v for k, v in sinks.items() if k in self.sinks})
        self._persist()
        self.changed.emit()

    def reset(self):
        self._apply_state(_default_state())
        self._persist()
        self.changed.emit()
