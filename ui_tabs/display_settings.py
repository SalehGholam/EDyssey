# -*- coding: utf-8 -*-
"""App-wide display-scale settings (ribbon text size, ribbon icon size,
ribbon height, plot font size, per-plot figure/canvas size) - one shared
singleton, changed from the Edit menu's Display Size dialog and applied
live by every other tab via TabBase.apply_display_settings().

Persisted to disk (unlike the rest of the app, which has no
settings-persistence mechanism otherwise): every Apply/Reset in the dialog
writes the full current state to SETTINGS_FILE, which is read back and
re-applied the next time the app starts - see __init__ below. DEFAULTS_FILE
is a separate, once-written snapshot of the built-in factory defaults, so
"Reset to Defaults" always lands on the same values regardless of whatever
the user has saved to SETTINGS_FILE in the meantime.
"""
import os
import json
import logging
from PyQt5.QtCore import QObject, pyqtSignal
from EDyssey.io_utils.app_dirs import writable_data_dir

logger = logging.getLogger('EDyssey.display_settings')

RIBBON_TEXT_SCALE_DEFAULT = 1.0
RIBBON_ICON_SIZE_DEFAULT = 26
RIBBON_HEIGHT_SCALE_DEFAULT = 1.0
PLOT_FONT_SCALE_DEFAULT = 1.0
FIGURE_SIZE_SCALE_DEFAULT = 1.0

# Every individually-resizable plot: (key, tab_name, figure_attr, label).
# `key` is the stable id used in figure_size_scales/JSON; `tab_name` is the
# TabBase._tab_name each one belongs to (see TabBase._display_settings_figures,
# which filters this list down to whichever figure(s) the current tab owns -
# 1 each for ROI on 4D/Navigator/SAM2, 2 for ROI Tracker's split canvas).
PLOT_DEFINITIONS = [
    ('roi4d', 'Tab_ROI_on_4D', 'figure', 'ROI on 4D'),
    ('navigator', 'Tab_Create_NavSignal', 'figure', 'Navigator'),
    ('tracker_nav', 'Tab_Tracking_CV2', 'figure_nav', 'ROI Tracker (Overview)'),
    ('tracker_extract', 'Tab_Tracking_CV2', 'figure_extract', 'ROI Tracker (Extracted Pattern)'),
    ('sam2', 'Tab_SAM2', 'figure', 'SAM2 Tracker'),
]
PLOT_KEYS = [key for key, *_ in PLOT_DEFINITIONS]

_CONFIG_DIR = os.path.join(writable_data_dir(), 'config')
DEFAULTS_FILE = os.path.join(_CONFIG_DIR, 'display_defaults.json')
SETTINGS_FILE = os.path.join(_CONFIG_DIR, 'display_settings.json')


def _default_state():
    """The built-in factory defaults, as a plain dict - the same shape
    read/written to DEFAULTS_FILE/SETTINGS_FILE."""
    return {
        'ribbon_text_scale': RIBBON_TEXT_SCALE_DEFAULT,
        'ribbon_icon_size': RIBBON_ICON_SIZE_DEFAULT,
        'ribbon_height_scale': RIBBON_HEIGHT_SCALE_DEFAULT,
        'plot_font_scale': PLOT_FONT_SCALE_DEFAULT,
        'figure_size_scales': {key: FIGURE_SIZE_SCALE_DEFAULT for key in PLOT_KEYS},
    }


def _load_json(path):
    """Returns the parsed dict, or None if the file is missing/unreadable/
    corrupt - callers fall back to built-in defaults rather than crashing
    over a hand-edited or partially-written settings file."""
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


class DisplaySettings(QObject):
    """Singleton - use DisplaySettings.instance(), never construct directly
    (a second instance would have its own `changed` signal nobody listens
    to, silently going out of sync with the one every tab subscribed to)."""
    changed = pyqtSignal()

    _instance = None

    def __init__(self):
        super().__init__()
        if _load_json(DEFAULTS_FILE) is None:
            # First run (or the file was deleted) - lay down today's
            # built-in defaults as the permanent reset target.
            _save_json(DEFAULTS_FILE, _default_state())
        self._apply_state(self._load_state_with_fallback())

    def _load_state_with_fallback(self):
        """SETTINGS_FILE (last-used/user-customized) if present and valid,
        else DEFAULTS_FILE, else the hard-coded constants - so a missing or
        corrupt file on disk never prevents the app from starting."""
        return (_load_json(SETTINGS_FILE) or _load_json(DEFAULTS_FILE)
                or _default_state())

    def _apply_state(self, state):
        """Copy `state` onto self's attributes, merged over the built-in
        defaults key-by-key (not a wholesale replace) - a JSON file saved by
        an older version of the app, missing a plot added since, still
        yields a complete, valid set of attributes instead of a KeyError or
        a silently-missing scale for the new plot."""
        defaults = _default_state()
        self.ribbon_text_scale = state.get('ribbon_text_scale', defaults['ribbon_text_scale'])
        self.ribbon_icon_size = state.get('ribbon_icon_size', defaults['ribbon_icon_size'])
        self.ribbon_height_scale = state.get('ribbon_height_scale', defaults['ribbon_height_scale'])
        self.plot_font_scale = state.get('plot_font_scale', defaults['plot_font_scale'])
        saved_scales = state.get('figure_size_scales') or {}
        self.figure_size_scales = {
            key: saved_scales.get(key, defaults['figure_size_scales'][key])
            for key in PLOT_KEYS
        }

    def _current_state(self):
        return {
            'ribbon_text_scale': self.ribbon_text_scale,
            'ribbon_icon_size': self.ribbon_icon_size,
            'ribbon_height_scale': self.ribbon_height_scale,
            'plot_font_scale': self.plot_font_scale,
            'figure_size_scales': dict(self.figure_size_scales),
        }

    def _persist(self):
        """Write the current state to SETTINGS_FILE - called after every
        Apply and every Reset, so whatever's on screen now is what the app
        opens with next time, with no separate "save" step for the user."""
        _save_json(SETTINGS_FILE, self._current_state())

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_values(self, ribbon_text_scale=None, ribbon_icon_size=None,
                   ribbon_height_scale=None, plot_font_scale=None, figure_size_scales=None):
        """Update whichever values are given (None = leave unchanged), then
        emit `changed` once for the whole batch and persist to disk - the
        Display Size dialog's "Apply" button calls this once with every
        control's current value, rather than each control pushing its own
        change live.

        `figure_size_scales`, if given, is a dict of {plot_key: scale} -
        only the keys present are updated, so the dialog can pass just the
        1-5 plots the user actually touched (or all of them, e.g. via its
        "Apply to All Plots" action).
        """
        if ribbon_text_scale is not None:
            self.ribbon_text_scale = ribbon_text_scale
        if ribbon_icon_size is not None:
            self.ribbon_icon_size = ribbon_icon_size
        if ribbon_height_scale is not None:
            self.ribbon_height_scale = ribbon_height_scale
        if plot_font_scale is not None:
            self.plot_font_scale = plot_font_scale
        if figure_size_scales:
            self.figure_size_scales.update(
                {k: v for k, v in figure_size_scales.items() if k in self.figure_size_scales})
        self._persist()
        self.changed.emit()

    def reset(self):
        """Reset every value to DEFAULTS_FILE's contents (falling back to
        the hard-coded constants if that file is somehow missing/corrupt at
        this exact moment) - always the same target regardless of whatever
        customized values are currently in SETTINGS_FILE - then persists the
        reset state too, so it sticks across restarts instead of the old
        customized values reappearing next launch."""
        self._apply_state(_load_json(DEFAULTS_FILE) or _default_state())
        self._persist()
        self.changed.emit()
