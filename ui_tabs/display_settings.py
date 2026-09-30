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
# Opacity of the translucent tracked-object/segmentation mask overlays drawn
# on ROI Tracker's, SAM2 Tracker's, and ROI on 4D's own main canvases (NOT
# the separate Fine-Tune Mask dialog, which has its own independent,
# per-session opacity spinboxes - see mask_edit_dialog.py). Replaces what
# used to be several independent hardcoded alphas (0.3-0.85, one or two per
# tab) with one shared, user-adjustable value.
MASK_ALPHA_DEFAULT = 0.5

# Curated, not matplotlib's full list - perceptually-uniform ones
# (viridis/inferno/plasma/magma/cividis) plus a few classic/high-contrast
# options - each paired with its own valid _r (reversed) variant, since
# matplotlib registers one for every named colormap.
COLORMAP_OPTIONS = [
    'gray', 'gray_r', 'viridis', 'viridis_r', 'inferno', 'inferno_r',
    'plasma', 'plasma_r', 'magma', 'magma_r', 'cividis', 'cividis_r',
    'turbo', 'turbo_r', 'hot', 'hot_r', 'bone', 'bone_r',
    'copper', 'copper_r', 'twilight', 'twilight_r',
]
# Every individually-resizable plot: (key, tab_name, figure_attr, label).
# `key` is the stable id used in figure_size_scales/JSON; `tab_name` is the
# TabBase._tab_name each one belongs to (see TabBase._display_settings_figures,
# which filters this list down to whichever figure(s) the current tab owns -
# 1 each for every tab, all sharing the single `figure`/`canvas` attribute
# name (ROI Tracker's 4 subplots - Nav./Tracking/Threshold/DP - used to be
# split across 2 figures, figure_nav/figure_extract, merged into one).
PLOT_DEFINITIONS = [
    ('roi4d', 'Tab_ROI_on_4D', 'figure', 'ROI on 4D'),
    ('navigator', 'Tab_Create_NavSignal', 'figure', 'Navigator'),
    ('tracker', 'Tab_Tracking_CV2', 'figure', 'ROI Tracker'),
    ('sam2', 'Tab_SAM2', 'figure', 'SAM2 Tracker'),
]
PLOT_KEYS = [key for key, *_ in PLOT_DEFINITIONS]

# Every individually-recolorable image plot across all 4 tabs: (key, label,
# default_colormap) - each plot's colormap is fully independent (no shared
# "Navigation Image"/"Diffraction Pattern" setting to fall back to by
# request - every plot gets its own combo in Display Preferences directly).
# default_colormap is just this plot's own starting value/"Reset" target,
# matching what it used to default to when nav-like/dp-like plots followed
# a shared viridis/inferno setting and mask/segmentation-overlay backgrounds
# were fixed to gray.
PLOT_COLORMAP_DEFINITIONS = [
    ('navigator_nav', 'Navigator - Navigation Image', 'viridis'),
    ('navigator_dp', 'Navigator - Summed DP', 'inferno'),
    ('roi4d_nav', 'ROI on 4D - Navigation Image', 'viridis'),
    ('roi4d_dp', 'ROI on 4D - Diffraction Pattern', 'inferno'),
    ('roi4d_nav_roi', 'ROI on 4D - Nav+ROI Overlay Background', 'gray'),
    ('tracker_nav', 'ROI Tracker - Navigation Image', 'viridis'),
    ('tracker_dp', 'ROI Tracker - Diffraction Pattern', 'inferno'),
    ('tracker_mask', 'ROI Tracker - Segmented Object Background', 'gray'),
    ('sam2_nav', 'SAM2 Tracker - Navigation Image Background', 'gray'),
    ('sam2_seg', 'SAM2 Tracker - Segmented Object Background', 'gray'),
    ('sam2_dp', 'SAM2 Tracker - Diffraction Pattern', 'inferno'),
]
PLOT_COLORMAP_KEYS = [key for key, *_ in PLOT_COLORMAP_DEFINITIONS]
_PLOT_COLORMAP_DEFAULTS = {key: default for key, _label, default in PLOT_COLORMAP_DEFINITIONS}


def _reversed_colormap_name(name):
    """The paired _r (reversed) variant of `name` - every entry in
    COLORMAP_OPTIONS has one (see its own comment) - or `name` with a
    trailing '_r' stripped if it's already the reversed one. Used wherever
    a "Revert Contrast"-style toggle needs to flip whichever colormap is
    ACTUALLY in effect (possibly a per-plot override, not always the
    hardcoded 'viridis' an in-effect colormap used to be assumed to be)."""
    return name[:-2] if name.endswith('_r') else name + '_r'

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
        'plot_colormaps': dict(_PLOT_COLORMAP_DEFAULTS),
        'mask_alpha': MASK_ALPHA_DEFAULT,
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
        # Each plot's own independent colormap - same merge-by-key/validate-
        # each-value convention as figure_size_scales above, so an old
        # settings file (missing a plot added since, still using the old
        # 'nav_colormap'/'dp_colormap'/'plot_colormap_overrides' shape from
        # before every plot got its own fully independent setting, or with a
        # hand-edited invalid name) never breaks loading - it just falls
        # back to that one plot's own built-in default.
        saved_cmaps = state.get('plot_colormaps') or {}
        self.plot_colormaps = {
            key: (saved_cmaps.get(key) if saved_cmaps.get(key) in COLORMAP_OPTIONS
                 else _PLOT_COLORMAP_DEFAULTS[key])
            for key in PLOT_COLORMAP_KEYS
        }
        # Clamped to [0, 1] - a hand-edited or out-of-range saved value would
        # otherwise pass straight through to matplotlib's own RGBA alpha
        # channel, which silently clips anyway but a slider/spinbox reading
        # it back needs an in-range value to show.
        mask_alpha = state.get('mask_alpha', defaults['mask_alpha'])
        try:
            self.mask_alpha = min(1.0, max(0.0, float(mask_alpha)))
        except (TypeError, ValueError):
            self.mask_alpha = defaults['mask_alpha']

    def _current_state(self):
        return {
            'ribbon_text_scale': self.ribbon_text_scale,
            'ribbon_icon_size': self.ribbon_icon_size,
            'ribbon_height_scale': self.ribbon_height_scale,
            'plot_font_scale': self.plot_font_scale,
            'figure_size_scales': dict(self.figure_size_scales),
            'plot_colormaps': dict(self.plot_colormaps),
            'mask_alpha': self.mask_alpha,
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
                   ribbon_height_scale=None, plot_font_scale=None, figure_size_scales=None,
                   plot_colormaps=None, mask_alpha=None):
        """Update whichever values are given (None = leave unchanged), then
        emit `changed` once for the whole batch and persist to disk - the
        Display Size dialog's "Apply" button calls this once with every
        control's current value, rather than each control pushing its own
        change live.

        `figure_size_scales`/`plot_colormaps`, if given, are dicts of
        {plot_key: value} - only the keys present are updated, so a caller
        can pass just the 1-11 plots the user actually touched (or all of
        them, e.g. Reset).
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
        if plot_colormaps:
            self.plot_colormaps.update(
                {k: v for k, v in plot_colormaps.items() if k in self.plot_colormaps})
        if mask_alpha is not None:
            self.mask_alpha = min(1.0, max(0.0, float(mask_alpha)))
        self._persist()
        self.changed.emit()

    def colormap_for(self, plot_key):
        """This plot's own current colormap - the single place every tab's
        own apply_display_settings() should read a per-plot colormap from
        (see PLOT_COLORMAP_DEFINITIONS - every plot is fully independent,
        no shared "Navigation Image"/"Diffraction Pattern" setting to fall
        back to)."""
        return self.plot_colormaps.get(plot_key, _PLOT_COLORMAP_DEFAULTS.get(plot_key, 'gray'))

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
