# -*- coding: utf-8 -*-
"""App-wide UI color theme - Dark (the app's original/default look) or
Light - one shared singleton, changed from the Edit menu's Display Size
dialog (see tab_edit.py) and applied live, three ways:

1. Every standard Qt widget (buttons, labels, tables, tabs, ...), via one
   QApplication-wide stylesheet (see build_stylesheet) - this is what
   EDyssey_MainWindow.py used to set once, hardcoded to dark, directly on
   itself; now generated from PALETTES and (re-)applied to the whole
   QApplication instead, so it reaches every dialog too, not just the main
   window's own children, and can change at runtime.
2. Every matplotlib Figure, via rcParams (see apply_qapp) for anything
   drawn from here on, plus TabBase.apply_theme() re-coloring each of the 4
   main tabs' own already-open Figure live (a dialog's own Figure, e.g.
   MaskEditDialog's, isn't re-colored retroactively if it's already open
   when the theme changes - it reads whatever's current the next time it's
   opened, same as any other freshly-created Figure).
3. The handful of custom QPainter-drawn widgets that can't be reached by a
   Qt stylesheet at all (RibbonPanel's own icons, frame_flag_bar.
   FrameFlagBar) - each subscribes to `changed` itself and re-reads
   color()/palette from here, so they remain theme-aware without every one
   of their many construction sites needing to remember to wire that up.

LogConsole (see logging_utils.py) is a deliberate exception - it keeps its
own fixed dark terminal-style palette regardless of this theme (the same
convention many IDEs' own terminal/log panels use), since its own error/
warning color-coding needs one stable, always-legible background rather
than following the rest of the app.

Persisted to disk the same way as display_settings.py - see that module's
own docstring for the DEFAULTS_FILE/SETTINGS_FILE rationale (this only
needs a single SETTINGS_FILE, no separate defaults snapshot - "reset" one
theme choice back to the other is just picking it from the same 2-item
list again, nothing to accidentally lose).
"""
import os
import json
import logging
import matplotlib.pyplot as plt
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import QObject, pyqtSignal
from EDyssey.io_utils.app_dirs import writable_data_dir

logger = logging.getLogger('EDyssey.app_theme')

THEME_DEFAULT = 'dark'

# Semantic palette per theme - every hardcoded color anywhere else in the
# app that should follow the theme (the QApplication stylesheet, RibbonPanel
# icons, FrameFlagBar, ...) reads from here instead of its own hex literal,
# so a new theme (or a tweak to an existing one) only needs changing here.
#
# 'mpl_style' picks matplotlib's own base style (only 'dark_background' and
# 'default' are used - both well-tuned, broadly-known styles). The named
# popular themes below (Solarized/Dracula/Nord/Monokai) aren't matplotlib
# styles of their own, so 'mpl_overrides' - applied on top of the base style
# in apply_qapp() - nudges just the handful of rcParams that matter for
# plots (figure/axes background, text/tick/spine/grid color) to match that
# theme's own bg/fg instead of generic style's own black/white. 'dark' and
# 'light' deliberately carry NO overrides - they stay exactly
# dark_background's/default's own colors, which is what keeps them
# byte-identical to this app's original, pre-theme-system look.
PALETTES = {
    'dark': {
        'bg': '#2b2b2b', 'bg_alt': '#3c3c3c', 'bg_panel': '#333333',
        'fg': '#f0f0f0', 'fg_dim': '#888888', 'fg_disabled': '#777777',
        'border': '#555555', 'button_border': '#666666',
        'accent': '#4a86c8', 'accent_hover': '#5a96d8',
        'accent_light': '#7fb8ec', 'accent_light_hover': '#8fc8fc',
        'button': '#4a4a4a', 'button_hover': '#5a5a5a', 'button_pressed': '#3a3a3a',
        'scrollbar': '#3c3c3c', 'scrollbar_handle': '#666666',
        'icon': '#f0f0f0',
        'bar_bg': '#3a3a3a',
        'mpl_style': 'dark_background',
    },
    'light': {
        'bg': '#f2f2f2', 'bg_alt': '#ffffff', 'bg_panel': '#e8e8e8',
        'fg': '#202020', 'fg_dim': '#606060', 'fg_disabled': '#a0a0a0',
        'border': '#b0b0b0', 'button_border': '#999999',
        'accent': '#3874b8', 'accent_hover': '#4a86c8',
        'accent_light': '#7fb8ec', 'accent_light_hover': '#6aa8dc',
        'button': '#e4e4e4', 'button_hover': '#d4d4d4', 'button_pressed': '#c4c4c4',
        'scrollbar': '#e0e0e0', 'scrollbar_handle': '#a0a0a0',
        'icon': '#202020',
        'bar_bg': '#c8c8c8',
        'mpl_style': 'default',
    },
    'solarized_dark': {
        'bg': '#002b36', 'bg_alt': '#073642', 'bg_panel': '#0a3d4a',
        'fg': '#93a1a1', 'fg_dim': '#657b83', 'fg_disabled': '#586e75',
        'border': '#073642', 'button_border': '#586e75',
        'accent': '#268bd2', 'accent_hover': '#3399e0',
        'accent_light': '#6ab0e0', 'accent_light_hover': '#8cc4e8',
        'button': '#073642', 'button_hover': '#0a4552', 'button_pressed': '#052831',
        'scrollbar': '#073642', 'scrollbar_handle': '#586e75',
        'icon': '#93a1a1',
        'bar_bg': '#073642',
        'mpl_style': 'dark_background',
        'mpl_overrides': {
            'figure.facecolor': '#002b36', 'axes.facecolor': '#002b36',
            'text.color': '#93a1a1', 'axes.edgecolor': '#93a1a1',
            'axes.labelcolor': '#93a1a1', 'xtick.color': '#93a1a1',
            'ytick.color': '#93a1a1', 'grid.color': '#073642',
        },
    },
    'solarized_light': {
        'bg': '#fdf6e3', 'bg_alt': '#eee8d5', 'bg_panel': '#f5efdc',
        'fg': '#586e75', 'fg_dim': '#657b83', 'fg_disabled': '#93a1a1',
        'border': '#eee8d5', 'button_border': '#d3cbb7',
        'accent': '#268bd2', 'accent_hover': '#2075b0',
        'accent_light': '#6ab0e0', 'accent_light_hover': '#4a9cd6',
        'button': '#eee8d5', 'button_hover': '#e4ddc7', 'button_pressed': '#d9d1ba',
        'scrollbar': '#eee8d5', 'scrollbar_handle': '#93a1a1',
        'icon': '#586e75',
        'bar_bg': '#eee8d5',
        'mpl_style': 'default',
        'mpl_overrides': {
            'figure.facecolor': '#fdf6e3', 'axes.facecolor': '#fdf6e3',
            'text.color': '#586e75', 'axes.edgecolor': '#586e75',
            'axes.labelcolor': '#586e75', 'xtick.color': '#586e75',
            'ytick.color': '#586e75', 'grid.color': '#eee8d5',
        },
    },
    'dracula': {
        'bg': '#282a36', 'bg_alt': '#343746', 'bg_panel': '#2f3140',
        'fg': '#f8f8f2', 'fg_dim': '#9099b8', 'fg_disabled': '#6272a4',
        'border': '#44475a', 'button_border': '#565973',
        'accent': '#bd93f9', 'accent_hover': '#cba6fb',
        'accent_light': '#d6b8fc', 'accent_light_hover': '#e2cafd',
        'button': '#3a3d4d', 'button_hover': '#454858', 'button_pressed': '#24252e',
        'scrollbar': '#343746', 'scrollbar_handle': '#6272a4',
        'icon': '#f8f8f2',
        'bar_bg': '#343746',
        'mpl_style': 'dark_background',
        'mpl_overrides': {
            'figure.facecolor': '#282a36', 'axes.facecolor': '#282a36',
            'text.color': '#f8f8f2', 'axes.edgecolor': '#f8f8f2',
            'axes.labelcolor': '#f8f8f2', 'xtick.color': '#f8f8f2',
            'ytick.color': '#f8f8f2', 'grid.color': '#44475a',
        },
    },
    'nord': {
        'bg': '#2e3440', 'bg_alt': '#3b4252', 'bg_panel': '#343a46',
        'fg': '#eceff4', 'fg_dim': '#9aa5b8', 'fg_disabled': '#4c566a',
        'border': '#4c566a', 'button_border': '#5e6779',
        'accent': '#88c0d0', 'accent_hover': '#8fbcbb',
        'accent_light': '#a3d4e0', 'accent_light_hover': '#b8dee8',
        'button': '#3b4252', 'button_hover': '#434c5e', 'button_pressed': '#2e3440',
        'scrollbar': '#3b4252', 'scrollbar_handle': '#4c566a',
        'icon': '#eceff4',
        'bar_bg': '#3b4252',
        'mpl_style': 'dark_background',
        'mpl_overrides': {
            'figure.facecolor': '#2e3440', 'axes.facecolor': '#2e3440',
            'text.color': '#eceff4', 'axes.edgecolor': '#eceff4',
            'axes.labelcolor': '#eceff4', 'xtick.color': '#eceff4',
            'ytick.color': '#eceff4', 'grid.color': '#4c566a',
        },
    },
    'monokai': {
        'bg': '#272822', 'bg_alt': '#3e3d32', 'bg_panel': '#33342c',
        'fg': '#f8f8f2', 'fg_dim': '#a6a89a', 'fg_disabled': '#75715e',
        'border': '#49483e', 'button_border': '#5b5a4e',
        'accent': '#66d9ef', 'accent_hover': '#7fe0f2',
        'accent_light': '#99e6f5', 'accent_light_hover': '#b3ecf8',
        'button': '#3e3d32', 'button_hover': '#49483e', 'button_pressed': '#232420',
        'scrollbar': '#3e3d32', 'scrollbar_handle': '#75715e',
        'icon': '#f8f8f2',
        'bar_bg': '#3e3d32',
        'mpl_style': 'dark_background',
        'mpl_overrides': {
            'figure.facecolor': '#272822', 'axes.facecolor': '#272822',
            'text.color': '#f8f8f2', 'axes.edgecolor': '#f8f8f2',
            'axes.labelcolor': '#f8f8f2', 'xtick.color': '#f8f8f2',
            'ytick.color': '#f8f8f2', 'grid.color': '#49483e',
        },
    },
}
THEME_LABELS = {
    'dark': 'Dark', 'light': 'Light',
    'solarized_dark': 'Solarized Dark', 'solarized_light': 'Solarized Light',
    'dracula': 'Dracula', 'nord': 'Nord', 'monokai': 'Monokai',
}

_CONFIG_DIR = os.path.join(writable_data_dir(), 'config')
SETTINGS_FILE = os.path.join(_CONFIG_DIR, 'theme_settings.json')


def _load_json(path):
    """Returns the parsed dict, or None if the file is missing/unreadable/
    corrupt - callers fall back to the built-in default rather than
    crashing over a hand-edited or partially-written settings file."""
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


def build_stylesheet(palette):
    """The whole-app QSS - ported from EDyssey_MainWindow's own original
    hardcoded (dark-only) stylesheet, with every hex literal replaced by
    the matching palette entry."""
    p = palette
    return f"""
        QMainWindow, QWidget {{
            background-color: {p['bg']}; color: {p['fg']};
        }}
        QTabWidget::pane {{ border: 1px solid {p['border']}; }}
        QTabBar::tab {{
            background: {p['bg_alt']}; color: {p['fg']};
            padding: 5px 12px; border: 1px solid {p['border']};
        }}
        QTabBar::tab:selected {{ background: {p['border']}; }}
        QGroupBox {{
            border: 1px solid {p['border']}; margin-top: 8px; color: {p['fg']};
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {p['fg']}; }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {p['bg_alt']}; color: {p['fg']};
            border: 1px solid {p['border']}; padding: 2px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {p['bg_alt']}; color: {p['fg']};
            selection-background-color: {p['accent']};
        }}
        QPushButton {{
            background-color: {p['button']}; color: {p['fg']};
            border: 1px solid {p['button_border']}; padding: 4px 8px;
        }}
        QPushButton:hover {{ background-color: {p['button_hover']}; }}
        QPushButton:pressed {{ background-color: {p['button_pressed']}; }}
        QPushButton:disabled {{ background-color: {p['button_pressed']}; color: {p['fg_disabled']}; }}
        QCheckBox {{ color: {p['fg']}; }}
        QLabel {{ color: {p['fg']}; }}
        QSlider::groove:horizontal {{ background: {p['bg_alt']}; height: 4px; }}
        QSlider::handle:horizontal {{
            background: {p['fg_dim']}; width: 12px; margin: -4px 0; border-radius: 6px;
        }}
        QProgressBar {{
            background-color: {p['bg_alt']}; color: {p['fg']};
            border: 1px solid {p['border']}; text-align: center;
        }}
        QProgressBar::chunk {{ background-color: {p['accent']}; }}
        QListWidget, QTreeWidget, QTableWidget {{
            background-color: {p['bg']}; color: {p['fg']};
            border: 1px solid {p['border']}; alternate-background-color: {p['bg_panel']};
        }}
        QListWidget::item:selected, QTreeWidget::item:selected {{
            background-color: {p['accent']};
        }}
        QHeaderView::section {{
            background-color: {p['bg_alt']}; color: {p['fg']};
            border: 1px solid {p['border']}; padding: 2px;
        }}
        QScrollBar:vertical {{ background: {p['scrollbar']}; width: 12px; }}
        QScrollBar::handle:vertical {{ background: {p['scrollbar_handle']}; min-height: 20px; }}
        QScrollBar:horizontal {{ background: {p['scrollbar']}; height: 12px; }}
        QScrollBar::handle:horizontal {{ background: {p['scrollbar_handle']}; min-width: 20px; }}
        QStatusBar {{ background-color: {p['bg']}; color: {p['fg_dim']}; }}
        QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: {p['border']}; }}
        QToolButton {{
            background-color: {p['bg']}; color: {p['fg']};
            border: 1px solid transparent;
        }}
        QToolButton:hover {{ background-color: {p['bg_alt']}; border: 1px solid {p['border']}; }}
        QToolButton:checked {{
            background-color: {p['accent']}; border: 1px solid {p['accent_light']};
        }}
        QToolButton:checked:hover {{
            background-color: {p['accent_hover']}; border: 1px solid {p['accent_light_hover']};
        }}
    """


class AppTheme(QObject):
    """Singleton - use AppTheme.instance(), never construct directly (a
    second instance would have its own `changed` signal nobody listens to,
    silently going out of sync with the one every subscriber connected to -
    same reasoning as DisplaySettings' identical singleton pattern)."""
    changed = pyqtSignal()

    _instance = None

    def __init__(self):
        super().__init__()
        name = (_load_json(SETTINGS_FILE) or {}).get('theme', THEME_DEFAULT)
        self.name = name if name in PALETTES else THEME_DEFAULT

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def palette(self):
        return PALETTES[self.name]

    def color(self, key):
        return self.palette[key]

    def set_theme(self, name):
        """Switch to `name` ('dark'/'light'), apply it live app-wide
        (QApplication stylesheet + matplotlib rcParams for anything drawn
        from now on - see apply_qapp), persist it, and notify every
        subscriber (TabBase.apply_theme, RibbonPanel, FrameFlagBar, ...) so
        already-built widgets/figures re-color too, without needing a
        restart. A no-op if `name` isn't a real theme or is already active."""
        if name not in PALETTES or name == self.name:
            return
        self.name = name
        self.apply_qapp()
        _save_json(SETTINGS_FILE, {'theme': self.name})
        self.changed.emit()

    def apply_qapp(self):
        """(Re-)apply this theme's stylesheet to the whole running
        QApplication (reaches every window/dialog, not just whichever ones
        happen to be children of MainWindow), and set matplotlib's own
        rcParams style so any Figure created from here on picks up the new
        theme automatically. Safe to call before a QApplication exists
        (does nothing) - lets __init__ below call it unconditionally."""
        app = qtw.QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(self.palette))
        # plt.style.use() resets EVERY rcParam to that style's own
        # defaults, including ones this app sets independently of theme
        # (tab_sam2.py/EDyssey.io_utils.video's own
        # 'animation.ffmpeg_path') - preserve those across the reset
        # instead of silently clobbering them.
        ffmpeg_path = plt.rcParams.get('animation.ffmpeg_path')
        plt.style.use(self.palette['mpl_style'])
        # Named themes that aren't matplotlib styles of their own (Solarized/
        # Dracula/Nord/Monokai) nudge the handful of rcParams that matter for
        # plots to match their own bg/fg - see PALETTES' own comment on why
        # 'dark'/'light' carry no overrides of their own.
        overrides = self.palette.get('mpl_overrides')
        if overrides:
            plt.rcParams.update(overrides)
        if ffmpeg_path:
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
