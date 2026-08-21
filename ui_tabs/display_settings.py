# -*- coding: utf-8 -*-
"""App-wide display-scale settings (ribbon text size, ribbon icon size,
plot font size) - one shared singleton, changed from the new Edit tab and
applied live by every other tab via TabBase.apply_display_settings().
Session-only (not persisted to disk) - matches the rest of the app, which
has no settings-persistence mechanism yet.
"""
from PyQt5.QtCore import QObject, pyqtSignal

RIBBON_TEXT_SCALE_DEFAULT = 1.0
RIBBON_ICON_SIZE_DEFAULT = 26
PLOT_FONT_SCALE_DEFAULT = 1.0


class DisplaySettings(QObject):
    """Singleton - use DisplaySettings.instance(), never construct directly
    (a second instance would have its own `changed` signal nobody listens
    to, silently going out of sync with the one every tab subscribed to)."""
    changed = pyqtSignal()

    _instance = None

    def __init__(self):
        super().__init__()
        self.ribbon_text_scale = RIBBON_TEXT_SCALE_DEFAULT
        self.ribbon_icon_size = RIBBON_ICON_SIZE_DEFAULT
        self.plot_font_scale = PLOT_FONT_SCALE_DEFAULT

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_values(self, ribbon_text_scale=None, ribbon_icon_size=None, plot_font_scale=None):
        """Update whichever values are given (None = leave unchanged), then
        emit `changed` once for the whole batch."""
        if ribbon_text_scale is not None:
            self.ribbon_text_scale = ribbon_text_scale
        if ribbon_icon_size is not None:
            self.ribbon_icon_size = ribbon_icon_size
        if plot_font_scale is not None:
            self.plot_font_scale = plot_font_scale
        self.changed.emit()

    def reset(self):
        self.ribbon_text_scale = RIBBON_TEXT_SCALE_DEFAULT
        self.ribbon_icon_size = RIBBON_ICON_SIZE_DEFAULT
        self.plot_font_scale = PLOT_FONT_SCALE_DEFAULT
        self.changed.emit()
