# -*- coding: utf-8 -*-
""""Per-Plot Colormaps..." dialog - the expansion opened from Display
Preferences' own "Colormaps" group (see tab_edit.py), listing every
individually-recolorable image plot across all 4 tabs (see
display_settings.PLOT_COLORMAP_DEFINITIONS) with its own combo box, so a
single plot can be pinned to a specific colormap independent of the shared
Navigation Image/Diffraction Pattern settings. Non-modal, live-apply (same
convention as the two colormap combos in the parent dialog) - there is no
separate Apply step here.
"""
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
from .display_settings import DisplaySettings, COLORMAP_OPTIONS, PLOT_COLORMAP_DEFINITIONS
from .app_theme import AppTheme

_FOLLOW_GLOBAL = '(Follow Global)'


class PlotColormapDialog(qtw.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Per-Plot Colormaps')
        self.setWindowFlags(Qt.Window)
        self.init_widget()

    def init_widget(self):
        layout = qtw.QVBoxLayout(self)

        intro = qtw.QLabel(
            'Pin an individual plot to its own colormap, overriding the shared '
            'Navigation Image/Diffraction Pattern settings just for that one plot. '
            'Applies immediately, and is remembered the next time EDyssey opens. '
            '"Reset All to Global Colormaps" clears every override below at once, '
            'going back to each plot following the shared settings again.')
        intro.setWordWrap(True)
        intro.setFixedWidth(420)
        intro.setStyleSheet(f"color: {AppTheme.instance().color('fg_dim')}; padding: 6px;")
        layout.addWidget(intro)

        # A scroll area, not a bare QFormLayout - the list is long enough
        # (one row per plot across all 4 tabs) that a short screen shouldn't
        # force this whole dialog to grow past it; only the row list itself
        # scrolls, not the intro text/bottom buttons.
        scroll = qtw.QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = qtw.QWidget()
        form = qtw.QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignRight)
        scroll.setWidget(form_widget)
        layout.addWidget(scroll, 1)

        self._combos = {}
        settings = DisplaySettings.instance()
        for key, label, _role, _fixed_default in PLOT_COLORMAP_DEFINITIONS:
            combo = qtw.QComboBox()
            combo.addItem(_FOLLOW_GLOBAL)
            combo.addItems(COLORMAP_OPTIONS)
            override = settings.plot_colormap_overrides.get(key)
            combo.setCurrentText(override if override else _FOLLOW_GLOBAL)
            combo.currentTextChanged.connect(lambda _text, k=key: self._on_combo_changed(k))
            form.addRow(label, combo)
            self._combos[key] = combo

        button_row = qtw.QHBoxLayout()
        layout.addLayout(button_row)
        button_row.addStretch(1)
        self.button_reset = qtw.QPushButton('Reset All to Global Colormaps')
        self.button_reset.setToolTip(
            'Clear every per-plot override above, so all plots follow the shared '
            'Navigation Image/Diffraction Pattern colormaps again')
        self.button_reset.clicked.connect(self.reset_all)
        button_row.addWidget(self.button_reset)
        self.button_close = qtw.QPushButton('Close')
        self.button_close.clicked.connect(self.close)
        button_row.addWidget(self.button_close)

        self.resize(480, 520)

    def _on_combo_changed(self, key):
        combo = self._combos[key]
        text = combo.currentText()
        DisplaySettings.instance().set_values(
            plot_colormap_overrides={key: None if text == _FOLLOW_GLOBAL else text})

    def reset_all(self):
        """"Reset All to Global Colormaps": clear every override at once,
        and re-sync every combo back to "(Follow Global)" to match."""
        DisplaySettings.instance().set_values(
            plot_colormap_overrides={key: None for key in self._combos})
        self.resync_combos()

    def resync_combos(self):
        """Re-read every combo's value from DisplaySettings - called after
        the parent Display Preferences dialog's own "Reset to Defaults"
        (which resets this dialog's overrides too, via DisplaySettings.
        reset(), without going through this dialog's own reset_all()) so
        the combos here don't show stale values if this dialog happens to
        already be open."""
        settings = DisplaySettings.instance()
        for key, combo in self._combos.items():
            override = settings.plot_colormap_overrides.get(key)
            combo.blockSignals(True)
            combo.setCurrentText(override if override else _FOLLOW_GLOBAL)
            combo.blockSignals(False)
