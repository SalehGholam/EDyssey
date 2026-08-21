# -*- coding: utf-8 -*-
"""Edit tab: lets the user interactively resize the top ribbon (text size)
and the plots (font size of titles/axis labels/ticks) across every other
tab, plus the vertical ribbon icon strip beside each canvas - see
display_settings.py for the shared state this writes to, and
TabBase.apply_display_settings() for how each of the other 4 tabs applies
it live.
"""
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
from .base_tab import TabBase
from .display_settings import DisplaySettings


class Tab_Edit(TabBase):
    def __init__(self, parent=None):
        super().__init__('Tab_Edit', parent, own_threadpool=False)
        self.init_widget()

    def init_widget(self):
        self.layout = qtw.QVBoxLayout(self)
        self.setLayout(self.layout)

        intro = qtw.QLabel(
            'Adjust the size of the top ribbon and the plots, live, on every other tab '
            '(ROI on 4D, Navigator, ROI Tracker, SAM2 Tracker) - including any duplicate '
            'tabs opened later. Nothing here is saved between sessions.')
        intro.setWordWrap(True)
        intro.setStyleSheet('color: #cccccc; padding: 6px;')
        self.layout.addWidget(intro)

        form_box = qtw.QGroupBox('Display Size')
        form = qtw.QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignRight)
        self.layout.addWidget(form_box)

        settings = DisplaySettings.instance()

        # Ribbon text scale (50%-200%) - the top parameter ribbon's font size.
        row_ribbon_text = qtw.QHBoxLayout()
        self.slider_ribbonText = qtw.QSlider(Qt.Horizontal)
        self.slider_ribbonText.setRange(50, 200)
        self.slider_ribbonText.setValue(round(settings.ribbon_text_scale * 100))
        self.slider_ribbonText.setTickInterval(10)
        self.slider_ribbonText.setTickPosition(qtw.QSlider.TicksBelow)
        row_ribbon_text.addWidget(self.slider_ribbonText, 1)
        self.spinbox_ribbonText = qtw.QSpinBox()
        self.spinbox_ribbonText.setRange(50, 200)
        self.spinbox_ribbonText.setSuffix(' %')
        self.spinbox_ribbonText.setValue(round(settings.ribbon_text_scale * 100))
        row_ribbon_text.addWidget(self.spinbox_ribbonText)
        form.addRow('Top Ribbon Text Size', row_ribbon_text)

        # Ribbon icon size (16-48 px) - the vertical icon strip beside each canvas.
        row_ribbon_icon = qtw.QHBoxLayout()
        self.slider_ribbonIcon = qtw.QSlider(Qt.Horizontal)
        self.slider_ribbonIcon.setRange(16, 48)
        self.slider_ribbonIcon.setValue(settings.ribbon_icon_size)
        self.slider_ribbonIcon.setTickInterval(4)
        self.slider_ribbonIcon.setTickPosition(qtw.QSlider.TicksBelow)
        row_ribbon_icon.addWidget(self.slider_ribbonIcon, 1)
        self.spinbox_ribbonIcon = qtw.QSpinBox()
        self.spinbox_ribbonIcon.setRange(16, 48)
        self.spinbox_ribbonIcon.setSuffix(' px')
        self.spinbox_ribbonIcon.setValue(settings.ribbon_icon_size)
        row_ribbon_icon.addWidget(self.spinbox_ribbonIcon)
        form.addRow('Ribbon Icon Size', row_ribbon_icon)

        # Plot font scale (50%-200%) - title/axis-label/tick text size on every plot.
        row_plot_font = qtw.QHBoxLayout()
        self.slider_plotFont = qtw.QSlider(Qt.Horizontal)
        self.slider_plotFont.setRange(50, 200)
        self.slider_plotFont.setValue(round(settings.plot_font_scale * 100))
        self.slider_plotFont.setTickInterval(10)
        self.slider_plotFont.setTickPosition(qtw.QSlider.TicksBelow)
        row_plot_font.addWidget(self.slider_plotFont, 1)
        self.spinbox_plotFont = qtw.QSpinBox()
        self.spinbox_plotFont.setRange(50, 200)
        self.spinbox_plotFont.setSuffix(' %')
        self.spinbox_plotFont.setValue(round(settings.plot_font_scale * 100))
        row_plot_font.addWidget(self.spinbox_plotFont)
        form.addRow('Plot Text Size (titles/labels/ticks)', row_plot_font)

        button_row = qtw.QHBoxLayout()
        self.layout.addLayout(button_row)
        button_row.addStretch(1)
        self.button_reset = qtw.QPushButton('Reset to Defaults')
        self.button_reset.clicked.connect(self.reset_defaults)
        button_row.addWidget(self.button_reset)

        self.layout.addStretch(1)

        # Slider <-> spinbox: kept in sync with each other (blockSignals to
        # avoid feedback loops), and either one changing pushes the new
        # value straight to DisplaySettings, which every other tab is
        # already listening to (see TabBase.apply_display_settings) - no
        # separate "Apply" step needed, changes are live.
        self.slider_ribbonText.valueChanged.connect(self._on_ribbon_text_changed)
        self.spinbox_ribbonText.valueChanged.connect(self._on_ribbon_text_changed)
        self.slider_ribbonIcon.valueChanged.connect(self._on_ribbon_icon_changed)
        self.spinbox_ribbonIcon.valueChanged.connect(self._on_ribbon_icon_changed)
        self.slider_plotFont.valueChanged.connect(self._on_plot_font_changed)
        self.spinbox_plotFont.valueChanged.connect(self._on_plot_font_changed)

    def _sync(self, slider, spinbox, value):
        slider.blockSignals(True)
        spinbox.blockSignals(True)
        slider.setValue(value)
        spinbox.setValue(value)
        slider.blockSignals(False)
        spinbox.blockSignals(False)

    def _on_ribbon_text_changed(self, value):
        self._sync(self.slider_ribbonText, self.spinbox_ribbonText, value)
        DisplaySettings.instance().set_values(ribbon_text_scale=value / 100)

    def _on_ribbon_icon_changed(self, value):
        self._sync(self.slider_ribbonIcon, self.spinbox_ribbonIcon, value)
        DisplaySettings.instance().set_values(ribbon_icon_size=value)

    def _on_plot_font_changed(self, value):
        self._sync(self.slider_plotFont, self.spinbox_plotFont, value)
        DisplaySettings.instance().set_values(plot_font_scale=value / 100)

    def reset_defaults(self):
        DisplaySettings.instance().reset()
        settings = DisplaySettings.instance()
        self._sync(self.slider_ribbonText, self.spinbox_ribbonText,
                  round(settings.ribbon_text_scale * 100))
        self._sync(self.slider_ribbonIcon, self.spinbox_ribbonIcon, settings.ribbon_icon_size)
        self._sync(self.slider_plotFont, self.spinbox_plotFont,
                  round(settings.plot_font_scale * 100))
