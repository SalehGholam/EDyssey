# -*- coding: utf-8 -*-
"""Display Size dialog: lets the user interactively resize the top ribbon
(text size, icon size, height) and the plots (font size of titles/axis
labels/ticks, and the figure/canvas size itself) across every other tab -
opened from the main window's "Edit" menu, not a tab of its own, so it
doesn't take up a permanent slot in the tab bar. Non-modal (.show(), not
.exec_()) so it can stay open while the user switches tabs to see the
effect. Changes only take effect on "Apply" - moving a slider previews
nothing live, unlike the shared widget's own instant-apply controls
elsewhere in the app. See display_settings.py for the shared state this
writes to, and TabBase.apply_display_settings() for how each of the other
4 tabs applies it.
"""
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
from .display_settings import DisplaySettings


class EditSettingsDialog(qtw.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Edit Display Size')
        # Not Qt.Dialog's default modality - a plain top-level window the
        # user can leave open (and move sliders on) while switching between
        # the other tabs, which is the whole point of live-adjusting these.
        self.setWindowFlags(Qt.Window)
        self.init_widget()

    def init_widget(self):
        self.layout = qtw.QVBoxLayout(self)
        self.setLayout(self.layout)

        intro = qtw.QLabel(
            'Adjust the size of the top ribbon and the plots on every other tab '
            '(ROI on 4D, Navigator, ROI Tracker, SAM2 Tracker) - including any duplicate '
            'tabs opened later. Changes only take effect once you click "Apply". '
            'Nothing here is saved between sessions.')
        intro.setWordWrap(True)
        intro.setFixedWidth(380)
        intro.setStyleSheet('color: #cccccc; padding: 6px;')
        self.layout.addWidget(intro)

        form_box = qtw.QGroupBox('Ribbon')
        form = qtw.QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignRight)
        self.layout.addWidget(form_box)

        settings = DisplaySettings.instance()

        self.slider_ribbonText, self.spinbox_ribbonText = self._add_percent_row(
            form, 'Top Ribbon Text Size', 50, 200, round(settings.ribbon_text_scale * 100))
        self.slider_ribbonIcon, self.spinbox_ribbonIcon = self._add_row(
            form, 'Ribbon Icon Size', 16, 48, settings.ribbon_icon_size, ' px')
        self.slider_ribbonHeight, self.spinbox_ribbonHeight = self._add_percent_row(
            form, 'Ribbon Height', 50, 200, round(settings.ribbon_height_scale * 100))

        form_box2 = qtw.QGroupBox('Plots')
        form2 = qtw.QFormLayout(form_box2)
        form2.setLabelAlignment(Qt.AlignRight)
        self.layout.addWidget(form_box2)

        self.slider_plotFont, self.spinbox_plotFont = self._add_percent_row(
            form2, 'Plot Text Size (titles/labels/ticks)', 50, 200,
            round(settings.plot_font_scale * 100))
        self.slider_figureSize, self.spinbox_figureSize = self._add_percent_row(
            form2, 'Figure/Plot Size', 50, 200, round(settings.figure_size_scale * 100))

        button_row = qtw.QHBoxLayout()
        self.layout.addLayout(button_row)
        button_row.addStretch(1)
        self.button_reset = qtw.QPushButton('Reset to Defaults')
        self.button_reset.setToolTip('Resets and applies immediately')
        self.button_reset.clicked.connect(self.reset_defaults)
        button_row.addWidget(self.button_reset)
        self.button_apply = qtw.QPushButton('Apply')
        self.button_apply.clicked.connect(self.apply_values)
        button_row.addWidget(self.button_apply)
        self.button_close = qtw.QPushButton('Close')
        self.button_close.clicked.connect(self.close)
        button_row.addWidget(self.button_close)

    def _add_row(self, form, label, minimum, maximum, value, suffix):
        """One slider+spinbox row (kept in sync with each other only - no
        DisplaySettings write here, see apply_values)."""
        row = qtw.QHBoxLayout()
        slider = qtw.QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setTickInterval(max((maximum - minimum) // 10, 1))
        slider.setTickPosition(qtw.QSlider.TicksBelow)
        row.addWidget(slider, 1)
        spinbox = qtw.QSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setSuffix(suffix)
        spinbox.setValue(value)
        row.addWidget(spinbox)
        slider.valueChanged.connect(lambda v: self._sync(slider, spinbox, v))
        spinbox.valueChanged.connect(lambda v: self._sync(slider, spinbox, v))
        form.addRow(label, row)
        return slider, spinbox

    def _add_percent_row(self, form, label, minimum, maximum, value):
        return self._add_row(form, label, minimum, maximum, value, ' %')

    def _sync(self, slider, spinbox, value):
        slider.blockSignals(True)
        spinbox.blockSignals(True)
        slider.setValue(value)
        spinbox.setValue(value)
        slider.blockSignals(False)
        spinbox.blockSignals(False)

    def apply_values(self):
        """Push every control's current value to DisplaySettings at once -
        the "Apply" button's slot (nothing here happens live/automatically
        as the sliders move)."""
        DisplaySettings.instance().set_values(
            ribbon_text_scale=self.spinbox_ribbonText.value() / 100,
            ribbon_icon_size=self.spinbox_ribbonIcon.value(),
            ribbon_height_scale=self.spinbox_ribbonHeight.value() / 100,
            plot_font_scale=self.spinbox_plotFont.value() / 100,
            figure_size_scale=self.spinbox_figureSize.value() / 100)

    def reset_defaults(self):
        """Reset every value to its default AND apply immediately - unlike
        the sliders above, Reset is already a single, deliberate action, so
        it doesn't need a separate Apply step."""
        DisplaySettings.instance().reset()
        settings = DisplaySettings.instance()
        self._sync(self.slider_ribbonText, self.spinbox_ribbonText,
                  round(settings.ribbon_text_scale * 100))
        self._sync(self.slider_ribbonIcon, self.spinbox_ribbonIcon, settings.ribbon_icon_size)
        self._sync(self.slider_ribbonHeight, self.spinbox_ribbonHeight,
                  round(settings.ribbon_height_scale * 100))
        self._sync(self.slider_plotFont, self.spinbox_plotFont,
                  round(settings.plot_font_scale * 100))
        self._sync(self.slider_figureSize, self.spinbox_figureSize,
                  round(settings.figure_size_scale * 100))
