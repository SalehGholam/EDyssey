# -*- coding: utf-8 -*-
"""Shared "Clipping Thresholds" widget: a vertical vmin/vmax slider pair
with a title, meant to sit directly beside a matplotlib image axis (a
navigation image or a diffraction pattern) rather than in the ribbon/
parameter panel - so the control is physically next to the image it clips.

Usage:
    self.clip_dp = ClippingThresholdsWidget()
    layout.addWidget(self.clip_dp)
    self.clip_dp.valueChanged.connect(lambda: self.update_canvas(ax='dp'))
    ...
    # whenever a new image is loaded:
    self.clip_dp.set_range(self.dp.min(), self.dp.max())
    ...
    # whenever the image is (re)drawn:
    vmin, vmax = self.clip_dp.values()
    self.img_display['dp'].set_clim(vmin, vmax)
"""
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, pyqtSignal


class ClippingThresholdsWidget(qtw.QWidget):
    """Vertical vmin/vmax sliders ("Clipping Thresholds") for one image
    axis. Both sliders share one narrow vertical column - a title on top,
    then the vmax slider (upper bound) and vmin slider (lower bound)
    stacked below it, each with a compact numeric readout, then a small
    Reset button."""
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(58)
        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        title = qtw.QLabel('Clipping\nThresholds')
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet('color: #999999; font-size: 8pt;')
        title.setWordWrap(True)
        layout.addWidget(title)

        self.label_vmax = qtw.QLabel('max: -')
        self.label_vmax.setAlignment(Qt.AlignHCenter)
        self.label_vmax.setStyleSheet('font-size: 8pt;')
        layout.addWidget(self.label_vmax)

        self.slider_vmax = qtw.QSlider(Qt.Vertical)
        self.slider_vmax.setRange(0, 0)
        layout.addWidget(self.slider_vmax, 1, alignment=Qt.AlignHCenter)

        self.slider_vmin = qtw.QSlider(Qt.Vertical)
        self.slider_vmin.setRange(0, 0)
        layout.addWidget(self.slider_vmin, 1, alignment=Qt.AlignHCenter)

        self.label_vmin = qtw.QLabel('min: -')
        self.label_vmin.setAlignment(Qt.AlignHCenter)
        self.label_vmin.setStyleSheet('font-size: 8pt;')
        layout.addWidget(self.label_vmin)

        self.button_reset = qtw.QPushButton('Reset')
        self.button_reset.setStyleSheet('font-size: 8pt;')
        self.button_reset.setToolTip('Reset both thresholds to the full data range (no clipping)')
        layout.addWidget(self.button_reset)
        self.button_reset.clicked.connect(self.reset)

        self.slider_vmin.valueChanged.connect(self._on_vmin_changed)
        self.slider_vmax.valueChanged.connect(self._on_vmax_changed)

    def set_range(self, vmin, vmax, reset=True):
        """(Re)anchor both sliders to a freshly-loaded image's raw
        intensity range - the previous image's thresholds would otherwise
        be meaningless (wrong scale, maybe outside the new range) on a
        different image. Resets to "no clipping" (full range) unless
        reset=False (e.g. a same-image redraw where the user's current
        thresholds should be kept)."""
        vmin, vmax = int(vmin), int(vmax)
        if vmax <= vmin:
            vmax = vmin + 1
        self.slider_vmin.blockSignals(True)
        self.slider_vmax.blockSignals(True)
        self.slider_vmin.setRange(vmin, vmax)
        self.slider_vmax.setRange(vmin, vmax)
        if reset:
            self.slider_vmin.setValue(vmin)
            self.slider_vmax.setValue(vmax)
        self.slider_vmin.blockSignals(False)
        self.slider_vmax.blockSignals(False)
        self._update_labels()

    def reset(self):
        """Reset both thresholds to the current full data range (i.e.
        disable clipping). Also the Reset button's slot."""
        self.slider_vmin.blockSignals(True)
        self.slider_vmax.blockSignals(True)
        self.slider_vmin.setValue(self.slider_vmin.minimum())
        self.slider_vmax.setValue(self.slider_vmax.maximum())
        self.slider_vmin.blockSignals(False)
        self.slider_vmax.blockSignals(False)
        self._update_labels()
        self.valueChanged.emit()

    def values(self):
        """Current (vmin, vmax), guaranteed vmin < vmax."""
        vmin, vmax = self.slider_vmin.value(), self.slider_vmax.value()
        if vmin >= vmax:
            vmin = vmax - 1
        return vmin, vmax

    def _on_vmin_changed(self, value):
        # Mirrors the crossing-guard tab_roi_4d.py's original DP sliders
        # used, now shared here: keep vmin strictly below vmax by nudging
        # vmin back down if a drag pushes it past vmax.
        if value >= self.slider_vmax.value():
            self.slider_vmin.blockSignals(True)
            self.slider_vmin.setValue(max(self.slider_vmin.minimum(), self.slider_vmax.value() - 1))
            self.slider_vmin.blockSignals(False)
        self._update_labels()
        self.valueChanged.emit()

    def _on_vmax_changed(self, value):
        if value <= self.slider_vmin.value():
            self.slider_vmax.blockSignals(True)
            self.slider_vmax.setValue(min(self.slider_vmax.maximum(), self.slider_vmin.value() + 1))
            self.slider_vmax.blockSignals(False)
        self._update_labels()
        self.valueChanged.emit()

    def _update_labels(self):
        self.label_vmin.setText(f'min: {self.slider_vmin.value():.0f}')
        self.label_vmax.setText(f'max: {self.slider_vmax.value():.0f}')
