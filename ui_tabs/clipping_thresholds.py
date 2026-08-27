# -*- coding: utf-8 -*-
"""Shared "Clipping Thresholds" widget: a single vertical dual-handle range
slider (plus two small spinboxes for direct value entry) meant to sit
directly beside a matplotlib image axis (a navigation image or a
diffraction pattern) rather than in the ribbon/parameter panel - so the
control is physically next to the image it clips.

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
from PyQt5.QtCore import Qt, QPointF, QSize, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor


class RangeSlider(qtw.QWidget):
    """A vertical slider with two draggable handles (low/high) sharing one
    track - "one slider with 2 adjustment buttons", replacing what used to
    be two separate QSliders (one for vmin, one for vmax). Values are
    integers; oriented bottom-to-top (minimum at the bottom, maximum at
    top), matching QSlider(Qt.Vertical)'s default orientation."""
    valueChanged = pyqtSignal()

    _HANDLE_R = 6
    _TRACK_W = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 1
        self._low = 0
        self._high = 1
        self._drag = None  # 'low', 'high', or None
        self.setMinimumWidth(20)
        self.setMinimumHeight(60)

    def sizeHint(self):
        return QSize(24, 120)

    def setRange(self, minimum, maximum):
        if maximum <= minimum:
            maximum = minimum + 1
        self._minimum = minimum
        self._maximum = maximum
        self._low = min(max(self._low, minimum), maximum)
        self._high = min(max(self._high, minimum), maximum)
        if self._low > self._high:
            self._low = self._high
        self.update()

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def low(self):
        return self._low

    def high(self):
        return self._high

    def setLow(self, value, emit=True):
        value = int(min(max(value, self._minimum), self._high))
        if value != self._low:
            self._low = value
            self.update()
            if emit:
                self.valueChanged.emit()

    def setHigh(self, value, emit=True):
        value = int(max(min(value, self._maximum), self._low))
        if value != self._high:
            self._high = value
            self.update()
            if emit:
                self.valueChanged.emit()

    def _usable_height(self):
        return max(self.height() - 2 * self._HANDLE_R, 1)

    def _value_to_y(self, value):
        span = self._maximum - self._minimum
        frac = (value - self._minimum) / span if span else 0
        return self.height() - self._HANDLE_R - frac * self._usable_height()

    def _y_to_value(self, y):
        frac = (self.height() - self._HANDLE_R - y) / self._usable_height()
        frac = min(max(frac, 0.0), 1.0)
        return self._minimum + frac * (self._maximum - self._minimum)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx = self.width() / 2
        y_low = self._value_to_y(self._low)
        y_high = self._value_to_y(self._high)

        pen_track = QPen(QColor('#555555'), self._TRACK_W)
        pen_track.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_track)
        painter.drawLine(QPointF(cx, self._HANDLE_R), QPointF(cx, self.height() - self._HANDLE_R))

        pen_selected = QPen(QColor('#5aa1e6'), self._TRACK_W)
        pen_selected.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_selected)
        painter.drawLine(QPointF(cx, y_high), QPointF(cx, y_low))

        painter.setPen(QPen(QColor('#222222'), 1))
        painter.setBrush(QColor('#f0f0f0'))
        painter.drawEllipse(QPointF(cx, y_low), self._HANDLE_R, self._HANDLE_R)
        painter.drawEllipse(QPointF(cx, y_high), self._HANDLE_R, self._HANDLE_R)
        painter.end()

    def mousePressEvent(self, event):
        y = event.pos().y()
        # Grab whichever handle is closer to the click - lets either one be
        # picked up without needing to land exactly on it.
        self._drag = ('low' if abs(y - self._value_to_y(self._low))
                      <= abs(y - self._value_to_y(self._high)) else 'high')
        self._apply_drag(y)

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            self._apply_drag(event.pos().y())

    def mouseReleaseEvent(self, event):
        self._drag = None

    def _apply_drag(self, y):
        value = self._y_to_value(y)
        if self._drag == 'low':
            self.setLow(value)
        else:
            self.setHigh(value)


class ClippingThresholdsWidget(qtw.QWidget):
    """One RangeSlider ("Clipping Thresholds") plus two small spinboxes for
    direct numeric entry, for one image axis - a title on top, the vmax
    spinbox, the dual-handle slider, the vmin spinbox, then a small Reset
    button."""
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

        self.spinbox_vmax = qtw.QSpinBox()
        self.spinbox_vmax.setRange(0, 0)
        self.spinbox_vmax.setSingleStep(1)  # rescaled in set_range() to ~8% of the data max
        self.spinbox_vmax.setAlignment(Qt.AlignCenter)
        self.spinbox_vmax.setButtonSymbols(qtw.QAbstractSpinBox.NoButtons)
        self.spinbox_vmax.setStyleSheet('font-size: 8pt;')
        self.spinbox_vmax.setToolTip('Upper threshold')
        layout.addWidget(self.spinbox_vmax)

        self.range_slider = RangeSlider()
        layout.addWidget(self.range_slider, 1, alignment=Qt.AlignHCenter)

        self.spinbox_vmin = qtw.QSpinBox()
        self.spinbox_vmin.setRange(0, 0)
        self.spinbox_vmin.setSingleStep(1)
        self.spinbox_vmin.setAlignment(Qt.AlignCenter)
        self.spinbox_vmin.setButtonSymbols(qtw.QAbstractSpinBox.NoButtons)
        self.spinbox_vmin.setStyleSheet('font-size: 8pt;')
        self.spinbox_vmin.setToolTip('Lower threshold')
        layout.addWidget(self.spinbox_vmin)

        self.button_reset = qtw.QPushButton('Reset')
        self.button_reset.setStyleSheet('font-size: 8pt;')
        self.button_reset.setToolTip('Reset both thresholds to the full data range (no clipping)')
        layout.addWidget(self.button_reset)
        self.button_reset.clicked.connect(self.reset)

        self.range_slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox_vmin.valueChanged.connect(self._on_spinbox_vmin_changed)
        self.spinbox_vmax.valueChanged.connect(self._on_spinbox_vmax_changed)

    def set_range(self, vmin, vmax, reset=True):
        """(Re)anchor the slider/spinboxes to a freshly-loaded image's raw
        intensity range - the previous image's thresholds would otherwise
        be meaningless (wrong scale, maybe outside the new range) on a
        different image. The slider's floor is 0, unless the data actually
        has negative values, in which case its true (negative) minimum is
        used instead - either way the floor is always <= the real data
        minimum, so it never clips away real data on its own. Resets to
        "no clipping" (floor..vmax) unless reset=False (e.g. a same-image
        redraw where the user's current thresholds should be kept).

        spinbox_vmax's step (wheel-scroll/arrow-key increment) is rescaled
        to ~8% of this new vmax each time - a step of 1 (spinbox_vmin's,
        unchanged) is next to useless for scrubbing through a range that
        can run into the thousands/millions."""
        vmin, vmax = int(vmin), int(vmax)
        floor = 0 if vmin >= 0 else vmin
        if vmax <= floor:
            vmax = floor + 1
        self.range_slider.setRange(floor, vmax)
        if reset:
            self.range_slider.setLow(floor, emit=False)
            self.range_slider.setHigh(vmax, emit=False)
        self.spinbox_vmin.blockSignals(True)
        self.spinbox_vmax.blockSignals(True)
        self.spinbox_vmin.setRange(floor, vmax)
        self.spinbox_vmax.setRange(floor, vmax)
        self.spinbox_vmax.setSingleStep(max(1, round(vmax * 0.08)))
        self.spinbox_vmin.setValue(self.range_slider.low())
        self.spinbox_vmax.setValue(self.range_slider.high())
        self.spinbox_vmin.blockSignals(False)
        self.spinbox_vmax.blockSignals(False)
        self.range_slider.update()

    def set_low(self, value):
        """Set just the lower threshold - e.g. Tab_ROI_on_4D/Navigator's
        "start DP clipping at 1, not the data's true minimum" convention."""
        self.range_slider.setLow(value, emit=False)
        self.spinbox_vmin.blockSignals(True)
        self.spinbox_vmin.setValue(self.range_slider.low())
        self.spinbox_vmin.blockSignals(False)
        self.valueChanged.emit()

    def reset(self):
        """Reset both thresholds to the current full slider range (i.e.
        disable clipping). Also the Reset button's slot."""
        self.range_slider.setLow(self.range_slider.minimum(), emit=False)
        self.range_slider.setHigh(self.range_slider.maximum(), emit=False)
        self._sync_spinboxes_from_slider()
        self.valueChanged.emit()

    def values(self):
        """Current (vmin, vmax), guaranteed vmin < vmax."""
        vmin, vmax = self.range_slider.low(), self.range_slider.high()
        if vmin >= vmax:
            vmin = vmax - 1
        return vmin, vmax

    def _sync_spinboxes_from_slider(self):
        self.spinbox_vmin.blockSignals(True)
        self.spinbox_vmax.blockSignals(True)
        self.spinbox_vmin.setValue(self.range_slider.low())
        self.spinbox_vmax.setValue(self.range_slider.high())
        self.spinbox_vmin.blockSignals(False)
        self.spinbox_vmax.blockSignals(False)

    def _on_slider_changed(self):
        self._sync_spinboxes_from_slider()
        self.valueChanged.emit()

    def _on_spinbox_vmin_changed(self, value):
        self.range_slider.setLow(value, emit=False)
        self.spinbox_vmin.blockSignals(True)
        self.spinbox_vmin.setValue(self.range_slider.low())
        self.spinbox_vmin.blockSignals(False)
        self.valueChanged.emit()

    def _on_spinbox_vmax_changed(self, value):
        self.range_slider.setHigh(value, emit=False)
        self.spinbox_vmax.blockSignals(True)
        self.spinbox_vmax.setValue(self.range_slider.high())
        self.spinbox_vmax.blockSignals(False)
        self.valueChanged.emit()
