# -*- coding: utf-8 -*-
"""Thin horizontal strip meant to sit directly under a tab's own frame
slider (slider_imgNo), marking flagged frames - see Tab_Tracking_CV2/
Tab_SAM2's own _compute_tracking_quality/EDyssey.io_utils.contrast.
flag_anomalous_mask_areas - as tick marks along the slider's own width.

A lightweight, passive "here's what might be worth a look" indicator for
whichever object is currently selected - distinct from mask_edit_dialog.
py's own _SegmentBar (which shows per-frame-range Dilate/Erode/Mesh
settings for ONE tracked object being fine-tuned in that dialog, not
tracking-quality flags for whichever object is selected in a main tab).
"""
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen
from .app_theme import AppTheme


class FrameFlagBar(qtw.QWidget):
    """set_range(n_frames) once a signal loads/changes length; set_flags
    (flagged_frame_indices) whenever the selected object (or its flags)
    changes - None/empty clears the bar back to a plain, unmarked strip.
    Click a tick to jump there - emits frameClicked(int), the same
    signal/argument shape as _SegmentBar's own, so a caller wires it up
    identically (e.g. `bar.frameClicked.connect(self.slider_imgNo.setValue)`)."""
    frameClicked = pyqtSignal(int)

    # Warm red-orange - reads as "needs a look", not a hard error - kept
    # fixed regardless of theme (unlike the background below), since it's
    # vivid/saturated enough to stay legible on either a dark or light one.
    _FLAG_COLOR = QColor('#e0533a')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(8)
        self.setCursor(Qt.PointingHandCursor)
        self._n_frames = 1
        self._flags = []
        self.setToolTip('No flagged frames')
        # Self-subscribing (see RibbonPanel's identical convention/its own
        # comment on why) - Qt auto-disconnects this once `self` is
        # destroyed.
        AppTheme.instance().changed.connect(self.update)

    def set_range(self, n_frames):
        self._n_frames = max(1, int(n_frames))
        self.update()

    def set_flags(self, flags):
        self._flags = sorted(flags) if flags else []
        if self._flags:
            shown = ', '.join(str(f) for f in self._flags[:20])
            more = f' (+{len(self._flags) - 20} more)' if len(self._flags) > 20 else ''
            self.setToolTip(
                f'{len(self._flags)} possibly mistracked frame(s) - click to jump: {shown}{more}')
        else:
            self.setToolTip('No flagged frames for the selected object')
        self.update()

    def _x_for_frame(self, frame):
        if self._n_frames <= 1:
            return 0
        return int(round(frame / (self._n_frames - 1) * (self.width() - 1)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(AppTheme.instance().color('bar_bg')))
        if self._flags:
            pen = QPen(self._FLAG_COLOR)
            pen.setWidth(2)
            painter.setPen(pen)
            for frame in self._flags:
                x = self._x_for_frame(frame)
                painter.drawLine(x, 0, x, self.height())
        painter.end()

    def mousePressEvent(self, event):
        if self._n_frames <= 1 or self.width() <= 1:
            return
        frac = event.pos().x() / (self.width() - 1)
        frame = int(round(frac * (self._n_frames - 1)))
        self.frameClicked.emit(max(0, min(frame, self._n_frames - 1)))
