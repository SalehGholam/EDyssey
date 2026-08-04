# -*- coding: utf-8 -*-
"""Frame-by-frame manual fine-tuning of a tracked object's per-frame mask
stack - grow/shrink the mask directionally (one pixel-row/column at a time,
via buttons) and optionally apply the same Edge Detection post-processing
the main tab uses, one frame at a time. Shared by Tab_SAM2 and
Tab_Tracking_CV2 (see their own open_fine_tune_mask_dialog())."""
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import py4DTomo.io_utils as io

_MASK_COLOR = np.array([1.0, 0.55, 0.0, 0.45])  # translucent orange overlay
_DIRECTIONS = [('Top', 270), ('Bottom', 90), ('Left', 180), ('Right', 0)]


class MaskEditDialog(qtw.QDialog):
    """Modal editor for a single tracked object's (N, H, W) boolean mask
    stack. `exec_()` returns QDialog.Accepted once the user confirms;
    `get_mask_stack()` then returns the edited (N, H, W) array to write back
    into the caller's own dataframe - editing happens on a private copy, so
    Cancel leaves the original stack untouched."""

    def __init__(self, parent, mask_stack, bg_stack=None, start_frame=0, logger=None):
        super().__init__(parent)
        self.setWindowTitle('Fine-Tune Mask')
        self.resize(720, 700)
        self.logger = logger
        self._original_stack = np.asarray(mask_stack).astype(bool)
        self.mask_stack = self._original_stack.copy()
        self.bg_stack = bg_stack
        self.n_frames = self.mask_stack.shape[0]
        self.frame = int(np.clip(start_frame, 0, self.n_frames - 1))

        layout = qtw.QVBoxLayout(self)

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        h, w = self.mask_stack.shape[1:]
        bg0 = self.bg_stack[self.frame] if self.bg_stack is not None else np.zeros((h, w))
        self.img_bg = self.ax.imshow(bg0, cmap='gray')
        self.img_mask = self.ax.imshow(self._mask_rgba(self.mask_stack[self.frame]))

        row_frame = qtw.QHBoxLayout()
        layout.addLayout(row_frame)
        self.label_frame = qtw.QLabel()
        row_frame.addWidget(self.label_frame)
        self.slider_frame = qtw.QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, self.n_frames - 1)
        self.slider_frame.setValue(self.frame)
        self.slider_frame.valueChanged.connect(self._on_frame_changed)
        row_frame.addWidget(self.slider_frame)

        box_directional = qtw.QGroupBox('Grow / Shrink Mask (1 px per click)')
        layout.addWidget(box_directional)
        grid = qtw.QGridLayout()
        box_directional.setLayout(grid)
        for row, (name, angle) in enumerate(_DIRECTIONS):
            grid.addWidget(qtw.QLabel(name), row, 0)
            button_add = qtw.QPushButton('+ Add')
            button_add.setToolTip(f'Add one row/column to the {name.lower()} side')
            button_add.clicked.connect(lambda _, a=angle: self._grow_shrink(a, grow=True))
            grid.addWidget(button_add, row, 1)
            button_remove = qtw.QPushButton('- Remove')
            button_remove.setToolTip(f'Remove one row/column from the {name.lower()} side')
            button_remove.clicked.connect(lambda _, a=angle: self._grow_shrink(a, grow=False))
            grid.addWidget(button_remove, row, 2)

        box_edge = qtw.QGroupBox('Edge Detection')
        layout.addWidget(box_edge)
        layout_edge = qtw.QVBoxLayout()
        box_edge.setLayout(layout_edge)

        row1 = qtw.QHBoxLayout()
        layout_edge.addLayout(row1)
        self.checkbox_edgeOnly = qtw.QCheckBox('Edge Detection')
        row1.addWidget(self.checkbox_edgeOnly)
        row1.addWidget(qtw.QLabel('Kernel'))
        self.spinbox_edgeKernel = qtw.QSpinBox()
        self.spinbox_edgeKernel.setRange(1, 99)
        self.spinbox_edgeKernel.setValue(3)
        row1.addWidget(self.spinbox_edgeKernel)
        self.checkbox_revertMask = qtw.QCheckBox('Revert Mask')
        row1.addWidget(self.checkbox_revertMask)
        row1.addStretch(1)

        row2 = qtw.QHBoxLayout()
        layout_edge.addLayout(row2)
        self.checkbox_edgeDirectional = qtw.QCheckBox('Directional')
        self.checkbox_edgeDirectional.stateChanged.connect(
            lambda: self.spinbox_edgeDirection.setEnabled(self.checkbox_edgeDirectional.isChecked()))
        row2.addWidget(self.checkbox_edgeDirectional)
        row2.addWidget(qtw.QLabel('Angle (°)'))
        self.spinbox_edgeDirection = qtw.QDoubleSpinBox()
        self.spinbox_edgeDirection.setRange(0, 359.9)
        self.spinbox_edgeDirection.setSingleStep(5)
        self.spinbox_edgeDirection.setDisabled(True)
        row2.addWidget(self.spinbox_edgeDirection)
        row2.addStretch(1)

        self.button_applyEdge = qtw.QPushButton('Apply Edge Detection to This Frame')
        self.button_applyEdge.setToolTip(
            'Bakes the edge-detection settings above into this frame\'s mask now '
            '(unlike the main tab, where it\'s only a display/export-time view)')
        self.button_applyEdge.clicked.connect(self._apply_edge_detection)
        layout_edge.addWidget(self.button_applyEdge)

        row_buttons = qtw.QHBoxLayout()
        layout.addLayout(row_buttons)
        self.button_resetFrame = qtw.QPushButton('Reset This Frame')
        self.button_resetFrame.setToolTip('Discard edits made to this frame only')
        self.button_resetFrame.clicked.connect(self._reset_frame)
        row_buttons.addWidget(self.button_resetFrame)
        row_buttons.addStretch(1)
        self.button_ok = qtw.QPushButton('Save && Close')
        self.button_ok.clicked.connect(self.accept)
        row_buttons.addWidget(self.button_ok)
        self.button_cancel_dlg = qtw.QPushButton('Cancel')
        self.button_cancel_dlg.clicked.connect(self.reject)
        row_buttons.addWidget(self.button_cancel_dlg)

        self._update_frame_label()

    def _mask_rgba(self, mask):
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = _MASK_COLOR
        return rgba

    def _update_frame_label(self):
        self.label_frame.setText(f'Frame {self.frame + 1} / {self.n_frames}')

    def _redraw_mask(self):
        self.img_mask.set_data(self._mask_rgba(self.mask_stack[self.frame]))
        self.canvas.draw_idle()

    def _on_frame_changed(self, value):
        self.frame = value
        self._update_frame_label()
        if self.bg_stack is not None:
            self.img_bg.set_data(self.bg_stack[self.frame])
        self._redraw_mask()

    def _grow_shrink(self, angle, grow):
        self.mask_stack[self.frame] = io.shift_mask_edge(
            self.mask_stack[self.frame], angle, grow=grow)
        self._redraw_mask()

    def _apply_edge_detection(self):
        if not self.checkbox_edgeOnly.isChecked():
            return
        direction = (self.spinbox_edgeDirection.value()
                    if self.checkbox_edgeDirectional.isChecked() else None)
        self.mask_stack[self.frame] = io.erode_mask_edge(
            self.mask_stack[self.frame], self.spinbox_edgeKernel.value(),
            direction=direction, revert=self.checkbox_revertMask.isChecked())
        self._redraw_mask()

    def _reset_frame(self):
        self.mask_stack[self.frame] = self._original_stack[self.frame].copy()
        self._redraw_mask()

    def get_mask_stack(self):
        return self.mask_stack
