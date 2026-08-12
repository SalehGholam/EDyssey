# -*- coding: utf-8 -*-
"""Frame-by-frame manual fine-tuning of a tracked object's per-frame mask
stack: grow/shrink the mask directionally (D-pad buttons), paint single
pixels or rectangular regions in/out with the mouse, and preview the same
Edge Detection post-processing the main tab uses - live, without baking it
into the stored mask. Shared by Tab_SAM2 and Tab_Tracking_CV2 (see their own
open_fine_tune_mask_dialog())."""
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import py4DTomo.io_utils as io

_MASK_COLOR = np.array([1.0, 0.55, 0.0, 0.45])  # translucent orange overlay
_DIRECTIONS = [('Top', 270), ('Bottom', 90), ('Left', 180), ('Right', 0)]
_TIP_TEXT = (
    'Tip: Ctrl+Scroll to zoom | Ctrl+Left/Right-click (or drag) to add/remove '
    'a pixel | Shift+Left/Right-drag to add/remove a rectangular region')


class MaskEditDialog(qtw.QDialog):
    """Modal editor for a single tracked object's (N, H, W) boolean mask
    stack. `exec_()` returns QDialog.Accepted once the user confirms;
    `get_mask_stack()` then returns the edited (N, H, W) array to write back
    into the caller's own dataframe - editing happens on a private copy, so
    Cancel leaves the original stack untouched. Edge Detection here is a
    live, non-destructive preview only (exactly like the main tab) - it is
    never baked into the returned mask, so toggling it never compounds
    across redraws or fine-tuning sessions.
    """

    def __init__(self, parent, mask_stack, bg_stack=None, start_frame=0, logger=None,
                 default_mask_stack=None, edge_settings=None):
        super().__init__(parent)
        self.setWindowTitle('Fine-Tune Mask')
        self.resize(720, 760)
        self.logger = logger
        self._original_stack = np.asarray(mask_stack).astype(bool)
        self.mask_stack = self._original_stack.copy()
        # The pristine tracking-derived stack (SAM2 output, or a freshly
        # recomputed cv2-tracking threshold mask) - independent of anything
        # edited in this dialog, this session or any previous one. Falls
        # back to this session's opening state if the caller has nothing
        # better to offer (e.g. no tracking result was ever cached).
        self._default_stack = (np.asarray(default_mask_stack).astype(bool)
                               if default_mask_stack is not None else None)
        self.bg_stack = bg_stack
        self.n_frames = self.mask_stack.shape[0]
        self.frame = int(np.clip(start_frame, 0, self.n_frames - 1))

        self._pixel_paint_value = None  # True/False while Ctrl+drag-painting pixels
        self._roi_drag = None  # (x0, y0, value) while Shift+drag-drawing a ROI
        self._roi_rect_artist = None
        self._roi_bg = None

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

        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

        row_frame = qtw.QHBoxLayout()
        layout.addLayout(row_frame)
        self.label_frame = qtw.QLabel()
        row_frame.addWidget(self.label_frame)
        self.slider_frame = qtw.QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, self.n_frames - 1)
        self.slider_frame.setValue(self.frame)
        self.slider_frame.valueChanged.connect(self._on_frame_changed)
        row_frame.addWidget(self.slider_frame)

        #%% D-pad grow/shrink buttons - arranged spatially (top/left/right/
        # bottom of a 3x3 grid) instead of a plain list, arrows pointing away
        # from center = grow, toward center = shrink.
        box_directional = qtw.QGroupBox('Grow / Shrink Mask (1 px per click)')
        layout.addWidget(box_directional)
        grid = qtw.QGridLayout()
        box_directional.setLayout(grid)

        def dpad_button(symbol, tooltip, angle, grow):
            btn = qtw.QPushButton(symbol)
            btn.setFixedSize(34, 28)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda _, a=angle, g=grow: self._grow_shrink(a, grow=g))
            return btn

        row_top = qtw.QHBoxLayout()
        row_top.addWidget(dpad_button('▲', 'Add one row to the top', 270, True))
        row_top.addWidget(dpad_button('▼', 'Remove one row from the top', 270, False))
        grid.addLayout(row_top, 0, 1)

        row_bottom = qtw.QHBoxLayout()
        row_bottom.addWidget(dpad_button('▲', 'Remove one row from the bottom', 90, False))
        row_bottom.addWidget(dpad_button('▼', 'Add one row to the bottom', 90, True))
        grid.addLayout(row_bottom, 2, 1)

        col_left = qtw.QVBoxLayout()
        col_left.addWidget(dpad_button('◀', 'Add one column to the left', 180, True))
        col_left.addWidget(dpad_button('▶', 'Remove one column from the left', 180, False))
        grid.addLayout(col_left, 1, 0)

        col_right = qtw.QVBoxLayout()
        col_right.addWidget(dpad_button('▶', 'Add one column to the right', 0, True))
        col_right.addWidget(dpad_button('◀', 'Remove one column from the right', 0, False))
        grid.addLayout(col_right, 1, 2)

        label_center = qtw.QLabel('Mask')
        label_center.setAlignment(Qt.AlignCenter)
        grid.addWidget(label_center, 1, 1)

        #%% edge detection - live preview only (see class docstring)
        box_edge = qtw.QGroupBox('Edge Detection')
        layout.addWidget(box_edge)
        layout_edge = qtw.QVBoxLayout()
        box_edge.setLayout(layout_edge)

        row1 = qtw.QHBoxLayout()
        layout_edge.addLayout(row1)
        self.checkbox_edgeOnly = qtw.QCheckBox('Edge Detection')
        self.checkbox_edgeOnly.setToolTip(
            'Live preview only, like the main tab - does not modify the mask '
            'returned by this dialog, only how it is shown here')
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
        row2.addWidget(self.checkbox_edgeDirectional)
        row2.addWidget(qtw.QLabel('Angle (°)'))
        self.spinbox_edgeDirection = qtw.QDoubleSpinBox()
        self.spinbox_edgeDirection.setRange(0, 359.9)
        self.spinbox_edgeDirection.setSingleStep(5)
        self.spinbox_edgeDirection.setDisabled(True)
        row2.addWidget(self.spinbox_edgeDirection)
        row2.addStretch(1)
        self.checkbox_edgeDirectional.stateChanged.connect(
            lambda: self.spinbox_edgeDirection.setEnabled(self.checkbox_edgeDirectional.isChecked()))

        if edge_settings:
            self.checkbox_edgeOnly.setChecked(bool(edge_settings.get('enabled', False)))
            self.spinbox_edgeKernel.setValue(int(edge_settings.get('kernel', 3)))
            self.checkbox_revertMask.setChecked(bool(edge_settings.get('revert', False)))
            self.checkbox_edgeDirectional.setChecked(bool(edge_settings.get('directional', False)))
            self.spinbox_edgeDirection.setValue(float(edge_settings.get('direction', 0)))
            self.spinbox_edgeDirection.setEnabled(self.checkbox_edgeDirectional.isChecked())

        # Any change just redraws (non-destructive) - unlike the removed
        # "Apply" button, nothing is ever baked into mask_stack here.
        for signal in (self.checkbox_edgeOnly.stateChanged, self.spinbox_edgeKernel.valueChanged,
                       self.checkbox_revertMask.stateChanged, self.checkbox_edgeDirectional.stateChanged,
                       self.spinbox_edgeDirection.valueChanged):
            signal.connect(lambda *_: self._redraw_mask())

        label_tip = qtw.QLabel(_TIP_TEXT)
        label_tip.setWordWrap(True)
        label_tip.setStyleSheet('color: gray; font-style: italic;')
        layout.addWidget(label_tip)

        row_buttons = qtw.QHBoxLayout()
        layout.addLayout(row_buttons)
        self.button_resetFrame = qtw.QPushButton('Reset This Frame')
        self.button_resetFrame.setToolTip('Discard edits made to this frame only')
        self.button_resetFrame.clicked.connect(self._reset_frame)
        row_buttons.addWidget(self.button_resetFrame)
        self.button_resetTracking = qtw.QPushButton('Reset to Tracking')
        self.button_resetTracking.setToolTip(
            'Discard every edit in this dialog, on every frame, and restore the '
            "object's original tracking-derived mask")
        self.button_resetTracking.clicked.connect(self._reset_to_tracking)
        row_buttons.addWidget(self.button_resetTracking)
        row_buttons.addStretch(1)
        self.button_ok = qtw.QPushButton('Save && Close')
        self.button_ok.clicked.connect(self.accept)
        row_buttons.addWidget(self.button_ok)
        self.button_cancel_dlg = qtw.QPushButton('Cancel')
        self.button_cancel_dlg.clicked.connect(self.reject)
        row_buttons.addWidget(self.button_cancel_dlg)

        self._update_frame_label()
        self._redraw_mask()

    def _mask_rgba(self, mask):
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = _MASK_COLOR
        return rgba

    def _update_frame_label(self):
        self.label_frame.setText(f'Frame {self.frame + 1} / {self.n_frames}')

    def _effective_mask(self, frame):
        """The mask as currently displayed: the editable base, plus the Edge
        Detection preview on top if its checkbox is checked - never written
        back to mask_stack itself."""
        mask = self.mask_stack[frame]
        if self.checkbox_edgeOnly.isChecked():
            direction = (self.spinbox_edgeDirection.value()
                        if self.checkbox_edgeDirectional.isChecked() else None)
            mask = io.erode_mask_edge(mask, self.spinbox_edgeKernel.value(),
                                      direction=direction, revert=self.checkbox_revertMask.isChecked())
        return mask

    def _redraw_mask(self):
        self.img_mask.set_data(self._mask_rgba(self._effective_mask(self.frame)))
        self.canvas.draw_idle()

    def _on_frame_changed(self, value):
        self.frame = value
        self._update_frame_label()
        if self.bg_stack is not None:
            self.img_bg.set_data(self.bg_stack[self.frame])
        self._cancel_drag()
        self._redraw_mask()

    def _grow_shrink(self, angle, grow):
        self.mask_stack[self.frame] = io.shift_mask_edge(
            self.mask_stack[self.frame], angle, grow=grow)
        self._redraw_mask()

    def _reset_frame(self):
        self.mask_stack[self.frame] = self._original_stack[self.frame].copy()
        self._redraw_mask()

    def _reset_to_tracking(self):
        source = self._default_stack if self._default_stack is not None else self._original_stack
        self.mask_stack = source.copy()
        self._redraw_mask()

    def get_mask_stack(self):
        return self.mask_stack

    #%% mouse interaction: Ctrl+Scroll zoom, Ctrl+Click(+drag) pixel paint,
    # Shift+drag rectangular region paint
    def _cancel_drag(self):
        self._pixel_paint_value = None
        self._roi_drag = None
        if self._roi_rect_artist is not None:
            try:
                self._roi_rect_artist.remove()
            except Exception:
                self.logger.debug('ROI drag rectangle already removed.', exc_info=True)
            self._roi_rect_artist = None

    def _on_scroll(self, event):
        """Zoom in/out on Ctrl+scroll wheel, centered on the cursor."""
        if event.inaxes != self.ax or event.xdata is None or 'ctrl' not in event.modifiers:
            return
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])
        self.ax.set_xlim([event.xdata - new_width * (1 - relx), event.xdata + new_width * relx])
        self.ax.set_ylim([event.ydata - new_height * (1 - rely), event.ydata + new_height * rely])
        self.canvas.draw_idle()

    def _paint_pixel(self, event):
        row, col = int(round(event.ydata)), int(round(event.xdata))
        h, w = self.mask_stack.shape[1:]
        if 0 <= row < h and 0 <= col < w:
            self.mask_stack[self.frame, row, col] = self._pixel_paint_value
            self._redraw_mask()

    def _on_press(self, event):
        """Start Ctrl+Click pixel painting (left=add, right=remove) or
        Shift+drag rectangular-region painting."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        mods = event.modifiers
        if 'ctrl' in mods and event.button in (1, 3):
            self._pixel_paint_value = (event.button == 1)
            self._paint_pixel(event)
        elif 'shift' in mods and event.button in (1, 3):
            self._roi_drag = (event.xdata, event.ydata, event.button == 1)
            color = 'lime' if event.button == 1 else 'red'
            self._roi_rect_artist = patches.Rectangle(
                (event.xdata, event.ydata), 0, 0, linewidth=1.5,
                edgecolor=color, facecolor='none')
            self.ax.add_patch(self._roi_rect_artist)
            self.canvas.draw()
            self._roi_bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def _on_motion(self, event):
        """Continue whichever paint mode on_press started - matches the rest
        of the app's convention of only gating the modifier key on press,
        not on every subsequent motion event."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self._pixel_paint_value is not None:
            self._paint_pixel(event)
        elif self._roi_drag is not None:
            x0, y0, _ = self._roi_drag
            try:
                self._roi_rect_artist.set_width(event.xdata - x0)
                self._roi_rect_artist.set_height(event.ydata - y0)
                self._roi_rect_artist.set_xy((x0, y0))
            except AttributeError:
                return
            self.canvas.restore_region(self._roi_bg)
            self.ax.draw_artist(self._roi_rect_artist)
            self.canvas.blit(self.ax.bbox)

    def _on_release(self, event):
        self._pixel_paint_value = None
        if self._roi_drag is None:
            return
        x0, y0, value = self._roi_drag
        self._roi_drag = None
        if self._roi_rect_artist is not None:
            try:
                self._roi_rect_artist.remove()
            except Exception:
                self.logger.debug('ROI drag rectangle already removed.', exc_info=True)
            self._roi_rect_artist = None
        if event.xdata is not None and event.ydata is not None:
            col0, col1 = sorted((int(round(x0)), int(round(event.xdata))))
            row0, row1 = sorted((int(round(y0)), int(round(event.ydata))))
            h, w = self.mask_stack.shape[1:]
            row0, row1 = max(row0, 0), min(row1 + 1, h)
            col0, col1 = max(col0, 0), min(col1 + 1, w)
            if row1 > row0 and col1 > col0:
                self.mask_stack[self.frame, row0:row1, col0:col1] = value
        self._redraw_mask()
