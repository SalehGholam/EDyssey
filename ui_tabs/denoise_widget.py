# -*- coding: utf-8 -*-
"""Shared "Denoise" widget: a method combo (EDyssey.io_utils.denoise.
DENOISE_METHODS) plus that method's one tunable parameter, and a "Check
Methods..." button that compares every method side by side on demand.

Used two ways:
- Nested inside ContrastScalingBox's "Adjust Contrast" box (ROI Tracker,
  SAM2 Tracker) - denoising applied on top of that box's own contrast
  stretch.
- Standalone, above the file list (ROI on 4D) - no contrast-stretch step
  there, so denoising runs directly on whatever 8-bit baseline the caller
  supplies to open_check_methods_dialog/apply.
"""
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import pyqtSignal, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import EDyssey.io_utils as io


def apply_denoise_to_array(img_8bit, method, param):
    """`io.denoise_image`, wrapped for a uint8 (or similar integer) image:
    normalizes to [0, 1] first (every denoise_image method's own default
    parameter assumes that range - see EDyssey.io_utils.denoise), then
    rescales the result back to the input's own dtype/range. A no-op for
    method == 'None'. A plain function (not a DenoiseBox method) so a
    background-worker job (e.g. ContrastScalingBox.rescale_async) can call
    it without touching any Qt widget from a non-GUI thread - the caller
    snapshots `method`/`param` from the widget on the GUI thread first."""
    if method == 'None':
        return img_8bit
    img_float = img_8bit.astype(np.float64) / 255.0
    out = io.denoise_image(img_float, method, param)
    return np.clip(out * 255.0, 0, 255).astype(img_8bit.dtype)


class DenoiseBox(qtw.QGroupBox):
    """A 'Denoise' QGroupBox: method combo + that method's single tunable
    parameter (hidden entirely for methods with none - 'None', 'Wavelet'),
    reconfigured (label/range/step/decimals/default - see
    io.DENOISE_PARAM_SPECS) each time the method changes, plus "Check
    Methods..." and (optional) "Apply to All Images" buttons.

    `settingsChanged` fires on every method/parameter change - deliberately
    scoped to mean "the live single-image preview should update", not "go
    recompute a whole stack": some methods (Non-Local Means, Wavelet, ...)
    are too slow to re-run on every frame for every parameter tweak, so a
    caller managing a multi-frame stack should react to this by refreshing
    only the one frame currently on screen, and leave the rest alone until
    the user explicitly asks via `applyAllRequested` (the "Apply to All
    Images" button - hidden if the caller passes `show_apply_all=False`,
    e.g. for a tab with no such stack to begin with).

    "Check Methods..." emits `checkMethodsRequested` rather than doing
    anything itself - this widget has no notion of which raw frame is
    "selected" in the caller's own UI, so the caller connects that signal
    to its own slot, which fetches an appropriate 8-bit baseline image and
    calls back into open_check_methods_dialog(base_img_8bit).
    """
    settingsChanged = pyqtSignal()
    checkMethodsRequested = pyqtSignal()
    applyAllRequested = pyqtSignal()

    def __init__(self, parent=None, title='Denoise', show_apply_all=True):
        super().__init__(title, parent)
        layout_box = qtw.QVBoxLayout()
        self.setLayout(layout_box)

        row1 = qtw.QHBoxLayout()
        layout_box.addLayout(row1)
        label_method = qtw.QLabel('Method')
        row1.addWidget(label_method)
        self.combo_method = qtw.QComboBox()
        self.combo_method.addItems(io.DENOISE_METHODS)
        self.combo_method.setToolTip(
            'Conventional denoising method - '
            "'None' (default) leaves the image untouched")
        row1.addWidget(self.combo_method)
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)

        row2 = qtw.QHBoxLayout()
        layout_box.addLayout(row2)
        self.label_param = qtw.QLabel()
        row2.addWidget(self.label_param)
        self.spinbox_param = qtw.QDoubleSpinBox()
        self.spinbox_param.setFixedWidth(70)
        row2.addWidget(self.spinbox_param)
        # valueChanged (not editingFinished) - every nudge (typed, spin
        # arrows, or scroll) should update the live single-frame preview
        # immediately; unlike a whole-stack recompute, one frame is cheap
        # enough that there's no need to wait for the field to lose focus.
        self.spinbox_param.valueChanged.connect(lambda *_: self.settingsChanged.emit())
        row2.addStretch(1)

        # Own row (not sharing row2 with the param spinbox) - Check
        # Methods.../Apply to All Images are both "act on the whole stack /
        # every method" actions, unlike the spinbox's per-parameter live
        # preview, so keeping them together reads more clearly than mixing
        # them in with the param controls.
        row3 = qtw.QHBoxLayout()
        layout_box.addLayout(row3)
        row3.addStretch(1)
        self.button_checkMethods = qtw.QPushButton('Test Methods')
        self.button_checkMethods.setToolTip(
            'Run every denoising method on the current raw image and '
            'compare them side by side in a separate window')
        self.button_checkMethods.clicked.connect(self.checkMethodsRequested.emit)
        row3.addWidget(self.button_checkMethods)

        if show_apply_all:
            self.button_applyAll = qtw.QPushButton('Apply to All')
            self.button_applyAll.setToolTip(
                'Run the current Denoise method on every frame, not just the '
                'one on screen - can take a while for a slower method on a '
                'long stack, so this only runs on demand rather than on '
                'every parameter change')
            self.button_applyAll.clicked.connect(self.applyAllRequested.emit)
            row3.addWidget(self.button_applyAll)
        else:
            self.button_applyAll = None

        self._update_param_visibility()

    def _on_method_changed(self):
        self._update_param_visibility()
        self.settingsChanged.emit()

    def _update_param_visibility(self):
        spec = io.DENOISE_PARAM_SPECS.get(self.combo_method.currentText())
        self.label_param.setVisible(spec is not None)
        self.spinbox_param.setVisible(spec is not None)
        if spec is None:
            return
        self.label_param.setText(spec['label'])
        self.spinbox_param.blockSignals(True)
        self.spinbox_param.setDecimals(spec['decimals'])
        self.spinbox_param.setRange(spec['min'], spec['max'])
        self.spinbox_param.setSingleStep(spec['step'])
        self.spinbox_param.setValue(spec['default'])
        self.spinbox_param.blockSignals(False)

    def set_busy(self, is_busy):
        """Toggle "Apply to All Images" between its normal state and a
        disabled "Applying..." state while a caller's own full-stack
        rescale is in flight - lets the user tell "still working" apart
        from "silently did nothing" while a slow method churns through a
        long stack. No-op if this box was built with show_apply_all=False."""
        if self.button_applyAll is None:
            return
        self.button_applyAll.setEnabled(not is_busy)
        self.button_applyAll.setText('Applying...' if is_busy else 'Apply to All Images')

    def get_method(self):
        return self.combo_method.currentText()

    def get_param(self):
        """The current method's adjustable parameter value, or None for a
        method with none ('None', 'Wavelet' - see io.DENOISE_PARAM_SPECS)."""
        if io.DENOISE_PARAM_SPECS.get(self.get_method()) is None:
            return None
        return self.spinbox_param.value()

    def apply(self, img_8bit):
        """The current method applied to a single uint8 2-D image - a
        no-op when the method is 'None'."""
        return apply_denoise_to_array(img_8bit, self.get_method(), self.get_param())

    def get_state(self):
        return {'method': self.get_method(), 'param': self.spinbox_param.value()}

    def set_state(self, state):
        """Restore a dict from get_state(). No-op on None/empty. Old states
        (saved before Denoise existed) simply lack these keys - default to
        'None'/untouched rather than raising a KeyError. Deliberately does
        NOT emit settingsChanged - a caller restoring a whole duplicated
        tab's state applies the already-processed images itself."""
        if not state:
            return
        idx = self.combo_method.findText(state.get('method', 'None'))
        if idx >= 0:
            self.combo_method.blockSignals(True)
            self.combo_method.setCurrentIndex(idx)
            self.combo_method.blockSignals(False)
        self._update_param_visibility()
        if 'param' in state and io.DENOISE_PARAM_SPECS.get(self.get_method()) is not None:
            self.spinbox_param.setValue(state['param'])

    def open_check_methods_dialog(self, base_img_8bit, parent=None):
        """Run every denoising method (io.DENOISE_METHODS, skipping 'None')
        on `base_img_8bit` and show them all next to it in a separate,
        non-modal window - lets the user compare which method actually
        works best on their own data before committing to one, and - via a
        parameter spinbox per method in the left panel - retune any one of
        them right there and see just that subplot update, without closing
        and reopening this dialog. Seeded with each method's own default
        parameter, except the currently-selected method (if any), which
        starts from whatever is presently dialed into this box's own
        spinbox_param - so opening this dialog while mid-tuning picks up
        where that tuning left off.

        Args:
            base_img_8bit: 2-D uint8 (or similar) array - the caller's own
                "currently selected" raw frame, already brought to a
                denoise_image-appropriate 8-bit-ish baseline (e.g. via
                ContrastScalingBox.get_kwargs()'s contrast stretch, or a
                plain default percentile stretch for a tab with no
                separate contrast box - see io.convert_img_to_8bit).
            parent: Optional parent widget for the dialog - also keeps it
                alive via Qt's own parent-child ownership; pass the calling
                tab (e.g. `self`) rather than leaving this None.

        Returns:
            The QDialog, already shown (non-modal) - the caller should keep
            a reference to it (e.g. `self._check_methods_dlg = ...`) so it
            isn't garbage-collected out from under the still-open window.
        """
        current_method = self.get_method()
        methods = [m for m in io.DENOISE_METHODS if m != 'None']

        dlg = qtw.QDialog(parent)
        dlg.setWindowTitle('Denoising Comparison')
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.resize(1400, 650)
        dlg_layout = qtw.QVBoxLayout(dlg)
        content_layout = qtw.QHBoxLayout()
        dlg_layout.addLayout(content_layout, 1)

        n_images = 1 + len(methods)
        n_cols = -(-n_images // 2)  # ceil division - 2 rows total
        figure = Figure(constrained_layout=True, figsize=(3.2 * n_cols, 6.4))
        canvas = FigureCanvas(figure)
        axes = figure.subplots(2, n_cols, sharex=True, sharey=True).ravel()

        axes[0].imshow(base_img_8bit, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Raw')

        # Left panel: one groupbox per method that actually has a tunable
        # parameter (skips 'Wavelet', which has none - see
        # io.DENOISE_PARAM_SPECS) - each spinbox recomputes and redraws just
        # its own subplot on change, live, so comparing and retuning don't
        # require closing/reopening this dialog.
        panel_content = qtw.QWidget()
        panel_layout = qtw.QVBoxLayout(panel_content)
        panel = qtw.QScrollArea()
        panel.setWidgetResizable(True)
        panel.setFixedWidth(190)
        panel.setWidget(panel_content)
        content_layout.addWidget(panel)
        content_layout.addWidget(canvas, 1)

        for ax, method in zip(axes[1:], methods):
            spec = io.DENOISE_PARAM_SPECS.get(method)
            param = (self.spinbox_param.value() if method == current_method and spec is not None
                    else (spec['default'] if spec is not None else None))
            result = apply_denoise_to_array(base_img_8bit, method, param)
            im = ax.imshow(result, cmap='gray', vmin=0, vmax=255)
            ax.set_title(method)

            if spec is not None:
                box_m = qtw.QGroupBox(method)
                layout_m = qtw.QVBoxLayout(box_m)
                layout_m.addWidget(qtw.QLabel(spec['label']))
                spinbox_m = qtw.QDoubleSpinBox()
                spinbox_m.setDecimals(spec['decimals'])
                spinbox_m.setRange(spec['min'], spec['max'])
                spinbox_m.setSingleStep(spec['step'])
                spinbox_m.setValue(param)
                layout_m.addWidget(spinbox_m)
                panel_layout.addWidget(box_m)

                def _on_param_changed(value, method=method, im=im):
                    im.set_data(apply_denoise_to_array(base_img_8bit, method, value))
                    canvas.draw_idle()

                spinbox_m.valueChanged.connect(_on_param_changed)
        panel_layout.addStretch(1)

        for ax in axes[n_images:]:
            ax.set_visible(False)
        for ax in axes[:n_images]:
            ax.set_xticks([])
            ax.set_yticks([])

        button_close = qtw.QPushButton('Close')
        button_close.clicked.connect(dlg.close)
        dlg_layout.addWidget(button_close, alignment=Qt.AlignRight)
        dlg.show()
        return dlg
