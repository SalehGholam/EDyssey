# -*- coding: utf-8 -*-
"""Shared "Adjust Contrast" widget: lets the user pick and tune how a loaded
navigation signal is contrast-stretched to 8-bit for display and downstream
processing (tracking/SAM2), instead of every tab hand-rolling its own copy
of the same combo box + parameter spinboxes. Nests a DenoiseBox (see
denoise_widget.py) below its own controls, applied on top of that stretch.

Also centralizes *how* a full signal is rescaled once settings change: a
single frame is rescaled synchronously (cheap, for instant visual feedback),
while the full stack is rescaled in a background QThreadPool worker (not
cheap for a long stack) via `rescale_async`. Each call to `rescale_async`
supersedes any still-running previous one - only the newest call's result
is ever delivered - so rapid retuning can't pile up worker threads.

Pairs with EDyssey.io_utils.io_utils_ui.convert_to_8bit/convert_img_to_8bit
(and their shared CONTRAST_METHODS/_contrast_bounds) - this widget owns the
UI plus this scheduling, but the actual pixel math still lives in
io_utils_ui, not here.
"""
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import pyqtSignal, Qt
import EDyssey.io_utils as io
from .worker_thread import WorkerThread_General
from .denoise_widget import DenoiseBox, apply_denoise_to_array


class ContrastScalingBox(qtw.QGroupBox):
    """An 'Adjust Contrast' QGroupBox: a method combo (Percentile/Min-Max/
    Std. Dev.) plus that method's tunable parameter(s), with irrelevant
    parameter fields hidden - plus a nested DenoiseBox (see
    denoise_widget.py), applied on top of the contrast stretch.

    Two separate change signals, deliberately not merged into one, since a
    contrast change and a denoise change warrant different responses on a
    multi-frame stack:
    - `settingsChanged`: a *contrast* control changed (method/percentiles/
      std-dev/clip sliders). Cheap - fire away, recompute the whole stack.
    - `denoisePreviewChanged`: box_denoise's own method/parameter changed.
      Some denoise methods are too slow to re-run on every frame for every
      tweak, so this means "refresh just the currently-displayed frame",
      NOT "recompute the whole stack" - see `denoiseApplyAllRequested`
      (relayed from box_denoise's "Apply to All Images" button) for the
      explicit, on-demand full-stack version, and rescale_async's own
      `denoise_method`/`denoise_param` snapshot for how that's applied.

    Usage (synchronous, single frame - e.g. instant preview of the frame
    currently on screen):
        self.box_contrast = ContrastScalingBox()
        layout.addWidget(self.box_contrast)
        frame_8bit = self.box_contrast.rescale_frame(raw_frame)

    Usage (asynchronous, full stack - e.g. after settingsChanged, to also
    keep tracking/SAM2 input up to date without blocking the GUI):
        self.box_contrast.settingsChanged.connect(self.rescale_nav_signal)
        self.box_contrast.denoisePreviewChanged.connect(self.refresh_current_frame)
        self.box_contrast.denoiseApplyAllRequested.connect(self.rescale_nav_signal)
        ...
        def rescale_nav_signal(self):
            self.box_contrast.rescale_async(
                self.s, self.threadpool, self.logger, on_done=self._on_stack_rescaled)

        def _on_stack_rescaled(self, s_8bit):
            self.s_8bit = s_8bit
            ...

    "Check Methods..." (box_denoise's own button): re-emitted here as this
    box's own `checkMethodsRequested`, rather than doing anything itself -
    this widget has no way to know which raw frame is "selected" in the
    caller's own UI, so each tab connects that signal to its own slot,
    which fetches the current raw frame and then calls
    open_check_methods_dialog(raw_frame) back on this widget.
    """
    settingsChanged = pyqtSignal()
    denoisePreviewChanged = pyqtSignal()
    denoiseApplyAllRequested = pyqtSignal()
    checkMethodsRequested = pyqtSignal()

    def __init__(self, parent=None, title='Adjust Contrast'):
        super().__init__(title, parent)
        layout_box = qtw.QVBoxLayout()
        self.setLayout(layout_box)

        layout_row1 = qtw.QHBoxLayout()
        layout_box.addLayout(layout_row1)
        label_method = qtw.QLabel('Method')
        label_method.setFixedWidth(55)
        layout_row1.addWidget(label_method)
        self.combo_method = qtw.QComboBox()
        self.combo_method.addItem('Percentile', 'percentile')
        self.combo_method.addItem('Min-Max', 'minmax')
        self.combo_method.addItem('Std. Dev.', 'std')
        self.combo_method.setToolTip(
            'Contrast-stretch to 8-bit:\n'
            '- Percentile: robust to hot/dead pixels\n'
            '- Min-Max: simple, sensitive to outliers (default)\n'
            '- Std. Dev.: mean ± N standard deviations')
        layout_row1.addWidget(self.combo_method)
        self.combo_method.setCurrentIndex(self.combo_method.findData('minmax'))
        self.combo_method.currentIndexChanged.connect(self._on_changed)

        layout_row2 = qtw.QHBoxLayout()
        layout_box.addLayout(layout_row2)

        self.label_low = qtw.QLabel('Low %')
        layout_row2.addWidget(self.label_low)
        self.spinbox_low = qtw.QDoubleSpinBox()
        self.spinbox_low.setRange(0.0, 100.0)
        self.spinbox_low.setValue(1.0)
        self.spinbox_low.setSingleStep(0.5)
        self.spinbox_low.setFixedWidth(60)
        layout_row2.addWidget(self.spinbox_low)
        self.spinbox_low.editingFinished.connect(self._on_changed)

        self.label_high = qtw.QLabel('High %')
        layout_row2.addWidget(self.label_high)
        self.spinbox_high = qtw.QDoubleSpinBox()
        self.spinbox_high.setRange(0.0, 100.0)
        self.spinbox_high.setValue(99.0)
        self.spinbox_high.setSingleStep(0.5)
        self.spinbox_high.setFixedWidth(60)
        layout_row2.addWidget(self.spinbox_high)
        self.spinbox_high.editingFinished.connect(self._on_changed)

        self.label_nstd = qtw.QLabel('N σ')
        layout_row2.addWidget(self.label_nstd)
        self.spinbox_nstd = qtw.QDoubleSpinBox()
        self.spinbox_nstd.setRange(0.1, 10.0)
        self.spinbox_nstd.setValue(3.0)
        self.spinbox_nstd.setSingleStep(0.5)
        self.spinbox_nstd.setFixedWidth(60)
        layout_row2.addWidget(self.spinbox_nstd)
        self.spinbox_nstd.editingFinished.connect(self._on_changed)
        layout_row2.addStretch(1)

        # Clip thresholds: independent of the method above - raw values below/
        # above these are clamped before the method's own stretch is computed
        # (e.g. to knock out a saturated beam stop or a dead-pixel border
        # without changing the percentile/min-max/std parameters). Integer-
        # valued (raw detector counts are integers) and anchored directly to
        # the loaded signal's own raw [min, max] via set_data_range(); until
        # that's called, the sliders are inert placeholders over a dummy 0-1
        # range.
        self._data_min = 0
        self._data_max = 1

        layout_row3 = qtw.QHBoxLayout()
        layout_box.addLayout(layout_row3)
        self.label_clip_low = qtw.QLabel('Clip low: -')
        self.label_clip_low.setFixedWidth(110)
        layout_row3.addWidget(self.label_clip_low)
        self.slider_clip_low = qtw.QSlider(Qt.Horizontal)
        self.slider_clip_low.setRange(self._data_min, self._data_max)
        self.slider_clip_low.setValue(self._data_min)
        self.slider_clip_low.setToolTip(
            'Raw values below this are clamped before contrast-stretching')
        layout_row3.addWidget(self.slider_clip_low)
        self.slider_clip_low.valueChanged.connect(self._on_clip_slider_changed)

        layout_row4 = qtw.QHBoxLayout()
        layout_box.addLayout(layout_row4)
        self.label_clip_high = qtw.QLabel('Clip high: -')
        self.label_clip_high.setFixedWidth(110)
        layout_row4.addWidget(self.label_clip_high)
        self.slider_clip_high = qtw.QSlider(Qt.Horizontal)
        self.slider_clip_high.setRange(self._data_min, self._data_max)
        self.slider_clip_high.setValue(self._data_max)
        self.slider_clip_high.setToolTip(
            'Raw values above this are clamped before contrast-stretching')
        layout_row4.addWidget(self.slider_clip_high)
        self.slider_clip_high.valueChanged.connect(self._on_clip_slider_changed)

        layout_row5 = qtw.QHBoxLayout()
        layout_box.addLayout(layout_row5)
        layout_row5.addStretch(1)
        self.button_reset_clip = qtw.QPushButton('Reset Clip Thresholds')
        # clicked emits a bool ("checked") - a lambda swallows it instead of
        # letting it land in reset_clip_thresholds' `emit` parameter, which
        # would otherwise silently suppress the settingsChanged the reset is
        # supposed to trigger (so the reset never visibly took effect).
        self.button_reset_clip.clicked.connect(lambda: self.reset_clip_thresholds())
        layout_row5.addWidget(self.button_reset_clip)

        self._update_clip_labels()

        #%% denoise - nested DenoiseBox, applied on top of the contrast
        # stretch above (see rescale_frame/rescale_async). 'None' (its own
        # default) is a no-op, so every existing caller of rescale_frame/
        # rescale_async keeps working unchanged until the user opts in.
        self.box_denoise = DenoiseBox()
        layout_box.addWidget(self.box_denoise)
        # NOT relayed into self.settingsChanged - see class docstring, these
        # two get their own distinct signals instead so a caller can treat
        # a denoise-only change (frame-only refresh) differently from a
        # contrast change (full-stack refresh).
        self.box_denoise.settingsChanged.connect(self.denoisePreviewChanged.emit)
        self.box_denoise.applyAllRequested.connect(self.denoiseApplyAllRequested.emit)
        self.box_denoise.checkMethodsRequested.connect(self.checkMethodsRequested.emit)

        self._update_param_visibility()
        self._job_id = 0  # bumped on every rescale_async() call; guards against stale results

    def set_data_range(self, vmin, vmax):
        """(Re)anchor the clip-threshold sliders to a freshly-loaded signal's
        raw intensity range, resetting them to "no clip" - the previous
        dataset's clip values would otherwise be meaningless (wrong scale, or
        even outside the new range) on a new signal. Raw detector counts are
        integers, so the sliders (and the thresholds they produce) are too."""
        self._data_min = int(round(vmin))
        self._data_max = int(round(vmax))
        if self._data_max <= self._data_min:
            self._data_max = self._data_min + 1
        self.slider_clip_low.setRange(self._data_min, self._data_max)
        self.slider_clip_high.setRange(self._data_min, self._data_max)
        self.reset_clip_thresholds(emit=False)

    def reset_clip_thresholds(self, emit=True):
        """Reset both clip thresholds to the current data's full range (i.e.
        disable clipping). Also the "Reset" button's slot."""
        self.slider_clip_low.blockSignals(True)
        self.slider_clip_high.blockSignals(True)
        self.slider_clip_low.setValue(self._data_min)
        self.slider_clip_high.setValue(self._data_max)
        self.slider_clip_low.blockSignals(False)
        self.slider_clip_high.blockSignals(False)
        self._update_clip_labels()
        if emit:
            self.settingsChanged.emit()

    def _update_clip_labels(self):
        self.label_clip_low.setText(f'Clip low: {self.clip_low_value:d}')
        self.label_clip_high.setText(f'Clip high: {self.clip_high_value:d}')

    def _on_clip_slider_changed(self):
        self._update_clip_labels()
        self.settingsChanged.emit()

    @property
    def clip_low_value(self):
        return self.slider_clip_low.value()

    @property
    def clip_high_value(self):
        return self.slider_clip_high.value()

    def _on_changed(self):
        self._update_param_visibility()
        self.settingsChanged.emit()

    def _update_param_visibility(self):
        method = self.combo_method.currentData()
        for wid in (self.label_low, self.spinbox_low, self.label_high, self.spinbox_high):
            wid.setVisible(method == 'percentile')
        for wid in (self.label_nstd, self.spinbox_nstd):
            wid.setVisible(method == 'std')

    def set_denoise_apply_all_busy(self, is_busy):
        """Passthrough to box_denoise.set_busy() - see there."""
        self.box_denoise.set_busy(is_busy)

    def apply_denoise(self, img_8bit):
        """box_denoise's current method applied to a single uint8 2-D image
        (already contrast-stretched, e.g. via rescale_frame) - a no-op when
        the method is 'None'."""
        return self.box_denoise.apply(img_8bit)

    def open_check_methods_dialog(self, raw_img, parent=None):
        """Contrast-stretch `raw_img` with this box's current settings, then
        delegate to box_denoise.open_check_methods_dialog for the actual
        comparison window - see that method for the full behavior/return
        value."""
        img_8bit = io.convert_img_to_8bit(raw_img, **self.get_kwargs())
        return self.box_denoise.open_check_methods_dialog(img_8bit, parent=parent)

    def get_kwargs(self):
        """Current method + its tunable parameter(s), as kwargs ready for
        io.convert_to_8bit/convert_img_to_8bit."""
        method = self.combo_method.currentData()
        if method == 'percentile':
            kwargs = {'method': method, 'plow': self.spinbox_low.value(),
                      'phigh': self.spinbox_high.value()}
        elif method == 'std':
            kwargs = {'method': method, 'n_std': self.spinbox_nstd.value()}
        else:
            kwargs = {'method': method}
        kwargs['clip_low'] = self.clip_low_value
        kwargs['clip_high'] = self.clip_high_value
        return kwargs

    def describe(self):
        """Human-readable one-liner of the current method/parameters, for logging."""
        kwargs = self.get_kwargs()
        method = kwargs['method']
        if method == 'percentile':
            desc = f"percentile (low={kwargs['plow']:.1f}%, high={kwargs['phigh']:.1f}%)"
        elif method == 'std':
            desc = f"std. dev. (±{kwargs['n_std']:.1f}σ)"
        else:
            desc = 'min-max'
        if self.clip_low_value != self._data_min or self.clip_high_value != self._data_max:
            desc += f', clipped to [{kwargs["clip_low"]:d}, {kwargs["clip_high"]:d}]'
        return desc

    def get_state(self):
        """Full widget state (method/parameters + clip thresholds + denoise,
        plus the clip sliders' own underlying data range) - restorable via
        set_state(), e.g. for "Duplicate Current Tab" (see
        EDyssey_MainWindow.duplicate_current_tab)."""
        return {
            'method': self.combo_method.currentData(),
            'low': self.spinbox_low.value(),
            'high': self.spinbox_high.value(),
            'nstd': self.spinbox_nstd.value(),
            'data_min': self._data_min,
            'data_max': self._data_max,
            'clip_low': self.clip_low_value,
            'clip_high': self.clip_high_value,
            'denoise': self.box_denoise.get_state(),
        }

    def set_state(self, state):
        """Restore a dict from get_state(). No-op on None/empty.
        Deliberately does NOT emit settingsChanged - a caller restoring a
        whole duplicated tab's state applies the already-rescaled images
        itself, so an extra background rescale here would just be
        redundant work racing that restore."""
        if not state:
            return
        idx = self.combo_method.findData(state['method'])
        if idx >= 0:
            self.combo_method.blockSignals(True)
            self.combo_method.setCurrentIndex(idx)
            self.combo_method.blockSignals(False)
        self.spinbox_low.setValue(state['low'])
        self.spinbox_high.setValue(state['high'])
        self.spinbox_nstd.setValue(state['nstd'])
        self._data_min = state['data_min']
        self._data_max = state['data_max']
        self.slider_clip_low.blockSignals(True)
        self.slider_clip_high.blockSignals(True)
        self.slider_clip_low.setRange(self._data_min, self._data_max)
        self.slider_clip_high.setRange(self._data_min, self._data_max)
        self.slider_clip_low.setValue(state['clip_low'])
        self.slider_clip_high.setValue(state['clip_high'])
        self.slider_clip_low.blockSignals(False)
        self.slider_clip_high.blockSignals(False)
        self._update_clip_labels()
        self._update_param_visibility()
        # Old states (saved before Denoise existed) simply lack this key.
        self.box_denoise.set_state(state.get('denoise'))

    def rescale_frame(self, raw_frame):
        """Synchronously contrast-stretch (then denoise, if enabled) a
        single 2-D frame with the current settings - cheap, for instant
        visual feedback while the (potentially much slower) full-stack
        rescale runs in the background via rescale_async."""
        img_8bit = io.convert_img_to_8bit(raw_frame, **self.get_kwargs())
        return self.apply_denoise(img_8bit)

    def rescale_async(self, raw_signal, threadpool, logger=None, on_done=None,
                       on_error=None, on_progress=None, label='navigation signal'):
        """Contrast-stretch (then denoise, if enabled) the full `raw_signal`
        (a HyperSpy Signal2D) in a background QThreadPool worker, current
        settings. Calling this again before a previous call has finished
        supersedes it - `on_done`/`on_error`/`on_progress` from the stale
        call is simply never invoked, so results can't arrive out of order
        and worker threads can't pile up from rapid retuning.

        Args:
            raw_signal: HyperSpy Signal2D holding the untouched raw data.
            threadpool: QThreadPool to run the worker on.
            logger: Optional logger for start/finish/failure messages.
            on_done: Callable(s_8bit) invoked on the GUI thread once this
                (still-current) call completes.
            on_error: Optional callable(traceback_text) invoked on the GUI
                thread if this (still-current) call raises - without this,
                a failure (e.g. an incompatible denoise-method parameter)
                would otherwise vanish silently at the QRunnable boundary,
                leaving the caller's own "is a full-stack rescale still
                pending" bookkeeping (e.g. a dirty flag) stuck as if the
                call had never happened.
            on_progress: Optional callable(current, total) invoked on the
                GUI thread as denoising works through the stack frame by
                frame - only fires when denoising is actually enabled (the
                contrast-only stretch below is vectorized, not a per-frame
                Python loop, so there's nothing incremental to report for
                it). Meant for a caller-owned progress bar (e.g. a tab's
                "Apply to All Images" button) - a slow per-frame method on a
                long stack can otherwise look hung for a while.
            label: What's being rescaled, for the log message.
        """
        self._job_id += 1
        job_id = self._job_id
        kwargs = self.get_kwargs()
        # Snapshot denoise settings here (GUI thread) rather than reading
        # box_denoise's own combo/spinbox from inside _job() (background
        # thread) - same reasoning as `kwargs` above.
        denoise_method = self.box_denoise.get_method()
        denoise_param = self.box_denoise.get_param()
        if logger is not None:
            logger.info('Rescaling %s contrast (%s)...', label, self.describe())

        def _job():
            s_8bit = io.convert_to_8bit(raw_signal, **kwargs)
            if denoise_method != 'None':
                data = s_8bit.data
                if data.ndim >= 3:
                    total = data.shape[0]
                    for i in range(total):
                        data[i] = apply_denoise_to_array(data[i], denoise_method, denoise_param)
                        # `worker` is assigned below, after this closure is
                        # defined, but not until it actually runs - fine,
                        # since Python closures resolve free variables at
                        # call time, and this only ever runs after that.
                        worker.signals.progress.emit(i + 1, total)
                else:
                    s_8bit.data = apply_denoise_to_array(data, denoise_method, denoise_param)
            return s_8bit

        def _on_result(result, index):
            if job_id != self._job_id:
                return  # superseded by a newer rescale_async() call - discard
            if on_done is not None:
                on_done(result)
            if logger is not None:
                logger.info('Contrast rescale applied to the full %s.', label)

        def _on_error(traceback_text, index):
            if job_id != self._job_id:
                return  # superseded by a newer rescale_async() call - discard
            if logger is not None:
                logger.error('Rescaling %s failed:\n%s', label, traceback_text)
            if on_error is not None:
                on_error(traceback_text)

        def _on_progress(current, total):
            if job_id != self._job_id:
                return  # superseded by a newer rescale_async() call - discard
            if on_progress is not None:
                on_progress(current, total)

        worker = WorkerThread_General(_job, 0)
        worker.signals.results.connect(_on_result)
        worker.signals.error.connect(_on_error)
        worker.signals.progress.connect(_on_progress)
        threadpool.start(worker)
