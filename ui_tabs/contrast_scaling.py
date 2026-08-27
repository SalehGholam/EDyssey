# -*- coding: utf-8 -*-
"""Shared "Display Contrast" widget: lets the user pick and tune how a
loaded navigation signal is contrast-stretched to 8-bit for display and
downstream processing (tracking/SAM2), instead of every tab hand-rolling
its own copy of the same combo box + parameter spinboxes.

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


class ContrastScalingBox(qtw.QGroupBox):
    """A 'Display Contrast' QGroupBox: a method combo (Percentile/Min-Max/
    Std. Dev.) plus that method's tunable parameter(s), with irrelevant
    parameter fields hidden. Emits `settingsChanged` whenever the method or
    an active parameter changes (parameter changes only fire on
    editingFinished, not per-keystroke, so retuning doesn't trigger a
    recompute on every digit typed).

    Usage (synchronous, single frame - e.g. instant preview of the frame
    currently on screen):
        self.box_contrast = ContrastScalingBox()
        layout.addWidget(self.box_contrast)
        frame_8bit = self.box_contrast.rescale_frame(raw_frame)

    Usage (asynchronous, full stack - e.g. after settingsChanged, to also
    keep tracking/SAM2 input up to date without blocking the GUI):
        self.box_contrast.settingsChanged.connect(self.rescale_nav_signal)
        ...
        def rescale_nav_signal(self):
            self.box_contrast.rescale_async(
                self.s, self.threadpool, self.logger, on_done=self._on_stack_rescaled)

        def _on_stack_rescaled(self, s_8bit):
            self.s_8bit = s_8bit
            ...
    """
    settingsChanged = pyqtSignal()

    def __init__(self, parent=None, title='Display Contrast'):
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

    def rescale_frame(self, raw_frame):
        """Synchronously contrast-stretch a single 2-D frame with the
        current settings - cheap, for instant visual feedback while the
        (potentially much slower) full-stack rescale runs in the background
        via rescale_async."""
        return io.convert_img_to_8bit(raw_frame, **self.get_kwargs())

    def rescale_async(self, raw_signal, threadpool, logger=None, on_done=None, label='navigation signal'):
        """Contrast-stretch the full `raw_signal` (a HyperSpy Signal2D) in a
        background QThreadPool worker, current settings. Calling this again
        before a previous call has finished supersedes it - `on_done` from
        the stale call is simply never invoked, so results can't arrive out
        of order and worker threads can't pile up from rapid retuning.

        Args:
            raw_signal: HyperSpy Signal2D holding the untouched raw data.
            threadpool: QThreadPool to run the worker on.
            logger: Optional logger for start/finish messages.
            on_done: Callable(s_8bit) invoked on the GUI thread once this
                (still-current) call completes.
            label: What's being rescaled, for the log message.
        """
        self._job_id += 1
        job_id = self._job_id
        kwargs = self.get_kwargs()
        if logger is not None:
            logger.info('Rescaling %s contrast (%s)...', label, self.describe())

        def _job():
            return io.convert_to_8bit(raw_signal, **kwargs)

        def _on_result(result, index):
            if job_id != self._job_id:
                return  # superseded by a newer rescale_async() call - discard
            if on_done is not None:
                on_done(result)
            if logger is not None:
                logger.info('Contrast rescale applied to the full %s.', label)

        worker = WorkerThread_General(_job, 0)
        worker.signals.results.connect(_on_result)
        threadpool.start(worker)
