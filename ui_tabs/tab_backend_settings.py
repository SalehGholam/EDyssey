# -*- coding: utf-8 -*-
"""Analysis Backend & Declustering dialog: choose which .tpx3 analysis
engine to use (old eventem / new, bug-fixed eventem / pyeventem) and
configure declustering - opened from the main window's "Edit" menu, not a
tab of its own. Non-modal, Apply-gated (these settings drive expensive
re-decodes, so no live-apply, unlike the Display Preferences dialog's
theme/colormap controls) - see tab_edit.py's EditSettingsDialog, whose shape
this mirrors closely.

See ui_tabs/analysis_backend_settings.py for the persisted state this
writes to, and EDyssey/io_utils/eventem_backend.py for what actually
consumes it.
"""
import os
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
from .analysis_backend_settings import AnalysisBackendSettings, SINK_KEYS, SINK_LABELS
from .app_theme import AppTheme
from EDyssey.io_utils import eventem_backend as eb

_CPU_COUNT = os.cpu_count() or 1


class BackendSettingsDialog(qtw.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Analysis Backend & Declustering')
        self.setWindowFlags(Qt.Window)
        self.init_widget()

    def init_widget(self):
        self.layout = qtw.QVBoxLayout(self)
        self.setLayout(self.layout)

        intro = qtw.QLabel(
            'Choose which engine analyzes .tpx3 files, and whether to decluster raw pixel '
            'activations into physical electron counts before Pacbed/Roi/vSTEM/Var see the '
            'data. Changes only take effect once you click "Apply", and are then remembered '
            'the next time EDyssey opens.')
        intro.setWordWrap(True)
        intro.setFixedWidth(440)
        intro.setStyleSheet(f"color: {AppTheme.instance().color('fg_dim')}; padding: 6px;")
        self.layout.addWidget(intro)

        settings = AnalysisBackendSettings.instance()

        # --- Analysis Backend -------------------------------------------------
        backend_box = qtw.QGroupBox('Analysis Backend')
        backend_layout = qtw.QVBoxLayout(backend_box)
        self._backend_group = qtw.QButtonGroup(self)
        self._backend_buttons = {}
        for key in eb.BACKENDS:
            radio = qtw.QRadioButton(eb.BACKEND_LABELS[key])
            if key == settings.backend:
                radio.setChecked(True)
            self._backend_group.addButton(radio)
            self._backend_buttons[key] = radio
            backend_layout.addWidget(radio)
        note = qtw.QLabel(
            "'Old eventem' has no declustering support - the Declustering section below is "
            "ignored while it's selected.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {AppTheme.instance().color('fg_dim')}; font-style: italic;")
        backend_layout.addWidget(note)

        new_eventem_note = qtw.QLabel(
            "'New eventem' runs in a separate background process for anything computed "
            "directly in the app (Sum DP, Test navigation image, Extract Current Frame, ...) - "
            "a one-time workaround for a crash in the compiled eventem_new build when it's "
            "imported in the main GUI process directly, pending a real fix upstream in the "
            "evenTem C++ build. Transparent otherwise, just a little slower to start each call.")
        new_eventem_note.setWordWrap(True)
        new_eventem_note.setStyleSheet(f"color: {AppTheme.instance().color('fg_dim')}; font-style: italic;")
        backend_layout.addWidget(new_eventem_note)

        cores_row = qtw.QHBoxLayout()
        cores_row.addWidget(qtw.QLabel('CPU cores:'))
        self.spinbox_nThreads = qtw.QSpinBox()
        self.spinbox_nThreads.setRange(0, _CPU_COUNT)
        self.spinbox_nThreads.setSpecialValueText('Auto')
        self.spinbox_nThreads.setValue(settings.n_threads or 0)
        self.spinbox_nThreads.setToolTip(
            f'Worker threads used to decode the file (this machine has {_CPU_COUNT}). '
            "\"Auto\" leaves it at the selected backend's own default. Old eventem uses this "
            "directly (n_threads) for its own internal thread pool, which - confirmed directly "
            "against the C++ source - does nothing at all for .tpx3 decode speed regardless of "
            "the value (a strictly sequential state machine). New eventem is identical *unless* "
            "\"Processes\" is selected below, where it instead becomes the process count for a "
            "real, checkpoint-based file split. pyeventem uses it to decode with several threads "
            "in parallel, but only while declustering is off - a declustered pyeventem run "
            'always decodes sequentially, since resolving clusters needs each hit in original '
            'time order.')
        cores_row.addWidget(self.spinbox_nThreads)
        cores_row.addStretch(1)
        backend_layout.addLayout(cores_row)

        # Threads vs. processes for splitting "CPU cores" above across
        # workers - meaningful for pyeventem (every sink) and, as of the
        # checkpoint-based Roi split, New eventem's Roi/Roi-masked
        # extraction too (see eb._new_eventem_run_roi_multiprocess's own
        # docstring - confirmed directly against the EvenTem C++ source:
        # its .tpx3 decode is a strictly sequential state machine, so
        # "Threads" here does *nothing* for New eventem, unlike pyeventem).
        # Old eventem has no equivalent split at all. Which choice is
        # actually recommended is backend-dependent - the label text below
        # updates live in _on_backend_changed rather than a static
        # "(recommended)" suffix that would be right for one backend and
        # actively misleading for the other.
        exec_row = qtw.QHBoxLayout()
        self.label_execStrategy = qtw.QLabel('Execution strategy:')
        exec_row.addWidget(self.label_execStrategy)
        self._exec_group = qtw.QButtonGroup(self)
        self._exec_buttons = {}
        for key in eb.EXECUTION_STRATEGIES:
            radio = qtw.QRadioButton(eb.EXECUTION_STRATEGY_LABELS[key])
            if key == settings.execution_strategy:
                radio.setChecked(True)
            self._exec_group.addButton(radio)
            self._exec_buttons[key] = radio
            exec_row.addWidget(radio)
        exec_row.addStretch(1)
        backend_layout.addLayout(exec_row)
        self.exec_note = qtw.QLabel()
        self.exec_note.setWordWrap(True)
        self.exec_note.setStyleSheet(f"color: {AppTheme.instance().color('fg_dim')}; font-style: italic;")
        backend_layout.addWidget(self.exec_note)

        self.layout.addWidget(backend_box)
        for radio in self._backend_buttons.values():
            radio.toggled.connect(self._on_backend_changed)
        # Not called yet here - _on_backend_changed also needs
        # self.decluster_box, built further down; called once, after that,
        # right below self._on_enabled_toggled's own identical initial call.

        # --- Declustering -------------------------------------------------
        self.decluster_box = decluster_box = qtw.QGroupBox('Declustering')
        decluster_layout = qtw.QVBoxLayout(decluster_box)

        self.checkbox_enabled = qtw.QCheckBox('Enable declustering')
        self.checkbox_enabled.setChecked(settings.decluster_enabled)
        self.checkbox_enabled.toggled.connect(self._on_enabled_toggled)
        decluster_layout.addWidget(self.checkbox_enabled)

        form = qtw.QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.spinbox_dspace = qtw.QSpinBox()
        self.spinbox_dspace.setRange(1, 64)
        self.spinbox_dspace.setValue(settings.dspace)
        self.spinbox_dspace.setToolTip('Max per-axis pixel distance from the seed hit to merge into its cluster.')
        form.addRow('Max Pixel Distance (dspace)', self.spinbox_dspace)

        self.spinbox_dtime = qtw.QDoubleSpinBox()
        self.spinbox_dtime.setRange(1.0, 100000.0)
        self.spinbox_dtime.setDecimals(1)
        self.spinbox_dtime.setSuffix(' ns')
        self.spinbox_dtime.setValue(settings.dtime_ns)
        self.spinbox_dtime.setToolTip('Max ToA distance from the seed hit to merge into its cluster.')
        form.addRow('Max Time Distance (dtime)', self.spinbox_dtime)

        self.spinbox_clusterRange = qtw.QSpinBox()
        self.spinbox_clusterRange.setRange(2, 65536)
        self.spinbox_clusterRange.setValue(settings.cluster_range)
        self.spinbox_clusterRange.setToolTip('How many subsequent hits are checked as merge candidates for each seed.')
        form.addRow('Cluster Range', self.spinbox_clusterRange)
        decluster_layout.addLayout(form)

        # Calibration mode: ToT-per-electron constant, or a LUT file - mutually
        # exclusive (a LUT file, when set, takes priority - see
        # eventem_backend._apply_decluster_eventem/_make_declusterer).
        calib_box = qtw.QGroupBox('Electron-Count Calibration')
        calib_layout = qtw.QVBoxLayout(calib_box)
        self._calib_group = qtw.QButtonGroup(self)
        self.radio_totPerElectron = qtw.QRadioButton('ToT per electron')
        self.radio_lutFile = qtw.QRadioButton('LUT file')
        self._calib_group.addButton(self.radio_totPerElectron)
        self._calib_group.addButton(self.radio_lutFile)
        using_lut = bool(settings.lut_file)
        self.radio_totPerElectron.setChecked(not using_lut)
        self.radio_lutFile.setChecked(using_lut)

        row_tot = qtw.QHBoxLayout()
        row_tot.addWidget(self.radio_totPerElectron)
        self.spinbox_totPerElectron = qtw.QDoubleSpinBox()
        self.spinbox_totPerElectron.setRange(0.1, 1_000_000.0)
        self.spinbox_totPerElectron.setDecimals(2)
        self.spinbox_totPerElectron.setValue(settings.tot_per_electron)
        row_tot.addWidget(self.spinbox_totPerElectron)
        row_tot.addStretch(1)
        calib_layout.addLayout(row_tot)

        row_lut = qtw.QHBoxLayout()
        row_lut.addWidget(self.radio_lutFile)
        self.lineedit_lutFile = qtw.QLineEdit(settings.lut_file)
        self.lineedit_lutFile.setPlaceholderText('Path to electron-count LUT file...')
        row_lut.addWidget(self.lineedit_lutFile, 1)
        self.button_browseLut = qtw.QPushButton('Browse...')
        self.button_browseLut.clicked.connect(self._browse_lut_file)
        row_lut.addWidget(self.button_browseLut)
        calib_layout.addLayout(row_lut)
        decluster_layout.addWidget(calib_box)

        # Sinks: which analyses declustering actually applies to.
        sinks_box = qtw.QGroupBox('Apply Declustering To')
        sinks_layout = qtw.QVBoxLayout(sinks_box)
        sinks_row = qtw.QHBoxLayout()
        self._sink_checkboxes = {}
        for key in SINK_KEYS:
            cb = qtw.QCheckBox(SINK_LABELS[key])
            cb.setChecked(settings.sinks.get(key, True))
            self._sink_checkboxes[key] = cb
            sinks_row.addWidget(cb)
        sinks_row.addStretch(1)
        sinks_layout.addLayout(sinks_row)

        select_row = qtw.QHBoxLayout()
        self.button_selectAllSinks = qtw.QPushButton('Select All')
        self.button_selectAllSinks.clicked.connect(lambda: self._set_all_sinks(True))
        select_row.addWidget(self.button_selectAllSinks)
        self.button_selectNoneSinks = qtw.QPushButton('Select None')
        self.button_selectNoneSinks.clicked.connect(lambda: self._set_all_sinks(False))
        select_row.addWidget(self.button_selectNoneSinks)
        select_row.addStretch(1)
        sinks_layout.addLayout(select_row)
        decluster_layout.addWidget(sinks_box)

        self.layout.addWidget(decluster_box)
        self._on_enabled_toggled(self.checkbox_enabled.isChecked())
        self._on_backend_changed()

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

    def _on_backend_changed(self):
        """Grey out the execution-strategy radios while 'Old eventem' is
        selected - it has no split API of any kind, unlike pyeventem
        (every sink) and New eventem (Roi/Roi-masked, via the checkpoint-
        based multi-process split - see eb._new_eventem_run_roi_multiprocess's
        own docstring). Also refreshes exec_note's text, since which choice
        is actually recommended is backend-dependent: pyeventem measured
        Threads faster in every configuration checked, but New eventem's
        .tpx3 decode is a strictly sequential state machine - Threads does
        *nothing* for it at all, Processes is the only way to get real
        parallelism.

        Also grey out the entire Declustering box while 'Old eventem' is
        selected - it has no declustering support at all (see
        eventem_backend._decluster_active, which silently no-ops it with
        just a warning-level log line). Previously the box stayed fully
        interactive regardless: a user could check "Enable declustering",
        tune every parameter, click Apply, and see literally zero effect
        on any computation - the only explanation was this dialog's own
        easy-to-miss italic note above. Confirmed as a real, reproducible
        case of exactly that (a user's persisted settings had backend=
        old_eventem and decluster_enabled=true at the same time - every
        past "with declustering" run had actually been running without
        it). Disabling the box doesn't touch the checkbox's own state - it
        re-enables, unchanged, the moment a declustering-capable backend
        is selected."""
        backend = self._selected_backend()
        for radio in self._exec_buttons.values():
            radio.setEnabled(backend != eb.BACKEND_OLD)
        if backend == eb.BACKEND_PYEVENTEM:
            self.exec_note.setText(
                "Measured directly (pyeventem's Examples/07_backend_performance.ipynb): 'Threads' "
                "wins in every configuration checked there - multiprocessing adds process-spawn/"
                "import overhead with nothing to show for it, since pyeventem's own decoding "
                "already releases the GIL. 'Processes' exists for comparison/testing, not because "
                "it's expected to win.")
        elif backend == eb.BACKEND_NEW:
            self.exec_note.setText(
                "New eventem's .tpx3 decode is a strictly sequential state machine (confirmed "
                "directly against the EvenTem C++ source and its own Examples/ROI.ipynb) - 'CPU "
                "cores' above never parallelizes it at all while 'Threads' is selected here, "
                "however high it's set. 'Processes' is the only way to get real parallelism: a "
                "checkpoint-based split of the file across real OS processes, for Roi/Roi-masked "
                "extraction only (Pacbed/vSTEM/Var have no equivalent C++ API yet).")
        else:
            self.exec_note.setText("Old eventem has no equivalent split - this setting is ignored.")

    def _on_enabled_toggled(self, checked):
        """Grey out every declustering sub-control while disabled, so it's
        visually clear they have no effect - matches how the sinks/
        calibration controls only matter once declustering itself is on."""
        for w in (self.spinbox_dspace, self.spinbox_dtime, self.spinbox_clusterRange,
                  self.radio_totPerElectron, self.spinbox_totPerElectron,
                  self.radio_lutFile, self.lineedit_lutFile, self.button_browseLut,
                  self.button_selectAllSinks, self.button_selectNoneSinks,
                  *self._sink_checkboxes.values()):
            w.setEnabled(checked)

    def _set_all_sinks(self, checked):
        for cb in self._sink_checkboxes.values():
            cb.setChecked(checked)

    def _browse_lut_file(self):
        path, _ = qtw.QFileDialog.getOpenFileName(
            self, 'Select Electron-Count LUT File', '', 'Text files (*.txt);;All files (*)')
        if path:
            self.lineedit_lutFile.setText(path)
            self.radio_lutFile.setChecked(True)

    def _selected_backend(self):
        for key, radio in self._backend_buttons.items():
            if radio.isChecked():
                return key
        return eb.BACKEND_OLD

    def _selected_execution_strategy(self):
        for key, radio in self._exec_buttons.items():
            if radio.isChecked():
                return key
        return eb.EXEC_THREADS

    def apply_values(self):
        """Push every control's current value to AnalysisBackendSettings at
        once - AnalysisBackendSettings.set_values() also persists this to
        disk, so it's still in effect next time EDyssey opens."""
        lut_file = self.lineedit_lutFile.text().strip() if self.radio_lutFile.isChecked() else ''
        AnalysisBackendSettings.instance().set_values(
            backend=self._selected_backend(),
            n_threads=self.spinbox_nThreads.value() or None,
            execution_strategy=self._selected_execution_strategy(),
            decluster_enabled=self.checkbox_enabled.isChecked(),
            dspace=self.spinbox_dspace.value(),
            dtime_ns=self.spinbox_dtime.value(),
            cluster_range=self.spinbox_clusterRange.value(),
            tot_per_electron=self.spinbox_totPerElectron.value(),
            lut_file=lut_file,
            sinks={key: cb.isChecked() for key, cb in self._sink_checkboxes.items()},
        )

    def reset_defaults(self):
        """Reset every value to its default AND apply immediately (a single,
        deliberate action, unlike the Apply-gated controls above)."""
        AnalysisBackendSettings.instance().reset()
        settings = AnalysisBackendSettings.instance()
        self._backend_buttons[settings.backend].setChecked(True)
        self.spinbox_nThreads.setValue(settings.n_threads or 0)
        self._exec_buttons[settings.execution_strategy].setChecked(True)
        self._on_backend_changed()
        self.checkbox_enabled.setChecked(settings.decluster_enabled)
        self.spinbox_dspace.setValue(settings.dspace)
        self.spinbox_dtime.setValue(settings.dtime_ns)
        self.spinbox_clusterRange.setValue(settings.cluster_range)
        self.spinbox_totPerElectron.setValue(settings.tot_per_electron)
        using_lut = bool(settings.lut_file)
        self.radio_totPerElectron.setChecked(not using_lut)
        self.radio_lutFile.setChecked(using_lut)
        self.lineedit_lutFile.setText(settings.lut_file)
        for key, cb in self._sink_checkboxes.items():
            cb.setChecked(settings.sinks.get(key, True))
