# -*- coding: utf-8 -*-
"""Shared "Make *.pts2" parameter popup, used by both Tab_SAM2 and
Tab_Tracking_CV2 so the two tabs don't duplicate the widget - same spirit as
ThresholdDialog (see ui_tabs/threshold_dialog.py).

Layout:
- A "Basic" section with the fields that matter for nearly every dataset
  (acceleration voltage, calibration, geometry, tilt ramp, center...).
- A "Show Advanced Parameters" toggle that extends the dialog with every
  other single-line PETS setting from a reference .pts2 file, each with that
  file's own comment as its tooltip.
- A separate "Configure Autotasks..." popup for the ordered list of tasks
  PETS should run automatically on open.

Every field except the per-study ones (center, alpha start/step) is cached
to `<repo>/temp/pets2_last_params.json` on OK and used to pre-fill the next
time this dialog opens (see py4DTomo/io_utils/pets.py).
"""
import re
import PyQt5.QtWidgets as qtw
import py4DTomo.io_utils as io

_INT_RE = re.compile(r'^[+-]?\d+$')


def _make_token_widget(token):
    """Build one small typed widget for a single whitespace-separated token
    from an ADVANCED_FIELDS default string, plus a zero-arg getter returning
    its current value as a string (in the same format PETS expects).

    Used to give every multi-parameter PETS setting (e.g. `noiseparameters
    5.0000 0.0000`, `peakanalysis 0 0 3.7400 0.0080`) its own row of separate
    entries instead of one raw text blob the user has to hand-edit - the
    PETS keyword documents each position's meaning, so it should be its own
    control. Single-token fields keep their own plain QLineEdit (built by the
    caller), so this is only used for genuinely multi-valued keywords.

    Type is guessed from the token's own text: yes/no -> checkbox, a bare
    integer -> QSpinBox, anything else parseable as a float -> QDoubleSpinBox
    (decimals matched to the token's own precision), otherwise a short
    QLineEdit for the free-form/enum word (e.g. 'auto', 'median', 'xplor').
    """
    if token.lower() in ('yes', 'no'):
        widget = qtw.QCheckBox()
        widget.setChecked(token.lower() == 'yes')
        return widget, (lambda: 'yes' if widget.isChecked() else 'no')

    if _INT_RE.match(token):
        widget = qtw.QSpinBox()
        widget.setRange(-1_000_000, 1_000_000)
        widget.setFixedWidth(70)
        widget.setValue(int(token))
        return widget, (lambda: str(widget.value()))

    try:
        float(token)
    except ValueError:
        pass
    else:
        decimals = len(token.split('.')[1]) if '.' in token else 4
        widget = qtw.QDoubleSpinBox()
        widget.setRange(-1e7, 1e7)
        widget.setDecimals(decimals)
        widget.setFixedWidth(90)
        widget.setValue(float(token))
        return widget, (lambda: f'{widget.value():.{decimals}f}')

    widget = qtw.QLineEdit(token)
    widget.setFixedWidth(max(40, 9 * len(token)))
    return widget, widget.text


class Pets2AutotaskDialog(qtw.QDialog):
    """Ordered-list editor for PETS' `autotask` block - the tasks PETS runs
    automatically when the project is opened."""

    def __init__(self, parent=None, tasks=None, keepautotasks='no'):
        super().__init__(parent)
        self.setWindowTitle('PETS2 Autotasks')
        self._tasks = list(tasks or [])

        layout = qtw.QVBoxLayout(self)
        layout.addWidget(qtw.QLabel(
            'Tasks PETS will run automatically, in order, when this project is opened.'))

        self.list_tasks = qtw.QListWidget()
        self.list_tasks.addItems(self._tasks)
        layout.addWidget(self.list_tasks)

        row_add = qtw.QHBoxLayout()
        layout.addLayout(row_add)
        self.combo_task = qtw.QComboBox()
        self.combo_task.addItems(io.AUTOTASK_OPTIONS)
        row_add.addWidget(self.combo_task)
        button_add = qtw.QPushButton('Add')
        button_add.clicked.connect(self._add_task)
        row_add.addWidget(button_add)

        row_edit = qtw.QHBoxLayout()
        layout.addLayout(row_edit)
        button_remove = qtw.QPushButton('Remove Selected')
        button_remove.clicked.connect(self._remove_selected)
        row_edit.addWidget(button_remove)
        button_up = qtw.QPushButton('Move Up')
        button_up.clicked.connect(lambda: self._move_selected(-1))
        row_edit.addWidget(button_up)
        button_down = qtw.QPushButton('Move Down')
        button_down.clicked.connect(lambda: self._move_selected(1))
        row_edit.addWidget(button_down)

        self.checkbox_keepautotasks = qtw.QCheckBox('Keep autotasks after all finished')
        self.checkbox_keepautotasks.setChecked(keepautotasks == 'yes')
        self.checkbox_keepautotasks.setToolTip(
            'If yes, list of tasks to be automaticaly performed after the start is not '
            'cleared when all tasks are finished.')
        layout.addWidget(self.checkbox_keepautotasks)

        buttons = qtw.QHBoxLayout()
        layout.addLayout(buttons)
        button_ok = qtw.QPushButton('OK')
        button_ok.clicked.connect(self.accept)
        buttons.addWidget(button_ok)
        button_cancel = qtw.QPushButton('Cancel')
        button_cancel.clicked.connect(self.reject)
        buttons.addWidget(button_cancel)

    def _add_task(self):
        self.list_tasks.addItem(self.combo_task.currentText())

    def _remove_selected(self):
        for item in self.list_tasks.selectedItems():
            self.list_tasks.takeItem(self.list_tasks.row(item))

    def _move_selected(self, direction):
        """Move the selected task by `direction` rows (-1 up, +1 down)."""
        row = self.list_tasks.currentRow()
        new_row = row + direction
        if row < 0 or not (0 <= new_row < self.list_tasks.count()):
            return
        item = self.list_tasks.takeItem(row)
        self.list_tasks.insertItem(new_row, item)
        self.list_tasks.setCurrentRow(new_row)

    def get_result(self):
        """Return (tasks, keepautotasks) reflecting the current list/checkbox state."""
        tasks = [self.list_tasks.item(i).text() for i in range(self.list_tasks.count())]
        keepautotasks = 'yes' if self.checkbox_keepautotasks.isChecked() else 'no'
        return tasks, keepautotasks


class Pets2ParamsDialog(qtw.QDialog):
    """Popup for the experiment parameters a PETS2 project file (.pts2)
    needs - acceleration voltage (converted to wavelength via the
    relativistic electron-wavelength formula), calibration, and tilt/
    rotation geometry, plus (behind "Show Advanced Parameters") every other
    single-line PETS setting."""

    _GEOMETRIES = ['continuous', 'precession', 'static']
    _DETECTORS = ['default', 'asi', 'olympus', 'cetad']

    def __init__(self, parent=None, voltage_kv=None, exposure_s=None, aperpixel=None,
                 center=None):
        """voltage_kv/exposure_s/aperpixel/center, when given, pre-fill their
        fields in place of the cached value from the last time this dialog was used."""
        super().__init__(parent)
        self.setWindowTitle('PETS2 Export Parameters')
        cache = io.load_pets2_defaults()
        self._autotask_list = cache.get('autotask', [])
        self._keepautotasks = cache.get('keepautotasks', 'no')

        layout = qtw.QVBoxLayout(self)
        form = qtw.QFormLayout()
        layout.addLayout(form)

        self.spinbox_voltage = qtw.QDoubleSpinBox()
        self.spinbox_voltage.setRange(1, 1000)
        self.spinbox_voltage.setDecimals(1)
        self.spinbox_voltage.setSuffix(' kV')
        self.spinbox_voltage.setValue(voltage_kv if voltage_kv is not None
                                       else cache.get('voltage_kv', 200.0))
        self.spinbox_voltage.setToolTip('wavelength in angstroms (entered here as '
            'acceleration voltage - converted via the relativistic wavelength formula)')
        form.addRow('Acceleration voltage', self.spinbox_voltage)

        self.label_wavelength = qtw.QLabel()
        form.addRow('Wavelength (Å)', self.label_wavelength)
        self.spinbox_voltage.valueChanged.connect(self._update_wavelength)
        self._update_wavelength()

        self.spinbox_aperpixel = qtw.QDoubleSpinBox()
        self.spinbox_aperpixel.setRange(0, 100)
        self.spinbox_aperpixel.setDecimals(6)
        self.spinbox_aperpixel.setValue(aperpixel if aperpixel is not None
                                         else cache.get('aperpixel', 0.0))
        self.spinbox_aperpixel.setToolTip('image calibration in reciprocal angstroms/pixel '
            '(pre-filled from the Recip. scale entered on the main tab)')
        form.addRow('aperpixel (1/Å per px)', self.spinbox_aperpixel)

        self.combo_geometry = qtw.QComboBox()
        self.combo_geometry.addItems(self._GEOMETRIES)
        self.combo_geometry.setToolTip('experimental geometry. Options: continuous,precession,static')
        form.addRow('geometry', self.combo_geometry)

        self.spinbox_phi = qtw.QDoubleSpinBox()
        self.spinbox_phi.setRange(0, 5)
        self.spinbox_phi.setDecimals(4)
        self.spinbox_phi.setSuffix(' deg')
        self.spinbox_phi.setValue(cache.get('phi', 0.0))
        self.spinbox_phi.setToolTip(
            'precession or integration (semi)angle. Must be zero for static geometry.')
        form.addRow('phi', self.spinbox_phi)
        # static geometry has no phi - force/lock it to 0 whenever that's
        # selected, instead of letting a stale nonzero value slip into the
        # written .pts2 (PETS itself is documented to require phi=0 there).
        self.combo_geometry.currentTextChanged.connect(self._on_geometry_changed)
        self.combo_geometry.setCurrentText(cache.get('geometry', 'continuous'))
        self._on_geometry_changed(self.combo_geometry.currentText())

        # Per-study tilt ramp - not cached between studies (see BASIC_FIELDS'
        # docstring note above), always starts blank/zero.
        self.spinbox_alpha_start = qtw.QDoubleSpinBox()
        self.spinbox_alpha_start.setRange(-180, 180)
        self.spinbox_alpha_start.setDecimals(4)
        self.spinbox_alpha_start.setSuffix(' deg')
        self.spinbox_alpha_start.setToolTip('Goniometer tilt angle (alpha) at frame 1 - '
            'not cached between studies, always starts blank.')
        form.addRow('Alpha start', self.spinbox_alpha_start)

        self.spinbox_alpha_step = qtw.QDoubleSpinBox()
        self.spinbox_alpha_step.setRange(-90, 90)
        self.spinbox_alpha_step.setDecimals(4)
        self.spinbox_alpha_step.setSuffix(' deg/frame')
        self.spinbox_alpha_step.setToolTip(
            'Alpha increment between consecutive frames - together with Alpha start, sets '
            'every frame\'s tilt angle in the .pts2 image list. Not cached between studies.')
        form.addRow('Alpha step', self.spinbox_alpha_step)

        self.spinbox_binning = qtw.QSpinBox()
        self.spinbox_binning.setRange(1, 64)
        self.spinbox_binning.setValue(int(cache.get('binning', 1)))
        self.spinbox_binning.setToolTip(
            'binning of the images after reading in. Should be an even number.')
        form.addRow('bin', self.spinbox_binning)

        for key, ph, default, comment in io.BASIC_FIELDS:
            spin = qtw.QDoubleSpinBox()
            spin.setRange(0, 1e6)
            spin.setDecimals(4)
            spin.setValue(float(cache.get(ph, default)))
            spin.setToolTip(comment)
            form.addRow(key, spin)
            setattr(self, f'spinbox_{ph}', spin)

        row_center = qtw.QHBoxLayout()
        self.checkbox_center_auto = qtw.QCheckBox('Auto-detect')
        self.checkbox_center_auto.setChecked(center is None)
        self.checkbox_center_auto.setToolTip(
            'center of the diffraction pattern. Options: AUTO=find center automatically; '
            'two numbers x y - position of the pattern center on the first frame in pixels.')
        row_center.addWidget(self.checkbox_center_auto)
        self.spinbox_center_x = qtw.QDoubleSpinBox()
        self.spinbox_center_x.setRange(0, 1e5)
        self.spinbox_center_x.setDecimals(2)
        self.spinbox_center_x.setPrefix('x=')
        self.spinbox_center_y = qtw.QDoubleSpinBox()
        self.spinbox_center_y.setRange(0, 1e5)
        self.spinbox_center_y.setDecimals(2)
        self.spinbox_center_y.setPrefix('y=')
        if center is not None:
            self.spinbox_center_x.setValue(center[0])
            self.spinbox_center_y.setValue(center[1])
        row_center.addWidget(self.spinbox_center_x)
        row_center.addWidget(self.spinbox_center_y)
        self.checkbox_center_auto.toggled.connect(
            lambda checked: (self.spinbox_center_x.setDisabled(checked),
                              self.spinbox_center_y.setDisabled(checked)))
        self.spinbox_center_x.setDisabled(self.checkbox_center_auto.isChecked())
        self.spinbox_center_y.setDisabled(self.checkbox_center_auto.isChecked())
        form.addRow('center (pre-filled from main UI)', row_center)

        row_autotask = qtw.QHBoxLayout()
        layout.addLayout(row_autotask)
        self.checkbox_use_autotask = qtw.QCheckBox('Run autotasks on open')
        self.checkbox_use_autotask.setChecked(bool(cache.get('autotask_enabled', True)))
        self.checkbox_use_autotask.setToolTip(
            'If unchecked, no autotask list is written to the .pts2 file at all, regardless '
            'of what is configured below - the configured list is kept (and still cached) '
            'but simply not applied.')
        row_autotask.addWidget(self.checkbox_use_autotask)
        self.button_autotasks = qtw.QPushButton('Configure Autotasks...')
        self.button_autotasks.clicked.connect(self._open_autotask_dialog)
        row_autotask.addWidget(self.button_autotasks)

        # Advanced section: built up-front (so get_params() can always read
        # it) but only added to the visible layout when toggled open.
        self.button_advanced = qtw.QPushButton('Show Advanced Parameters >>')
        self.button_advanced.setCheckable(True)
        self.button_advanced.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.button_advanced)

        self._advanced_edits = {}
        advanced_container = qtw.QWidget()
        advanced_form = qtw.QFormLayout(advanced_container)

        self.spinbox_omega = qtw.QDoubleSpinBox()
        self.spinbox_omega.setRange(-180, 180)
        self.spinbox_omega.setDecimals(4)
        self.spinbox_omega.setSuffix(' deg')
        self.spinbox_omega.setValue(cache.get('omega', 0.0))
        self.spinbox_omega.setToolTip(
            'angle between the positive horizontal axis on the image and the projection of '
            'the tilt axis on the image. Two keys define, if global search and local '
            'refinement should be performed.')
        row_omega = qtw.QHBoxLayout()
        row_omega.addWidget(self.spinbox_omega)
        self.checkbox_omega_global = qtw.QCheckBox('Global search')
        self.checkbox_omega_global.setChecked(bool(int(cache.get('omega_global_search', 1))))
        self.checkbox_omega_global.setToolTip(
            'If checked, PETS performs a global search for omega on open (recommended default '
            'for a fresh study). Written as the first key after the omega angle.')
        row_omega.addWidget(self.checkbox_omega_global)
        self.checkbox_omega_local = qtw.QCheckBox('Local refinement')
        self.checkbox_omega_local.setChecked(bool(int(cache.get('omega_local_refine', 1))))
        self.checkbox_omega_local.setToolTip(
            'If checked, PETS locally refines omega around the found/entered value. Written '
            'as the second key after the omega angle.')
        row_omega.addWidget(self.checkbox_omega_local)
        advanced_form.addRow('omega', row_omega)

        self.spinbox_exposure = qtw.QDoubleSpinBox()
        self.spinbox_exposure.setRange(0, 3600)
        self.spinbox_exposure.setDecimals(6)
        self.spinbox_exposure.setSuffix(' s')
        self.spinbox_exposure.setValue(exposure_s if exposure_s is not None
                                        else cache.get('exposure_s', 0.0))
        self.spinbox_exposure.setToolTip(
            'Exposure time per frame in seconds. Used by the detector non-linearity correction.')
        advanced_form.addRow('Exposure per frame', self.spinbox_exposure)

        self.combo_detector = qtw.QComboBox()
        self.combo_detector.addItems(self._DETECTORS)
        self.combo_detector.setCurrentText(cache.get('detector', 'default'))
        self.combo_detector.setToolTip(
            'detector type. Options: default, asi, olympus, cetad. More to come...')
        advanced_form.addRow('detector', self.combo_detector)

        for key, ph, default, comment in io.ADVANCED_FIELDS:
            default_str = str(cache.get(ph, default))
            tokens = default_str.split()
            if len(tokens) > 1:
                # Several PETS parameters packed onto one line - give each
                # its own typed entry instead of one free-text blob (see
                # _make_token_widget).
                row_widget = qtw.QWidget()
                row_layout = qtw.QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_widget.setToolTip(comment)
                getters = []
                for token in tokens:
                    token_widget, getter = _make_token_widget(token)
                    token_widget.setToolTip(comment)
                    row_layout.addWidget(token_widget)
                    getters.append(getter)
                row_layout.addStretch(1)
                advanced_form.addRow(key, row_widget)
                self._advanced_edits[ph] = (lambda getters=getters:
                                             ' '.join(g() for g in getters))
            else:
                edit = qtw.QLineEdit(default_str)
                edit.setToolTip(comment)
                advanced_form.addRow(key, edit)
                self._advanced_edits[ph] = edit.text
        self.scroll_advanced = qtw.QScrollArea()
        self.scroll_advanced.setWidgetResizable(True)
        self.scroll_advanced.setWidget(advanced_container)
        self.scroll_advanced.setMinimumHeight(300)
        self.scroll_advanced.setVisible(False)
        layout.addWidget(self.scroll_advanced)

        buttons = qtw.QHBoxLayout()
        layout.addLayout(buttons)
        self.button_ok = qtw.QPushButton('OK')
        self.button_ok.clicked.connect(self._on_ok)
        buttons.addWidget(self.button_ok)
        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.button_cancel)

    def _toggle_advanced(self, checked):
        self.scroll_advanced.setVisible(checked)
        self.button_advanced.setText(
            'Hide Advanced Parameters <<' if checked else 'Show Advanced Parameters >>')
        self.adjustSize()

    def _on_geometry_changed(self, geometry):
        """static geometry has no phi - PETS requires it to be zero there, so
        force/lock the field instead of writing out a stale nonzero value."""
        is_static = (geometry == 'static')
        if is_static:
            self.spinbox_phi.setValue(0.0)
        self.spinbox_phi.setDisabled(is_static)

    def _open_autotask_dialog(self):
        dialog = Pets2AutotaskDialog(self, tasks=self._autotask_list,
                                      keepautotasks=self._keepautotasks)
        if dialog.exec_() == qtw.QDialog.Accepted:
            self._autotask_list, self._keepautotasks = dialog.get_result()

    def _update_wavelength(self):
        lam = io.electron_wavelength_angstrom(self.spinbox_voltage.value())
        self.label_wavelength.setText(f'{lam:.6f}')

    def _on_ok(self):
        io.save_pets2_defaults(self._cacheable_params())
        self.accept()

    def _cacheable_params(self):
        """Every field except the per-study ones (center, alpha start/step),
        which are intentionally never pre-filled from a previous study."""
        params = {
            'voltage_kv': self.spinbox_voltage.value(),
            'aperpixel': self.spinbox_aperpixel.value(),
            'geometry': self.combo_geometry.currentText(),
            'phi': self.spinbox_phi.value(),
            'binning': self.spinbox_binning.value(),
            'omega': self.spinbox_omega.value(),
            'omega_global_search': int(self.checkbox_omega_global.isChecked()),
            'omega_local_refine': int(self.checkbox_omega_local.isChecked()),
            'exposure_s': self.spinbox_exposure.value(),
            'detector': self.combo_detector.currentText(),
            'autotask': self._autotask_list,
            'keepautotasks': self._keepautotasks,
            'autotask_enabled': self.checkbox_use_autotask.isChecked(),
        }
        for _key, ph, _default, _comment in io.BASIC_FIELDS:
            params[ph] = getattr(self, f'spinbox_{ph}').value()
        for ph, get_value in self._advanced_edits.items():
            params[ph] = get_value()
        return params

    def get_params(self):
        """Return the collected parameters as kwargs for `io.write_pts2`
        (everything except `fn`/`n_frames`/`frame_shape`, which are only
        known at save time)."""
        center = None
        if not self.checkbox_center_auto.isChecked():
            center = (self.spinbox_center_x.value(), self.spinbox_center_y.value())

        advanced_overrides = {}
        for key, ph, _default, _comment in io.ADVANCED_FIELDS:
            advanced_overrides[key] = self._advanced_edits[ph]()

        geometry = self.combo_geometry.currentText()
        params = {
            'voltage_kv': self.spinbox_voltage.value(),
            'aperpixel': self.spinbox_aperpixel.value(),
            'geometry': geometry,
            'tilt_start_deg': self.spinbox_alpha_start.value(),
            'tilt_step_deg': self.spinbox_alpha_step.value(),
            'omega': self.spinbox_omega.value(),
            'omega_global_search': int(self.checkbox_omega_global.isChecked()),
            'omega_local_refine': int(self.checkbox_omega_local.isChecked()),
            # Belt-and-suspenders on top of the UI lock in _on_geometry_changed:
            # static geometry must never write a nonzero phi.
            'phi': 0.0 if geometry == 'static' else self.spinbox_phi.value(),
            'binning': self.spinbox_binning.value(),
            'exposure_s': self.spinbox_exposure.value(),
            'detector': self.combo_detector.currentText(),
            'center': center,
            # Unchecked: write no autotask list at all, regardless of what's
            # configured in the sub-dialog (that list is kept/cached, just
            # not applied here).
            'autotask': self._autotask_list if self.checkbox_use_autotask.isChecked() else [],
            'keepautotasks': self._keepautotasks,
            'advanced_overrides': advanced_overrides,
        }
        for _key, ph, _default, _comment in io.BASIC_FIELDS:
            params[ph] = getattr(self, f'spinbox_{ph}').value()
        return params
