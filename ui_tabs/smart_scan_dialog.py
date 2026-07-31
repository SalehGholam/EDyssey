# -*- coding: utf-8 -*-
""""Check Files" popup for smart-scanned (pattern-file) acquisitions -
reviews the automatic detection/acquisition/pattern-file match for every
tilt angle in a folder (see py4DTomo/io_utils/smart_scan.py) before it's
used for a batch nav-image calculation (Tab_Create_NavSignal) or 3DED
extraction (Tab_Tracking_CV2/Tab_SAM2), and lets bad rows be fixed by hand -
the same file-by-file checking these tabs' docstrings describe doing
manually today.
"""
import os
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import py4DTomo.io_utils as io

_COLS = ['Include', 'Angle', 'Detection File', 'Acquisition File', 'Pattern File', 'Status']
_FILE_COLS = {2: 'detection_file', 3: 'acquisition_file', 4: 'pattern_file'}

_BAD_ROW_COLOR = QColor('#5a2d2d')
_OK_ROW_COLOR = QColor('#2b2b2b')


class SmartScanCheckDialog(qtw.QDialog):
    """Editable table of every tilt angle found in `data_dir`, matched to
    its detection/acquisition data files and pattern file. `exec_()` returns
    QDialog.Accepted once the user confirms; `self.rows` then holds the
    reviewed match table (see `py4DTomo.io_utils.smart_scan.match_tilt_files`
    for the row shape) - pass it to `resolve_smart_scan_files` to get the
    ordered per-role file list to actually use."""

    def __init__(self, parent, data_dir, data_ext, pattern_dir=None, rows=None):
        super().__init__(parent)
        self.setWindowTitle('Check Smart-Scan Files')
        self.resize(900, 500)
        self.data_dir = data_dir
        self.data_ext = data_ext
        self.pattern_dir = pattern_dir
        self.rows = rows if rows is not None else []

        layout = qtw.QVBoxLayout(self)

        info = qtw.QLabel(
            'One row per detected tilt angle. Rows with a problem are unchecked and '
            'highlighted - double-click a file cell to browse for a replacement, or '
            'check/uncheck "Include" to use/skip a row as-is.')
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = qtw.QTableWidget()
        self.table.setColumnCount(len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(qtw.QAbstractItemView.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        layout.addWidget(self.table)

        row_buttons = qtw.QHBoxLayout()
        layout.addLayout(row_buttons)
        self.button_rescan = qtw.QPushButton('Rescan Folder')
        self.button_rescan.clicked.connect(self.rescan)
        row_buttons.addWidget(self.button_rescan)
        self.label_summary = qtw.QLabel('')
        row_buttons.addWidget(self.label_summary)
        row_buttons.addStretch(1)

        buttons = qtw.QHBoxLayout()
        layout.addLayout(buttons)
        buttons.addStretch(1)
        self.button_confirm = qtw.QPushButton('Confirm && Use')
        self.button_confirm.clicked.connect(self.accept)
        buttons.addWidget(self.button_confirm)
        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.button_cancel)

        if not self.rows:
            self.rescan()
        else:
            self._populate_table()

    def rescan(self):
        try:
            self.rows = io.match_tilt_files(self.data_dir, self.data_ext,
                                            pattern_dir=self.pattern_dir)
        except Exception as e:
            qtw.QMessageBox.critical(self, 'Scan Failed', f'Could not scan {self.data_dir}:\n{e}')
            self.rows = []
        self._populate_table()

    def _populate_table(self):
        self.table.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            checkbox = qtw.QCheckBox()
            checkbox.setChecked(not row['excluded'])
            checkbox.stateChanged.connect(lambda state, r=r: self._on_include_toggled(r, state))
            cell = qtw.QWidget()
            cell_layout = qtw.QHBoxLayout(cell)
            cell_layout.addWidget(checkbox)
            cell_layout.setAlignment(Qt.AlignCenter)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(r, 0, cell)

            angle_text = f"{row['angle']:.2f}" if row['angle'] is not None else '?'
            self.table.setItem(r, 1, qtw.QTableWidgetItem(angle_text))
            for col, key in _FILE_COLS.items():
                fn = row.get(key)
                text = os.path.basename(fn) if fn else '(missing)'
                self.table.setItem(r, col, qtw.QTableWidgetItem(text))
            status_text = ', '.join(io.STATUS_LABELS.get(s, s) for s in row['status'])
            if row.get('extra_files'):
                status_text += ' (' + ', '.join(os.path.basename(f) for f in row['extra_files']) + ')'
            self.table.setItem(r, 5, qtw.QTableWidgetItem(status_text))
            self._color_row(r)
        self.table.resizeColumnsToContents()
        self._update_summary()

    def _color_row(self, r):
        color = _OK_ROW_COLOR if not self.rows[r]['excluded'] else _BAD_ROW_COLOR
        for c in range(1, len(_COLS)):
            item = self.table.item(r, c)
            if item is not None:
                item.setBackground(color)

    def _update_summary(self):
        n_ok = sum(1 for row in self.rows if not row['excluded'])
        self.label_summary.setText(f'{n_ok} / {len(self.rows)} angle(s) included')

    def _on_include_toggled(self, r, state):
        self.rows[r]['excluded'] = (state != Qt.Checked)
        self._color_row(r)
        self._update_summary()

    def _on_cell_double_clicked(self, r, c):
        key = _FILE_COLS.get(c)
        if key is None:
            return
        start_dir = self.pattern_dir if key == 'pattern_file' else self.data_dir
        file_filter = ('Text files (*.txt)' if key == 'pattern_file'
                       else f'Data files (*{self.data_ext})')
        path, _ = qtw.QFileDialog.getOpenFileName(
            self, f'Select replacement for {_COLS[c]}', start_dir or '', file_filter)
        if not path:
            return
        self.rows[r][key] = path
        # Re-derive this row's status from its (possibly now-fixed) files,
        # keeping its originally-detected angle and any extra_files as-is.
        row = self.rows[r]
        status = []
        if row['detection_file'] is None:
            status.append(io.STATUS_MISSING_DETECTION)
        if row['acquisition_file'] is None:
            status.append(io.STATUS_MISSING_ACQUISITION)
        if row['pattern_file'] is None:
            status.append(io.STATUS_MISSING_PATTERN)
        if row.get('extra_files'):
            status.append(io.STATUS_EXTRA_FILES)
        row['status'] = status or [io.STATUS_OK]
        row['excluded'] = status != []
        self.table.item(r, c).setText(os.path.basename(path))
        status_text = ', '.join(io.STATUS_LABELS.get(s, s) for s in row['status'])
        self.table.item(r, 5).setText(status_text)
        # Reflect the new status in the "Include" checkbox too.
        cell = self.table.cellWidget(r, 0)
        checkbox = cell.findChild(qtw.QCheckBox)
        checkbox.blockSignals(True)
        checkbox.setChecked(not row['excluded'])
        checkbox.blockSignals(False)
        self._color_row(r)
        self._update_summary()
