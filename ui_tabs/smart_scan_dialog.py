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
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QColor, QDrag
import py4DTomo.io_utils as io

_COLS = ['Include', 'Angle', 'Detection File', 'Acquisition File', 'Pattern File', 'Status']
_FILE_COLS = {2: 'detection_file', 3: 'acquisition_file', 4: 'pattern_file'}
_ANGLE_COL = 1

_BAD_ROW_COLOR = QColor('#5a2d2d')
_OK_ROW_COLOR = QColor('#2b2b2b')


class _FileCellTable(qtw.QTableWidget):
    """QTableWidget where dragging one file cell (Detection/Acquisition/
    Pattern column) onto another swaps their file assignments.

    Qt's default drag-and-drop for QTableWidget moves/copies
    QTableWidgetItem text directly, which doesn't understand this table's
    semantics (every cell mirrors a field in the dialog's own self.rows
    list, not freestanding text) - so both ends of the drag are handled
    manually here and delegated back to the dialog via `on_swap(src_row,
    src_col, dst_row, dst_col)`.
    """

    def __init__(self, on_swap, parent=None):
        super().__init__(parent)
        self._on_swap = on_swap
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

    def startDrag(self, supportedActions):
        """Begin dragging the current file cell, encoding its (row, column) as mime text."""
        item = self.currentItem()
        if item is None or item.column() not in _FILE_COLS:
            return
        mime = QMimeData()
        mime.setText(f'{item.row()},{item.column()}')
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Only accept the drag while hovering another file-column cell."""
        item = self.itemAt(event.pos())
        if event.mimeData().hasText() and item is not None and item.column() in _FILE_COLS:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Decode the dragged (row, column) from mime text and delegate the
        swap against the drop target's cell to on_swap."""
        text = event.mimeData().text()
        try:
            src_row, src_col = (int(x) for x in text.split(','))
        except ValueError:
            event.ignore()
            return
        target_item = self.itemAt(event.pos())
        if target_item is None or target_item.column() not in _FILE_COLS:
            event.ignore()
            return
        dst_row, dst_col = target_item.row(), target_item.column()
        if (src_row, src_col) == (dst_row, dst_col):
            event.ignore()
            return
        event.acceptProposedAction()
        self._on_swap(src_row, src_col, dst_row, dst_col)


class SmartScanCheckDialog(qtw.QDialog):
    """Editable table of every tilt angle found in `data_dir`, matched to
    its detection/acquisition data files and pattern file. `exec_()` returns
    QDialog.Accepted once the user confirms; `self.rows` then holds the
    reviewed match table (see `py4DTomo.io_utils.smart_scan.match_tilt_files`
    for the row shape) - pass it to `resolve_smart_scan_files` to get the
    ordered per-role file list to actually use."""

    def __init__(self, parent, data_dir, data_ext, pattern_dir=None, detection_dir=None, rows=None):
        super().__init__(parent)
        self.setWindowTitle('Check Smart-Scan Files')
        self.resize(1000, 520)
        self.data_dir = data_dir
        self.data_ext = data_ext
        self.pattern_dir = pattern_dir
        self.detection_dir = detection_dir
        self.rows = rows if rows is not None else []

        layout = qtw.QVBoxLayout(self)

        info = qtw.QLabel(
            'One row per detected tilt angle. Rows with a problem are unchecked and '
            'highlighted. Double-click the Angle cell to correct it, or a file cell to '
            'browse for a replacement. Drag one file cell onto another to swap them. '
            'Right-click a row/cell for more options (clear, swap Detection/Acquisition, '
            'shift a column up/down to close a gap left by a stray file).')
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = _FileCellTable(self._on_cell_swap)
        self.table.setColumnCount(len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(qtw.QAbstractItemView.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.table)

        row_buttons = qtw.QHBoxLayout()
        layout.addLayout(row_buttons)
        self.button_rescan = qtw.QPushButton('Rescan Folder')
        self.button_rescan.clicked.connect(self.rescan)
        row_buttons.addWidget(self.button_rescan)
        self.button_select_all = qtw.QPushButton('Select All')
        self.button_select_all.clicked.connect(lambda: self._set_all_included(True))
        row_buttons.addWidget(self.button_select_all)
        self.button_select_none = qtw.QPushButton('Select None')
        self.button_select_none.clicked.connect(lambda: self._set_all_included(False))
        row_buttons.addWidget(self.button_select_none)
        self.button_swap_all = qtw.QPushButton('Swap Detection ↔ Acquisition (All Rows)')
        self.button_swap_all.setToolTip(
            'The auto-match assumes, per tilt angle, the alphabetically-first file is '
            'the detection one and the second is the acquisition one - if that\'s '
            'backwards for this whole dataset, swap every row at once instead of one '
            'by one.')
        self.button_swap_all.clicked.connect(self._swap_all_rows)
        row_buttons.addWidget(self.button_swap_all)
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
        """Re-run match_tilt_files over the configured folders and repopulate the table."""
        try:
            self.rows = io.match_tilt_files(self.data_dir, self.data_ext,
                                            pattern_dir=self.pattern_dir,
                                            detection_dir=self.detection_dir)
        except Exception as e:
            qtw.QMessageBox.critical(self, 'Scan Failed', f'Could not scan {self.data_dir}:\n{e}')
            self.rows = []
        self._populate_table()

    def _populate_table(self):
        """Rebuild the whole table (rows, checkboxes, cell contents) from self.rows."""
        self.table.setRowCount(len(self.rows))
        for r in range(len(self.rows)):
            checkbox = qtw.QCheckBox()
            checkbox.stateChanged.connect(lambda state, r=r: self._on_include_toggled(r, state))
            cell = qtw.QWidget()
            cell_layout = qtw.QHBoxLayout(cell)
            cell_layout.addWidget(checkbox)
            cell_layout.setAlignment(Qt.AlignCenter)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(r, 0, cell)
            for c in range(1, len(_COLS)):
                self.table.setItem(r, c, qtw.QTableWidgetItem(''))
            self._refresh_row_display(r)
        self.table.resizeColumnsToContents()
        self._update_summary()

    def _refresh_row_display(self, r):
        """Repaint row `r`'s cells/checkbox/color from self.rows[r] - shared
        by initial population and every edit (angle/file/swap/shift/clear)."""
        row = self.rows[r]
        angle_text = f"{row['angle']:.2f}" if row['angle'] is not None else '?'
        self.table.item(r, _ANGLE_COL).setText(angle_text)
        for col, key in _FILE_COLS.items():
            fn = row.get(key)
            text = os.path.basename(fn) if fn else '(missing)'
            self.table.item(r, col).setText(text)
        status_text = ', '.join(io.STATUS_LABELS.get(s, s) for s in row['status'])
        if row.get('extra_files'):
            status_text += ' (' + ', '.join(os.path.basename(f) for f in row['extra_files']) + ')'
        self.table.item(r, 5).setText(status_text)

        cell = self.table.cellWidget(r, 0)
        checkbox = cell.findChild(qtw.QCheckBox)
        checkbox.blockSignals(True)
        checkbox.setChecked(not row['excluded'])
        checkbox.blockSignals(False)

        self._color_row(r)

    def _recompute_status(self, r):
        """Re-derive row `r`'s 'status'/'excluded' from its current
        detection/acquisition/pattern files - called after any edit that
        changes which files a row points to (browse, clear, swap, shift).
        A missing detection file only counts as a problem for formats where
        one is structurally expected (.tpx3) - see
        io.smart_scan.detection_required."""
        row = self.rows[r]
        status = []
        if row['detection_file'] is None and io.detection_required(self.data_ext):
            status.append(io.STATUS_MISSING_DETECTION)
        if row['acquisition_file'] is None:
            status.append(io.STATUS_MISSING_ACQUISITION)
        if row['pattern_file'] is None:
            status.append(io.STATUS_MISSING_PATTERN)
        if row.get('extra_files'):
            status.append(io.STATUS_EXTRA_FILES)
        row['status'] = status or [io.STATUS_OK]
        row['excluded'] = status != []

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

    def _set_all_included(self, included):
        for r in range(len(self.rows)):
            self.rows[r]['excluded'] = not included
        self._populate_table()

    def _on_cell_double_clicked(self, r, c):
        if c == _ANGLE_COL:
            self._edit_angle(r)
        elif c in _FILE_COLS:
            self._browse_replacement(r, c)

    def _edit_angle(self, r):
        """Prompt to edit row `r`'s angle, then re-sort rows by the new value."""
        row = self.rows[r]
        current = row['angle'] if row['angle'] is not None else 0.0
        value, ok = qtw.QInputDialog.getDouble(
            self, 'Edit Tilt Angle', 'Angle (degrees):', current, -360, 360, 2)
        if not ok:
            return
        row['angle'] = value
        self.rows.sort(key=lambda rw: (rw['angle'] is None, rw['angle']))
        self._populate_table()

    def _browse_replacement(self, r, c):
        """Prompt for a replacement file for row `r`, column `c`, with a file
        filter chosen by which role (pattern/detection/acquisition) it is."""
        key = _FILE_COLS[c]
        if key == 'pattern_file':
            start_dir = self.pattern_dir or self.data_dir
            file_filter = 'Text files (*.txt)'
        elif key == 'detection_file':
            start_dir = self.detection_dir or self.data_dir
            # Detection may be a same-format file (.tpx3, or occasionally
            # .mib/.hspy/.zspy too) or a separate HAADF image - offer both.
            haadf_pattern = ' '.join(f'*{ext}' for ext in io.DETECTION_EXTENSIONS)
            file_filter = (f'Detection files (*{self.data_ext} {haadf_pattern});;'
                           f'Data files (*{self.data_ext});;'
                           f'HAADF images ({haadf_pattern})')
        else:
            start_dir = self.data_dir
            file_filter = f'Data files (*{self.data_ext})'
        path, _ = qtw.QFileDialog.getOpenFileName(
            self, f'Select replacement for {_COLS[c]}', start_dir or '', file_filter)
        if not path:
            return
        self.rows[r][key] = path
        self._recompute_status(r)
        self._refresh_row_display(r)
        self._update_summary()

    def _on_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if item is None:
            return
        r, c = item.row(), item.column()
        menu = qtw.QMenu(self)

        if c in _FILE_COLS:
            key = _FILE_COLS[c]
            action_browse = menu.addAction(f'Browse for {_COLS[c]}...')
            action_browse.triggered.connect(lambda: self._browse_replacement(r, c))
            if self.rows[r].get(key):
                action_clear = menu.addAction(f'Clear {_COLS[c]}')
                action_clear.triggered.connect(lambda: self._clear_cell(r, key))
            menu.addSeparator()
            action_shift_up = menu.addAction(
                f'Remove & Shift {_COLS[c]} Column Up (rows below move up one)')
            action_shift_up.triggered.connect(lambda: self._shift_column(r, key, 'up'))
            action_shift_down = menu.addAction(
                f'Insert Gap & Shift {_COLS[c]} Column Down (rows below move down one)')
            action_shift_down.triggered.connect(lambda: self._shift_column(r, key, 'down'))
            menu.addSeparator()

        action_swap_row = menu.addAction('Swap Detection ↔ Acquisition (This Row)')
        action_swap_row.triggered.connect(lambda: self._swap_row(r))

        menu.exec_(self.table.viewport().mapToGlobal(pos))

    def _clear_cell(self, r, key):
        self.rows[r][key] = None
        self._recompute_status(r)
        self._refresh_row_display(r)
        self._update_summary()

    def _swap_row(self, r):
        """Swap row `r`'s detection/acquisition file assignment."""
        row = self.rows[r]
        row['detection_file'], row['acquisition_file'] = (
            row['acquisition_file'], row['detection_file'])
        self._recompute_status(r)
        self._refresh_row_display(r)
        self._update_summary()

    def _swap_all_rows(self):
        """Swap detection/acquisition file assignment for every row."""
        for row in self.rows:
            row['detection_file'], row['acquisition_file'] = (
                row['acquisition_file'], row['detection_file'])
        for r in range(len(self.rows)):
            self._recompute_status(r)
        self._populate_table()

    def _on_cell_swap(self, src_row, src_col, dst_row, dst_col):
        """A file cell was dragged onto another (same or different column) -
        swap the two cells' file assignments. Dragging within the same
        column is the common case (re-pairing which file belongs to which
        angle, e.g. fixing a run of rows that all got shifted by one);
        dragging across columns (e.g. Detection -> Acquisition) reassigns a
        file's role at the same time."""
        src_key = _FILE_COLS[src_col]
        dst_key = _FILE_COLS[dst_col]
        self.rows[src_row][src_key], self.rows[dst_row][dst_key] = (
            self.rows[dst_row][dst_key], self.rows[src_row][src_key])
        self._recompute_status(src_row)
        if (dst_row, dst_col) != (src_row, src_col):
            self._recompute_status(dst_row)
        self._refresh_row_display(src_row)
        self._refresh_row_display(dst_row)
        self._update_summary()

    def _shift_column(self, r, key, direction):
        """Remove the file at row `r` in column `key` (detection_file/
        acquisition_file/pattern_file) - closing the gap by shifting every
        later row's value in that same column up by one ('up': the common
        case, an extra/misattributed file at `r` pushed everything after it
        one row out of alignment with its angle), or opening a gap by
        shifting every later row's value down by one, discarding the last
        row's old value ('down': fixes the mirror case, where the file that
        belongs at `r` is currently one row too far down)."""
        n = len(self.rows)
        if direction == 'up':
            for i in range(r, n - 1):
                self.rows[i][key] = self.rows[i + 1][key]
            self.rows[n - 1][key] = None
        else:
            for i in range(n - 1, r, -1):
                self.rows[i][key] = self.rows[i - 1][key]
            self.rows[r][key] = None
        for i in range(r, n):
            self._recompute_status(i)
        self._populate_table()
