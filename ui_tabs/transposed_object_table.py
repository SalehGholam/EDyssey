# -*- coding: utf-8 -*-
"""A QTableWidget that lays the object list out transposed from the usual
QTreeWidget arrangement: property NAMES run down the fixed first column
(Qt's own vertical header - "Use"/"Idx"/"End"/... - rather than across the
top), and each tracked object is one COLUMN instead of one row, so adding an
object adds a column. Shared by Tab_Tracking_CV2 and Tab_SAM2's own
tree_objects (still named that in both, despite no longer literally being a
QTreeWidget - renaming every one of the many existing references in both
tabs wasn't worth it for a name that was already a bit generic).

TransposedObjectTable/_TransposedColumnItem deliberately mimic
QTreeWidget's/QTreeWidgetItem's own method names and (int-column-indexed)
signatures - topLevelItem()/addTopLevelItem()/takeTopLevelItem()/
indexOfTopLevelItem()/selectedItems()/clear(), and item.text(row)/
setText(row, val)/checkState(row)/setCheckState(row, val)/setIcon(row,
val)/data(row, role)/setData(row, role, val) - so the many existing
per-object call sites in both tabs (add_item_tree, duplicate_row,
delete_row, on_item_check_changed, _refresh_ref_combos, toggle_tree_icon,
...), all originally written against a real QTreeWidgetItem's identical
API, kept working with at most a one-line change (constructing the item via
`self.tree_objects.addTopLevelItem()` instead of a standalone
`qtw.QTreeWidgetItem()` + a separate `addTopLevelItem(item)` call) once the
table itself was transposed - not a general-purpose abstraction, just this
specific compatibility surface.
"""
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt


class _TransposedColumnItem:
    """One object's column, proxied to look like a QTreeWidgetItem - see
    module docstring. `row` arguments below are plain ints (the same
    `cols[...]` values already used everywhere - e.g. cols['idx'] - since
    TransposedObjectTable's own rows are declared in that exact order), not
    string keys, so item.text(cols['idx']) etc. reads identically to the
    original QTreeWidgetItem-based code.

    Wraps a real QTableWidgetItem placed in the table's own row 0 purely so
    .column() - a genuine Qt method, automatically kept correct by Qt
    itself as other columns are inserted/removed - gives this proxy's own
    CURRENT column on every access, never a stale cached index. That's the
    same guarantee a real QTreeWidgetItem's own identity gave the original
    per-row-object code (e.g. indexOfTopLevelItem(item), looked up fresh at
    call time rather than captured once, so a closure over `item` stays
    correct even after some other row/column was deleted in between)."""

    def __init__(self, table, anchor_item):
        self._table = table
        self._anchor = anchor_item

    @property
    def col(self):
        return self._anchor.column()

    def _cell(self, row, create=True):
        col = self.col
        item = self._table.item(row, col)
        if item is None and create:
            item = qtw.QTableWidgetItem()
            self._table.setItem(row, col, item)
        return item

    def text(self, row):
        item = self._cell(row, create=False)
        return item.text() if item is not None else ''

    def setText(self, row, value):
        self._cell(row).setText(str(value))

    def checkState(self, row):
        item = self._cell(row, create=False)
        return item.checkState() if item is not None else Qt.Unchecked

    def setCheckState(self, row, state):
        item = self._cell(row)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(state)

    def setIcon(self, row, icon):
        self._cell(row).setIcon(icon)

    def setData(self, row, role, value):
        self._cell(row).setData(role, value)

    def data(self, row, role):
        item = self._cell(row, create=False)
        return item.data(role) if item is not None else None

    def setSelected(self, selected):
        if selected:
            self._table.selectColumn(self.col)
        else:
            self._table.clearSelection()

    def setFlags(self, flags):
        # QTreeWidgetItem-style "whole item" flags (e.g. the
        # Qt.ItemIsUserCheckable the original add_item_tree() set once,
        # up front, before any cell existed) don't map cleanly onto a
        # per-cell QTableWidgetItem model - setCheckState() above already
        # applies that specific flag to its own cell directly wherever
        # it's actually needed, which is the only flag the real call
        # sites relied on, so this is just a safe no-op otherwise.
        pass


class TransposedObjectTable(qtw.QTableWidget):
    """See module docstring. `row_keys` (e.g. ['use', 'idx', 'init', ...])
    fixes the row order (and is what `cols[...]` in each tab's own
    add_item_tree already builds an int-lookup dict from); `row_labels` are
    the matching vertical-header captions shown in column 0."""

    def __init__(self, row_keys, row_labels, parent=None):
        super().__init__(parent)
        self.setRowCount(len(row_keys))
        self.setVerticalHeaderLabels(row_labels)
        self.horizontalHeader().setVisible(False)
        self.setSelectionBehavior(qtw.QAbstractItemView.SelectColumns)
        self.setSelectionMode(qtw.QAbstractItemView.SingleSelection)
        self.setEditTriggers(qtw.QAbstractItemView.NoEditTriggers)

    def clear(self):
        self.setColumnCount(0)

    def topLevelItemCount(self):
        return self.columnCount()

    def topLevelItem(self, col):
        if col < 0 or col >= self.columnCount():
            return None
        anchor = self.item(0, col)
        if anchor is None:
            anchor = qtw.QTableWidgetItem()
            self.setItem(0, col, anchor)
        return _TransposedColumnItem(self, anchor)

    def addTopLevelItem(self, width=90):
        """Unlike QTreeWidget's addTopLevelItem(item) (which takes an
        already-built item), this one BUILDS the new column itself and
        returns its item proxy directly - a QTableWidgetItem can't exist
        detached from the table the way a QTreeWidgetItem can, so there's
        no equivalent "construct it standalone, add it after" two-step."""
        col = self.columnCount()
        self.insertColumn(col)
        self.setColumnWidth(col, width)
        anchor = qtw.QTableWidgetItem()
        self.setItem(0, col, anchor)
        return _TransposedColumnItem(self, anchor)

    def takeTopLevelItem(self, col):
        self.removeColumn(col)

    def indexOfTopLevelItem(self, item):
        return item.col

    def selectedItems(self):
        cols = sorted({index.column() for index in self.selectedIndexes()})
        return [self.topLevelItem(c) for c in cols]

    def currentItem(self):
        col = self.currentColumn()
        return self.topLevelItem(col) if col >= 0 else None

    def setCurrentItem(self, item):
        self.setCurrentCell(0, item.col)

    def setItemWidget(self, item, row, widget):
        self.setCellWidget(row, item.col, widget)

    def itemWidget(self, item, row):
        return self.cellWidget(row, item.col)
