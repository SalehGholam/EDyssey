# -*- coding: utf-8 -*-
"""Shared vertical icon-strip ("ribbon") widget docked to the right of each
tab's canvas - an additional, click-driven entry point for the same actions
already reachable via Ctrl/Shift-click and keyboard modifiers on the canvas
itself (see each tab's on_press/on_click handlers). Existing modifier-key
interactions are left untouched; a ribbon "tool" button only changes what a
*plain* click/drag on the canvas does, by setting `active_tool`, which each
tab's mouse handlers additionally check alongside their existing modifier
checks. No bundled icon set exists in this repo yet (see ui_tabs/logo/, just
the app icon/splash) - buttons use short Unicode glyphs/text instead of
image icons, kept deliberately simple rather than blocking this on artwork.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import pyqtSignal, Qt


@dataclass
class RibbonTool:
    """One ribbon entry.

    id: Stable identifier - the value `RibbonPanel.active_tool` takes when
        this tool is selected (only meaningful for kind='tool').
    label: Short glyph/text shown on the button.
    tooltip: Full description shown on hover.
    kind: 'tool' - checkable, mutually exclusive with every other 'tool' in
        the same panel; selecting it sets `active_tool` and stays pressed
        until another tool (or the same one again, to deselect) is clicked.
        'action' - a momentary click that does not change `active_tool`.
        'separator' - a thin dividing line; every other field is ignored.
    callback: Called with no arguments on click, for kind='action' only.
    """
    id: str
    label: str = ''
    tooltip: str = ''
    kind: str = 'tool'
    callback: Optional[Callable[[], None]] = None


class RibbonPanel(qtw.QWidget):
    """Vertical strip of QToolButtons docked to the right of a tab's canvas.

    At most one 'tool' button is checked at a time; selecting one updates
    `active_tool` and emits toolChanged. Re-clicking the active tool's own
    button deselects it (active_tool becomes None) - a plain click/drag on
    the canvas then falls back to whatever it already did before this panel
    existed (nothing, or matplotlib's own toolbar mode).
    """
    toolChanged = pyqtSignal(object)  # new active_tool id (str), or None

    def __init__(self, tools, parent=None):
        super().__init__(parent)
        self._active_tool = None
        self._tool_buttons = {}
        self.setFixedWidth(64)
        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignTop)

        for tool in tools:
            if tool.kind == 'separator':
                line = qtw.QFrame()
                line.setFrameShape(qtw.QFrame.HLine)
                line.setFrameShadow(qtw.QFrame.Sunken)
                layout.addWidget(line)
                continue

            btn = qtw.QToolButton()
            btn.setText(tool.label)
            btn.setToolTip(tool.tooltip)
            btn.setFixedSize(56, 40)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            layout.addWidget(btn)

            if tool.kind == 'tool':
                btn.setCheckable(True)
                btn.clicked.connect(lambda checked, tid=tool.id: self._on_tool_clicked(tid, checked))
                self._tool_buttons[tool.id] = btn
            elif tool.kind == 'action':
                if tool.callback is not None:
                    btn.clicked.connect(tool.callback)
            else:
                raise ValueError(f"Unknown RibbonTool.kind {tool.kind!r} for tool {tool.id!r}")

        layout.addStretch(1)

    def _on_tool_clicked(self, tool_id, checked):
        # QToolButton has already toggled itself by the time this slot runs
        # (`checked` reflects the click's result), so every *other* tool
        # button is unchecked explicitly here - a plain QButtonGroup would
        # enforce mutual exclusivity but doesn't allow "none selected" once
        # one member has been checked, which re-clicking the active tool
        # needs (see class docstring).
        for tid, btn in self._tool_buttons.items():
            if tid != tool_id:
                btn.setChecked(False)
        self._active_tool = tool_id if checked else None
        self.toolChanged.emit(self._active_tool)

    @property
    def active_tool(self):
        return self._active_tool

    def clear_active_tool(self):
        """Deselect whichever tool button is currently checked, if any -
        for callers that want a tool to stop being "armed" once its action
        has fired (e.g. after Segment Image runs)."""
        if self._active_tool is None:
            return
        btn = self._tool_buttons.get(self._active_tool)
        if btn is not None:
            btn.setChecked(False)
        self._active_tool = None
        self.toolChanged.emit(None)
