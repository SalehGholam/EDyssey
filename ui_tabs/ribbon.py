# -*- coding: utf-8 -*-
"""Shared vertical icon-strip ("ribbon") widget docked to the right of each
tab's canvas - an additional entry point for interacting with the plot
itself (pan/zoom/select/point-placement), alongside the Ctrl/Shift-click
modifiers already used for the same actions on the canvas (see each tab's
on_press/on_click handlers). It deliberately does NOT duplicate buttons that
already exist in each tab's left panel (Track, Save, Segment, ...) - only
things that act directly on the subplot/canvas belong here.

Icons: Pan/Zoom/Home reuse matplotlib's own bundled toolbar icons (the same
images NavigationToolbar2QT itself shows on the toolbar strip below each
canvas), so they read as the same action in both places. The handful of
tools with no matplotlib equivalent (Select ROI, Add point, Remove point)
are drawn on the fly with QPainter instead - this repo has no bundled icon
set of its own (see ui_tabs/logo/, just the app icon/splash) to draw from.
"""
import os
from dataclasses import dataclass
from typing import Callable, Optional
import matplotlib
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QSize, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QColor

_MPL_IMAGE_DIR = os.path.join(matplotlib.get_data_path(), 'images')
_MPL_ICON_FILES = {
    'pan': 'move.png',
    'zoom': 'zoom_to_rect.png',
    'home': 'home.png',
}
_ICON_COLOR = QColor('#f0f0f0')  # matches the app's light-on-dark theme
_ICON_SIZE = 26


def _mpl_icon(key):
    path = os.path.join(_MPL_IMAGE_DIR, _MPL_ICON_FILES[key])
    return QIcon(path) if os.path.isfile(path) else QIcon()


def _drawn_icon(kind, size):
    """Render a small QPainter-drawn icon for a tool with no matplotlib
    equivalent - a light-colored glyph on a transparent background, so it
    sits directly on the (dark) QToolButton surface like a normal icon."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(_ICON_COLOR)
    pen.setWidthF(1.8)
    painter.setPen(pen)
    margin = size * 0.18
    cx = cy = size / 2

    if kind == 'select_roi':
        # A dashed selection-marquee rectangle.
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(QRectF(margin, margin, size - 2 * margin, size - 2 * margin))
    elif kind == 'add_point':
        # A target/crosshair - a circle with a small "+" at its center.
        r = size / 2 - margin
        painter.drawEllipse(QPointF(cx, cy), r, r)
        d = r * 0.5
        painter.drawLine(QPointF(cx, cy - d), QPointF(cx, cy + d))
        painter.drawLine(QPointF(cx - d, cy), QPointF(cx + d, cy))
    elif kind == 'remove_point':
        # The same target, minus the "+" (a plain ring reads as "undo the
        # last point placed" paired with add_point's filled crosshair).
        r = size / 2 - margin
        painter.drawEllipse(QPointF(cx, cy), r, r)
        d = r * 0.5
        painter.drawLine(QPointF(cx - d, cy), QPointF(cx + d, cy))
    elif kind == 'clear_roi':
        # A selection rectangle crossed out - "remove the current selection".
        painter.drawRect(QRectF(margin, margin, size - 2 * margin, size - 2 * margin))
        painter.drawLine(QPointF(margin, margin), QPointF(size - margin, size - margin))
        painter.drawLine(QPointF(size - margin, margin), QPointF(margin, size - margin))
    elif kind == 'center_recip':
        # Bullseye (ring + filled center dot) - the diffraction pattern's own
        # beam center, used for the reciprocal-space scale rings.
        r = size / 2 - margin
        painter.drawEllipse(QPointF(cx, cy), r, r)
        painter.setBrush(_ICON_COLOR)
        painter.drawEllipse(QPointF(cx, cy), r * 0.22, r * 0.22)
    elif kind == 'center_mask':
        # Concentric double ring + center "+" - the virtual detector's own
        # inner/outer annulus shape, visually distinct from center_recip's
        # plain beam-center bullseye since the two centers can differ.
        r_out = size / 2 - margin
        r_in = r_out * 0.5
        painter.drawEllipse(QPointF(cx, cy), r_out, r_out)
        painter.drawEllipse(QPointF(cx, cy), r_in, r_in)
        d = r_in * 0.6
        painter.drawLine(QPointF(cx, cy - d), QPointF(cx, cy + d))
        painter.drawLine(QPointF(cx - d, cy), QPointF(cx + d, cy))
    else:
        raise ValueError(f'Unknown drawn-icon kind {kind!r}')

    painter.end()
    return QIcon(pixmap)


_DRAWN_ICON_KINDS = {'select_roi', 'add_point', 'remove_point', 'clear_roi',
                      'center_recip', 'center_mask'}


def build_icon(key, size=_ICON_SIZE):
    """QIcon for ribbon icon key `key` - one of _MPL_ICON_FILES' keys
    (reuses matplotlib's bundled toolbar images, always their own native
    resolution - QToolButton.setIconSize scales them regardless) or
    _DRAWN_ICON_KINDS (hand-drawn via QPainter at `size`, see _drawn_icon -
    RibbonPanel.set_icon_size() re-renders these at a new size on demand,
    see the Edit tab's "Ribbon Icon Size" control)."""
    if key in _MPL_ICON_FILES:
        return _mpl_icon(key)
    if key in _DRAWN_ICON_KINDS:
        return _drawn_icon(key, size)
    raise ValueError(f'Unknown ribbon icon key {key!r}')


@dataclass
class RibbonTool:
    """One ribbon entry.

    id: Stable identifier - the value `RibbonPanel.active_tool` takes when
        this tool is selected (only meaningful for kind='tool').
    icon: Key into build_icon() - required for kind in ('tool', 'action').
    tooltip: Full description shown on hover (the button itself shows only
        the icon - see RibbonPanel).
    kind: 'tool' - checkable, mutually exclusive with every other 'tool' in
        the same panel; selecting it sets `active_tool` and stays pressed
        until another tool (or the same one again, to deselect) is clicked.
        'action' - a momentary click that does not change `active_tool`.
        'separator' - a thin dividing line; every other field is ignored.
    callback: Called with no arguments on click, for kind='action' only.
    """
    id: str
    icon: str = ''
    tooltip: str = ''
    kind: str = 'tool'
    callback: Optional[Callable[[], None]] = None


class RibbonPanel(qtw.QWidget):
    """Vertical strip of icon QToolButtons docked to the right of a tab's
    canvas.

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
        self._icon_size = _ICON_SIZE
        # Every non-separator button, with the icon key used to build it -
        # set_icon_size() (see the Edit tab's "Ribbon Icon Size" control)
        # re-renders each one's icon at a new size from this list.
        self._icon_buttons = []
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
            btn.setIcon(build_icon(tool.icon, self._icon_size))
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setToolTip(tool.tooltip)
            layout.addWidget(btn)
            self._icon_buttons.append((btn, tool.icon))

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
        self._apply_icon_size()

    def set_icon_size(self, size):
        """Re-render every button's icon at `size` px and resize the panel/
        buttons to match - the Edit tab's "Ribbon Icon Size" control calls
        this on every RibbonPanel instance whenever it changes."""
        if size == self._icon_size:
            return
        self._icon_size = size
        for btn, icon_key in self._icon_buttons:
            btn.setIcon(build_icon(icon_key, size))
        self._apply_icon_size()

    def _apply_icon_size(self):
        size = self._icon_size
        for btn, _icon_key in self._icon_buttons:
            btn.setIconSize(QSize(size, size))
            btn.setFixedSize(size + 14, size + 10)
        self.setFixedWidth(size + 22)

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
        has fired."""
        if self._active_tool is None:
            return
        btn = self._tool_buttons.get(self._active_tool)
        if btn is not None:
            btn.setChecked(False)
        self._active_tool = None
        self.toolChanged.emit(None)
