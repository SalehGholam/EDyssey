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
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional
import matplotlib
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QSize, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QColor
from .app_theme import AppTheme

_MPL_IMAGE_DIR = os.path.join(matplotlib.get_data_path(), 'images')
_MPL_ICON_FILES = {
    'pan': 'move.png',
    'zoom': 'zoom_to_rect.png',
    'home': 'home.png',
}
_ICON_SIZE = 26


def _icon_color():
    """The current theme's own icon color (see app_theme.py) - a function,
    not a module-level constant, so every drawn icon (re-)built after a
    theme change picks up the new color instead of the one active when
    this module was first imported."""
    return QColor(AppTheme.instance().color('icon'))


def _mpl_icon(key):
    path = os.path.join(_MPL_IMAGE_DIR, _MPL_ICON_FILES[key])
    return QIcon(path) if os.path.isfile(path) else QIcon()


def _drawn_icon(kind, size):
    """Render a small QPainter-drawn icon for a tool with no matplotlib
    equivalent - a light-colored glyph on a transparent background, so it
    sits directly on the (dark) QToolButton surface like a normal icon.

    Rendered at the app's current device pixel ratio (>1 on a HiDPI/scaled
    monitor - e.g. 2.0 at 200% Windows scaling) so it stays crisp instead
    of being upscaled/blurry there - setting the QPixmap's own device pixel
    ratio *before* painting on it makes QPainter operate in logical (size x
    size) coordinates regardless, so nothing below needs to know about it."""
    app = qtw.QApplication.instance()
    dpr = app.devicePixelRatio() if app is not None else 1.0
    pixmap = QPixmap(round(size * dpr), round(size * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(_icon_color())
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
    elif kind in ('remove_point', 'undo', 'redo'):
        # A counter-clockwise "undo" arrow ('remove_point'/'undo' - both
        # read as "step back to the previous state", just in different
        # contexts) - reads unambiguously as "undo/remove the last point
        # placed" for remove_point specifically. A previous design (a
        # plain ring with one horizontal line through it, echoing
        # add_point's "+" minus its vertical stroke) looked too much like
        # a bare minus sign, easily mistaken for "add a NEGATIVE point"
        # instead of "remove the last point" - add_point's own left/right-
        # click already covers positive/negative, so this icon must read
        # as neither. 'redo' is this same arrow's exact horizontal mirror
        # (via a painter transform, not separately re-derived geometry, so
        # the two are guaranteed to read as consistent opposites) - "step
        # forward again" through whatever undo just stepped back from.
        if kind == 'redo':
            painter.save()
            painter.translate(size, 0)
            painter.scale(-1, 1)
        r = size / 2 - margin
        rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        # Qt angles are in 1/16th of a degree, counter-clockwise from the
        # 3-o'clock position - drawn from 40 to 320 degrees, leaving a gap
        # at the bottom-right for the arrowhead below.
        start_deg, span_deg = 40, 280
        painter.drawArc(rect, start_deg * 16, span_deg * 16)
        ang = math.radians(start_deg)
        tip = QPointF(cx + r * math.cos(ang), cy - r * math.sin(ang))
        tangent = ang + math.pi / 2  # direction of travel at the arc's start
        head_len = r * 0.6
        head_angle = math.radians(28)
        p1 = QPointF(tip.x() - head_len * math.cos(tangent - head_angle),
                     tip.y() + head_len * math.sin(tangent - head_angle))
        p2 = QPointF(tip.x() - head_len * math.cos(tangent + head_angle),
                     tip.y() + head_len * math.sin(tangent + head_angle))
        painter.drawLine(tip, p1)
        painter.drawLine(tip, p2)
        if kind == 'redo':
            painter.restore()
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
        painter.setBrush(_icon_color())
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
    elif kind == 'hide_mask':
        # center_mask's same concentric-ring annulus, struck through - "hide
        # the virtual detector overlay" (a checkable toggle, not a one-shot
        # action - see RibbonTool.kind == 'toggle').
        r_out = size / 2 - margin
        r_in = r_out * 0.5
        painter.drawEllipse(QPointF(cx, cy), r_out, r_out)
        painter.drawEllipse(QPointF(cx, cy), r_in, r_in)
        painter.drawLine(QPointF(margin, margin), QPointF(size - margin, size - margin))
    elif kind == 'paint_in':
        # A solid filled square - "paint pixels IN (add to the mask)",
        # paired with paint_out's hollow counterpart below (mask_edit_dialog
        # ribbon only).
        rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
        painter.setBrush(_icon_color())
        painter.drawRect(rect)
    elif kind == 'paint_out':
        # paint_in's hollow counterpart - "paint pixels OUT (remove from
        # the mask)".
        rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
        painter.drawRect(rect)
    elif kind == 'rect_in':
        # select_roi's own dashed marquee, plus a small "+" at its center -
        # "paint a rectangular region IN", paired with rect_out's "-"
        # counterpart below (mask_edit_dialog ribbon only).
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(QRectF(margin, margin, size - 2 * margin, size - 2 * margin))
        pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)
        d = size * 0.12
        painter.drawLine(QPointF(cx, cy - d), QPointF(cx, cy + d))
        painter.drawLine(QPointF(cx - d, cy), QPointF(cx + d, cy))
    elif kind == 'rect_out':
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(QRectF(margin, margin, size - 2 * margin, size - 2 * margin))
        pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)
        d = size * 0.12
        painter.drawLine(QPointF(cx - d, cy), QPointF(cx + d, cy))
    elif kind == 'help':
        # A plain "?" in a circle - opens this tab's Shortcuts/Controls
        # dialog (see TabBase.show_shortcuts_dialog), which is where the
        # mouse/keyboard hints that used to run underneath each subplot as
        # xlabel text now live, instead of crowding the canvas (adjacent
        # axes' multi-line hints used to visibly run into each other on a
        # narrow window).
        r = size / 2 - margin
        painter.drawEllipse(QPointF(cx, cy), r, r)
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(round(size * 0.5))
        painter.setFont(font)
        painter.drawText(QRectF(0, 0, size, size), Qt.AlignCenter, '?')
    else:
        raise ValueError(f'Unknown drawn-icon kind {kind!r}')

    painter.end()
    return QIcon(pixmap)


_DRAWN_ICON_KINDS = {'select_roi', 'add_point', 'remove_point', 'clear_roi',
                      'center_recip', 'center_mask', 'hide_mask', 'help',
                      'paint_in', 'paint_out', 'rect_in', 'rect_out',
                      'undo', 'redo'}


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
        'toggle' - checkable like 'tool', but independent of it: any number
        of 'toggle' buttons (and the one 'tool' selection) can be pressed at
        once, since these represent persistent on/off display state (e.g.
        "hide virtual detectors") rather than a mutually-exclusive
        interaction mode.
        'separator' - a thin dividing line; every other field is ignored.
    callback: Called with no arguments on click for kind='action'; called
        with the new checked state (bool) for kind='toggle'.
    """
    id: str
    icon: str = ''
    tooltip: str = ''
    kind: str = 'tool'
    callback: Optional[Callable[[], None]] = None


class RibbonPanel(qtw.QWidget):
    """Strip of icon QToolButtons - vertical and docked to the right of a
    tab's canvas by default (`orientation='vertical'`); pass
    `orientation='horizontal'` for a horizontal strip instead (used under
    the canvas in MaskEditDialog, where a tall narrow dock doesn't fit the
    dialog's single-column layout).

    At most one 'tool' button is checked at a time; selecting one updates
    `active_tool` and emits toolChanged. Re-clicking the active tool's own
    button deselects it (active_tool becomes None) - a plain click/drag on
    the canvas then falls back to whatever it already did before this panel
    existed (nothing, or matplotlib's own toolbar mode).
    """
    toolChanged = pyqtSignal(object)  # new active_tool id (str), or None

    def __init__(self, tools, parent=None, orientation='vertical'):
        super().__init__(parent)
        self._active_tool = None
        self._tool_buttons = {}
        # Every non-separator button, any kind - lets a caller reach a
        # specific button after construction (see get_button), e.g. to
        # enable/disable an 'action' button depending on some external
        # state (MaskEditDialog's Undo/Redo, greyed out while their own
        # history stacks are empty).
        self._buttons_by_id = {}
        self._icon_size = _ICON_SIZE
        self._orientation = orientation
        # Every non-separator button, with the icon key used to build it -
        # set_icon_size() (see the Edit tab's "Ribbon Icon Size" control)
        # re-renders each one's icon at a new size from this list.
        self._icon_buttons = []
        horizontal = orientation == 'horizontal'
        layout = qtw.QHBoxLayout(self) if horizontal else qtw.QVBoxLayout(self)
        if horizontal:
            layout.setContentsMargins(4, 2, 4, 2)
        else:
            layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignLeft if horizontal else Qt.AlignTop)

        for tool in tools:
            if tool.kind == 'separator':
                line = qtw.QFrame()
                line.setFrameShape(qtw.QFrame.VLine if horizontal else qtw.QFrame.HLine)
                line.setFrameShadow(qtw.QFrame.Sunken)
                layout.addWidget(line)
                continue

            btn = qtw.QToolButton()
            btn.setIcon(build_icon(tool.icon, self._icon_size))
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setToolTip(tool.tooltip)
            layout.addWidget(btn)
            self._icon_buttons.append((btn, tool.icon))
            self._buttons_by_id[tool.id] = btn

            if tool.kind == 'tool':
                btn.setCheckable(True)
                btn.clicked.connect(lambda checked, tid=tool.id: self._on_tool_clicked(tid, checked))
                self._tool_buttons[tool.id] = btn
            elif tool.kind == 'action':
                if tool.callback is not None:
                    btn.clicked.connect(tool.callback)
            elif tool.kind == 'toggle':
                btn.setCheckable(True)
                if tool.callback is not None:
                    btn.toggled.connect(tool.callback)
            else:
                raise ValueError(f"Unknown RibbonTool.kind {tool.kind!r} for tool {tool.id!r}")

        layout.addStretch(1)
        self._apply_icon_size()
        # Subscribes itself, rather than relying on every one of this
        # class's many construction sites (each main tab, MaskEditDialog,
        # BlobSegmentationDialog, ...) to remember to wire this up - Qt
        # auto-disconnects this once `self` is destroyed, so a closed
        # dialog's own RibbonPanel doesn't linger as a stale subscriber.
        AppTheme.instance().changed.connect(self.refresh_theme)

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

    def refresh_theme(self):
        """Re-render every button's icon at the SAME size, picking up
        AppTheme's current icon color (see _icon_color/build_icon) -
        connected to AppTheme.changed above. The QToolButton chrome around
        each icon (background/border/checked-state colors) already
        re-colors itself for free via the QApplication-wide stylesheet
        AppTheme.apply_qapp() sets; only the hand-drawn icon glyphs
        themselves need rebuilding here."""
        for btn, icon_key in self._icon_buttons:
            btn.setIcon(build_icon(icon_key, self._icon_size))

    def _apply_icon_size(self):
        size = self._icon_size
        for btn, _icon_key in self._icon_buttons:
            btn.setIconSize(QSize(size, size))
            btn.setFixedSize(size + 14, size + 10)
        if self._orientation == 'horizontal':
            self.setFixedHeight(size + 16)
        else:
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

    def get_button(self, tool_id):
        """The QToolButton for `tool_id` (any kind - 'tool', 'action', or
        'toggle') - None if there's no such id (e.g. a typo, or a
        separator, which has no id at all)."""
        return self._buttons_by_id.get(tool_id)

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
