# -*- coding: utf-8 -*-
"""Shared base class for the 4 top-level tab widgets (Tab_ROI_on_4D,
Tab_Create_NavSignal, Tab_Tracking_CV2, Tab_SAM2). Each used to hand-copy
the same handful of lines into its own __init__ - a per-tab logger, a
threadpool, and the `_cancelling` guard-flag convention used throughout
this app to suppress the error popups a deliberate Cancel would otherwise
trigger (see each tab's own cancel_running_work()). This just centralizes
that so the four copies can't drift out of sync with each other (as
tab_create_navSignal.py's `_stopping`/`stop_worker` naming had, before
being unified to match the other three here).

Each tab still builds its own layout entirely itself, in its own
init_widget() - besides the one shared piece of boilerplate factored out
below (build_left_panel()), this module owns none of that.
"""
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QThreadPool
from .logging_utils import get_tab_logger


def compute_left_panel_width(base=440, min_width=420, max_width=480, fraction=0.22):
    """Left input panel width (px) for the current primary screen - a fixed
    (non-draggable) value like the `width_userInput` constant every tab used
    to hard-code, but scaled to the display instead of being the same
    literal number on a small laptop screen and a large monitor alike.

    `base` is returned unchanged if no screen can be queried (e.g. running
    headless) - the previous hard-coded behaviour. Otherwise the panel is
    sized to `fraction` of the primary screen's available width, clamped to
    [`min_width`, `max_width`] so it neither eats most of a small screen nor
    goes needlessly wide on a large one.

    The defaults (and `min_width` especially) are sized to comfortably fit
    the widest row across all 4 tabs' panels (long smart-scan/tracker rows
    with several labels+combos+buttons) without clipping - each tab's own
    QScrollArea wrapper (see tab_*.py's init_widget()) only scrolls
    vertically, so anything wider than this budget would otherwise be
    hidden behind the scrollbar rather than reachable at all.
    """
    screen = qtw.QApplication.primaryScreen()
    if screen is None:
        return base
    avail_width = screen.availableGeometry().width()
    if avail_width <= 0:
        return base
    return max(min_width, min(max_width, int(avail_width * fraction)))


def build_left_panel(splitter, width_userInput):
    """Build the scrollable left parameter panel and add it to `splitter` -
    the boilerplate every tab used to hand-copy identically into its own
    init_widget() (a QScrollArea wrapper, sized to `width_userInput` - see
    compute_left_panel_width() - so the panel scrolls vertically instead of
    squeezing every box into whatever height the window happens to have).

    `width_userInput` is padded by the vertical scrollbar's own width so the
    viewport (where everything actually gets laid out) still gets the full
    `width_userInput` this panel's widgets are sized for - otherwise the
    scrollbar itself would eat into that budget and the rightmost widgets in
    each row would get clipped/hidden behind it.

    Returns the QVBoxLayout to add each tab's own boxes into.
    """
    left_widget = qtw.QWidget()
    left_scroll = qtw.QScrollArea()
    left_scroll.setWidget(left_widget)
    left_scroll.setWidgetResizable(True)
    scrollbar_w = qtw.QApplication.style().pixelMetric(qtw.QStyle.PM_ScrollBarExtent)
    left_scroll.setFixedWidth(width_userInput + scrollbar_w)
    left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    splitter.addWidget(left_scroll)
    return qtw.QVBoxLayout(left_widget)


def get_existing_directory(parent, caption, start_dir=''):
    """Like `QFileDialog.getExistingDirectory()`, but shows files (not just
    subfolders) while browsing, instead of hiding them entirely - the
    built-in convenience function always sets ShowDirsOnly, which makes it
    impossible to visually confirm a folder actually holds the files you're
    looking for (e.g. .tpx3/.zspy 4D signals) before selecting it.
    Selection is still restricted to directories; the return value matches
    getExistingDirectory's own contract (the chosen path, or '' if cancelled).
    """
    dialog = qtw.QFileDialog(parent, caption, start_dir)
    dialog.setFileMode(qtw.QFileDialog.Directory)
    dialog.setOption(qtw.QFileDialog.ShowDirsOnly, False)
    if dialog.exec_() == qtw.QFileDialog.Accepted:
        selected = dialog.selectedFiles()
        return selected[0] if selected else ''
    return ''


class TabBase(qtw.QWidget):
    def __init__(self, tab_name, parent=None, own_threadpool=True):
        """
        Args:
            tab_name: Passed to get_tab_logger() - also the name shown in
                the Qt log console for this tab's messages.
            own_threadpool: Most tabs get their own QThreadPool (isolates
                one tab's background work from the others' queue). Pass
                False for a tab that deliberately shares
                QThreadPool.globalInstance() instead (e.g. the navigator
                tab, whose batch nav-image jobs are dispatched via
                QProcess anyway and only use the global pool for small
                one-off clip/frame-export workers) - in that case no
                `self.threadpool` attribute is set here, and the subclass
                is expected to call QThreadPool.globalInstance() itself.
        """
        super().__init__(parent)
        self.logger = get_tab_logger(tab_name)
        if own_threadpool:
            self.threadpool = QThreadPool()
        self._cancelling = False  # set by cancel_running_work(); suppresses error popups it causes

    def cancel_running_work(self):
        """Stop this tab's running background work. The base implementation
        just raises the flag other code already checks against - subclasses
        that actually have work to cancel (QProcess pools, SAM2
        segmentation, etc.) override this, calling super().cancel_running_work()
        first, then killing/discarding whatever they own."""
        self._cancelling = True

    def cleanup(self):
        """Release resources held by this tab. Called by MainWindow's
        closeEvent on every tab, unconditionally - subclasses with real
        resources (running subprocesses, matplotlib figures, log console
        subscriptions) override this, and should still call
        super().cleanup() (currently a no-op, kept for future shared
        cleanup and so every override reads the same way)."""
        pass
