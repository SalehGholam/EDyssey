# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:36:57 2024

@author: Saleh Gholam

"""

import os
import sys
file_path = os.path.abspath(__file__)
# In a PyInstaller-frozen build, __file__ resolves next to the bootloader
# exe, not the actual bundled package tree - sys._MEIPASS is where the
# real EDyssey/ui_tabs/worker_*.py layout lives instead (same pattern the
# window icon below already relies on).
fld_path = getattr(sys, '_MEIPASS', os.path.dirname(file_path))
os.chdir(fld_path)
_path_io_utils = os.path.join(fld_path, r'EDyssey\io_utils')
if _path_io_utils not in sys.path:
    sys.path.append(_path_io_utils)

# Dispatch to a worker_*.py subprocess (see ui_tabs/worker_launch.py /
# worker_dispatch.py) before any of the GUI-only imports below - both to
# avoid paying for PyQt5/matplotlib/the 4 tab classes in every short-lived
# worker subprocess, and because this is what makes worker subprocesses work
# at all in a frozen build (no bundled python.exe to hand a bare .py path
# to, so the frozen exe re-invokes itself with `--worker <name> <args>`
# instead).
if len(sys.argv) > 1 and sys.argv[1] == '--worker':
    from worker_dispatch import run_worker
    run_worker(sys.argv[2], sys.argv[3:])
    # run_worker always calls sys.exit(...) itself; this is unreachable but
    # keeps the module import below from happening if that ever changes.
    raise SystemExit(0)

import gc
import logging
import re
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
from ui_tabs import (Tab_Create_NavSignal, Tab_Tracking_CV2,
                     Tab_ROI_on_4D, Tab_SAM2, EditSettingsDialog)
from ui_tabs.logging_utils import install_excepthook, shutdown_qt_log_handler
from PyQt5.QtGui import QIcon, QCursor, QPixmap
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# Version is a plain "last updated" timestamp (year.month.day.hourminute,
# local time, zero-padded) rather than a semantic x.y.z number - update
# this whenever a user-facing change is made anywhere in the app, so
# Help > About always reflects how current the running build actually is.
# A plain source constant (not computed at build/run time) so it's visible
# directly in the repo on GitHub, not just at runtime. Shown only in the
# About dialog (Help > About EDyssey) - not duplicated elsewhere in the UI.
APP_VERSION = '2026.08.23.1130'

#%% window
class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()

        # Deliberately NOT setting Qt.WA_DeleteOnClose here: verified by
        # hand that it makes Tab_Tracking_CV2 (matplotlib canvas + widget
        # tree) crash the interpreter as soon as a *second* MainWindow is
        # constructed after the first was actually destroyed (reproduced
        # offscreen, bisected to Qt/matplotlib teardown, independent of any
        # of this file's own cleanup code). So a closed/abandoned window
        # still only hides its Qt C++ object rather than destroying it —
        # closeEvent below does the parts of cleanup that ARE safe without
        # forcing destruction (dropping the log-signal connection, clearing
        # thread pools, killing subprocesses).
        self.init_ui()
    
    def init_ui(self):
        self.resize(1000, 800)  # Width, Height in pixels
        self.setWindowTitle("EDyssey")
        self.tabs = qtw.QTabWidget()
        self.tab_roi_on_4D = Tab_ROI_on_4D()
        self.tabs.addTab(self.tab_roi_on_4D, 'ROI on 4D')
        self.tab_create_navSignal = Tab_Create_NavSignal()
        self.tabs.addTab(self.tab_create_navSignal, 'Navigator')
        self.tab_tracking_cv2 = Tab_Tracking_CV2()
        self.tabs.addTab(self.tab_tracking_cv2, 'ROI Tracker')
        self.tab_sam2 = Tab_SAM2()
        self.tabs.addTab(self.tab_sam2, 'SAM2 Tracker')
        # Display Size lives in the Edit menu (see show_display_size_dialog),
        # not a tab of its own - created lazily on first use.
        self._display_size_dialog = None

        # Every tab instance that currently exists: the 4 fixed ones above,
        # plus any duplicates opened via the File menu - closeEvent below
        # cleans up all of them, not just the original 4, so a duplicate
        # left open at exit doesn't leak its threadpool/subprocesses/
        # matplotlib figure. self._primary_tabs (a fixed snapshot of just
        # the original 4) is what _on_tab_close_requested refuses to close -
        # closing e.g. self.tab_roi_on_4D itself would leave that attribute
        # pointing at a destroyed widget.
        self._all_tabs = [self.tab_roi_on_4D, self.tab_create_navSignal,
                          self.tab_tracking_cv2, self.tab_sam2]
        self._primary_tabs = set(self._all_tabs)

        self.tabs.setTabsClosable(True)
        # setTabsClosable(True) adds a close ("x") button to every tab that
        # exists right now, and to every one added afterward - the 4
        # original tabs shouldn't offer one (see _on_tab_close_requested,
        # which refuses to close them anyway), so it's stripped back off
        # just for these 4 indices; duplicates added later keep theirs.
        for i in range(self.tabs.count()):
            self.tabs.tabBar().setTabButton(i, qtw.QTabBar.RightSide, None)
            self.tabs.tabBar().setTabButton(i, qtw.QTabBar.LeftSide, None)
        self.tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self._build_menu()

        # In a PyInstaller-frozen build, bundled data files (this icon
        # included) are extracted under sys._MEIPASS, not next to __file__ -
        # sys._MEIPASS doesn't exist at all in a normal (non-frozen) run.
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        fn_icon = os.path.join(base_dir, 'ui_tabs', 'logo', 'EDyssey_logo.ico')
        self.setWindowIcon(QIcon(fn_icon))

        central = qtw.QWidget()
        central_layout = qtw.QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.tabs)

        # Each tab embeds its own log console below its own plot area (see
        # LogConsole in ui_tabs/logging_utils.py) rather than one shared
        # console living here below the whole window - that way the left
        # parameter panel of whichever tab is active can span the full
        # window height instead of being squeezed by a full-width log strip.
        self.setCentralWidget(central)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #f0f0f0;
            }
            QTabWidget::pane { border: 1px solid #555; }
            QTabBar::tab {
                background: #3c3c3c; color: #f0f0f0;
                padding: 5px 12px; border: 1px solid #555;
            }
            QTabBar::tab:selected { background: #555; }
            QGroupBox {
                border: 1px solid #555; margin-top: 8px; color: #f0f0f0;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #f0f0f0; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c; color: #f0f0f0;
                border: 1px solid #555; padding: 2px;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c; color: #f0f0f0;
                selection-background-color: #4a86c8;
            }
            QPushButton {
                background-color: #4a4a4a; color: #f0f0f0;
                border: 1px solid #666; padding: 4px 8px;
            }
            QPushButton:hover { background-color: #5a5a5a; }
            QPushButton:pressed { background-color: #3a3a3a; }
            QPushButton:disabled { background-color: #3a3a3a; color: #777; }
            QCheckBox { color: #f0f0f0; }
            QLabel { color: #f0f0f0; }
            QSlider::groove:horizontal { background: #3c3c3c; height: 4px; }
            QSlider::handle:horizontal {
                background: #888; width: 12px; margin: -4px 0; border-radius: 6px;
            }
            QProgressBar {
                background-color: #3c3c3c; color: #f0f0f0;
                border: 1px solid #555; text-align: center;
            }
            QProgressBar::chunk { background-color: #4a86c8; }
            QListWidget, QTreeWidget, QTableWidget {
                background-color: #2b2b2b; color: #f0f0f0;
                border: 1px solid #555; alternate-background-color: #333;
            }
            QListWidget::item:selected, QTreeWidget::item:selected {
                background-color: #4a86c8;
            }
            QHeaderView::section {
                background-color: #3c3c3c; color: #f0f0f0;
                border: 1px solid #555; padding: 2px;
            }
            QScrollBar:vertical { background: #3c3c3c; width: 12px; }
            QScrollBar::handle:vertical { background: #666; min-height: 20px; }
            QScrollBar:horizontal { background: #3c3c3c; height: 12px; }
            QScrollBar::handle:horizontal { background: #666; min-width: 20px; }
            QStatusBar { background-color: #2b2b2b; color: #888; }
            QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #555; }
            QToolButton {
                background-color: #2b2b2b; color: #f0f0f0;
                border: 1px solid transparent;
            }
            QToolButton:hover { background-color: #3c3c3c; border: 1px solid #555; }
            QToolButton:checked {
                background-color: #4a86c8; border: 1px solid #7fb8ec;
            }
            QToolButton:checked:hover { background-color: #5a96d8; border: 1px solid #8fc8fc; }
        """)

    def _build_menu(self):
        menu_file = self.menuBar().addMenu('&File')

        action_duplicate = menu_file.addAction('Duplicate Current Tab')
        action_duplicate.setShortcut('Ctrl+Shift+D')
        action_duplicate.setToolTip(
            'Open a new, empty tab of the same type as the currently active one - '
            'e.g. a second "ROI on 4D" tab, for working on two signals side by side')
        action_duplicate.triggered.connect(self.duplicate_current_tab)

        action_close = menu_file.addAction('Close Current Tab')
        action_close.setShortcut('Ctrl+W')
        action_close.triggered.connect(
            lambda: self._on_tab_close_requested(self.tabs.currentIndex()))

        menu_file.addSeparator()
        action_exit = menu_file.addAction('Exit')
        action_exit.setShortcut('Ctrl+Q')
        action_exit.triggered.connect(self.close)

        menu_edit = self.menuBar().addMenu('&Edit')
        action_display_size = menu_edit.addAction('Display Size...')
        action_display_size.setToolTip(
            'Adjust the top ribbon text size, ribbon icon size, and plot text size, '
            'live, across every tab')
        action_display_size.triggered.connect(self.show_display_size_dialog)

        menu_help = self.menuBar().addMenu('&Help')
        action_about = menu_help.addAction('About EDyssey')
        action_about.triggered.connect(self.show_about_dialog)

    def duplicate_current_tab(self):
        """Open a new, empty tab of the same type as the currently active
        one, appended at the end and switched to - e.g. a second
        independent "ROI on 4D" tab, for comparing two signals side by side
        without losing the first one's state. The new tab is a fresh
        instance (its own class's __init__, same as the 4 original tabs
        get at startup), not a copy of the current tab's loaded data."""
        current = self.tabs.currentWidget()
        if current is None:
            return
        base_label = re.sub(r' \(\d+\)$', '', self.tabs.tabText(self.tabs.currentIndex()))
        try:
            new_tab = type(current)()
        except Exception:
            logging.getLogger('EDyssey.app').exception(
                'Failed to duplicate tab %s', type(current).__name__)
            qtw.QMessageBox.critical(self, 'Duplicate Failed',
                f'Could not create a new {base_label} tab - see the log for details.')
            return
        existing = sum(1 for i in range(self.tabs.count())
                      if re.sub(r' \(\d+\)$', '', self.tabs.tabText(i)) == base_label)
        index = self.tabs.addTab(new_tab, f'{base_label} ({existing + 1})')
        self._all_tabs.append(new_tab)
        self.tabs.setCurrentIndex(index)

    def _on_tab_close_requested(self, index):
        """Slot for the tab bar's close ('x') button and "Close Current
        Tab" - refuses to close any of the 4 original tabs (see
        self._primary_tabs), since self.tab_roi_on_4D etc. would then point
        at a destroyed widget; only tabs opened via duplicate_current_tab
        can actually be closed."""
        if index < 0 or index >= self.tabs.count():
            return
        widget = self.tabs.widget(index)
        if widget in self._primary_tabs:
            qtw.QMessageBox.information(self, 'Cannot Close',
                'The original tabs can\'t be closed - use "Duplicate Current Tab" '
                '(File menu, Ctrl+Shift+D) first if you want a closable copy.')
            return
        try:
            widget.cleanup()
        except Exception:
            logging.getLogger('EDyssey.app').exception(
                'Error cleaning up %s on tab close', type(widget).__name__)
        self.tabs.removeTab(index)
        if widget in self._all_tabs:
            self._all_tabs.remove(widget)
        widget.deleteLater()

    def show_about_dialog(self):
        qtw.QMessageBox.about(self, 'About EDyssey',
            f'EDyssey\nVersion {APP_VERSION}\n\n'
            '4D-STEM and 4D-STEM Tomography analysis toolkit.')

    def show_display_size_dialog(self):
        """Open (or re-raise, if already open) the singleton Display Size
        dialog - Edit menu > Display Size... Non-modal, so it can stay open
        while the user switches tabs to see the live effect."""
        if self._display_size_dialog is None:
            self._display_size_dialog = EditSettingsDialog(self)
        self._display_size_dialog.show()
        self._display_size_dialog.raise_()
        self._display_size_dialog.activateWindow()

    def closeEvent(self, event):
        if self._display_size_dialog is not None:
            self._display_size_dialog.close()
        for tab in self._all_tabs:
            try:
                tab.cleanup()
            except Exception:
                logging.getLogger('EDyssey.app').exception(
                    'Error cleaning up %s on close', type(tab).__name__)

        # Must happen here, while the QApplication/Qt objects are still
        # fully alive - see shutdown_qt_log_handler's docstring for why
        # leaving this to logging's own atexit-registered shutdown() prints
        # an "Exception ignored in atexit callback" RuntimeError on every exit.
        shutdown_qt_log_handler()

        gc.collect()
        event.accept()

if __name__ == "__main__":
    # Must be set before the QApplication is constructed. Without these,
    # Qt falls back to raw-pixel rendering on a high-DPI (e.g. 4K) display -
    # every widget (including every setFixedSize/setFixedWidth value used
    # throughout the tabs) is then sized in *physical* pixels instead of
    # *logical* ones, which is why it looked either tiny or - if Windows
    # compensated with its own bitmap stretching instead - blurry with
    # widgets overlapping. Enabling this makes Qt do the scaling itself, so
    # every existing fixed pixel size scales consistently with the display.
    qtw.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    qtw.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Without this, a window dragged between two monitors with different
    # (non-integer-ratio) scale factors - e.g. a 100% laptop screen and a
    # 150%/200% external 4K one - gets Qt's *rounded* scale factor per
    # monitor rather than the exact one, which is what actually produces
    # visibly wrong/inconsistent widget sizes and overlap right after such
    # a move. PassThrough uses the real factor instead of rounding it.
    qtw.QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = qtw.QApplication([])

    # Shown while MainWindow() builds its four tabs (matplotlib canvases
    # included) below - the heavy hyperspy/dask/etc. imports above already
    # happened before this point (Python runs all module-level imports
    # before __main__ starts), so this covers UI construction time, not
    # library import time, but that's still a real few-second gap with
    # nothing else on screen otherwise.
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    fn_splash = os.path.join(base_dir, 'ui_tabs', 'logo', 'EDyssey_logo.png')
    splash_pixmap = QPixmap(fn_splash).scaledToWidth(420, Qt.SmoothTransformation)
    splash = qtw.QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    # Must come after the QApplication is constructed - get_qt_log_handler()
    # instantiates a QObject with signals, which needs QApplication to
    # already exist for cross-thread delivery to work (see its docstring).
    # Without this call, install_excepthook (imported above) was dead code:
    # any uncaught exception during startup vanished silently instead of
    # reaching logs/app.log - the only trace of a crash when run via
    # `pythonw` (no console attached there, unlike a frozen build - see
    # EDyssey.spec's console=True).
    install_excepthook()
    window = MainWindow()

    # A fresh top-level window's default position, as Qt/Windows computes it
    # at construction time, can land outside every screen's visible bounds
    # on some multi-monitor / high-DPI combinations (a known rough edge when
    # AA_EnableHighDpiScaling meets mismatched per-monitor scale factors) -
    # the process keeps running and its taskbar entry stays put, but the
    # window itself is simply never visible anywhere to click on. Force it
    # onto whichever screen the mouse is currently on (falling back to the
    # primary screen), sized/centered within that screen's available area.
    screen = qtw.QApplication.screenAt(QCursor.pos()) or qtw.QApplication.primaryScreen()
    if screen is not None:
        geo = screen.availableGeometry()
        w = min(window.width(), geo.width())
        h = min(window.height(), geo.height())
        window.resize(w, h)
        window.move(geo.x() + (geo.width() - w) // 2, geo.y() + (geo.height() - h) // 2)

    window.show()
    window.raise_()
    window.activateWindow()
    splash.finish(window)

    if getattr(sys, 'frozen', False):
        # EDyssey.spec's console=True gives a frozen build a real console
        # window (raw prints, torch/CUDA's own stderr chatter, tracebacks
        # that don't reach the Qt log console) - push it behind the main
        # window so it doesn't steal focus/cover the app on startup. Still
        # reachable via the taskbar or Alt+Tab to check. GetConsoleWindow()
        # returns 0 if there's no console for some reason - SetWindowPos
        # with hwnd 0 would be a no-op anyway, but skip explicitly rather
        # than rely on that.
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW('EDyssey Console (diagnostic output)')
        console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if console_hwnd:
            HWND_BOTTOM, SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 1, 0x0001, 0x0002, 0x0010
            ctypes.windll.user32.SetWindowPos(
                console_hwnd, HWND_BOTTOM, 0, 0, 0, 0,
                SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)

    app.exec_()
