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
# workers/ on sys.path (for ui_tabs' own bare worker_* imports) is set up
# in ui_tabs/__init__.py instead - run_worker() below resolves its own
# script paths directly, no sys.path needed here.

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

# A batch worker (worker_nav_img_batch.py/worker_extract_frame_batch.py,
# dispatched via --worker above) runs its own multiprocessing.Pool
# internally - on Windows/frozen, each pool worker re-invokes this same exe
# with `--multiprocessing-fork ...` (multiprocessing's own bootstrap, not
# ours). That has to be caught here too, at the same module level and for
# the same reason as the --worker check above: left to fall through, every
# pool worker would pay the full GUI-stack import cost below before
# multiprocessing.freeze_support() (called from __main__, see the bottom of
# this file) ever gets a chance to intercept it.
if len(sys.argv) > 1 and sys.argv[1] == '--multiprocessing-fork':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(0)

import gc
import logging
import re
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
from ui_tabs import (Tab_Create_NavSignal, Tab_Tracking_CV2,
                     Tab_ROI_on_4D, Tab_SAM2, EditSettingsDialog)
from ui_tabs.logging_utils import install_excepthook, shutdown_qt_log_handler
from ui_tabs.app_theme import AppTheme, THEME_LABELS
from PyQt5.QtGui import QIcon, QCursor, QPixmap
# Sets matplotlib's own style (dark_background/default) to match whichever
# theme was last saved (see AppTheme) - the QApplication-wide stylesheet
# for every other (Qt, not matplotlib) widget is applied once a
# QApplication actually exists, in main()/__main__ below, since
# apply_qapp() needs one to already be running.
AppTheme.instance().apply_qapp()

# Semantic-versioning-shaped, but the 3rd/4th parts are a build timestamp
# rather than counts: MAJOR.MINOR.YYYYMMDD.HHMM (local time, zero-padded) -
# MAJOR.MINOR is bumped by hand for real milestones (this is 2.1: Blob
# Selection segmentation methods - Watershed/K-Means/GMM - plus the
# transposed object-list tables), the date/time update on every user-facing
# change anywhere in the app, so Help > About always reflects how current
# the running build actually is, without needing a separate build/release
# process to compute it. A plain source constant (not computed at run
# time) so it's visible directly in the repo on GitHub, not just at
# runtime. Shown only in the About dialog (Help > About EDyssey).
APP_VERSION = '2.1.20260907.1302'

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
        # Frozen: staged under EDyssey/ui_tabs/logo (see EDyssey.spec's
        # datas=), grouped with the rest of EDyssey's own loose files;
        # dev-mode's git checkout still has ui_tabs/logo/ at the repo root.
        if hasattr(sys, '_MEIPASS'):
            fn_icon = os.path.join(sys._MEIPASS, 'EDyssey', 'ui_tabs', 'logo', 'EDyssey_logo.ico')
        else:
            fn_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'ui_tabs', 'logo', 'EDyssey_logo.ico')
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
        # Applies the whole-app QSS stylesheet (see app_theme.build_stylesheet)
        # to the now-running QApplication - the module-level apply_qapp()
        # call up top only managed matplotlib's own style, since no
        # QApplication existed yet at that point.
        AppTheme.instance().apply_qapp()

    def _build_menu(self):
        menu_file = self.menuBar().addMenu('&File')

        action_duplicate = menu_file.addAction('Duplicate Current Tab')
        action_duplicate.setShortcut('Ctrl+Shift+D')
        action_duplicate.setToolTip('Open a new, empty tab of the same type as the active one')
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
        action_display_size = menu_edit.addAction('Display Preferences...')
        action_display_size.setToolTip('Theme, colormaps, and ribbon/plot text/icon size, across every tab')
        action_display_size.triggered.connect(self.show_display_size_dialog)

        menu_edit.addSeparator()
        self._build_theme_menu(menu_edit)

        menu_help = self.menuBar().addMenu('&Help')
        action_sam2_setup = menu_help.addAction('Set Up SAM2...')
        action_sam2_setup.setToolTip(
            'Install torch/sam2 for the SAM2 tab - not needed with the offline installer')
        action_sam2_setup.triggered.connect(self.show_sam2_setup_dialog)
        action_ffmpeg_setup = menu_help.addAction('Download ffmpeg (for faster video export)...')
        action_ffmpeg_setup.setToolTip(
            'Video export works without this (falls back to GIF) - ffmpeg makes it .mp4 and faster')
        action_ffmpeg_setup.triggered.connect(self.download_ffmpeg)
        menu_help.addSeparator()
        action_about = menu_help.addAction('About EDyssey')
        action_about.triggered.connect(self.show_about_dialog)

    def _build_theme_menu(self, menu_edit):
        """Edit > Color Theme - a checkable submenu, one radio-style entry
        per AppTheme.PALETTES key, so the current theme is always visible
        (checked) right in the menu bar, not just buried in the Display
        Size dialog's own combo (which still exists too, and stays in sync
        with this - both write through the same AppTheme.set_theme)."""
        menu_theme = menu_edit.addMenu('Color Theme')
        group = qtw.QActionGroup(self)
        group.setExclusive(True)
        self._theme_actions = {}
        for key, label in THEME_LABELS.items():
            action = menu_theme.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == AppTheme.instance().name)
            action.triggered.connect(lambda _checked, k=key: AppTheme.instance().set_theme(k))
            group.addAction(action)
            self._theme_actions[key] = action
        # AppTheme.instance() is a QObject with no parent, so it outlives
        # this MainWindow's own construction - keep it self-subscribing
        # here too (same convention as RibbonPanel/FrameFlagBar) so this
        # menu re-checks the right entry if the theme changes from
        # elsewhere (e.g. the Display Size dialog's own combo).
        AppTheme.instance().changed.connect(self._sync_theme_menu)

    def _sync_theme_menu(self):
        current = AppTheme.instance().name
        for key, action in self._theme_actions.items():
            action.setChecked(key == current)

    def duplicate_current_tab(self):
        """Open a new tab of the same type as the currently active one,
        appended at the end and switched to - e.g. a second independent
        "ROI on 4D" tab, for comparing two signals side by side without
        losing the first one's state. The new tab is a fresh instance (its
        own class's __init__, same as the 4 original tabs get at startup),
        but if the current tab has an in-progress analysis (a loaded
        signal, computed images, tracked/segmented objects, ...), that
        state is copied into the new tab too - see each tab class's own
        get_duplicate_state()/apply_duplicate_state() (all 4 implement
        this same pair, so no per-type branching is needed here). A tab
        with nothing loaded yet still duplicates fine - it just starts
        empty, same as before this state-copying existed."""
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
        get_state = getattr(current, 'get_duplicate_state', None)
        if get_state is not None:
            state = get_state()
            if state:
                apply_state = getattr(new_tab, 'apply_duplicate_state', None)
                if apply_state is not None:
                    try:
                        apply_state(state)
                    except Exception:
                        logging.getLogger('EDyssey.app').exception(
                            'Failed to copy analysis state into duplicated tab %s',
                            type(current).__name__)
                        qtw.QMessageBox.warning(self, 'Duplicate Tab',
                            'The new tab was created, but copying its analysis state failed - '
                            'see the log for details. It opened empty instead.')
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

    def show_sam2_setup_dialog(self):
        """Help > Set Up SAM2... - see ui_tabs/sam2_setup_dialog.py. Imported
        lazily so a plain menu click doesn't cost anything at startup."""
        from ui_tabs.sam2_setup_dialog import SAM2SetupDialog
        SAM2SetupDialog(self).exec_()

    def download_ffmpeg(self):
        """Help > Download ffmpeg... - a one-time, explicit action (like Set
        Up SAM2...) rather than trying to intercept the moment of first use,
        since video export runs on background worker threads (created from
        several different tabs) that can't show a Qt confirm/progress
        dialog themselves. video.py's create_clip_*() functions just prefer
        whatever this downloads, falling back to a system PATH ffmpeg or a
        GIF export if neither is available, exactly as before this existed."""
        from PyQt5.QtCore import QThreadPool
        from EDyssey.tracking_utils import asset_fetch
        from ui_tabs.asset_download_dialog import confirm_and_download

        existing = asset_fetch.resolve_ffmpeg_exe()
        if existing is not None:
            qtw.QMessageBox.information(self, 'ffmpeg', f'Already downloaded:\n{existing}')
            return

        def _ready(_path):
            qtw.QMessageBox.information(self, 'ffmpeg', 'Downloaded - fast .mp4 video export is now available.')

        def _failed(error_msg):
            if error_msg:
                qtw.QMessageBox.warning(self, 'Download Failed', f'Could not download ffmpeg:\n{error_msg}')

        confirm_and_download(
            self, QThreadPool.globalInstance(), 'Download ffmpeg',
            'Download ffmpeg (~110 MB) for faster .mp4 video export? Without it, video export '
            'still works, just as a slower/larger .gif instead. An internet connection is needed.',
            asset_fetch.ensure_ffmpeg, _ready, _failed)

    def show_about_dialog(self):
        """Help > About EDyssey - QMessageBox.about() renders this as rich
        text (Qt auto-detects the HTML), and its label has clickable-link
        support on by default, so the mailto:/https: links below open the
        user's mail client/browser directly. Links get an explicit light
        blue (Qt's default link blue is too dark to read against this
        dialog's own dark/black background - inherited from the app's
        overall dark palette)."""
        link = 'style="color: #6db3ff;"'
        qtw.QMessageBox.about(self, 'About EDyssey',
            '<h3>EDyssey</h3>'
            f'<p>Version {APP_VERSION}</p>'
            '<p>4D-STEM and 4D-STEM Tomography analysis toolkit.</p>'
            '<p>Developed by Saleh Gholam at the EMAT group, University of Antwerp.<br>'
            f'Contact: <a href="mailto:saleh.gholam@uantwerpen.be" {link}>saleh.gholam@uantwerpen.be</a><br>'
            f'Software: <a href="https://github.com/SalehGholam/EDyssey" {link}>'
            'github.com/SalehGholam/EDyssey</a><br>'
            f'Paper: <a href="https://arxiv.org/abs/2602.09768" {link}>arxiv.org/abs/2602.09768</a></p>'
            '<p>&copy; 2024&ndash;2026 Saleh Gholam. Released as open-source software - '
            'see the GitHub repository for license terms.</p>'
            '<p style="font-size: small; color: gray;">Built with NumPy, PyQt5, HyperSpy, '
            'Dask, OpenCV (opencv-contrib-python), Matplotlib, matplotlib-scalebar, '
            'scikit-image, PyWavelets, SciPy, pandas, Pillow, tifffile, h5py, tqdm, PyTorch, '
            "and Meta AI's Segment Anything 2 (SAM2), plus eventem/pacbed for .tpx3 loading - "
            'each remains the property of its respective authors under its own '
            'open-source license.</p>')

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
    # Belt-and-suspenders alongside the --multiprocessing-fork check above
    # (which already handles the frozen-build spawn-bootstrap case before
    # any GUI imports happen) - a cheap no-op here otherwise, per the
    # standard PyInstaller + multiprocessing guidance.
    import multiprocessing
    multiprocessing.freeze_support()

    qtw.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    qtw.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    qtw.QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = qtw.QApplication([])

    # Shown while MainWindow() builds its four tabs (matplotlib canvases
    # included) below - the heavy hyperspy/dask/etc. imports above already
    # happened before this point (Python runs all module-level imports
    # before __main__ starts), so this covers UI construction time, not
    # library import time, but that's still a real few-second gap with
    # nothing else on screen otherwise.
    if hasattr(sys, '_MEIPASS'):
        fn_splash = os.path.join(sys._MEIPASS, 'EDyssey', 'ui_tabs', 'logo', 'EDyssey_logo.png')
    else:
        fn_splash = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'ui_tabs', 'logo', 'EDyssey_logo.png')
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
