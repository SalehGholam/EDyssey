# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:36:57 2024

@author: Saleh Gholam

"""

import os
import sys
file_path = os.path.abspath(__file__)
fld_path = os.path.dirname(file_path)
os.chdir(fld_path)
_path_io_utils = os.path.join(fld_path, r'py4DTomo\io_utils')
if _path_io_utils not in sys.path:
    sys.path.append(_path_io_utils)
import gc
import logging
import PyQt5.QtWidgets as qtw
from ui_tabs import (Tab_Create_NavSignal, Tab_Tracking_CV2,
                     Tab_ROI_on_4D, Tab_SAM2)
from ui_tabs.logging_utils import install_excepthook
from PyQt5.QtGui import QIcon
import matplotlib.pyplot as plt
plt.style.use('dark_background')

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
        self.setWindowTitle("5DED Analysis")
        self.tabs = qtw.QTabWidget()
        # self.tab_converter = Tab_Converter()
        # self.tabs.addTab(self.tab_converter, 'tpx3 Converter')
        self.tab_roi_on_4D = Tab_ROI_on_4D()
        self.tabs.addTab(self.tab_roi_on_4D, 'ROI on 4D')
        self.tab_create_navSignal = Tab_Create_NavSignal()
        self.tabs.addTab(self.tab_create_navSignal, 'Make Nav. Sig.')
        self.tab_tracking_cv2 = Tab_Tracking_CV2()
        self.tabs.addTab(self.tab_tracking_cv2, 'Tracking by CV2')
        self.tab_sam2 = Tab_SAM2()
        self.tabs.addTab(self.tab_sam2, 'SAM2 Seg.')
        
        file_path = os.path.split(os.path.abspath(__file__))[0]
        fn_icon = os.path.join(file_path, 'ui_tabs',
                               'logo', 'Scream_logo.ico')
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
        """)

    def closeEvent(self, event):
        # Each tab's own cleanup() (below) disconnects its LogConsole from
        # the shared log signal - see LogConsole.disconnect_log() - so a
        # closed-but-not-destroyed window (see __init__ above) doesn't keep
        # pushing log lines into hidden widgets.

        # These are the *safe* parts of cleanup: they don't destroy any
        # widget, only stop queued threadpool work, kill running
        # subprocesses, and no-op-close matplotlib figures. See __init__
        # for why we deliberately don't go further and force the window
        # itself to be destroyed.
        for tab in (self.tab_roi_on_4D, self.tab_create_navSignal,
                    self.tab_tracking_cv2, self.tab_sam2):
            try:
                tab.cleanup()
            except Exception:
                logging.getLogger('py5DED.app').exception(
                    'Error cleaning up %s on close', type(tab).__name__)

        gc.collect()
        event.accept()

if __name__ == "__main__":
    # Reuse an existing QApplication instead of unconditionally constructing
    # a new one — Qt only supports one QApplication per process, and this
    # script may be re-run in the same console/kernel (e.g. Spyder's "Run
    # File") without the interpreter restarting.
    app = qtw.QApplication.instance()
    if app is None:
        app = qtw.QApplication([])
    install_excepthook()

    # If a MainWindow from an earlier run in this console is still alive,
    # reuse it (just bring it to the front) instead of constructing another
    # one. This is deliberate, not an oversight: closing that old window and
    # building a fresh MainWindow in its place was verified by hand to
    # crash the interpreter (a separate, deeper pre-existing bug in how this
    # app's widget tree — Tab_Tracking_CV2's matplotlib canvas in
    # particular — tears down and gets reconstructed in the same process).
    # Never rebuilding it at all sidesteps that crash entirely, and as a
    # side effect also stops the resource accumulation (duplicate
    # threadpools, matplotlib figures, hyperspy signal handles) that
    # building a brand new MainWindow on every rerun used to cause.
    existing_window = next(
        (w for w in app.topLevelWidgets() if isinstance(w, MainWindow)), None)
    if existing_window is not None:
        existing_window.show()
        existing_window.raise_()
        existing_window.activateWindow()
        window = existing_window
    else:
        window = MainWindow()
        window.show()
    app.exec_()
