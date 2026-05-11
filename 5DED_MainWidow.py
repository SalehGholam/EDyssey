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
sys.path.append(os.path.join(fld_path, 'py4DTomo\io_utils'))
import gc
import PyQt5.QtWidgets as qtw
from ui_tabs import (Tab_Create_NavSignal, Tab_Tracking_CV2,
                     Tab_ROI_on_4D, Tab_SAM2)
from PyQt5.QtGui import QIcon
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#%% window
class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
    
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
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage('Ready')
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

    def closeEvent(self,event):
        gc.collect()
        event.accept()
        
if __name__ == "__main__":
    app = qtw.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
    