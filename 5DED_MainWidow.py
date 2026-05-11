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
# from ui_tabs import (Tab_Create_NavSignal, Tab_Tracking_CV2,
#                      Tab_ROI_on_4D)
from ui_tabs import (Tab_Create_NavSignal, Tab_Tracking_CV2,
                     Tab_ROI_on_4D, Tab_SAM2)
from PyQt5.QtGui import QIcon

# import matplotlib.pyplot as plt
# plt.style.use('dark_background')

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
        
    def closeEvent(self,event):
        gc.collect()
        event.accept()
        
if __name__ == "__main__":
    app = qtw.QApplication([])
    # app.setWindowIcon(QIcon('Scream_logo.ico'))
    from PyQt5.QtGui import QPalette, QColor
    from PyQt5.QtCore import Qt
    
    app = qtw.QApplication([])
    # app.setStyle("Fusion")
    
    # palette = QPalette()
    
    # palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # palette.setColor(QPalette.WindowText, Qt.white)
    # palette.setColor(QPalette.Base, QColor(25, 25, 25))
    # palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    # palette.setColor(QPalette.ToolTipBase, Qt.white)
    # palette.setColor(QPalette.ToolTipText, Qt.white)
    # palette.setColor(QPalette.Text, Qt.white)
    # palette.setColor(QPalette.Button, QColor(53, 53, 53))
    # palette.setColor(QPalette.ButtonText, Qt.white)
    # palette.setColor(QPalette.BrightText, Qt.red)
    # palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    # palette.setColor(QPalette.HighlightedText, Qt.black)
    
    # app.setPalette(palette)
    
    
    window = MainWindow()
    window.show()
    app.exec_()
    