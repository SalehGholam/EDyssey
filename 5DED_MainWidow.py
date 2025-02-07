# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:36:57 2024

@author: Saleh Gholam

version 2:
    1. Fix ROIs plotting for bottom to up drawings in tab roi4D and CV2
    2. Fix loading hdf5 in io_utils
    3. Debug reset ROIs button in tab cv2
    4. Add delete 1 ROI from selection on tab cv2
    5. Support of pyLiveprocession for a range of python versions
    6. improve GPU management for SAM2
    7. tab for data conversion
    8. roi in roi in tab cv2
    9. completion of sam2 widget
        todo. VDF
        todo. is hyperspy signals necessary for reading files? specially hdf5.
"""

import os
import PyQt5.QtWidgets as qtw
import gc
from tab_create_navSignal import Tab_Create_NavSignal
from tab_tracking_cv2 import Tab_Tracking_CV2
from tab_roi_4d import Tab_ROI_on_4D
from tab_sam2 import Tab_SAM2
# from tab_converter import Tab_Converter
from PyQt5.QtGui import QIcon

file_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(file_path))
#%% window
class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
    
        self.init_ui()
    
    def init_ui(self):
        self.resize(800, 600)  # Width, Height in pixels
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
        
        
        self.setWindowIcon(QIcon('Scream_logo.ico'))
        self.setCentralWidget(self.tabs)
        
    def closeEvent(self,event):
        self.tab_sam2.clear_model()
        gc.collect()
        event.accept()
        
if __name__ == "__main__":
    app = qtw.QApplication([])
    # app.setWindowIcon(QIcon('Scream_logo.ico'))
    window = MainWindow()
    window.show()
    app.exec_()