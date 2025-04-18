# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:49:33 2025

@author: sgholam
"""

import os
import sys
from PyQt5.QtCore import Qt, QThreadPool, QSize
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIntValidator, QDoubleValidator
import numpy as np
import gc
import py4DTomo.io_utils as io
from .worker_thread import WorkerThread_General
import hyperspy.api as hs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import time
from matplotlib_scalebar.scalebar import ScaleBar
#%%
class LoadNPlot(qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
            
        self.init_ui()
        
        # threadpool to use in the entire tab
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(5)
    
    def init_ui(self):
        self.central_widget = qtw.QWidget(self)
        # self.setCentralWidget(self.central_widget)
        self.layout = qtw.QVBoxLayout(self)
        self.setLayout(self.layout)
        
        # layout top
        layout_top = qtw.QHBoxLayout()
        self.layout.addLayout(layout_top)
        #%% directory
        self.box_dir = qtw.QGroupBox('Directories', self)
        layout_top.addWidget(self.box_dir)
        layout_dir = qtw.QVBoxLayout()
        self.box_dir.setLayout(layout_dir)
        
        label_dir = qtw.QLabel('Dir.')
        label_dir.setFixedWidth(25)
        self.lineEdit_dir = qtw.QLineEdit()
        self.lineEdit_dir.setFixedWidth(150)
        self.button_dir = qtw.QPushButton('...')
        self.button_dir.setFixedWidth(25)
        self.button_dir.clicked.connect(self.show_dialog)
        #%% list of files
        layout_fileList = qtw.QVBoxLayout()
        layout_top.addLayout(layout_fileList)
        
        self.file_list_widget = qtw.QListWidget()
        layout_fileList.addWidget(self.file_list_widget)
        self.file_list_widget.setFixedSize(400, 150)
        # self.file_list_widget.setSelectionMode(qtw.QListWidget.MultiSelection)  # Allow multiple selections
        self.file_list_widget.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        #%% plotting utils
        self.box_plotUtils = qtw.QGroupBox('Plotting')
        label_cmap = qtw.QLabel('Color Map')
        self.combo_cmap = qtw.QComboBox()
    #%% functions
    def show_dialog(self):
        # file_filter = "supported signals (*.zspy *.hspy *.hdf5 *.tpx3);;All Files (*)"
        path = qtw.QFileDialog.getExistingDirectory(self, 'Select folder')
        if path:
            self.lineEdit_dir_signal.setText(path[0])

    def populate_file_list(self):
        ext_filter = ['.tpx3', '.hdf5', '.zspy', '.hspy', '.pmf']
        directory = self.lineEdit_dir.text()
        # Clear the current list
        if os.path.isdir(directory):
            self.file_list_widget.clear()
            
            # List all files in the directory
            items = os.listdir(directory)
            items.sort()
            for f in items:
                if os.path.splitext(f)[1] in ext_filter:
                    self.file_list_widget.addItem(f)
            
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    
    # Create and show the main window
    window = LoadNPlot()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())
