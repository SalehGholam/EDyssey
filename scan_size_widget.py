# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 00:21:54 2025

@author: sgholam
"""

#TODO load calibration file => calib function for fov nm/pixel, list of mag

import os
# from PyQt5.QtCore import (pyqtSignal, Qt, QRunnable, QObject, QThreadPool)
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIntValidator, QDoubleValidator
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import sys


class ScanSizeWidget(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = qtw.QWidget()
        self.setCentralWidget(central_widget)
        self.setWindowTitle('Ideal Scan Size')
        # self.setCentralWidget(self.central_widget)
        self.layout = qtw.QVBoxLayout(central_widget)
        central_widget.setLayout(self.layout)
        
        layout_top = qtw.QHBoxLayout()
        self.layout.addLayout(layout_top)
        
        label_microscope = qtw.QLabel('Microscope')
        self.combo_microscope = qtw.QComboBox()
        
        label_fwhm = qtw.QLabel('Probe FWHM (nm)')
        self.lineEdit_fwhm = qtw.QLineEdit()
        
        label_mag = qtw.QLabel('Get Scan Size for Mag (kX)')
        self.lineEdit_mag = qtw.QLineEdit()
        self.label_scanSize = qtw.QLabel('Ideal Scan Size:')
        for wid in [label_microscope, self.combo_microscope, label_fwhm, 
                    self.lineEdit_fwhm]:
            layout_top.addWidget(wid)
            
        self.combo_microscope.addItems(['Tecnai', 'Titan 1'])
        self.combo_microscope.currentTextChanged.connect(self.get_calib)
        self.lineEdit_fwhm.setFixedWidth(100)
        self.lineEdit_fwhm.setValidator(QDoubleValidator(0.01, 1e3, 2))
        self.lineEdit_fwhm.textChanged.connect(self.get_calib)
        
        self.lineEdit_mag.textChanged.connect(self.get_scanSize_manual)
        self.lineEdit_mag.setValidator(QDoubleValidator(0.01, 1e9, 2))
        self.lineEdit_mag.setFixedWidth(100)
        layout_top.addStretch(1)
        
        layout_top_2 = qtw.QHBoxLayout()
        self.layout.addLayout(layout_top_2)
        self.checkbox_logy = qtw.QCheckBox('Log Y axis')
        for wid in [self.checkbox_logy, label_mag, self.lineEdit_mag,
                    self.label_scanSize]:
            layout_top_2.addWidget(wid)
        self.checkbox_logy.clicked.connect(self.update_plot)
        layout_top_2.addStretch(1)
        #%% canvas
        layout_canvas = qtw.QVBoxLayout()
        self.layout.addLayout(layout_canvas)
        
        self.figure = Figure()
        self.figure = Figure(constrained_layout=True)
        # self.figure = Figure(figsize=(16,8)) # with figsize
        self.canvas = FigureCanvas(self.figure)
        layout_canvas.addWidget(NavigationToolbar(self.canvas, self))
        self.ax = self.figure.add_subplot(111)
        self.plot, = self.ax.plot([], 'o--',)
        # self.plot_holz, = self.ax.plot([], 'co', label='HOLZ')
        self.ax.set(xlabel='Log Mag. (kX)', ylabel='Ideal Scan Size')
        self.ax.tick_params(axis='x', rotation=-45)
        # self.ax.set_yscale('log')
        # self.ax.legend()

        # self.figure.tight_layout()
        layout_canvas.addWidget(self.canvas)
        
        self.label_info = qtw.QLabel('')
        self.layout.addWidget(self.label_info)
    
    def calibration(self, mag, microscope='Tecnai'):
        if microscope == 'Tecnai':
            return 222.71 / mag #Tecnai at 200 kV for scan size 512 
        elif microscope == 'Titan 1':
            return 142.44 * mag**(-0.974)
            
    def get_calib(self):
        self.fwhm = self.lineEdit_fwhm.text()
        if self.fwhm == '':
            return
        self.fwhm = float(self.fwhm)
        self.microscope = self.combo_microscope.currentText()
        if self.microscope == 'Tecnai':
            self.mag_list = np.array([5.1, 7.2, 10, 14.5, 20.5, 29, 41, 58, 81, 115, 165, 
                                      230, 330, 460, 650,])
        elif self.microscope == 'Titan 1':
            self.mag_list = np.array([10, 14, 20, 28.5, 40, 57, 80, 115])
        self.calib = self.calibration(self.mag_list, self.microscope)
        self.fov = self.calib * 512
        self.probe_size = 2.4 * self.fwhm
        self.resolution = self.probe_size / 2
        self.scanSize = self.fov / self.resolution
        
        self.label_info.setText(f'Probe Size = {self.probe_size:.1f} nm; '
                                f'Rayleigh Resolution = {self.resolution:.1f} nm')
        self.update_plot()
    
    def get_scanSize_manual(self):
        mag = self.lineEdit_mag.text()
        try:
            mag = float(mag)
        except:
            return
        calib = self.calibration(mag)
        fov = calib * 512
        probe_size = 2.4 * self.fwhm
        resolution = probe_size / 2
        scanSize = fov / resolution
        self.label_scanSize.setText(f'Ideal Scan Size: {scanSize:0.0f}')
        
    
    def update_plot(self):
        self.plot.set_data(np.log(self.mag_list), self.scanSize)
        self.ax.set_xticks(np.log(self.mag_list), self.mag_list)
        if self.checkbox_logy.isChecked():
            self.ax.set_yscale('log')
        else:
            self.ax.set_yscale('linear')
            
        self.ax.relim()           # Recompute the data limits
        self.ax.autoscale_view()  # Update the view limits
        self.ax.grid(visible=True, which='major')
        self.ax.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.ax.axhline(1024, c='tab:orange')
        self.ax.axhline(512, c='tab:orange')
        self.ax.axhline(256, c='tab:orange')
        self.canvas.draw()
    
    def closeEvent(self, event):
        # empty_cache()
        event.accept()        
    
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    
    # Create the main window and show it
    window = ScanSizeWidget()
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec_())