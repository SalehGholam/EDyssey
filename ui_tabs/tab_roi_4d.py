# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:34:09 2024

@author: SGholam
"""



import os
from PyQt5.QtCore import (pyqtSignal, Qt, QRunnable, QObject, QThreadPool)
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from matplotlib.colors import SymLogNorm
import numpy as np
import py4DTomo.io_utils as io
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
# import matplotlib.gridspec as gridspec
# from skimage.filters import threshold_otsu, threshold_li, threshold_mean, threshold_yen
# from skimage import exposure
#%% wdiget
class Tab_ROI_on_4D(qtw.QWidget):
# class Tab_Create_NavSignal(qtw.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.threadpool = QThreadPool()
        self.init_widget()
        
    def init_widget(self):
        self.central_widget = qtw.QWidget(self)
        # self.setCentralWidget(self.central_widget)
        self.layout = qtw.QVBoxLayout(self)
        self.setLayout(self.layout)
        #%% directory
        layout_dir = qtw.QHBoxLayout(self)
        self.layout.addLayout(layout_dir)
        
        # nav signal dir
        label_dir = qtw.QLabel('Nav. Signal')
        layout_dir.addWidget(label_dir)
        
        self.lineEdit_dir_signal = qtw.QLineEdit()
        layout_dir.addWidget(self.lineEdit_dir_signal)
        self.lineEdit_dir_signal.textChanged.connect(lambda:self.enable_dwellTime_spinbox(
            self.lineEdit_dir_signal.text()))
        
        self.button_dir_navSignal = qtw.QPushButton('...')
        layout_dir.addWidget(self.button_dir_navSignal)
        self.button_dir_navSignal.clicked.connect(self.show_dialog)
        
        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        layout_dir.addWidget(self.button_loadNavigation)
        self.button_loadNavigation.clicked.connect(self.get_nav_image)
        #%% input layout, box scan size
        layout_input_info = qtw.QHBoxLayout()
        # layout_input_info.setAlignment(Qt.AlignLeft)
        self.layout.addLayout(layout_input_info)
        
        self.box_scanSize = qtw.QGroupBox('Scan Size')
        self.box_scanSize.setFixedSize(450, 60)
        layout_box_scanSize = qtw.QHBoxLayout()
        self.box_scanSize.setLayout(layout_box_scanSize)
        layout_input_info.addWidget(self.box_scanSize)
        
        # label_scanSize = qtw.QLabel('Scan Size')
        # layout_box_scanSize.addWidget(label_scanSize)
        
        self.checkbox_scanSize = qtw.QCheckBox('Auto')
        layout_box_scanSize.addWidget(self.checkbox_scanSize)
        self.checkbox_scanSize.setChecked(True)

        self.lineEdit_scanSize_x = qtw.QLineEdit()
        self.lineEdit_scanSize_x.setAlignment(Qt.AlignLeft)
        layout_box_scanSize.addWidget(self.lineEdit_scanSize_x)
        self.lineEdit_scanSize_x.setFixedWidth(50)
        self.lineEdit_scanSize_x.setValidator(QIntValidator(0,99999))
        
        label_cross = qtw.QLabel('X')
        layout_box_scanSize.addWidget(label_cross)
        
        self.lineEdit_scanSize_y = qtw.QLineEdit()
        layout_box_scanSize.addWidget(self.lineEdit_scanSize_y)
        self.lineEdit_scanSize_y.setFixedWidth(50)
        self.lineEdit_scanSize_y.setValidator(QIntValidator(0,99999))

        self.activate_lineEdit_scanSize()
        self.checkbox_scanSize.stateChanged.connect(self.activate_lineEdit_scanSize)
        
        label_dwellTime = qtw.QLabel('Dwell Time (\u03BCsec)')
        self.spinbox_dwellTime = qtw.QSpinBox()
        self.spinbox_dwellTime.setFixedWidth(100)
        self.spinbox_dwellTime.setRange(1, 99999999)
        self.spinbox_dwellTime.setDisabled(True)
        for wid in [label_dwellTime, self.spinbox_dwellTime]:
            layout_box_scanSize.addWidget(wid)
        
        
        
# =============================================================================
#         # spacer
#         width = self.width()
#         # h_spacer = qtw.QSpacerItem(width-50, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
#         h_spacer = qtw.QSpacerItem(2000, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
#         layout_input_info.addItem(h_spacer)
# =============================================================================
        #%% box for scales
        self.box_scale = qtw.QGroupBox('Scale bars')
        self.box_scale.setFixedSize(250, 60)
        layout_box_scale = qtw.QHBoxLayout()
        # self.box_scale.setFixedWidth(150)
        self.box_scale.setLayout(layout_box_scale)
        layout_input_info.addWidget(self.box_scale)
        
        # real space
        self.double_validator = QDoubleValidator(0.0, 1e5, 5)
        layout_scale_real = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_real)
        label_scale_real = qtw.QLabel('Real (nm)')
        layout_scale_real.addWidget(label_scale_real)
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        layout_scale_real.addWidget(self.lineEdit_scale_real)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        # reciprocal space
        layout_scale_recip = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_recip)
        label_scale_recip = qtw.QLabel('Recip. (\u00C5<sup>-1</sup>)')
        layout_scale_recip.addWidget(label_scale_recip)
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        layout_scale_recip.addWidget(self.lineEdit_scale_recip)
        self.lineEdit_scale_recip.setValidator(self.double_validator)
        
        self.lineEdit_scale_recip.textChanged.connect(self.update_canvas)
        self.lineEdit_scale_real.textChanged.connect(self.update_canvas)
        
        layout_input_info.addStretch(1)
        #%% canvas layout
        layout_canvas = qtw.QHBoxLayout()
        self.layout.addLayout(layout_canvas)
        
        # self.figure = Figure(figsize=(5,5))
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(NavigationToolbar(self.canvas, self))
        self.layout.addWidget(self.canvas)
        
        self.ax_nav = self.figure.add_subplot(131)
        self.ax_nav_roi = self.figure.add_subplot(132)
        self.ax_dp = self.figure.add_subplot(133)
        
# =============================================================================
#         gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1])  # Top row: equal, bottom: double height
#         self.ax_nav = self.figure.add_subplot(gs[0, 0])
#         self.ax_nav_roi = self.figure.add_subplot(gs[0, 1])
#         self.ax_dp = self.figure.add_subplot(gs[1, :])
# =============================================================================
        
        
        # self.ax_nav_roi.set_position([0.75, 0.75, 0.15, 0.15])
        # self.ax_nav.set_position([0.05, 0.1, 0.4, 0.8])
        # self.ax_dp.set_position([0.5, 0.1, 0.4, 0.8])
        
        self.figure.tight_layout()
        self.reset_canvas()
        
        self.rect = None            # Currently drawn rectangle
        self.press = None           # Mouse press coordinates
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        #%% slider layout
        layout_slider = qtw.QHBoxLayout(self)
        self.layout.addLayout(layout_slider)
        
        self.label_vmin = qtw.QLabel('vmin')
        layout_slider.addWidget(self.label_vmin)
        
        self.slider_vmin = qtw.QSlider(self)
        self.slider_vmin.setOrientation(1)  # Horizontal slider
        self.slider_vmin.setRange(0,0)
        layout_slider.addWidget(self.slider_vmin)
        
        self.label_vmax = qtw.QLabel('vmax')
        layout_slider.addWidget(self.label_vmax)
        
        self.slider_vmax = qtw.QSlider(self)
        self.slider_vmax.setOrientation(1)  # Horizontal slider
        self.slider_vmax.setRange(0,0)
        layout_slider.addWidget(self.slider_vmax)

        self.slider_vmin.valueChanged.connect(lambda: self.update_canvas(ax='dp'))
        self.slider_vmax.valueChanged.connect(lambda: self.update_canvas(ax='dp'))
        
        self.button_reset_slider = qtw.QPushButton('Reset Sliders')
        layout_slider.addWidget(self.button_reset_slider)
        self.button_reset_slider.clicked.connect(self.reset_sliders)
#%% functions
    def activate_lineEdit_scanSize(self):
        if self.checkbox_scanSize.isChecked():
            self.lineEdit_scanSize_x.setDisabled(True)
            self.lineEdit_scanSize_y.setDisabled(True)
        else:
            self.lineEdit_scanSize_x.setEnabled(True)
            self.lineEdit_scanSize_y.setEnabled(True)

    def get_scan_size(self):
        if not self.checkbox_scanSize.isChecked(): # get scan size
            x = int(self.lineEdit_scanSize_x.text())
            y = int(self.lineEdit_scanSize_y.text())
            scanSize = (x,y)
        else:
            scanSize = None
        return scanSize
    
    def get_nav_image(self):
        self.reset_canvas()
        self.fn = self.lineEdit_dir_signal.text()
        self.scanSize = self.get_scan_size()
        self.dwellTime = self.spinbox_dwellTime.value()
        dtype = os.path.splitext(self.fn)[-1]
        if dtype == '.tpx3' and self.scanSize == None:
            self.message_box_tpx3()
            return
        
        
        worker = Worker_NavImg(self.fn, self.scanSize, self.dwellTime)
        
        worker.signals.result.connect(self.image_handler)  # Connect to result signal
        self.threadpool.start(worker)

# =============================================================================
#         self.navImg = io.calculate_nav_signal(self.fn, scanSize=scanSize)
#         self.update_canvas('nav')
# =============================================================================

    def image_handler(self, result):
        self.navImg = result
        self.update_canvas('nav')

    def show_dialog(self):
        file_filter = "supported signals (*.zspy *.hspy *.hdf5 *.h5 *.tpx3 *.pmf);;All Files (*)"
        path = qtw.QFileDialog.getOpenFileName(self, "Select 4D Signals Folder", '', file_filter)
        # if path and os.path.isdir(path[0]):
        if path:
            self.lineEdit_dir_signal.setText(path[0])
    
    def enable_dwellTime_spinbox(self, txt):
        enable = False
        if os.path.isfile(txt):
            dtype = os.path.splitext(txt)[1]
            if dtype == '.tpx3':
                enable = True
        for wid in self.box_scanSize.findChildren(qtw.QWidget):
            wid.setEnabled(enable)
            # print(e)
        self.checkbox_scanSize.setChecked(not enable)
        self.checkbox_scanSize.setDisabled(enable)
            
    
    def on_press(self, event):
        # Mouse press event: record the starting point
        if event.inaxes == self.ax_nav:
            self.press = (event.xdata, event.ydata)
            if self.rect is not None:
                self.rect.remove()
            self.rect = patches.Rectangle(self.press, 0, 0, linewidth=1, 
                                          edgecolor='r', facecolor='none')
            self.ax_nav.add_patch(self.rect)
            self.canvas.draw()

    def on_motion(self, event):
        # Mouse motion event: update the rectangle size as the mouse moves
        if self.press is None or event.inaxes is None:
            return
        x0, y0 = self.press
        width = event.xdata - x0
        height = event.ydata - y0
        self.rect.set_width(width)
        self.rect.set_height(height)
        self.rect.set_xy((x0, y0))
        self.canvas.draw()

    def on_release(self, event):
        # Mouse release event: finalize the rectangle
        if self.press is None or event.inaxes is None:
            return
        
        # Finalize the rectangle and store it
        x0, y0 = self.press
        width = event.xdata - x0
        height = event.ydata - y0
        # ROI might be drawn reversed
        if width < 0:
            width = abs(width)
            x0 = event.xdata
        if height < 0:
            height = abs(height)
            y0 = event.ydata
            
        if width==0:
            width = 1
        if height==0:
            height = 1
        self.roi = (int(x0), int(y0), int(width), int(height))
        
        
        # Set the press attribute to None for future drawings
        self.press = None
        self.canvas.draw()
        print(self.roi)
        if not hasattr(self, 'dwellTime'):
            try:
                self.dwellTime = self.spinbox_dwellTime.value()
            except:
                self.dwellTime = None
        worker = Worker_CalculateDP(self.fn, self.roi, self.scanSize, self.dwellTime)
        worker.signals.result.connect(self.get_dp)
        self.threadpool.start(worker)
    
    def get_dp(self, result):
# =============================================================================
#         s_cut = io.load_signal(self.fn, lazy=False, roi=roi, scanSize=scanSize, dwellTime=dwellTime)
#         self.navImg_cut = s_cut.sum(axis=(2,3)).data
#         self.dp = s_cut.sum(axis=(0,1)).data
#         if hasattr(self.dp, 'compute'): # lazy signals
#             self.dp.compute()
# =============================================================================
        self.dp, self.navImg_cut = result
        print(f'max = {self.dp.max()}')
        self.ax_nav_roi.set_title(f'ROI Image: {self.roi}')
        self.update_slider_range()
        self.slider_vmax.setValue(self.dp.max())
        self.slider_vmin.setValue(1)
        self.update_canvas(roiUpdate=True)
    
    def reset_sliders(self):
        self.slider_vmin.setValue(0)
        self.slider_vmax.setValue(self.dp.max())
        self.update_canvas(roiUpdate=True)
    
    def reset_canvas(self):
        self.img_display = {}
        img_temp = np.zeros((512,512), dtype='uint16')
        self.img_display['nav'] = self.ax_nav.imshow(img_temp, cmap='viridis')
        self.img_display['nav_roi'] = self.ax_nav_roi.imshow(img_temp, cmap='viridis')
        # self.img_display['nav'].set_axis_off()

        self.img_display['dp'] = self.ax_dp.imshow(img_temp, cmap='viridis')
        self.img_display['dp'].set_norm(SymLogNorm(linthresh=1))
        
        self.ax_nav.set_title('Nav. Image')
        self.ax_nav_roi.set_title('ROI Image')
        self.ax_dp.set_title('Dif. Pattern')
        
        self.ax_nav.set_axis_off()
        self.ax_nav_roi.set_axis_off()
        self.ax_dp.set_axis_off()

        self.figure.tight_layout()
    
    def update_slider_range(self):
        self.slider_vmin.setRange(0, int(self.dp.max()/2))
        self.slider_vmax.setRange(1, self.dp.max())
        self.slider_vmin.setSingleStep(1)
        self.slider_vmax.setSingleStep(1)
    
    def update_canvas(self, ax='dp', roiUpdate=False):
        if ax == 'dp':
            vmax = self.slider_vmax.value()
            self.label_vmax.setText(f'vmax: {vmax:.0f}')
            vmin = self.slider_vmin.value()
            if vmin >= vmax:
                vmin = vmax - 1
                self.slider_vmin.setValue(vmin)
            self.label_vmin.setText(f'vmin: {vmin:.0f}')
            
            self.img_display['dp'].set_data(self.dp)
            
            self.img_display['dp'].set_clim(vmin, vmax)
            # self.img_display['dp'].set_norm(SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax))
            shape_x, shape_y = self.dp.shape
            self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])
            self.img_display['dp'].set_clim(self.dp.min(), self.dp.max())

            if roiUpdate:
                self.img_display['nav_roi'].set_data(self.navImg_cut)
                self.img_display['nav_roi'].set_clim(self.navImg_cut.min(), self.navImg_cut.max())
                shape_x, shape_y = self.navImg_cut.shape
                self.img_display['nav_roi'].set_extent([0, shape_y, shape_x, 0])

        elif ax == 'nav':
            self.img_display['nav'].set_data(self.navImg)
            shape_x, shape_y = self.navImg.shape
            self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
            self.img_display['nav'].set_clim(vmin=self.navImg.min(), vmax=self.navImg.max())
        
        # scale bars 
        #TODO adding and removing the artist is not efficient
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
            for ax in [self.ax_nav, self.ax_nav_roi]:
                scalebar_real = ScaleBar(scale_real, 'nm', dimension='si-length', location='lower left')
                for artist in ax.artists:
                    if isinstance(artist, ScaleBar):
                        artist.remove()
                ax.add_artist(scalebar_real)
        except Exception as e:
            # print(e)
            pass
        scale_recip = self.lineEdit_scale_recip.text()
        try:
            scale_recip = float(scale_recip)
            if scale_recip != 0:
                scalebar_recip = ScaleBar(scale_recip*10, '1/nm', dimension='si-length-reciprocal', location='lower left',
                                    scale_formatter=lambda value, unit:  f'{value / 10}'r' $\AA^{-1}$', fixed_value=5)
                for artist in self.ax_dp.artists:
                        if isinstance(artist, ScaleBar):
                            artist.remove()
                self.ax_dp.add_artist(scalebar_recip)
        except Exception as e:
            # print(e)
            pass
        
        self.canvas.draw()
    
    def message_box_tpx3(self):
       msg = qtw.QMessageBox()
       msg.setWindowTitle("Scan Size Error!")
       msg.setText("(Currently,) tpx3 conversion requires scan size input!")
       msg.setInformativeText("Enter scan size and try again.")
       msg.setStandardButtons(qtw.QMessageBox.Ok)
       msg.setIcon(qtw.QMessageBox.Critical)
       retval = msg.exec_()
        
class WorkerSignals(QObject):
    finished = pyqtSignal()  # Signal to indicate task completion
    result = pyqtSignal(object)  # Signal to emit the result of the task

# Step 2: Create a WorkerThread class that runs the task in the background
class Worker_NavImg(QRunnable):
    def __init__(self, fn, scanSize=None, dwellTime=None):
        super().__init__()
        print('Calculating navigation image...')
        self.fn = fn
        self.scanSize = scanSize
        self.dwellTime = dwellTime
        self.signals = WorkerSignals()  # Create an instance of WorkerSignals
        
    def run(self):
        # Simulate a task (e.g., calculating the sum of numbers)
        navImg = io.calculate_nav_signal(self.fn, scanSize=self.scanSize, 
                                         dwellTime=self.dwellTime)
        
        # Emit the result when the task is done
        self.signals.result.emit(navImg)
        print('Plotted Navigation Image')
        self.signals.finished.emit()  # Emit the finished signal when done
    
class Worker_CalculateDP(QRunnable):
    def __init__(self, fn, roi, scanSize, dwellTime):
        super().__init__()
        print('calculating the dp...')
        self.fn = fn
        self.roi = roi
        self.scanSize = scanSize
        self.dwellTime = dwellTime
        
        self.signals = WorkerSignals()
    
    def run(self):
        s_cut = io.load_signal(self.fn, lazy=False, roi=self.roi, 
                               scanSize=self.scanSize, dwellTime=self.dwellTime)
        navImg_cut = s_cut.sum(axis=(2,3)).data
        dp = s_cut.sum(axis=(0,1)).data
        if hasattr(dp, 'compute'): # lazy signals
            dp.compute()
        if hasattr(navImg_cut, 'compute'): # lazy signals
            navImg_cut = navImg_cut.compute()
        self.signals.result.emit((dp, navImg_cut))
        # self.signals.finished.emit()
# =============================================================================
# if __name__ == "__main__":
#     app = qtw.QApplication(sys.argv)
#     
#     # Create and show the main window
#     window = Tab_ROI_on_4D()
#     window.show()
#     
#     # Start the event loop
#     sys.exit(app.exec_())
# =============================================================================
