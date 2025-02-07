# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:53:46 2024

@author: SGholam
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
#%% class
class Tab_Create_NavSignal(qtw.QWidget):
# class Tab_Create_NavSignal(qtw.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
            
        self.init_widget()
        
        # threadpool to use in the entire tab
        self.threadpool = QThreadPool()
        # self.threadpool = QThreadPool.globalInstance()
        # logical_processors = os.cpu_count()
        
# =============================================================================
#         if logical_processors > 2:
#             self.threadpool.setMaxThreadCount(logical_processors - 2)
# =============================================================================
        self.threadpool.setMaxThreadCount(5)

    def init_widget(self):
        self.central_widget = qtw.QWidget(self)
        # self.setCentralWidget(self.central_widget)
        self.layout = qtw.QVBoxLayout(self)
        self.setLayout(self.layout)
        #%% directory
        layout_dir = qtw.QHBoxLayout(self)
        self.layout.addLayout(layout_dir)
        
        label_dir = qtw.QLabel('Signals Directory')
        layout_dir.addWidget(label_dir)
        
        self.lineEdit_dir_signal = qtw.QLineEdit()
        layout_dir.addWidget(self.lineEdit_dir_signal)
        
        self.button_dir = qtw.QPushButton('...')
        layout_dir.addWidget(self.button_dir)
        self.button_dir.clicked.connect(lambda: self.show_dialog('file'))
        
        self.lineEdit_dir_signal.textChanged.connect(self.populate_file_list)
        
        layout_dir_save = qtw.QHBoxLayout(self)
        self.layout.addLayout(layout_dir_save)
        
        label_dir_save = qtw.QLabel('Save Directory')
        layout_dir_save.addWidget(label_dir_save)
        
        self.lineEdit_dir_save = qtw.QLineEdit()
        layout_dir_save.addWidget(self.lineEdit_dir_save)
        
        self.button_dir_save = qtw.QPushButton('...')
        layout_dir_save.addWidget(self.button_dir_save)
        self.button_dir_save.clicked.connect(lambda: self.show_dialog('folder'))
        #%% input
        layout_input_info = qtw.QHBoxLayout()
        # layout_input_info.setAlignment(Qt.AlignLeft)
        self.layout.addLayout(layout_input_info)
        
        self.checkbox_selectAll = qtw.QCheckBox('All files')
        layout_input_info.addWidget(self.checkbox_selectAll)
        self.checkbox_selectAll.setChecked(True)
        
        self.combo_dtype = qtw.QComboBox()
        layout_input_info.addWidget(self.combo_dtype)
        # self.combo_dtype.addItems(['.tpx3', '.hdf5', '.hspy', '.zspy', '.h5', '.pmf']) # TODO
        self.combo_dtype.addItems(['.tpx3', '.hdf5', '.hspy', '.zspy'])
        self.combo_dtype.setDisabled(True)
        self.checkbox_selectAll.stateChanged.connect(self.activate_combo_dtype)
        
        vline = qtw.QFrame()
        vline.setFrameShape(qtw.QFrame.VLine)
        vline.setFrameShadow(qtw.QFrame.Sunken)
        layout_input_info.addWidget(vline)
        
        label_scanSize = qtw.QLabel('Scan Size')
        layout_input_info.addWidget(label_scanSize)
        
        self.checkbox_scanSize = qtw.QCheckBox('Auto')
        layout_input_info.addWidget(self.checkbox_scanSize)
        self.checkbox_scanSize.setChecked(True)

        self.lineEdit_scanSize_x = qtw.QLineEdit()
        self.lineEdit_scanSize_x.setAlignment(Qt.AlignLeft)
        layout_input_info.addWidget(self.lineEdit_scanSize_x)
        self.lineEdit_scanSize_x.setFixedWidth(50)
        self.lineEdit_scanSize_x.setValidator(QIntValidator(0,99999))
        
        label_cross = qtw.QLabel('X')
        layout_input_info.addWidget(label_cross)
        
        self.lineEdit_scanSize_y = qtw.QLineEdit()
        layout_input_info.addWidget(self.lineEdit_scanSize_y)
        self.lineEdit_scanSize_y.setFixedWidth(50)
        self.lineEdit_scanSize_y.setValidator(QIntValidator(0,99999))
        self.activate_lineEdit_scanSize()
        self.checkbox_scanSize.stateChanged.connect(self.activate_lineEdit_scanSize)

        vline = qtw.QFrame()
        vline.setFrameShape(qtw.QFrame.VLine)
        vline.setFrameShadow(qtw.QFrame.Sunken)
        layout_input_info.addWidget(vline)
        
        label_dwellTime = qtw.QLabel('Dwell Time (usec)')
        self.spinbox_dwellTime = qtw.QSpinBox()
        self.spinbox_dwellTime.setFixedWidth(60)
        self.spinbox_dwellTime.setRange(1, 99999999)
        for wid in [label_dwellTime, self.spinbox_dwellTime]:
            layout_input_info.addWidget(wid)

        vline = qtw.QFrame()
        vline.setFrameShape(qtw.QFrame.VLine)
        vline.setFrameShadow(qtw.QFrame.Sunken)
        layout_input_info.addWidget(vline)
        
        # scale
        self.double_validator = QDoubleValidator(0.0, 1e5, 5)
        label_scale_real = qtw.QLabel('Scale (nm)')
        layout_input_info.addWidget(label_scale_real)
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        layout_input_info.addWidget(self.lineEdit_scale_real)
        self.lineEdit_scale_real.setFixedWidth(50)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        self.lineEdit_scale_real.textChanged.connect(lambda: self.update_canvas(
            self.slider_imgNo.value()))

        layout_input_info.addStretch(1)
        #%% list of files
        layout_fileList = qtw.QHBoxLayout()
        self.layout.addLayout(layout_fileList)
        
        self.file_list_widget = qtw.QListWidget()
        self.file_list_widget.setFixedHeight(200)
        # self.file_list_widget.setSelectionMode(qtw.QListWidget.MultiSelection)  # Allow multiple selections
        self.file_list_widget.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        
        # self.file_list_widget.setResizeMode(qtw.QListWidget.Adjust)
        
        # self.layout.addWidget(self.file_list_widget)
        layout_fileList.addWidget(self.file_list_widget)
        #%% calculate button
        # layout_calculate_buttons = qtw.QHBoxLayout()
        # self.layout.addLayout(layout_calculate_buttons)
        layout_calculate_buttons = qtw.QVBoxLayout()
        layout_fileList.addLayout(layout_calculate_buttons)
        
        self.button_calculate = qtw.QPushButton('Calculate')
        layout_calculate_buttons.addWidget(self.button_calculate)
        self.button_calculate.clicked.connect(self.calculate_button)

        self.button_stop = qtw.QPushButton('Stop')
        layout_calculate_buttons.addWidget(self.button_stop)
        self.button_stop.setStyleSheet("background-color: red; color: white;")
        self.button_stop.setDisabled(True)
        self.button_stop.clicked.connect(self.stop_worker)
        #%% progress bar/status bar
# =============================================================================
#         #status bar        
#         self.statusBar = qtw.QStatusBar(self)
#         self.layout.addWidget(self.statusBar)
# =============================================================================
        layout_progress_bar = qtw.QHBoxLayout()
        self.layout.addLayout(layout_progress_bar)
        
        self.progress_bar = qtw.QProgressBar()
        layout_progress_bar.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)
        #%% canvas
        # self.figure = Figure(figsize=(5,5))
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(NavigationToolbar(self.canvas, self))
        self.ax = self.figure.add_subplot()
        self.img_display = self.ax.imshow(np.zeros((512,512), dtype='int16'), cmap='viridis')
        # self.figure.tight_layout()
        self.layout.addWidget(self.canvas)
        #%% slider layout
        layout_slider = qtw.QHBoxLayout(self)
        self.layout.addLayout(layout_slider)
        
        self.label_imgCounter = qtw.QLabel('Img No.')
        layout_slider.addWidget(self.label_imgCounter)
        
        self.slider_imgNo = qtw.QSlider(self)
        self.slider_imgNo.setOrientation(1)  # Horizontal slider
        self.slider_imgNo.setRange(0,0)
        layout_slider.addWidget(self.slider_imgNo)
        
        # self.update_canvas(0)
        self.slider_imgNo.valueChanged.connect(self.update_canvas)
    #%% functions
    def show_dialog(self, f):
        sender = self.sender()
        if sender == self.button_dir:
            file_filter = "supported signals (*.zspy *.hspy *.hdf5 *.tpx3);;All Files (*)"
            # path = qtw.QFileDialog.getOpenFileNames(self, "Select 4D Signals Folder", '', file_filter)
            path = qtw.QFileDialog.getOpenFileName(self, "Select 4D Signals Folder", '', file_filter)
            if path:
                path = os.path.split(path[0])[0] # save it as the main directory
                self.lineEdit_dir_signal.setText(path)
        elif sender == self.button_dir_save:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select Destination Folder")
            if path:
                self.lineEdit_dir_save.setText(path)
                
    def populate_file_list(self):
        ext_filter = ['.tpx3', '.hdf5', '.zspy', '.hspy', '.pmf']
        directory = self.lineEdit_dir_signal.text()
        # print(directory)
        # Clear the current list
        if os.path.isdir(directory):
            self.file_list_widget.clear()
            
            # List all files in the directory
            items = os.listdir(directory)
            items.sort()
            for f in items:
                if os.path.splitext(f)[1] in ext_filter:
                    self.file_list_widget.addItem(f)
            self.set_save_directory()
    
    def set_save_directory(self):
        p = self.lineEdit_dir_signal.text()
        self.path_save = os.path.dirname(p)
        self.lineEdit_dir_save.setText(self.path_save)
    
    def activate_lineEdit_scanSize(self):
        if self.checkbox_scanSize.isChecked():
            self.lineEdit_scanSize_x.setDisabled(True)
            self.lineEdit_scanSize_y.setDisabled(True)
        else:
            self.lineEdit_scanSize_x.setEnabled(True)
            self.lineEdit_scanSize_y.setEnabled(True)
            
# =============================================================================
#     def update_statusBar(self, msg):
#         self.statusBar.showMessage(msg)
# =============================================================================
    
    def get_all_item_names(self):
        item_names = []
        # Iterate over all items in the QListWidget
        for index in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(index)
            if item:
                item_names.append(item.text())
        return item_names
    
    def activate_combo_dtype(self, state):
        if state == 0:
            self.combo_dtype.setEnabled(True)
        else:
            self.combo_dtype.setDisabled(True)
    
    def calculate_button(self):
        self.path_main = self.lineEdit_dir_signal.text()
        if self.checkbox_selectAll.isChecked():
            fns = self.get_all_item_names()
        else:
            fns = self.file_list_widget.selectedItems()
            fns = [item.text() for item in fns]
            if len(fns) == 0:
                dtype = self.combo_dtype.currentText()
                fns = self.get_all_item_names()
                fns = [fn for fn in fns if os.path.splitext(fn) == dtype]
        fns = [os.path.join(self.path_main, fn) for fn in fns]
        
        # check data types in the folder
        dtype = [os.path.splitext(fn)[1] for fn in fns]
        dtype = np.array(dtype)
        dtype = np.unique(dtype)
        if len(dtype) != 1:
            self.update_statusBar(f"Files with different extensions were found in the directory: {list(dtype)}")
            return
        dtype = dtype[0]
        
        dwellTime = self.spinbox_dwellTime.value()
        if self.checkbox_scanSize.isChecked():
            scanSize = None
        else:
            try:
                scanSize = (int(self.lineEdit_scanSize_x.text()), int(self.lineEdit_scanSize_y.text()))
            except: # line edits might be empty
                scanSize = None
        if dtype == '.tpx3' and scanSize == None:
            self.message_box_tpx3()
            return
        
        self.pathSave = self.lineEdit_dir_save.text()
        if not os.path.isdir(self.pathSave):
            os.mkdir(self.pathSave)

        self.button_stop.setEnabled(True)
        self.create_navigation_signal(fns, dtype, scanSize, dwellTime)
            
    def create_navigation_signal(self, fns, dtype, scanSize, dwellTime):
        if scanSize == None:
            scanSize = io.get_scan_size(fns[0])
        self.nav_imgs = np.zeros((len(fns), scanSize[0], scanSize[1]), dtype='int32')
        self.counter_nav = 0
        self.counter_nav_total = len(fns)   
        self.workers = []
        for i, fn in enumerate(fns):
            worker = WorkerThread_General(io.calculate_nav_signal, i, fn, dtype
                                          , scanSize)
            worker.signals.results.connect(self.get_nav_imgs)
            worker.signals.stopped.connect(lambda: self.update_progress_bar(0, 1))
            self.workers.append(worker)
            self.threadpool.start(worker)
            
# =============================================================================
#         self.worker_thread = CreateNavigationSignal(d_input)
#         self.worker_thread.start()
#         self.start_time = QTime.currentTime()
#         self.worker_thread.progress_update.connect(self.update_progress_bar)
#         self.worker_thread.progress_update.connect(self.update_timer)
#         self.worker_thread.nav_imgs.connect(self.get_images)
#     def get_images(self, imgs):
#         self.nav_imgs = imgs
#         self.update_canvas(0)
#         self.slider_imgNo.setRange(0, len(imgs)-1)
# =============================================================================

    def get_nav_imgs(self, result, index):
        self.nav_imgs[index] = result
        # progress bar update
        self.counter_nav += 1
        for worker in self.workers: # if the workers are stopped, don't update it
            if worker.is_running == True:
                self.update_progress_bar(self.counter_nav, self.counter_nav_total)
        # plot results at the end
        if self.counter_nav == self.counter_nav_total:
            self.update_canvas(0)
            self.slider_imgNo.setRange(0, len(self.nav_imgs)-1)
            self.save_results()
            
    def save_results(self):
        s = hs.signals.Signal2D(self.nav_imgs)
        s.save(os.path.join(self.pathSave, 'navigation_signal.hspy'), overwrite=True)
        # save tif images
        path_imgs = os.path.join(self.pathSave, 'navigation_images')
        if os.path.isdir(path_imgs):
            [os.remove(os.path.join(path_imgs, fn)) for fn in os.listdir(path_imgs)]
        else:
            os.mkdir(path_imgs)
        worker_frames = WorkerThread_General(io.create_frames, 0, path_imgs, s)
        self.threadpool.start(worker_frames)
        # save clip
        # fn_clip = os.path.join(self.pathSave, 'navigation_images_clip')
        fn_clip = self.pathSave + '\\navigation_images_clip'
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
        except:
            scale_real = None
        worker_clip = WorkerThread_General(io.create_clip_tracking, 0, fn_clip,
                                           s.data, None, scale_real)
        self.threadpool.start(worker_clip)
            
    def update_progress_bar(self, value, total):
        value = value / total * 100
        value = int(value)
        self.progress_bar.setValue(value)
    
    def update_canvas(self, imgNo):
        if hasattr(self, 'nav_imgs'):
            self.img_display.set_data(self.nav_imgs[imgNo])
            self.img_display.set_clim(self.nav_imgs.min(), self.nav_imgs.max())
            shape_x, shape_y = self.nav_imgs[imgNo].shape
            self.img_display.set_extent([0, shape_y, shape_x, 0])
            self.ax.set_title(f'Image No. {imgNo+1:d}')
            scale_real = self.lineEdit_scale_real.text()
            try:
                scale_real = float(scale_real)
                scalebar_real = ScaleBar(scale_real, 'nm', dimension='si-length', location='lower left')
                for artist in self.ax.artists:
                    if isinstance(artist, ScaleBar):
                        artist.remove()
                self.ax.add_artist(scalebar_real)
            except Exception as e:
                # print(e)
                pass        
            
            self.canvas.draw()
    
    def stop_worker(self):
        for worker in self.workers:
            worker.stop()
        self.button_stop.setDisabled(True)

    def message_box_tpx3(self):
       msg = qtw.QMessageBox()
       msg.setWindowTitle("Scan Size Error!")
       msg.setText("(Currently,) tpx3 conversion requires scan size input!")
       msg.setInformativeText("Enter scan size and try again.")
       msg.setStandardButtons(qtw.QMessageBox.Ok)
       msg.setIcon(qtw.QMessageBox.Critical)
       retval = msg.exec_()
    


# =============================================================================
# if __name__ == "__main__":
#     app = qtw.QApplication(sys.argv)
#     
#     # Create and show the main window
#     window = Tab_Create_NavSignal()
#     window.show()
#     
#     # Start the event loop
#     sys.exit(app.exec_())
# =============================================================================
