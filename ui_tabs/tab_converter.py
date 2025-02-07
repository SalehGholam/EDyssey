<<<<<<< Updated upstream
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:42:26 2025

@author: sgholam
"""
import os
import sys
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QTime, QThreadPool
from PyQt5.QtCore import Qt, QRunnable, QObject, pyqtSignal
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIntValidator
import gc
# import py4DTomo.io_utils as io
from worker_thread import WorkerThread_General
import hyperspy.api as hs
from glob import glob
path = os.getcwd()
path = os.path.join(path, 'py4DTomo', 'io_utils')
sys.path.append(path)
# import pyLiveProcessing as pyLP
import eventem as pyLP
from time import sleep
from multiprocessing import Pool, Manager
from multiprocessing import Process, Queue
#%% class
class Tab_Converter(qtw.QWidget):
# class Tab_Create_NavSignal(qtw.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.init_widget()
        
        # threadpool to use in the entire tab
        self.threadpool = QThreadPool()
        # self.threadpool = QThreadPool.globalInstance()
# =============================================================================
#         logical_processors = os.cpu_count()
#         
# # =============================================================================
# #         if logical_processors > 2:
# #             self.threadpool.setMaxThreadCount(logical_processors - 2)
# # =============================================================================
#         self.threadpool.setMaxThreadCount(5)
# =============================================================================


    def init_widget(self):
        self.central_widget = qtw.QWidget(self)
        # self.setCentralWidget(self.central_widget)
        self.layout = qtw.QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
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
        
        layout_dir_save = qtw.QHBoxLayout(self)
        self.layout.addLayout(layout_dir_save)
        
        label_dir_save = qtw.QLabel('Save Directory')
        layout_dir_save.addWidget(label_dir_save)
        
        self.lineEdit_dir_save = qtw.QLineEdit()
        layout_dir_save.addWidget(self.lineEdit_dir_save)
        
        self.button_dir_save = qtw.QPushButton('...')
        layout_dir_save.addWidget(self.button_dir_save)
        self.button_dir_save.clicked.connect(lambda: self.show_dialog('folder'))        
        #%% input parameter
        layout_input = qtw.QHBoxLayout()
        self.layout.addLayout(layout_input)
        label_det_shape = qtw.QLabel('Detector Size')
        layout_input.addWidget(label_det_shape)
        
        self.combo_detSize = qtw.QComboBox()
        layout_input.addWidget(self.combo_detSize)
        self.combo_detSize.addItems([str(i) for i in [512,256,1024]])
        
        label_scanSize = qtw.QLabel('Scan Size')
        self.lineEdit_scan_x = qtw.QLineEdit()
        label_x = qtw.QLabel('X')
        self.lineEdit_scan_y = qtw.QLineEdit()
        label_dwellTime = qtw.QLabel('Dwell Time')
        self.lineEdit_dwellTime = qtw.QLineEdit()
        for item in [label_scanSize, self.lineEdit_scan_x, label_x, self.lineEdit_scan_y,
                     label_dwellTime, self.lineEdit_dwellTime]:
            layout_input.addWidget(item)
        for item in [self.lineEdit_scan_x, self.lineEdit_scan_y,
                     self.lineEdit_dwellTime]:
            item.setValidator(QIntValidator(0,9999999))
        
        layout_conversion = qtw.QHBoxLayout()
        self.layout.addLayout(layout_conversion)
        
        label_convert_from = qtw.QLabel('Convert from')
        self.combo_convert_from = qtw.QComboBox()
        self.combo_convert_from.addItems(['.tpx3']) #TODO add hdf5
        label_convert_to = qtw.QLabel('Convert to')
        self.combo_convert_to = qtw.QComboBox()
        self.combo_convert_to.addItems(['.hdf5', '.zspy', '.hspy'])
        label_bitDepth = qtw.QLabel('Bit Depth')
        self.combo_bitDepth = qtw.QComboBox()
        self.combo_bitDepth.addItems([str(i) for i in [8,16,32]])
        label_no_process = qtw.QLabel('No of Processes')
        self.spinbox_no_processes = qtw.QSpinBox()
        self.spinbox_no_processes.setRange(1, 20)
        self.button_convert = qtw.QPushButton('Convert!')
        for item in [label_convert_from, self.combo_convert_from, self.combo_bitDepth,
                     label_convert_to, self.combo_convert_to, label_bitDepth, 
                     self.combo_bitDepth, label_no_process, self.spinbox_no_processes, self.button_convert]:
            layout_conversion.addWidget(item)
        
        self.button_convert.clicked.connect(self.convert_datasets)
        #%% progress bar
        layout_progress_bar = qtw.QHBoxLayout()
        self.layout.addLayout(layout_progress_bar)
        
        self.progress_bar = qtw.QProgressBar()
        layout_progress_bar.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)
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
                
    def populate_file_list(self): # TODO
        ext_filter = ['.tpx3', '.hdf5', '.zspy', '.hspy', '.pmf']
        directory = self.lineEdit_dir_signal.text()
        # print(directory)
        # Clear the current list
        if os.path.isdir(directory):
            self.file_list_widget.clear()
            
            # List all files in the directory
            for f in os.listdir(directory):
                if os.path.splitext(f)[1] in ext_filter:
                    self.file_list_widget.addItem(f)
            
            self.set_save_directory()

    def convert_datasets(self):
        path_from = self.lineEdit_dir_signal.text()
        path_to = self.lineEdit_dir_save.text()
        det_size = int(self.combo_detSize.currentText())
        bit_depth = self.combo_bitDepth.currentText()
        bit_depth = int(bit_depth)
        scanSize_x = int(self.lineEdit_scan_x.text())
        scanSize_y = int(self.lineEdit_scan_y.text())
        scanSize = (scanSize_x, scanSize_y)
        dwellTime = self.lineEdit_dwellTime.text()
        dwellTime = int(dwellTime)
        dtype_from = self.combo_convert_from.currentText()
        dtype_to = self.combo_convert_to.currentText()
        no_processes = self.spinbox_no_processes.text()
        no_processes = int(no_processes)
        # self.set_thread_no(no_processes)
        

        # self.queue = Queue()
        # self.process = Process(target=process_file, args=)
        self.pool = Pool(no_processes)
        # Communicator for handling signals
        self.communicator = Communicator()
        self.communicator.finished.connect(self.onFinished)
        args = []
        chunksize = 8
        det_bin = 1
        scan_bin = 1
        compression_factor = 4
        self.fns = glob(os.path.join(path_from, '*.tpx3'))
        self.fns_save = [os.path.split(fn)[1] for fn in self.fns]
        self.fns_save = [os.path.join(path_to, os.path.splitext(fn)[0]) for fn in self.fns_save]
        for i, _ in enumerate(self.fns):
            # index = (i+1, self.total_no)
            args.append((self.fns[i], self.fns_save[i], scanSize, bit_depth, 
                         chunksize, det_size, det_bin, scan_bin, compression_factor))
        
        self.pool.map_async(self.process_file, args)

    def process_file(self, in_file, out_file, scan_size, bitdepth=8, chunksize=8, 
                     det_size=512, det_bin=1, scan_bin=1, compression_factor=4, counter=0):
        print('pass_3')
        # print(f"RAM requirement of 4D chunk: {(scan_size//scan_bin)*(chunksize//scan_bin) * (det_size//det_bin)**2 * (bitdepth/8) * 2 /1000000000} GB per process") #also ram used by raw data that is processed
        scan_size_x, scan_size_y = scan_size

        if bitdepth == 8:
            FourD = pyLP.FourD8
        elif bitdepth == 16:
            FourD = pyLP.FourD16
        elif bitdepth == 32:
            FourD = pyLP.FourD32
        fourD = FourD(output_filename = out_file,repetitions = 1, bitdepth=bitdepth, compression_factor=compression_factor) # create a new instance of the FourD class with the output file name
        fourD.set_file(in_file) # set the input file
        fourD.detector_size = det_size
        fourD.det_bin = det_bin 
        fourD.chunksize = chunksize 
        fourD.nx = scan_size_x
        fourD.ny = scan_size_y
        fourD.allocate_chunk() # allocate the memory for the 4D array
        fourD.init_4D_file() # initialize the hdf5 file writing
        fourD.run() # run the processing
        fourD.save_dose_image() # save the dose image
        return 'DONE'
    
    def onFinished(self, message):
        print(message)
    
    def update_progressbar(self, result):
        no, total = result
        value = no / total * 100
        value = int(value)
        self.progress_bar.setValue(value)
    
    
# =============================================================================
#     def closeEvent(self,event):
#         gc.collect()
#         event.accept()
# 
# if __name__ == "__main__":
#     app = qtw.QApplication(sys.argv)
#     
#     # Create and show the main window
#     window = Tab_Converter()
#     window.show()
#     
#     # Start the event loop
#     sys.exit(app.exec_())
# =============================================================================
    
class WorkerThread_processing(QRunnable):
    finished = pyqtSignal(object)

    def __init__(self, path_tpx3, path_save, N_processes=3, det_size=512, 
                 bitdepth=8, scan_size=(512,512)):
        super(WorkerThread_processing, self).__init__()
        print('pass_1')
        self.fns = glob(os.path.join(path_tpx3, '*.tpx3'))
        self.fns_save = [os.path.split(fn)[1] for fn in self.fns]
        self.fns_save = [os.path.join(path_save, os.path.splitext(fn)[0]) for fn in self.fns_save]
        self.N_processes = N_processes
        # self.det_size = det_size
        
        chunksize = 8
        compression_factor =  4 #1 is least compression, 9 is most compression
        det_bin = 1
        scan_bin = 1
        self.total_no = len(self.fns)

        self.args = []
        for i, _ in enumerate(self.fns):
            # index = (i+1, self.total_no)
            self.args.append((self.fns[i], self.fns_save[i], scan_size, bitdepth, 
                         chunksize, det_size, det_bin, scan_bin, compression_factor))
        self.signals = WorkerSignals()
# =============================================================================
#         manager = Manager()
#         self.counter = manager.Value('i', 0)
#         self.total_no = len(self.args)
# =============================================================================
        



    def run(self):
        print('pass_2')
        # l = len(self.fns)
        pool = Pool(self.N_processes)
        for i, arg in enumerate(self.args):
            pool.apply_async(self.process_file, args=arg + (i+1,)) # , callback=self.callback
        pool.close()
        pool.join()
        print('pass_2.2')

    def process_file(self, in_file, out_file, scan_size, bitdepth=8, chunksize=8, 
                     det_size=512, det_bin=1, scan_bin=1, compression_factor=4, counter=0):
        print('pass_3')
        # print(f"RAM requirement of 4D chunk: {(scan_size//scan_bin)*(chunksize//scan_bin) * (det_size//det_bin)**2 * (bitdepth/8) * 2 /1000000000} GB per process") #also ram used by raw data that is processed
        scan_size_x, scan_size_y = scan_size

        if bitdepth == 8:
            FourD = pyLP.FourD8
        elif bitdepth == 16:
            FourD = pyLP.FourD16
        elif bitdepth == 32:
            FourD = pyLP.FourD32
        fourD = FourD(output_filename = out_file,repetitions = 1, bitdepth=bitdepth, compression_factor=compression_factor) # create a new instance of the FourD class with the output file name
        fourD.set_file(in_file) # set the input file
        fourD.detector_size = det_size
        fourD.det_bin = det_bin 
        fourD.chunksize = chunksize 
        fourD.nx = scan_size_x
        fourD.ny = scan_size_y
        fourD.allocate_chunk() # allocate the memory for the 4D array
        fourD.init_4D_file() # initialize the hdf5 file writing
        fourD.run() # run the processing
        fourD.save_dose_image() # save the dose image

class Communicator(QObject):
    finished = pyqtSignal(str)
            
=======
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:42:26 2025

@author: sgholam
"""
import os
import sys
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QTime, QThreadPool
from PyQt5.QtCore import Qt, QRunnable, QObject, pyqtSignal
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIntValidator
import gc
# import py4DTomo.io_utils as io
from worker_thread import WorkerThread_General
import hyperspy.api as hs
from glob import glob
path = os.getcwd()
path = os.path.join(path, 'py4DTomo', 'io_utils')
sys.path.append(path)
# import pyLiveProcessing as pyLP
import eventem as pyLP
from time import sleep
from multiprocessing import Pool, Manager
from multiprocessing import Process, Queue
#%% class
class Tab_Converter(qtw.QWidget):
# class Tab_Create_NavSignal(qtw.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.init_widget()
        
        # threadpool to use in the entire tab
        self.threadpool = QThreadPool()
        # self.threadpool = QThreadPool.globalInstance()
# =============================================================================
#         logical_processors = os.cpu_count()
#         
# # =============================================================================
# #         if logical_processors > 2:
# #             self.threadpool.setMaxThreadCount(logical_processors - 2)
# # =============================================================================
#         self.threadpool.setMaxThreadCount(5)
# =============================================================================


    def init_widget(self):
        self.central_widget = qtw.QWidget(self)
        # self.setCentralWidget(self.central_widget)
        self.layout = qtw.QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
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
        
        layout_dir_save = qtw.QHBoxLayout(self)
        self.layout.addLayout(layout_dir_save)
        
        label_dir_save = qtw.QLabel('Save Directory')
        layout_dir_save.addWidget(label_dir_save)
        
        self.lineEdit_dir_save = qtw.QLineEdit()
        layout_dir_save.addWidget(self.lineEdit_dir_save)
        
        self.button_dir_save = qtw.QPushButton('...')
        layout_dir_save.addWidget(self.button_dir_save)
        self.button_dir_save.clicked.connect(lambda: self.show_dialog('folder'))        
        #%% input parameter
        layout_input = qtw.QHBoxLayout()
        self.layout.addLayout(layout_input)
        label_det_shape = qtw.QLabel('Detector Size')
        layout_input.addWidget(label_det_shape)
        
        self.combo_detSize = qtw.QComboBox()
        layout_input.addWidget(self.combo_detSize)
        self.combo_detSize.addItems([str(i) for i in [512,256,1024]])
        
        label_scanSize = qtw.QLabel('Scan Size')
        self.lineEdit_scan_x = qtw.QLineEdit()
        label_x = qtw.QLabel('X')
        self.lineEdit_scan_y = qtw.QLineEdit()
        label_dwellTime = qtw.QLabel('Dwell Time')
        self.lineEdit_dwellTime = qtw.QLineEdit()
        for item in [label_scanSize, self.lineEdit_scan_x, label_x, self.lineEdit_scan_y,
                     label_dwellTime, self.lineEdit_dwellTime]:
            layout_input.addWidget(item)
        for item in [self.lineEdit_scan_x, self.lineEdit_scan_y,
                     self.lineEdit_dwellTime]:
            item.setValidator(QIntValidator(0,9999999))
        
        layout_conversion = qtw.QHBoxLayout()
        self.layout.addLayout(layout_conversion)
        
        label_convert_from = qtw.QLabel('Convert from')
        self.combo_convert_from = qtw.QComboBox()
        self.combo_convert_from.addItems(['.tpx3']) #TODO add hdf5
        label_convert_to = qtw.QLabel('Convert to')
        self.combo_convert_to = qtw.QComboBox()
        self.combo_convert_to.addItems(['.hdf5', '.zspy', '.hspy'])
        label_bitDepth = qtw.QLabel('Bit Depth')
        self.combo_bitDepth = qtw.QComboBox()
        self.combo_bitDepth.addItems([str(i) for i in [8,16,32]])
        label_no_process = qtw.QLabel('No of Processes')
        self.spinbox_no_processes = qtw.QSpinBox()
        self.spinbox_no_processes.setRange(1, 20)
        self.button_convert = qtw.QPushButton('Convert!')
        for item in [label_convert_from, self.combo_convert_from, self.combo_bitDepth,
                     label_convert_to, self.combo_convert_to, label_bitDepth, 
                     self.combo_bitDepth, label_no_process, self.spinbox_no_processes, self.button_convert]:
            layout_conversion.addWidget(item)
        
        self.button_convert.clicked.connect(self.convert_datasets)
        #%% progress bar
        layout_progress_bar = qtw.QHBoxLayout()
        self.layout.addLayout(layout_progress_bar)
        
        self.progress_bar = qtw.QProgressBar()
        layout_progress_bar.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)
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
                
    def populate_file_list(self): # TODO
        ext_filter = ['.tpx3', '.hdf5', '.zspy', '.hspy', '.pmf']
        directory = self.lineEdit_dir_signal.text()
        # print(directory)
        # Clear the current list
        if os.path.isdir(directory):
            self.file_list_widget.clear()
            
            # List all files in the directory
            for f in os.listdir(directory):
                if os.path.splitext(f)[1] in ext_filter:
                    self.file_list_widget.addItem(f)
            
            self.set_save_directory()

    def convert_datasets(self):
        path_from = self.lineEdit_dir_signal.text()
        path_to = self.lineEdit_dir_save.text()
        det_size = int(self.combo_detSize.currentText())
        bit_depth = self.combo_bitDepth.currentText()
        bit_depth = int(bit_depth)
        scanSize_x = int(self.lineEdit_scan_x.text())
        scanSize_y = int(self.lineEdit_scan_y.text())
        scanSize = (scanSize_x, scanSize_y)
        dwellTime = self.lineEdit_dwellTime.text()
        dwellTime = int(dwellTime)
        dtype_from = self.combo_convert_from.currentText()
        dtype_to = self.combo_convert_to.currentText()
        no_processes = self.spinbox_no_processes.text()
        no_processes = int(no_processes)
        # self.set_thread_no(no_processes)
        

        # self.queue = Queue()
        # self.process = Process(target=process_file, args=)
        self.pool = Pool(no_processes)
        # Communicator for handling signals
        self.communicator = Communicator()
        self.communicator.finished.connect(self.onFinished)
        args = []
        chunksize = 8
        det_bin = 1
        scan_bin = 1
        compression_factor = 4
        self.fns = glob(os.path.join(path_from, '*.tpx3'))
        self.fns_save = [os.path.split(fn)[1] for fn in self.fns]
        self.fns_save = [os.path.join(path_to, os.path.splitext(fn)[0]) for fn in self.fns_save]
        for i, _ in enumerate(self.fns):
            # index = (i+1, self.total_no)
            args.append((self.fns[i], self.fns_save[i], scanSize, bit_depth, 
                         chunksize, det_size, det_bin, scan_bin, compression_factor))
        
        self.pool.map_async(self.process_file, args)

    def process_file(self, in_file, out_file, scan_size, bitdepth=8, chunksize=8, 
                     det_size=512, det_bin=1, scan_bin=1, compression_factor=4, counter=0):
        print('pass_3')
        # print(f"RAM requirement of 4D chunk: {(scan_size//scan_bin)*(chunksize//scan_bin) * (det_size//det_bin)**2 * (bitdepth/8) * 2 /1000000000} GB per process") #also ram used by raw data that is processed
        scan_size_x, scan_size_y = scan_size

        if bitdepth == 8:
            FourD = pyLP.FourD8
        elif bitdepth == 16:
            FourD = pyLP.FourD16
        elif bitdepth == 32:
            FourD = pyLP.FourD32
        fourD = FourD(output_filename = out_file,repetitions = 1, bitdepth=bitdepth, compression_factor=compression_factor) # create a new instance of the FourD class with the output file name
        fourD.set_file(in_file) # set the input file
        fourD.detector_size = det_size
        fourD.det_bin = det_bin 
        fourD.chunksize = chunksize 
        fourD.nx = scan_size_x
        fourD.ny = scan_size_y
        fourD.allocate_chunk() # allocate the memory for the 4D array
        fourD.init_4D_file() # initialize the hdf5 file writing
        fourD.run() # run the processing
        fourD.save_dose_image() # save the dose image
        return 'DONE'
    
    def onFinished(self, message):
        print(message)
    
    def update_progressbar(self, result):
        no, total = result
        value = no / total * 100
        value = int(value)
        self.progress_bar.setValue(value)
    
    
# =============================================================================
#     def closeEvent(self,event):
#         gc.collect()
#         event.accept()
# 
# if __name__ == "__main__":
#     app = qtw.QApplication(sys.argv)
#     
#     # Create and show the main window
#     window = Tab_Converter()
#     window.show()
#     
#     # Start the event loop
#     sys.exit(app.exec_())
# =============================================================================
    
class WorkerThread_processing(QRunnable):
    finished = pyqtSignal(object)

    def __init__(self, path_tpx3, path_save, N_processes=3, det_size=512, 
                 bitdepth=8, scan_size=(512,512)):
        super(WorkerThread_processing, self).__init__()
        print('pass_1')
        self.fns = glob(os.path.join(path_tpx3, '*.tpx3'))
        self.fns_save = [os.path.split(fn)[1] for fn in self.fns]
        self.fns_save = [os.path.join(path_save, os.path.splitext(fn)[0]) for fn in self.fns_save]
        self.N_processes = N_processes
        # self.det_size = det_size
        
        chunksize = 8
        compression_factor =  4 #1 is least compression, 9 is most compression
        det_bin = 1
        scan_bin = 1
        self.total_no = len(self.fns)

        self.args = []
        for i, _ in enumerate(self.fns):
            # index = (i+1, self.total_no)
            self.args.append((self.fns[i], self.fns_save[i], scan_size, bitdepth, 
                         chunksize, det_size, det_bin, scan_bin, compression_factor))
        self.signals = WorkerSignals()
# =============================================================================
#         manager = Manager()
#         self.counter = manager.Value('i', 0)
#         self.total_no = len(self.args)
# =============================================================================
        



    def run(self):
        print('pass_2')
        # l = len(self.fns)
        pool = Pool(self.N_processes)
        for i, arg in enumerate(self.args):
            pool.apply_async(self.process_file, args=arg + (i+1,)) # , callback=self.callback
        pool.close()
        pool.join()
        print('pass_2.2')

    def process_file(self, in_file, out_file, scan_size, bitdepth=8, chunksize=8, 
                     det_size=512, det_bin=1, scan_bin=1, compression_factor=4, counter=0):
        print('pass_3')
        # print(f"RAM requirement of 4D chunk: {(scan_size//scan_bin)*(chunksize//scan_bin) * (det_size//det_bin)**2 * (bitdepth/8) * 2 /1000000000} GB per process") #also ram used by raw data that is processed
        scan_size_x, scan_size_y = scan_size

        if bitdepth == 8:
            FourD = pyLP.FourD8
        elif bitdepth == 16:
            FourD = pyLP.FourD16
        elif bitdepth == 32:
            FourD = pyLP.FourD32
        fourD = FourD(output_filename = out_file,repetitions = 1, bitdepth=bitdepth, compression_factor=compression_factor) # create a new instance of the FourD class with the output file name
        fourD.set_file(in_file) # set the input file
        fourD.detector_size = det_size
        fourD.det_bin = det_bin 
        fourD.chunksize = chunksize 
        fourD.nx = scan_size_x
        fourD.ny = scan_size_y
        fourD.allocate_chunk() # allocate the memory for the 4D array
        fourD.init_4D_file() # initialize the hdf5 file writing
        fourD.run() # run the processing
        fourD.save_dose_image() # save the dose image

class Communicator(QObject):
    finished = pyqtSignal(str)
            
>>>>>>> Stashed changes
