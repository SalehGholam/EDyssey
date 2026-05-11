# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:53:46 2024

@author: SGholam
"""

import os
import sys
from collections import deque
from PyQt5.QtCore import Qt, QProcess
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIntValidator, QDoubleValidator
import numpy as np
import gc
import py4DTomo.io_utils as io
import hyperspy.api as hs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
#%% class
class Tab_Create_NavSignal(qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_widget()

    def init_widget(self):
        self.central_widget = qtw.QWidget(self)
        self.layout = qtw.QHBoxLayout(self)
        self.setLayout(self.layout)
        self._splitter = qtw.QSplitter(Qt.Horizontal)
        self.layout.addWidget(self._splitter)
        self._left_widget = qtw.QWidget()
        self._splitter.addWidget(self._left_widget)

        width_userInput = 300

        # layout top
        layout_userInput = qtw.QVBoxLayout(self._left_widget)
        #%% directory
        layout_dir_scanSize = qtw.QVBoxLayout()
        layout_userInput.addLayout(layout_dir_scanSize)

        self.box_dir = qtw.QGroupBox('Directories', self)
        layout_dir = qtw.QVBoxLayout()
        layout_dir_scanSize.addWidget(self.box_dir)
        self.box_dir.setLayout(layout_dir)

        layout_dir_4d = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_4d)

        label_dir_4d = qtw.QLabel('4D Signals')
        label_dir_4d.setFixedWidth(55)
        layout_dir_4d.addWidget(label_dir_4d)

        self.lineEdit_dir_signal = qtw.QLineEdit()
        layout_dir_4d.addWidget(self.lineEdit_dir_signal)

        self.button_dir = qtw.QPushButton('...')
        layout_dir_4d.addWidget(self.button_dir)
        self.button_dir.clicked.connect(lambda: self.show_dialog('file'))

        self.lineEdit_dir_signal.textChanged.connect(self.populate_file_list)

        layout_dir_save = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_save)

        label_dir_save = qtw.QLabel('Save Path')
        label_dir_save.setFixedWidth(55)
        layout_dir_save.addWidget(label_dir_save)

        self.lineEdit_dir_save = qtw.QLineEdit()
        layout_dir_save.addWidget(self.lineEdit_dir_save)

        self.button_dir_save = qtw.QPushButton('...')
        layout_dir_save.addWidget(self.button_dir_save)
        self.button_dir_save.clicked.connect(lambda: self.show_dialog('folder'))

        #%% scale
        layout_scale = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_scale)
        self.double_validator = QDoubleValidator(0.0, 1e5, 5)
        label_scale_real = qtw.QLabel('Scale (nm)')
        layout_scale.addWidget(label_scale_real)
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        layout_scale.addWidget(self.lineEdit_scale_real)
        self.lineEdit_scale_real.setFixedWidth(50)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        self.lineEdit_scale_real.textChanged.connect(lambda: self.update_canvas(
            self.slider_imgNo.value()))
        layout_scale.addStretch(1)
        #%% scan size
        self.box_scanSize = qtw.QGroupBox('Scan Size')
        layout_dir_scanSize.addWidget(self.box_scanSize)
        layout_scanSize = qtw.QHBoxLayout()
        self.box_scanSize.setLayout(layout_scanSize)

        self.checkbox_scanSize = qtw.QCheckBox('Auto')
        layout_scanSize.addWidget(self.checkbox_scanSize)
        self.checkbox_scanSize.setChecked(True)

        self.lineEdit_scanSize_x = qtw.QLineEdit()
        self.lineEdit_scanSize_x.setAlignment(Qt.AlignLeft)
        layout_scanSize.addWidget(self.lineEdit_scanSize_x)
        self.lineEdit_scanSize_x.setFixedWidth(50)
        self.lineEdit_scanSize_x.setValidator(QIntValidator(0,99999))

        label_cross = qtw.QLabel('X')
        layout_scanSize.addWidget(label_cross)

        self.lineEdit_scanSize_y = qtw.QLineEdit()
        layout_scanSize.addWidget(self.lineEdit_scanSize_y)
        self.lineEdit_scanSize_y.setFixedWidth(50)
        self.lineEdit_scanSize_y.setValidator(QIntValidator(0,99999))
        self.activate_lineEdit_scanSize()
        self.checkbox_scanSize.stateChanged.connect(self.activate_lineEdit_scanSize)

        label_dwellTime = qtw.QLabel('Dwell Time (usec)')
        self.spinbox_dwellTime = qtw.QSpinBox()
        self.spinbox_dwellTime.setFixedWidth(60)
        self.spinbox_dwellTime.setRange(1, 99999999)
        for wid in [label_dwellTime, self.spinbox_dwellTime]:
            layout_scanSize.addWidget(wid)
        #%% list of files
        layout_fileList = qtw.QVBoxLayout()
        layout_userInput.addLayout(layout_fileList)

        self.box_dtype = qtw.QGroupBox('Data Type')
        layout_fileList.addWidget(self.box_dtype)
        layout_dtype = qtw.QHBoxLayout()
        self.box_dtype.setLayout(layout_dtype)

        self.checkbox_selectAll = qtw.QCheckBox('All files')
        layout_dtype.addWidget(self.checkbox_selectAll)
        self.checkbox_selectAll.setChecked(True)

        self.combo_dtype = qtw.QComboBox()
        layout_dtype.addWidget(self.combo_dtype)
        self.combo_dtype.addItems(['.tpx3', '.hdf5', '.hspy', '.zspy'])
        self.combo_dtype.setDisabled(True)
        self.checkbox_selectAll.stateChanged.connect(self.activate_combo_dtype)

        #%% calculate button + CPU cores
        layout_calculate_buttons = qtw.QHBoxLayout()
        layout_userInput.addLayout(layout_calculate_buttons)

        self.button_calculate = qtw.QPushButton('Calculate')
        self.button_calculate.setFixedSize(100, 50)
        layout_calculate_buttons.addWidget(self.button_calculate)
        self.button_calculate.clicked.connect(self.calculate_button)

        self.button_stop = qtw.QPushButton('Stop')
        self.button_stop.setFixedSize(100, 50)
        layout_calculate_buttons.addWidget(self.button_stop)
        self.button_stop.setStyleSheet("background-color: red; color: white;")
        self.button_stop.setDisabled(True)
        self.button_stop.clicked.connect(self.stop_worker)

        layout_cores = qtw.QVBoxLayout()
        layout_calculate_buttons.addLayout(layout_cores)
        label_cores = qtw.QLabel('CPU Cores')
        label_cores.setAlignment(Qt.AlignCenter)
        self.spinbox_cpuCores = qtw.QSpinBox()
        self.spinbox_cpuCores.setRange(1, os.cpu_count() or 1)
        self.spinbox_cpuCores.setValue(max(1, (os.cpu_count() or 2) - 2))
        self.spinbox_cpuCores.setToolTip('Number of parallel worker processes for nav image computation')
        for wid in [label_cores, self.spinbox_cpuCores]:
            layout_cores.addWidget(wid)

        layout_fps = qtw.QVBoxLayout()
        layout_calculate_buttons.addLayout(layout_fps)
        label_fps = qtw.QLabel('FPS')
        label_fps.setAlignment(Qt.AlignCenter)
        self.spinbox_fps = qtw.QSpinBox()
        self.spinbox_fps.setRange(1, 60)
        self.spinbox_fps.setValue(5)
        self.spinbox_fps.setToolTip('Frames per second for the navigation clip')
        for wid in [label_fps, self.spinbox_fps]:
            layout_fps.addWidget(wid)

        #%% list of files
        self.file_list_widget = qtw.QListWidget()
        layout_userInput.addWidget(self.file_list_widget)
        self.file_list_widget.setMinimumWidth(150)
        self.file_list_widget.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        #%% canvas
        self._right_widget = qtw.QWidget()
        self._splitter.addWidget(self._right_widget)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([300, 900])
        layout_canvas = qtw.QVBoxLayout(self._right_widget)
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot()
        self.img_display = self.ax.imshow(np.zeros((512,512), dtype='int16'), cmap='viridis')
        self.ax.set_axis_off()
        layout_canvas.addWidget(self.canvas)
        #%% slider layout
        layout_slider = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_slider)

        self.label_imgCounter = qtw.QLabel('Img No.')
        layout_slider.addWidget(self.label_imgCounter)

        self.slider_imgNo = qtw.QSlider(self)
        self.slider_imgNo.setOrientation(1)  # Horizontal slider
        self.slider_imgNo.setRange(0,0)
        layout_slider.addWidget(self.slider_imgNo)

        self.slider_imgNo.valueChanged.connect(self.update_canvas)
        layout_canvas.addWidget(NavigationToolbar(self.canvas, self))
        #%% progress bar
        layout_progress_bar = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_progress_bar)

        self.progress_bar = qtw.QProgressBar()
        layout_progress_bar.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)

    #%% functions
    def show_dialog(self, f):
        sender = self.sender()
        if sender == self.button_dir:
            file_filter = "supported signals (*.zspy *.hspy *.hdf5 *.tpx3);;All Files (*)"
            path = qtw.QFileDialog.getOpenFileName(self, "Select 4D Signals Folder", '', file_filter)
            if path:
                path = os.path.split(path[0])[0]
                self.lineEdit_dir_signal.setText(path)
        elif sender == self.button_dir_save:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select Destination Folder")
            if path:
                self.lineEdit_dir_save.setText(path)

    def populate_file_list(self):
        ext_filter = ['.tpx3', '.hdf5', '.zspy', '.hspy', '.pmf']
        directory = self.lineEdit_dir_signal.text()
        if os.path.isdir(directory):
            self.file_list_widget.clear()
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

    def get_all_item_names(self):
        item_names = []
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
        fns.sort()

        dtype = [os.path.splitext(fn)[1] for fn in fns]
        dtype = np.array(dtype)
        dtype = np.unique(dtype)
        if len(dtype) != 1:
            qtw.QMessageBox.warning(self, 'Mixed File Types',
                f'Files with different extensions found in directory: {list(dtype)}\n'
                'Select a single file type and try again.')
            return
        dtype = dtype[0]

        dwellTime = self.spinbox_dwellTime.value()
        if self.checkbox_scanSize.isChecked():
            scanSize = None
        else:
            try:
                scanSize = (int(self.lineEdit_scanSize_x.text()), int(self.lineEdit_scanSize_y.text()))
            except:
                scanSize = None
        if dtype == '.tpx3' and scanSize is None:
            self.message_box_tpx3()
            return

        self.pathSave = self.lineEdit_dir_save.text()
        if not os.path.isdir(self.pathSave):
            os.mkdir(self.pathSave)

        self.button_stop.setEnabled(True)
        self.create_navigation_signal(fns, dtype, scanSize, dwellTime)

    def create_navigation_signal(self, fns, dtype, scanSize, dwellTime):
        self.nav_imgs = [None] * len(fns)
        self.nav_counter = 0
        self.nav_counter_total = len(fns)
        self.tasks = deque()
        for i, fn in enumerate(fns):
            self.tasks.append((fn, dtype, scanSize, dwellTime, i))
        self.running_processes = []
        self.process_task_map = {}
        self.process_output_buffers = {}
        self.max_processes = self.spinbox_cpuCores.value()
        self.update_progress_bar(0, self.nav_counter_total)
        for _ in range(min(self.max_processes, len(self.tasks))):
            self.launch_next_nav_task()

    def launch_next_nav_task(self):
        if not self.tasks or len(self.running_processes) >= self.max_processes:
            return
        fn, dtype, scanSize, dwellTime, i_index = self.tasks.popleft()
        scanSize_str = str(scanSize) if scanSize is not None else 'None'
        process = QProcess()
        process.setProgram(sys.executable)
        process.setArguments(['worker_nav_img.py', fn, dtype, scanSize_str,
                              str(dwellTime), str(i_index)])
        process.readyReadStandardOutput.connect(lambda: self._accumulate_output_nav(process))
        process.readyReadStandardError.connect(lambda: self.handle_error_nav(process))
        process.finished.connect(lambda: self.handle_finished_nav(process))
        process.errorOccurred.connect(self.process_failed_nav)
        self.running_processes.append(process)
        self.process_task_map[process] = i_index
        self.process_output_buffers[process] = bytearray()
        process.start()

    def _accumulate_output_nav(self, process):
        self.process_output_buffers[process] += process.readAllStandardOutput().data()

    def handle_error_nav(self, process):
        err = process.readAllStandardError().data().decode().strip()
        if err:
            qtw.QMessageBox.warning(self, 'Worker Error', err[:500])

    def handle_finished_nav(self, process):
        if process in self.running_processes:
            self.running_processes.remove(process)
        i_index = self.process_task_map.pop(process, None)
        # drain any remaining bytes not yet signalled
        self.process_output_buffers[process] += process.readAllStandardOutput().data()
        raw = bytes(self.process_output_buffers.pop(process, b'')).decode().strip()
        try:
            import base64, pickle
            nav_img, i_index = pickle.loads(base64.b64decode(raw))
            self.nav_imgs[i_index] = nav_img
        except Exception as e:
            print(f'Failed to decode nav image for index {i_index}: {e}')
        process.deleteLater()
        self.nav_counter += 1
        self.update_progress_bar(self.nav_counter, self.nav_counter_total)
        if self.nav_counter >= self.nav_counter_total:
            valid = [img for img in self.nav_imgs if img is not None]
            if not valid:
                qtw.QMessageBox.critical(self, 'No Results', 'All worker processes failed to produce output.')
                self.button_stop.setDisabled(True)
                return
            self.nav_imgs = np.stack(valid)
            self.update_canvas(0)
            self.slider_imgNo.setRange(0, len(self.nav_imgs) - 1)
            self.button_stop.setDisabled(True)
            self.save_results()
        else:
            self.launch_next_nav_task()

    def process_failed_nav(self, error):
        self.button_stop.setDisabled(True)
        qtw.QMessageBox.critical(self, 'Process Error',
            f'A worker process failed to start (error code {error}).\n'
            'Check that Python is on PATH and worker_nav_img.py exists.')

    def stop_worker(self):
        self.tasks.clear()
        for p in list(self.running_processes):
            p.kill()
        self.running_processes.clear()
        self.process_task_map.clear()
        self.button_stop.setDisabled(True)

    def save_results(self):
        from .worker_thread import WorkerThread_General
        from PyQt5.QtCore import QThreadPool
        threadpool = QThreadPool.globalInstance()
        s = hs.signals.Signal2D(self.nav_imgs)
        s.save(os.path.join(self.pathSave, 'navigation_signal.hspy'), overwrite=True)
        path_imgs = os.path.join(self.pathSave, 'navigation_images')
        if os.path.isdir(path_imgs):
            [os.remove(os.path.join(path_imgs, fn)) for fn in os.listdir(path_imgs)]
        else:
            os.mkdir(path_imgs)
        worker_frames = WorkerThread_General(io.create_frames, 0, path_imgs, s.data)
        threadpool.start(worker_frames)
        fn_clip = self.pathSave + '\\navigation_images_clip'
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
        except:
            scale_real = None
        worker_clip = WorkerThread_General(io.create_clip_tracking, 0, fn_clip,
                                           s.data, None, scale_real,
                                           fps=self.spinbox_fps.value())
        threadpool.start(worker_clip)

    def update_progress_bar(self, value, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f'%v / {total}')

    def update_canvas(self, imgNo):
        if hasattr(self, 'nav_imgs') and isinstance(self.nav_imgs, np.ndarray):
            self.img_display.set_data(self.nav_imgs[imgNo])
            self.img_display.set_clim(self.nav_imgs.min(), self.nav_imgs.max())
            shape_x, shape_y = self.nav_imgs[imgNo].shape
            self.img_display.set_extent([0, shape_y, shape_x, 0])
            self.ax.set_title(f'Image No. {imgNo+1:d}')
            scale_real = self.lineEdit_scale_real.text()
            try:
                scale_real = float(scale_real)
                scalebar_real = ScaleBar(scale_real, 'nm', dimension='si-length',
                                         location='lower left', box_alpha=0, color='w')
                for artist in self.ax.artists:
                    if isinstance(artist, ScaleBar):
                        artist.remove()
                self.ax.add_artist(scalebar_real)
            except Exception:
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
