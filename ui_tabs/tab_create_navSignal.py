# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:53:46 2024

@author: SGholam
"""

import os
import sys
from time import perf_counter
from collections import deque
from PyQt5.QtCore import Qt, QProcess, QThreadPool
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIntValidator, QDoubleValidator
import numpy as np
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import py4DTomo.io_utils as io
import hyperspy.api as hs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
from .logging_utils import get_tab_logger, LogConsole
from .worker_thread import WorkerThread_General
from .threshold_dialog import ThresholdDialog
from worker_extract_frame import load_dp
#%% class
class Tab_Create_NavSignal(qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_tab_logger('Tab_Create_NavSignal')
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

        label_scale_recip = qtw.QLabel('Recip. (Å<sup>-1</sup>)')
        layout_scale.addWidget(label_scale_recip)
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        layout_scale.addWidget(self.lineEdit_scale_recip)
        self.lineEdit_scale_recip.setFixedWidth(50)
        self.lineEdit_scale_recip.setValidator(self.double_validator)
        self.lineEdit_scale_recip.setToolTip(
            'Reciprocal-space calibration (1/A per pixel) for the PACBED preview - '
            'drawn as concentric dashed rings every 1 1/A, centered on the found center')
        self.lineEdit_scale_recip.textChanged.connect(self.update_recip_scale_circles)
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
        self.combo_dtype.addItems(['.tpx3', '.hdf5', '.hspy', '.zspy', '.mib'])
        self.combo_dtype.setDisabled(True)
        self.checkbox_selectAll.stateChanged.connect(self.activate_combo_dtype)
        self.combo_dtype.currentIndexChanged.connect(self.refresh_file_list)

        #%% calculate button + CPU cores
        layout_calculate_buttons = qtw.QHBoxLayout()
        layout_userInput.addLayout(layout_calculate_buttons)

        self.button_calculate = qtw.QPushButton('Calculate All')
        self.button_calculate.setFixedSize(100, 50)
        layout_calculate_buttons.addWidget(self.button_calculate)
        self.button_calculate.clicked.connect(self.calculate_button)
        self.button_calculate.setToolTip('Run the full batch over every listed (or selected) file')

        self.button_testFile = qtw.QPushButton('Test File')
        self.button_testFile.setFixedSize(100, 50)
        layout_calculate_buttons.addWidget(self.button_testFile)
        self.button_testFile.clicked.connect(lambda: self.test_selected_file(None))
        self.button_testFile.setToolTip(
            'Compute/preview the navigation image for the selected (or first) '
            'file only, without running the full batch - same as double-clicking it')

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
        label_fps = qtw.QLabel('Clip FPS')
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
        self.file_list_widget.setToolTip('Double-click a file to test/preview its navigation image')
        self.file_list_widget.itemDoubleClicked.connect(self.test_selected_file)
        #%% virtual detector mask
        self.box_mask = qtw.QGroupBox('Virtual Detector Mask')
        layout_userInput.addWidget(self.box_mask)
        layout_mask = qtw.QVBoxLayout()
        self.box_mask.setLayout(layout_mask)

        self.checkbox_useMask = qtw.QCheckBox('Use Virtual Mask (annular detector)')
        layout_mask.addWidget(self.checkbox_useMask)
        self.checkbox_useMask.setToolTip(
            'When checked, the navigation image sums each diffraction pattern only '
            'within the annular region below, instead of the whole detector')

        layout_mask_center = qtw.QHBoxLayout()
        layout_mask.addLayout(layout_mask_center)
        label_centerX = qtw.QLabel('Center X')
        layout_mask_center.addWidget(label_centerX)
        self.spinbox_centerX = qtw.QSpinBox()
        self.spinbox_centerX.setRange(0, 8192)
        self.spinbox_centerX.setValue(256)
        layout_mask_center.addWidget(self.spinbox_centerX)
        label_centerY = qtw.QLabel('Center Y')
        layout_mask_center.addWidget(label_centerY)
        self.spinbox_centerY = qtw.QSpinBox()
        self.spinbox_centerY.setRange(0, 8192)
        self.spinbox_centerY.setValue(256)
        layout_mask_center.addWidget(self.spinbox_centerY)

        layout_mask_radii = qtw.QHBoxLayout()
        layout_mask.addLayout(layout_mask_radii)
        label_rIn = qtw.QLabel('Inner R')
        layout_mask_radii.addWidget(label_rIn)
        self.spinbox_rIn = qtw.QSpinBox()
        self.spinbox_rIn.setRange(0, 4096)
        self.spinbox_rIn.setValue(0)
        self.spinbox_rIn.setSingleStep(10)
        layout_mask_radii.addWidget(self.spinbox_rIn)
        label_rOut = qtw.QLabel('Outer R')
        layout_mask_radii.addWidget(label_rOut)
        self.spinbox_rOut = qtw.QSpinBox()
        self.spinbox_rOut.setRange(1, 4096)
        self.spinbox_rOut.setValue(256)
        self.spinbox_rOut.setSingleStep(10)
        layout_mask_radii.addWidget(self.spinbox_rOut)
        for sb in (self.spinbox_rIn, self.spinbox_rOut):
            sb.setToolTip('Up/down arrows step by 10; type a value directly for finer control')

        for sb in (self.spinbox_centerX, self.spinbox_centerY, self.spinbox_rIn, self.spinbox_rOut):
            sb.valueChanged.connect(self.update_mask_overlay)

        layout_mask_buttons = qtw.QHBoxLayout()
        layout_mask.addLayout(layout_mask_buttons)
        self.button_computePacbed = qtw.QPushButton('Compute PACBED')
        layout_mask_buttons.addWidget(self.button_computePacbed)
        self.button_computePacbed.clicked.connect(self.compute_pacbed)
        self.button_computePacbed.setToolTip(
            'Sum all diffraction patterns of the selected (or first) file to find '
            'the detector center - needed to place the virtual mask')

        self.button_autoCenter = qtw.QPushButton('Auto-Find Center')
        layout_mask_buttons.addWidget(self.button_autoCenter)
        self.button_autoCenter.clicked.connect(self.auto_find_center)
        self.button_autoCenter.setToolTip('Auto-detect the direct-beam center from the PACBED preview')

        layout_mask_threshold = qtw.QHBoxLayout()
        layout_mask.addLayout(layout_mask_threshold)
        self.button_pacbedFromThreshold = qtw.QPushButton('PACBED from Threshold...')
        layout_mask_threshold.addWidget(self.button_pacbedFromThreshold)
        self.button_pacbedFromThreshold.clicked.connect(self.open_threshold_dialog)
        self.button_pacbedFromThreshold.setToolTip(
            'Open a window to check/adjust a real-space threshold on the last '
            'tested navigation image, then sum diffraction patterns only at the '
            'scan positions above it, instead of the whole scan - for checking '
            'purposes only, not a substitute for "Compute PACBED"')

        self.button_pacbedFromRoi = qtw.QPushButton('PACBED from ROI')
        layout_mask_threshold.addWidget(self.button_pacbedFromRoi)
        self.button_pacbedFromRoi.clicked.connect(self.compute_pacbed_from_roi)
        self.button_pacbedFromRoi.setDisabled(True)
        self.button_pacbedFromRoi.setToolTip(
            'Sum diffraction patterns only over the scan-space rectangle drawn '
            'on the navigation/test image (hold Ctrl and drag there), instead '
            'of the whole scan')

        # The PACBED preview canvas itself lives in the main window, beside
        # the navigation image (see #%% canvas below), so it has more room
        # to work with and its own navigation toolbar.
        self._mask_artists = []
        self._mask_drag_mode = None
        # Scan-space ROI drawn (Ctrl+drag) on the nav/test image, used only
        # by "PACBED from ROI" - never by "Compute PACBED", which always
        # sums the whole scan regardless of whether a ROI is currently drawn.
        self.roi_navsig = None
        self.rect_navsig = None
        self._navsig_press = None
        self._navsig_bg = None
        #%% canvas
        self._right_widget = qtw.QWidget()
        self._splitter.addWidget(self._right_widget)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([300, 900])
        layout_canvas = qtw.QVBoxLayout(self._right_widget)

        # Both plots share a single figure/canvas (side-by-side subplots)
        # rather than two separate canvases, with one shared toolbar below.
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(121)
        self.ax_mask_preview = self.figure.add_subplot(122)

        self.img_display = self.ax.imshow(np.zeros((512,512), dtype='int16'), cmap='viridis')
        # ax keeps its x-axis label visible (for the ROI interaction hint
        # below), so its ticks/spines are hidden individually instead of via
        # set_axis_off(), which would hide the label too (see the identical
        # pattern/comment on ax_mask_preview just below).
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax.set_xlabel(
            'Hold Ctrl and drag to draw a scan-space ROI (auto-loads its PACBED).\n'
            'Right-click to remove it.',
            fontsize=7)
        self.ax.xaxis.label.set_visible(True)
        self.colorbar_nav = self.figure.colorbar(
            self.img_display, ax=self.ax, fraction=0.046, pad=0.04)

        self.img_display_mask = self.ax_mask_preview.imshow(
            np.zeros((512, 512), dtype='uint16'), cmap='inferno')
        self.img_display_mask.set_norm(SymLogNorm(linthresh=1))
        self.ax_mask_preview.set_title('PACBED (sum of all DPs)', fontsize=9)
        # ax_mask_preview keeps its x-axis label visible (for the
        # interaction hint below), so its ticks/spines are hidden
        # individually instead of via set_axis_off() - that sets
        # axison=False, which suppresses the *entire* axis decoration set at
        # draw time (including the xlabel) regardless of the label artist's
        # own set_visible(True).
        for spine in self.ax_mask_preview.spines.values():
            spine.set_visible(False)
        self.ax_mask_preview.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax_mask_preview.set_xlabel(
            'Hold Ctrl and drag the center (+) or a circle edge to move/resize '
            'the virtual mask.\nCtrl+Scroll to zoom either plot; plain click+drag pans (toolbar).',
            fontsize=7)
        self.ax_mask_preview.xaxis.label.set_visible(True)
        self.colorbar_mask = self.figure.colorbar(
            self.img_display_mask, ax=self.ax_mask_preview, fraction=0.046, pad=0.04)

        layout_canvas.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout_canvas.addWidget(self.toolbar)

        layout_slider = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_slider)
        self.label_imgCounter = qtw.QLabel('Img No.')
        layout_slider.addWidget(self.label_imgCounter)
        self.slider_imgNo = qtw.QSlider(self)
        self.slider_imgNo.setOrientation(1)  # Horizontal slider
        self.slider_imgNo.setRange(0,0)
        layout_slider.addWidget(self.slider_imgNo)
        self.slider_imgNo.valueChanged.connect(self.update_canvas)

        self.canvas.mpl_connect('scroll_event', self.on_scroll_mask)
        self.canvas.mpl_connect('button_press_event', self.on_press_mask)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_mask)
        self.canvas.mpl_connect('button_release_event', self.on_release_mask)
        # Separate handlers for the nav/test image's own Ctrl+drag ROI
        # (used by "PACBED from ROI") - each checks event.inaxes for its own
        # target axis, so both sets of callbacks can coexist on one canvas.
        self.canvas.mpl_connect('button_press_event', self.on_press_navsig)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_navsig)
        self.canvas.mpl_connect('button_release_event', self.on_release_navsig)
        self.update_mask_overlay()

        #%% progress bar
        layout_progress_bar = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_progress_bar)

        self.progress_bar = qtw.QProgressBar()
        layout_progress_bar.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)

        # The app-wide log console lives here (below this tab's own plot
        # columns) rather than under the whole window, so the left
        # parameter panel (a separate splitter pane) can span the full
        # window height.
        self.log_console = LogConsole(self)
        layout_canvas.addWidget(self.log_console)

    #%% functions
    def show_dialog(self, f):
        sender = self.sender()
        if sender == self.button_dir:
            file_filter = "supported signals (*.zspy *.hspy *.hdf5 *.tpx3 *.mib);;All Files (*)"
            path = qtw.QFileDialog.getOpenFileName(self, "Select 4D Signals Folder", '', file_filter)
            if path:
                path = os.path.split(path[0])[0]
                self.lineEdit_dir_signal.setText(path)
        elif sender == self.button_dir_save:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select Destination Folder")
            if path:
                self.lineEdit_dir_save.setText(path)

    def populate_file_list(self):
        directory = self.lineEdit_dir_signal.text()
        if os.path.isdir(directory):
            self.set_save_directory()
        self.refresh_file_list()

    def refresh_file_list(self):
        """(Re)populate the file list from the current directory, filtered
        to the data type selected in combo_dtype whenever "All files" is
        unchecked."""
        directory = self.lineEdit_dir_signal.text()
        if not os.path.isdir(directory):
            return
        if self.checkbox_selectAll.isChecked():
            ext_filter = ['.tpx3', '.hdf5', '.zspy', '.hspy', '.mib', '.pmf']
        else:
            ext_filter = [self.combo_dtype.currentText()]
        self.file_list_widget.clear()
        for f in sorted(os.listdir(directory)):
            if os.path.splitext(f)[1] in ext_filter:
                self.file_list_widget.addItem(f)

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
        self.refresh_file_list()

#%% single-file test / virtual mask
    def _get_current_scanSize(self):
        if self.checkbox_scanSize.isChecked():
            return None
        try:
            return (int(self.lineEdit_scanSize_x.text()), int(self.lineEdit_scanSize_y.text()))
        except Exception:
            return None

    def _get_test_fn(self, item=None):
        """Resolve the file to use for test/PACBED actions: the item that
        was clicked, else the current selection, else the first file."""
        if item is None:
            selected = self.file_list_widget.selectedItems()
            if selected:
                item = selected[0]
        if item is None:
            names = self.get_all_item_names()
            if not names:
                return None
            name = names[0]
        else:
            name = item.text()
        return os.path.join(self.lineEdit_dir_signal.text(), name)

    def test_selected_file(self, item):
        """Compute and preview the navigation image for a single file
        (double-clicked in the file list), without touching the full batch
        result - lets the user check dwell time / scan size / virtual mask
        parameters before committing to the full "Calculate" run."""
        fn = self._get_test_fn(item)
        if fn is None or not os.path.isfile(fn):
            self.logger.error('Cannot test file: %s', fn)
            return
        dtype = os.path.splitext(fn)[-1]
        scanSize = self._get_current_scanSize()
        if dtype == '.tpx3' and scanSize is None:
            self.logger.warning('Cannot test %s: scan size is required for .tpx3 files.', fn)
            self.message_box_tpx3()
            return

        dwellTime = self.spinbox_dwellTime.value()
        self.logger.info('Testing navigation image for %s...', fn)
        if self.checkbox_useMask.isChecked():
            worker = WorkerThread_General(
                io.calculate_nav_img_masked, 0, fn, dtype=dtype, scanSize=scanSize,
                dwellTime=dwellTime, r_in=self.spinbox_rIn.value(),
                r_out=self.spinbox_rOut.value(),
                center=(self.spinbox_centerX.value(), self.spinbox_centerY.value()),
                logger=self.logger)
        else:
            worker = WorkerThread_General(io.calculate_nav_img, 0, fn, dtype=dtype,
                                          scanSize=scanSize, dwellTime=dwellTime,
                                          logger=self.logger)
        worker.signals.results.connect(lambda result, idx, fn=fn: self._on_test_result(result, fn))
        QThreadPool.globalInstance().start(worker)

    def _on_test_result(self, result, fn):
        self._last_test_fn = fn
        self._last_test_img = result
        self.img_display.set_data(result)
        self.img_display.set_clim(result.min(), result.max())
        shape_x, shape_y = result.shape
        self.img_display.set_extent([0, shape_y, shape_x, 0])
        self.ax.set_xlim(0, shape_y)
        self.ax.set_ylim(shape_x, 0)
        self.ax.set_title(f'TEST: {os.path.basename(fn)}')
        self.canvas.draw_idle()
        self.logger.info('Test navigation image computed successfully for %s.', fn)

    def compute_pacbed(self):
        """Sum all diffraction patterns of one representative file to get a
        PACBED - the reference image used to place the virtual mask. Always
        uses the whole scan, regardless of any ROI currently drawn on the
        nav/test image - for a restricted PACBED instead, use "PACBED from
        Threshold..." or "PACBED from ROI"."""
        fn = self._get_test_fn()
        if fn is None or not os.path.isfile(fn):
            qtw.QMessageBox.critical(self, 'No File',
                'Load a directory (and optionally select a file) before computing a PACBED.')
            return
        dtype = os.path.splitext(fn)[-1]
        scanSize = self._get_current_scanSize()
        if dtype == '.tpx3' and scanSize is None:
            self.logger.warning('Cannot compute PACBED for %s: scan size is required for .tpx3 files.', fn)
            self.message_box_tpx3()
            return

        self.logger.info('Computing PACBED (summed diffraction pattern) from %s...', fn)
        self.button_computePacbed.setDisabled(True)
        self._pacbed_tic = perf_counter()
        worker = WorkerThread_General(io.get_pacbed, 0, fn, dtype=dtype, scanSize=scanSize,
                                      dwellTime=self.spinbox_dwellTime.value(), roi=None,
                                      logger=self.logger)
        worker.signals.results.connect(self._on_pacbed_computed)
        worker.signals.error.connect(self._on_pacbed_failed)
        QThreadPool.globalInstance().start(worker)

    def compute_pacbed_from_roi(self):
        """Sum diffraction patterns only over the scan-space rectangle drawn
        (Ctrl+drag) on the nav/test image - "Compute PACBED" itself always
        ignores this ROI and sums the whole scan instead."""
        if not self.roi_navsig:
            qtw.QMessageBox.critical(self, 'No ROI',
                'Hold Ctrl and drag on the navigation/test image to draw a '
                'scan-space ROI first.')
            return
        fn = self._get_test_fn()
        if fn is None or not os.path.isfile(fn):
            qtw.QMessageBox.critical(self, 'No File',
                'Load a directory (and optionally select a file) before computing a PACBED.')
            return
        dtype = os.path.splitext(fn)[-1]
        scanSize = self._get_current_scanSize()
        if dtype == '.tpx3' and scanSize is None:
            self.logger.warning('Cannot compute PACBED for %s: scan size is required for .tpx3 files.', fn)
            self.message_box_tpx3()
            return

        self.logger.info('Computing PACBED from ROI %s of %s...', self.roi_navsig, fn)
        self.button_computePacbed.setDisabled(True)
        self.button_pacbedFromRoi.setDisabled(True)
        self._pacbed_tic = perf_counter()
        worker = WorkerThread_General(io.get_pacbed, 0, fn, dtype=dtype, scanSize=scanSize,
                                      dwellTime=self.spinbox_dwellTime.value(),
                                      roi=self.roi_navsig, logger=self.logger)
        worker.signals.results.connect(self._on_pacbed_from_roi_computed)
        worker.signals.error.connect(self._on_pacbed_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_pacbed_from_roi_computed(self, result, index):
        self.button_pacbedFromRoi.setEnabled(True)
        self._on_pacbed_computed(result, index)  # also logs completion + duration

    def _on_pacbed_failed(self, traceback_text, index):
        # Without this, a failure in the background computation (which
        # WorkerThread_General now catches instead of letting it vanish
        # silently) would otherwise leave the triggering button disabled
        # forever, with no way to retry and no visible sign anything went wrong.
        self.button_computePacbed.setEnabled(True)
        self.button_pacbedFromThreshold.setEnabled(True)
        self.button_pacbedFromRoi.setEnabled(True)
        self.logger.error('Failed to compute PACBED:\n%s', traceback_text)
        qtw.QMessageBox.critical(self, 'PACBED Failed',
            'Computing the PACBED failed - see the log for details.')

    def _on_pacbed_computed(self, result, index):
        self.button_computePacbed.setEnabled(True)
        self.pacbed = result
        det_y, det_x = self.pacbed.shape
        for sb, val in ((self.spinbox_centerX, det_x), (self.spinbox_centerY, det_y),
                       (self.spinbox_rIn, max(det_x, det_y)), (self.spinbox_rOut, max(det_x, det_y))):
            sb.setMaximum(val)
        self.img_display_mask.set_data(self.pacbed)
        self.img_display_mask.set_clim(vmin=1, vmax=self.pacbed.max())
        self.img_display_mask.set_extent([0, det_x, det_y, 0])
        self.ax_mask_preview.set_xlim(0, det_x)
        self.ax_mask_preview.set_ylim(det_y, 0)
        self.update_mask_overlay()
        duration = perf_counter() - self._pacbed_tic if hasattr(self, '_pacbed_tic') else None
        if duration is not None:
            self.logger.info('PACBED computed successfully (%d x %d px) in %.1f s.',
                             det_x, det_y, duration)
        else:
            self.logger.info('PACBED computed successfully (%d x %d px).', det_x, det_y)

    def update_mask_overlay(self):
        """Redraw the inner/outer radius circles and center marker on the
        PACBED preview to match the current spinbox values."""
        for artist in self._mask_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._mask_artists = []
        cx = self.spinbox_centerX.value()
        cy = self.spinbox_centerY.value()
        r_in = self.spinbox_rIn.value()
        r_out = self.spinbox_rOut.value()

        circle_out = patches.Circle((cx, cy), r_out, fill=False, edgecolor='lime', linewidth=1.5)
        self.ax_mask_preview.add_patch(circle_out)
        self._mask_artists.append(circle_out)
        if r_in > 0:
            circle_in = patches.Circle((cx, cy), r_in, fill=False, edgecolor='red', linewidth=1.5)
            self.ax_mask_preview.add_patch(circle_in)
            self._mask_artists.append(circle_in)
        # green stands out clearly against inferno (whose brightest values
        # are yellow/white, near the direct beam where the center usually is)
        center_marker = self.ax_mask_preview.scatter([cx], [cy], color='lime', marker='+', s=80, linewidth=2)
        self._mask_artists.append(center_marker)
        self.update_recip_scale_circles()
        self.canvas.draw_idle()

    def update_recip_scale_circles(self):
        """Draw concentric dashed 1/A rings on the PACBED preview, centered
        on the found center (spinbox_centerX/Y) - in place of a conventional
        scale bar, which doesn't read naturally on a radially-symmetric
        diffraction pattern."""
        if not hasattr(self, 'pacbed'):
            return
        center = (self.spinbox_centerX.value(), self.spinbox_centerY.value())
        self._pacbed_recip_circles = io.draw_reciprocal_scale_circles(
            self.ax_mask_preview, self.lineEdit_scale_recip.text(), self.pacbed.shape,
            center=center, old_artists=getattr(self, '_pacbed_recip_circles', None))
        self.canvas.draw_idle()

    def auto_find_center(self):
        if not hasattr(self, 'pacbed'):
            qtw.QMessageBox.critical(self, 'No PACBED',
                'Click "Compute PACBED" first so there is a summed diffraction '
                'pattern to search for the center in.')
            return
        r_mask = self.spinbox_rOut.value() or 50
        y, x = io.find_dp_center(self.pacbed, r_mask=r_mask, det_shape=self.pacbed.shape)
        self.spinbox_centerX.setValue(int(round(x)))
        self.spinbox_centerY.setValue(int(round(y)))
        self.logger.info('Auto-found diffraction pattern center at (%.0f, %.0f).', x, y)

    def open_threshold_dialog(self):
        """Open the ThresholdDialog popup to check/adjust the real-space
        threshold on the last-tested navigation image *before* committing
        to the actual (file-reading) PACBED computation - kept off the main
        navigation-image plot, per user request."""
        if not hasattr(self, '_last_test_fn'):
            qtw.QMessageBox.critical(self, 'No Test Image',
                'Double-click a file in the list to compute a test navigation '
                'image first - the threshold is applied to that image.')
            return
        dlg = ThresholdDialog(self, self._last_test_img, self._last_test_fn)
        if dlg.exec_() == qtw.QDialog.Accepted:
            self.compute_pacbed_from_threshold(dlg.fn, dlg.mask, dlg.combo_threshMethod.currentText())

    def compute_pacbed_from_threshold(self, fn, mask, method):
        """Sum diffraction patterns only at the scan positions in `mask`
        (confirmed via the ThresholdDialog popup), instead of the whole
        scan - e.g. to exclude vacuum/background regions from the PACBED
        used to find the diffraction center."""
        dtype = os.path.splitext(fn)[-1]
        scanSize = self._get_current_scanSize()
        self.logger.info(
            'Computing PACBED from %s-thresholded scan positions of %s...', method, fn)
        self.button_pacbedFromThreshold.setDisabled(True)
        self._pacbed_tic = perf_counter()
        worker = WorkerThread_General(self._pacbed_from_mask_worker, 0, fn, dtype, scanSize,
                                      mask, self.logger)
        worker.signals.results.connect(self._on_pacbed_from_threshold_computed)
        worker.signals.error.connect(self._on_pacbed_failed)
        QThreadPool.globalInstance().start(worker)

    @staticmethod
    def _pacbed_from_mask_worker(fn, dtype, scanSize, mask, logger=None):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        y0, y1 = int(y_idx[0]), int(y_idx[-1])
        x0, x1 = int(x_idx[0]), int(x_idx[-1])
        roi = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        dp = load_dp(fn, roi=roi, mask=mask, dtype=dtype, scanSize=scanSize, dwellTime=1)
        if hasattr(dp, 'compute'):
            with io.LoggingProgressBar(logger, 'Thresholded PACBED'):
                dp = dp.compute()
        return np.asarray(dp)

    def _on_pacbed_from_threshold_computed(self, result, index):
        self.button_pacbedFromThreshold.setEnabled(True)
        self._on_pacbed_computed(result, index)  # also logs completion + duration

    def on_press_mask(self, event):
        """Grab the center marker or a circle edge for dragging - gated
        behind Ctrl so plain click/drag stays available for the toolbar's
        Pan/Zoom tool, matching the rest of the app's convention."""
        self._mask_drag_mode = None
        if (event.inaxes != self.ax_mask_preview or event.xdata is None
                or 'ctrl' not in event.modifiers):
            return
        cx = self.spinbox_centerX.value()
        cy = self.spinbox_centerY.value()
        r_in = self.spinbox_rIn.value()
        r_out = self.spinbox_rOut.value()

        # Hit-test in pixel space (via a 1-data-unit reference vector) so the
        # threshold doesn't depend on the current zoom level.
        p0 = self.ax_mask_preview.transData.transform((0, 0))
        p1 = self.ax_mask_preview.transData.transform((1, 0))
        px_per_data = np.hypot(p1[0] - p0[0], p1[1] - p0[1]) or 1
        threshold_data = 8 / px_per_data

        dist = np.hypot(event.xdata - cx, event.ydata - cy)
        if dist < threshold_data:
            self._mask_drag_mode = 'center'
        elif abs(dist - r_out) < threshold_data:
            self._mask_drag_mode = 'r_out'
        elif r_in > 0 and abs(dist - r_in) < threshold_data:
            self._mask_drag_mode = 'r_in'

    def on_motion_mask(self, event):
        if (self._mask_drag_mode is None or event.inaxes != self.ax_mask_preview
                or event.xdata is None or event.ydata is None):
            return
        if self._mask_drag_mode == 'center':
            self.spinbox_centerX.setValue(int(round(event.xdata)))
            self.spinbox_centerY.setValue(int(round(event.ydata)))
        else:
            cx = self.spinbox_centerX.value()
            cy = self.spinbox_centerY.value()
            r = int(round(np.hypot(event.xdata - cx, event.ydata - cy)))
            if self._mask_drag_mode == 'r_out':
                self.spinbox_rOut.setValue(max(r, self.spinbox_rIn.value() + 1))
            elif self._mask_drag_mode == 'r_in':
                self.spinbox_rIn.setValue(min(r, self.spinbox_rOut.value() - 1))
        # The spinboxes' valueChanged -> update_mask_overlay connection
        # (wired at creation time) redraws the circles/marker automatically.

    def on_release_mask(self, event):
        self._mask_drag_mode = None

    def on_scroll_mask(self, event):
        """Zoom either plot in/out on Ctrl+scroll wheel, centered on the
        cursor position (despite the name, this handles both self.ax and
        self.ax_mask_preview - it dispatches on event.inaxes)."""
        ax = event.inaxes
        if (ax is None or event.xdata is None or event.ydata is None
                or 'ctrl' not in event.modifiers):
            return
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([event.xdata - new_width * (1 - relx), event.xdata + new_width * relx])
        ax.set_ylim([event.ydata - new_height * (1 - rely), event.ydata + new_height * rely])
        self.canvas.draw_idle()

    def on_press_navsig(self, event):
        """Start dragging a scan-space ROI on the nav/test image (Ctrl+drag),
        mirroring the ROI-drawing convention already used on tab ROI on 4D -
        this ROI only ever feeds "PACBED from ROI", never "Compute PACBED".
        A plain right-click instead deletes the currently-drawn ROI."""
        if event.inaxes != self.ax:
            self._navsig_press = None
            return
        if event.button == 3:
            self._navsig_press = None
            self.clear_navsig_roi()
            return
        if event.xdata is None or 'ctrl' not in event.modifiers or event.button != 1:
            self._navsig_press = None
            return
        self._navsig_press = (event.xdata, event.ydata)
        if self.rect_navsig is not None:
            try:
                self.rect_navsig.remove()
            except Exception:
                pass
        self.rect_navsig = patches.Rectangle(self._navsig_press, 0, 0, linewidth=1,
                                             edgecolor='r', facecolor='none')
        self.ax.add_patch(self.rect_navsig)
        self.canvas.draw()
        self._navsig_bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def on_motion_navsig(self, event):
        if (self._navsig_press is None or event.inaxes != self.ax
                or event.xdata is None or event.ydata is None):
            return
        x0, y0 = self._navsig_press
        width = event.xdata - x0
        height = event.ydata - y0
        try:
            self.rect_navsig.set_width(width)
            self.rect_navsig.set_height(height)
            self.rect_navsig.set_xy((x0, y0))
        except AttributeError:
            self._navsig_press = None
            return
        self.canvas.restore_region(self._navsig_bg)
        self.ax.draw_artist(self.rect_navsig)
        self.canvas.blit(self.ax.bbox)

    def on_release_navsig(self, event):
        if self._navsig_press is None or event.inaxes is None:
            return
        x0, y0 = self._navsig_press
        width = event.xdata - x0
        height = event.ydata - y0
        if width < 0:
            width = abs(width)
            x0 = event.xdata
        if height < 0:
            height = abs(height)
            y0 = event.ydata
        width = max(int(width), 1)
        height = max(int(height), 1)
        self.roi_navsig = (int(x0), int(y0), width, height)
        self._navsig_press = None
        self.canvas.draw()
        self.button_pacbedFromRoi.setEnabled(True)
        self.logger.info('Nav. signal ROI: %s', self.roi_navsig)
        # Load the PACBED for this region immediately, as a live preview -
        # the user can still re-click "PACBED from ROI" later (e.g. after
        # switching files) to recompute it for the same region.
        self.compute_pacbed_from_roi()

    def clear_navsig_roi(self):
        """Remove the drawn scan-space ROI (right-click on the nav/test
        image) so it stops feeding "PACBED from ROI" - "Compute PACBED"
        itself never uses this ROI regardless."""
        had_roi = self.roi_navsig is not None
        self.roi_navsig = None
        if self.rect_navsig is not None:
            try:
                self.rect_navsig.remove()
            except Exception:
                pass
            self.rect_navsig = None
            self.canvas.draw_idle()
        self.button_pacbedFromRoi.setDisabled(True)
        if had_roi:
            self.logger.info('Cleared nav. signal ROI.')

    def calculate_button(self):
        self.path_main = self.lineEdit_dir_signal.text()
        if self.checkbox_selectAll.isChecked():
            fns = self.get_all_item_names()
        else:
            fns = self.file_list_widget.selectedItems()
            fns = [item.text() for item in fns]
            if len(fns) == 0:
                # The list is already filtered to this dtype by
                # refresh_file_list(); this is just a defensive fallback.
                dtype = self.combo_dtype.currentText()
                fns = self.get_all_item_names()
                fns = [fn for fn in fns if os.path.splitext(fn)[1] == dtype]
        fns = [os.path.join(self.path_main, fn) for fn in fns]
        fns.sort()

        dtype = [os.path.splitext(fn)[1] for fn in fns]
        dtype = np.array(dtype)
        dtype = np.unique(dtype)
        if len(dtype) != 1:
            self.logger.warning('Mixed file types found in directory: %s', list(dtype))
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
            self.logger.warning('Cannot start: scan size is required for .tpx3 files.')
            self.message_box_tpx3()
            return

        mask_params = None
        if self.checkbox_useMask.isChecked():
            r_in = self.spinbox_rIn.value()
            r_out = self.spinbox_rOut.value()
            if r_out <= r_in:
                self.logger.warning(
                    'Cannot start: virtual mask outer radius (%d) must be greater than '
                    'the inner radius (%d).', r_out, r_in)
                qtw.QMessageBox.critical(self, 'Invalid Virtual Mask',
                    'Outer radius must be greater than the inner radius.')
                return
            mask_params = (r_in, r_out, (self.spinbox_centerX.value(), self.spinbox_centerY.value()))
            self.logger.info(
                'Using virtual mask: center=%s, inner radius=%d, outer radius=%d.',
                mask_params[2], r_in, r_out)

        self.pathSave = self.lineEdit_dir_save.text()
        if not os.path.isdir(self.pathSave):
            os.mkdir(self.pathSave)

        self.button_stop.setEnabled(True)
        self.create_navigation_signal(fns, dtype, scanSize, dwellTime, mask_params)

    def create_navigation_signal(self, fns, dtype, scanSize, dwellTime, mask_params=None):
        self.nav_imgs = [None] * len(fns)
        self.nav_counter = 0
        self.nav_counter_total = len(fns)
        self._nav_tic = perf_counter()
        self._nav_failed = False
        self.logger.info('Starting navigation signal creation for %d file(s)...', len(fns))
        self.tasks = deque()
        for i, fn in enumerate(fns):
            self.tasks.append((fn, dtype, scanSize, dwellTime, i, mask_params))
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
        fn, dtype, scanSize, dwellTime, i_index, mask_params = self.tasks.popleft()
        scanSize_str = str(scanSize) if scanSize is not None else 'None'
        args = ['worker_nav_img.py', fn, dtype, scanSize_str, str(dwellTime), str(i_index)]
        if mask_params is not None:
            r_in, r_out, center = mask_params
            args += [str(r_in), str(r_out), str(center)]
        process = QProcess()
        process.setProgram(sys.executable)
        process.setArguments(args)
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
            self._nav_failed = True
            self.logger.error('Worker ERROR: %s', err)
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
            self.logger.error('Failed to decode nav image for index %s: %s', i_index, e)
        process.deleteLater()
        self.nav_counter += 1
        self.update_progress_bar(self.nav_counter, self.nav_counter_total)
        if self.nav_counter >= self.nav_counter_total:
            duration = perf_counter() - self._nav_tic
            valid = [img for img in self.nav_imgs if img is not None]
            if not valid:
                self.logger.error(
                    'Navigation signal creation failed: all worker processes '
                    'failed to produce output (after %.1f s).', duration)
                qtw.QMessageBox.critical(self, 'No Results', 'All worker processes failed to produce output.')
                self.button_stop.setDisabled(True)
                return
            self.nav_imgs = np.stack(valid)
            self.update_canvas(0)
            self.slider_imgNo.setRange(0, len(self.nav_imgs) - 1)
            self.button_stop.setDisabled(True)
            if self._nav_failed or len(valid) < self.nav_counter_total:
                self.logger.error(
                    'Navigation signal creation finished with errors: %d/%d file(s) '
                    'succeeded in %.1f s.', len(valid), self.nav_counter_total, duration)
            else:
                self.logger.info(
                    'Navigation signal creation completed successfully for %d file(s) in %.1f s.',
                    self.nav_counter_total, duration)
            self.save_results()
        else:
            self.launch_next_nav_task()

    def process_failed_nav(self, error):
        self._nav_failed = True
        self.logger.error("QProcess error occurred: %s", error)
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

    def cleanup(self):
        """Release resources held by this tab. Called by MainWindow.closeEvent
        so repeated runs of the app in the same console/kernel don't leave
        running subprocesses and matplotlib figures alive."""
        if hasattr(self, 'running_processes'):
            for p in list(self.running_processes):
                p.kill()
        self.log_console.disconnect_log()
        plt.close(self.figure)

    def save_results(self):
        tic = perf_counter()
        try:
            self._save_results_impl()
        except Exception:
            self.logger.exception('Failed to save navigation signal after %.1f s.',
                                   perf_counter() - tic)
            return
        self.logger.info(
            'Navigation signal saved successfully in %.1f s (background frame/clip '
            'generation continues asynchronously).', perf_counter() - tic)

    def _save_results_impl(self):
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
