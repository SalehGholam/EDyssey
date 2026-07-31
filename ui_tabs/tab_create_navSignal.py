# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:53:46 2024

@author: SGholam
"""

import os
import shutil
import tempfile
import json
import datetime
from time import perf_counter
from collections import deque
from PyQt5.QtCore import Qt, QProcess, QThreadPool
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QKeySequence
from PyQt5.QtWidgets import QShortcut
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
from .logging_utils import LogConsole
from .base_tab import TabBase
from .worker_thread import WorkerThread_General, ProcessStderrBuffer
from .worker_launch import worker_command
from .threshold_dialog import ThresholdDialog
from worker_extract_frame import load_dp
#%% class
class Tab_Create_NavSignal(TabBase):
    def __init__(self, parent=None):
        # own_threadpool=False: this tab's batch nav-image jobs are
        # dispatched via QProcess, not QThreadPool - the only QThreadPool
        # work it has is small one-off clip/frame-export jobs, which
        # deliberately share QThreadPool.globalInstance() with everything
        # else instead of getting a dedicated pool.
        super().__init__('Tab_Create_NavSignal', parent, own_threadpool=False)
        self._stderr_buffer = ProcessStderrBuffer()
        self.init_widget()

    def init_widget(self):
        self.central_widget = qtw.QWidget(self)
        self.layout = qtw.QHBoxLayout(self)
        self.setLayout(self.layout)
        self._splitter = qtw.QSplitter(Qt.Horizontal)
        self.layout.addWidget(self._splitter)

        width_userInput = 320
        button_w = 90
        button_h_lrg = 50

        self._left_widget = qtw.QWidget()
        self._left_widget.setFixedWidth(width_userInput)
        self._splitter.addWidget(self._left_widget)

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

        layout_project_name = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_project_name)

        label_project_name = qtw.QLabel('Project')
        label_project_name.setFixedWidth(55)
        layout_project_name.addWidget(label_project_name)

        self.lineEdit_projectName = qtw.QLineEdit()
        layout_project_name.addWidget(self.lineEdit_projectName)
        self.lineEdit_projectName.setToolTip(
            'Results are saved to "<Save Path>/<Project>_5DED Analysis/navigator signal/'
            '<timestamp>/" - defaults to the 4D signals folder name whenever that '
            'folder changes, but can be edited freely before saving.')

        layout_save_name = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_save_name)

        label_save_name = qtw.QLabel('Save Name')
        label_save_name.setFixedWidth(55)
        layout_save_name.addWidget(label_save_name)

        self.lineEdit_saveName = qtw.QLineEdit('navigation_signal')
        layout_save_name.addWidget(self.lineEdit_saveName)
        self.lineEdit_saveName.setToolTip(
            'Filename (without extension) for the saved navigation signal. Other tabs\' '
            '"Load Saved Analysis" looks for the default name "navigation_signal" '
            'specifically - rename only if you don\'t need that auto-discovery.')

        #%% box for scales
        self.box_scale = qtw.QGroupBox('Scale bars')
        layout_box_scale = qtw.QVBoxLayout()
        self.box_scale.setLayout(layout_box_scale)
        layout_dir.addWidget(self.box_scale)

        self.double_validator = QDoubleValidator(0.0, 1e5, 5)
        # real space
        layout_scale_real = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_real)
        label_scale_real = qtw.QLabel('Real (nm)')
        label_scale_real.setFixedWidth(55)
        layout_scale_real.addWidget(label_scale_real)
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        layout_scale_real.addWidget(self.lineEdit_scale_real)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        self.lineEdit_scale_real.textChanged.connect(lambda: self.update_canvas(
            self.slider_imgNo.value()))

        # reciprocal space
        layout_scale_recip = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_recip)
        label_scale_recip = qtw.QLabel('Recip. (Å<sup>-1</sup>)')
        label_scale_recip.setFixedWidth(55)
        layout_scale_recip.addWidget(label_scale_recip)
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        layout_scale_recip.addWidget(self.lineEdit_scale_recip)
        self.lineEdit_scale_recip.setValidator(self.double_validator)
        self.lineEdit_scale_recip.setToolTip(
            'Reciprocal-space calibration (1/A per pixel) for the Summed DP preview - '
            'drawn as concentric dashed rings every 1 1/A, centered on the found center')
        self.lineEdit_scale_recip.textChanged.connect(self.update_recip_scale_circles)
        #%% scan size
        self.box_scanSize = qtw.QGroupBox('Scan Size')
        layout_dir_scanSize.addWidget(self.box_scanSize)
        layout_scanSize = qtw.QVBoxLayout()
        self.box_scanSize.setLayout(layout_scanSize)

        layout_scanSize_row1 = qtw.QHBoxLayout()
        layout_scanSize.addLayout(layout_scanSize_row1)

        self.checkbox_scanSize = qtw.QCheckBox('Auto')
        layout_scanSize_row1.addWidget(self.checkbox_scanSize)
        self.checkbox_scanSize.setChecked(True)

        self.lineEdit_scanSize_x = qtw.QLineEdit()
        self.lineEdit_scanSize_x.setAlignment(Qt.AlignLeft)
        layout_scanSize_row1.addWidget(self.lineEdit_scanSize_x)
        self.lineEdit_scanSize_x.setFixedWidth(40)
        self.lineEdit_scanSize_x.setValidator(QIntValidator(0,99999))

        label_cross = qtw.QLabel('X')
        layout_scanSize_row1.addWidget(label_cross)

        self.lineEdit_scanSize_y = qtw.QLineEdit()
        layout_scanSize_row1.addWidget(self.lineEdit_scanSize_y)
        self.lineEdit_scanSize_y.setFixedWidth(40)
        self.lineEdit_scanSize_y.setValidator(QIntValidator(0,99999))
        self.activate_lineEdit_scanSize()
        self.checkbox_scanSize.stateChanged.connect(self.activate_lineEdit_scanSize)

        label_dwellTime = qtw.QLabel('Dwell T. (\u03BCs)')
        label_dwellTime.setToolTip('Dwell time in microseconds')
        self.spinbox_dwellTime = qtw.QSpinBox()
        self.spinbox_dwellTime.setFixedWidth(60)
        self.spinbox_dwellTime.setRange(1, 99999999)
        for wid in [label_dwellTime, self.spinbox_dwellTime]:
            layout_scanSize_row1.addWidget(wid)
        layout_scanSize_row1.addStretch(1)

        # metadata (comment.txt) auto-fill - tpx3 acquisitions log scan
        # size/dwell time there, alongside the .tpx3 file(s).
        layout_scanSize_row2 = qtw.QHBoxLayout()
        layout_scanSize.addLayout(layout_scanSize_row2)

        label_metadataCount = qtw.QLabel('Block #')
        label_metadataCount.setToolTip(
            'Which 0-indexed metadata block to read from comment.txt. Only '
            'enabled when comment.txt logs more than one measurement')
        layout_scanSize_row2.addWidget(label_metadataCount)
        self.spinbox_metadataCount = qtw.QSpinBox()
        self.spinbox_metadataCount.setFixedWidth(50)
        self.spinbox_metadataCount.setRange(0, 99999)
        self.spinbox_metadataCount.setValue(0)
        self.spinbox_metadataCount.setDisabled(True)  # re-enabled once >1 block is found
        layout_scanSize_row2.addWidget(self.spinbox_metadataCount)

        self.button_loadMetadata = qtw.QPushButton('Load')
        self.button_loadMetadata.setToolTip(
            'Fill scan size / dwell time from comment.txt in the 4D signals '
            'folder (tpx3 acquisitions only)')
        layout_scanSize_row2.addWidget(self.button_loadMetadata)
        self.button_loadMetadata.clicked.connect(lambda: self.load_metadata(silent=False))

        self.button_browseMetadata = qtw.QPushButton('...')
        self.button_browseMetadata.setFixedWidth(30)
        self.button_browseMetadata.setToolTip(
            'Browse for the metadata file (defaults to comment.txt in the 4D signals folder)')
        layout_scanSize_row2.addWidget(self.button_browseMetadata)
        self.button_browseMetadata.clicked.connect(self.browse_metadata_file)
        layout_scanSize_row2.addStretch(1)

        self.metadata_path_override = None  # set by browse_metadata_file(); cleared on new 4D folder
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
        self.combo_dtype.setMaximumWidth(130)
        layout_dtype.addWidget(self.combo_dtype)
        self.combo_dtype.addItems(['.tpx3', '.hdf5', '.hspy', '.zspy', '.mib'])
        self.combo_dtype.setDisabled(True)
        self.checkbox_selectAll.stateChanged.connect(self.activate_combo_dtype)
        self.combo_dtype.currentIndexChanged.connect(self.refresh_file_list)

        #%% calculate button + CPU cores
        # Two rows, not one wide one: 3 buttons at 100px each already equal
        # the whole panel's default width on their own, so packing the
        # cores/fps spinboxes into that same row too (as before) forced the
        # panel's natural minimum width far past 300px - the underlying
        # cause of the panel-width/line-edit-sizing complaints.
        layout_calculate_buttons = qtw.QHBoxLayout()
        layout_userInput.addLayout(layout_calculate_buttons)
        layout_calculate_buttons.setAlignment(Qt.AlignLeft)

        self.button_testFile = qtw.QPushButton('Test File')
        self.button_testFile.setFixedSize(button_w, button_h_lrg)
        layout_calculate_buttons.addWidget(self.button_testFile)
        self.button_testFile.clicked.connect(lambda: self.test_selected_file(None))
        self.button_testFile.setToolTip(
            'Compute/preview the navigation image for the selected (or first) '
            'file only, without running the full batch - same as double-clicking it')

        self.button_calculate = qtw.QPushButton('Calculate All')
        self.button_calculate.setFixedSize(button_w, button_h_lrg)
        layout_calculate_buttons.addWidget(self.button_calculate)
        self.button_calculate.clicked.connect(self.calculate_button)
        self.button_calculate.setToolTip('Run the full batch over every listed (or selected) file')

        layout_calculate_buttons2 = qtw.QHBoxLayout()
        layout_userInput.addLayout(layout_calculate_buttons2)

        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.setFixedSize(button_w, button_h_lrg)
        layout_calculate_buttons.addWidget(self.button_cancel)
        self.button_cancel.setStyleSheet("background-color: red; color: white;")
        self.button_cancel.setDisabled(True)
        self.button_cancel.clicked.connect(self.cancel_running_work)
        self.button_cancel.setToolTip(
            'Stop the running navigation signal creation. Already-running '
            'background computations finish silently; their results are discarded.')


        #%% save
        layout_save = qtw.QHBoxLayout()
        layout_calculate_buttons2.addLayout(layout_save)
        
        self.button_save_results = qtw.QPushButton('Save Results')
        self.button_save_results.setFixedSize(button_w, button_h_lrg)
        layout_save.addWidget(self.button_save_results)
        self.button_save_results.clicked.connect(self.save_results)
        self.button_save_results.setToolTip(
            'Save the navigation signal, frames, and clip to the Save Path above')
        self.button_save_results.setDisabled(True)
        
        layout_save_options = qtw.QVBoxLayout()
        layout_save.addLayout(layout_save_options)
        layout_save_cpu = qtw.QHBoxLayout()
        layout_save_cpu.setAlignment(Qt.AlignLeft)
        layout_save_options.addLayout(layout_save_cpu)

        label_cores = qtw.QLabel('CPU Cores')
        label_cores.setAlignment(Qt.AlignLeft)
        label_cores.setFixedWidth(60)
        self.spinbox_cpuCores = qtw.QSpinBox()
        self.spinbox_cpuCores.setFixedWidth(40)
        self.spinbox_cpuCores.setRange(1, os.cpu_count() or 1)
        self.spinbox_cpuCores.setValue(1)
        self.spinbox_cpuCores.setToolTip('Number of parallel worker processes for nav image computation')
        for wid in [label_cores, self.spinbox_cpuCores]:
            layout_save_cpu.addWidget(wid)


        layout_save_clip = qtw.QHBoxLayout()
        layout_save_clip.setAlignment(Qt.AlignLeft)
        layout_save_options.addLayout(layout_save_clip)
        label_fps = qtw.QLabel('Clip FPS')
        label_fps.setAlignment(Qt.AlignLeft)
        label_fps.setFixedWidth(60)
        self.spinbox_fps = qtw.QSpinBox()
        self.spinbox_fps.setFixedWidth(40)
        self.spinbox_fps.setRange(1, 60)
        self.spinbox_fps.setValue(5)
        self.spinbox_fps.setToolTip('Frames per second for the navigation clip')
        for wid in [label_fps, self.spinbox_fps]:
            layout_save_clip.addWidget(wid)

        self.checkbox_autosave = qtw.QCheckBox('Autosave')
        layout_save_options.addWidget(self.checkbox_autosave)
        self.checkbox_autosave.setToolTip(
            'Automatically save when "Calculate All" finishes, instead of needing '
            'to click "Save Results" manually')

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

        self.checkbox_useMask = qtw.QCheckBox('Use Virtual Mask')
        layout_mask.addWidget(self.checkbox_useMask)
        self.checkbox_useMask.setToolTip(
            'When checked, the navigation image sums each diffraction pattern only '
            'within the annular region below, instead of the whole detector')

        layout_mask_center = qtw.QHBoxLayout()
        layout_mask.addLayout(layout_mask_center)
        label_centerX = qtw.QLabel('Center X')
        layout_mask_center.addWidget(label_centerX)
        self.spinbox_centerX = qtw.QSpinBox()
        self.spinbox_centerX.setMaximumWidth(70)
        self.spinbox_centerX.setRange(0, 8192)
        self.spinbox_centerX.setValue(256)
        layout_mask_center.addWidget(self.spinbox_centerX)
        label_centerY = qtw.QLabel('Center Y')
        layout_mask_center.addWidget(label_centerY)
        self.spinbox_centerY = qtw.QSpinBox()
        self.spinbox_centerY.setMaximumWidth(70)
        self.spinbox_centerY.setRange(0, 8192)
        self.spinbox_centerY.setValue(256)
        layout_mask_center.addWidget(self.spinbox_centerY)

        layout_mask_radii = qtw.QHBoxLayout()
        layout_mask.addLayout(layout_mask_radii)
        label_rIn = qtw.QLabel('Inner R')
        layout_mask_radii.addWidget(label_rIn)
        self.spinbox_rIn = qtw.QSpinBox()
        self.spinbox_rIn.setMaximumWidth(70)
        self.spinbox_rIn.setRange(0, 4096)
        self.spinbox_rIn.setValue(0)
        self.spinbox_rIn.setSingleStep(10)
        layout_mask_radii.addWidget(self.spinbox_rIn)
        label_rOut = qtw.QLabel('Outer R')
        layout_mask_radii.addWidget(label_rOut)
        self.spinbox_rOut = qtw.QSpinBox()
        self.spinbox_rOut.setMaximumWidth(70)
        self.spinbox_rOut.setRange(1, 4096)
        self.spinbox_rOut.setValue(256)
        self.spinbox_rOut.setSingleStep(10)
        layout_mask_radii.addWidget(self.spinbox_rOut)
        for sb in (self.spinbox_rIn, self.spinbox_rOut):
            sb.setToolTip('Up/down arrows step by 10; type a value directly for finer control')

        for sb in (self.spinbox_centerX, self.spinbox_centerY, self.spinbox_rIn, self.spinbox_rOut):
            sb.valueChanged.connect(self.update_mask_overlay)

        # Each row here holds at most one long-label button - a long label
        # can't shrink below its text's natural width (no eliding on
        # QPushButton), so packing several per row is what forced the whole
        # panel wider than intended; \n-wrapping the longer labels keeps
        # each button itself narrower too.
        layout_mask_buttons = qtw.QHBoxLayout()
        layout_mask.addLayout(layout_mask_buttons)

        self.button_autoCenter = qtw.QPushButton('Auto-Find\nCenter')
        self.button_autoCenter.setFixedSize(button_w-10, button_h_lrg)
        layout_mask_buttons.addWidget(self.button_autoCenter)
        self.button_autoCenter.clicked.connect(self.auto_find_center)
        self.button_autoCenter.setToolTip('Auto-detect the direct-beam center from the Summed DP preview')

        self.checkbox_autoCenterDp = qtw.QCheckBox('Auto-center')
        self.checkbox_autoCenterDp.setChecked(True)
        self.checkbox_autoCenterDp.setToolTip(
            'When checked, the center is re-found automatically (large-sigma blur) '
            'every time a new Summed DP is computed. When unchecked, set the center '
            'manually via the spinboxes above or by Ctrl+dragging the "+" marker.')
        layout_mask_buttons.addWidget(self.checkbox_autoCenterDp)
        self.checkbox_autoCenterDp.stateChanged.connect(self.update_recip_scale_circles)


        layout_sum_dp = qtw.QHBoxLayout()
        layout_mask.addLayout(layout_sum_dp)
        layout_sum_dp.setAlignment(Qt.AlignLeft)

        self.button_computeSumDp = qtw.QPushButton('Compute\nSummed DP')
        self.button_computeSumDp.setFixedSize(button_w-10, button_h_lrg)
        layout_sum_dp.addWidget(self.button_computeSumDp)
        self.button_computeSumDp.clicked.connect(self.compute_sum_dp)
        self.button_computeSumDp.setToolTip(
            'Sum all diffraction patterns of the selected (or first) file to find '
            'the detector center - needed to place the virtual mask')

        self.button_sumDpFromThreshold = qtw.QPushButton('Summed DP\nby Threshold')
        self.button_sumDpFromThreshold.setFixedSize(button_w-10, button_h_lrg)
        layout_sum_dp.addWidget(self.button_sumDpFromThreshold)
        self.button_sumDpFromThreshold.clicked.connect(self.open_threshold_dialog)
        self.button_sumDpFromThreshold.setToolTip(
            'Open a window to check/adjust a real-space threshold on the last '
            'tested navigation image, then sum diffraction patterns only at the '
            'scan positions above it, instead of the whole scan - for checking '
            'purposes only, not a substitute for "Compute Summed DP"')

        self.button_sumDpFromRoi = qtw.QPushButton('Summed DP\nfrom ROI')
        self.button_sumDpFromRoi.setFixedSize(button_w-10, button_h_lrg)
        layout_sum_dp.addWidget(self.button_sumDpFromRoi)
        self.button_sumDpFromRoi.clicked.connect(self.compute_sum_dp_from_roi)
        self.button_sumDpFromRoi.setDisabled(True)
        self.button_sumDpFromRoi.setToolTip(
            'Sum diffraction patterns only over the scan-space rectangle drawn '
            'on the navigation/test image (hold Ctrl and drag there), instead '
            'of the whole scan')

        # The Summed DP preview canvas itself lives in the main window, beside
        # the navigation image (see #%% canvas below), so it has more room
        # to work with and its own navigation toolbar.
        self._mask_artists = []
        self._mask_drag_mode = None
        # Scan-space ROI drawn (Ctrl+drag) on the nav/test image, used only
        # by "Summed DP from ROI" - never by "Compute Summed DP", which always
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
            'Hold Ctrl and drag to draw a scan-space ROI (auto-loads its Summed DP).\n'
            'Right-click to remove it.',
            fontsize=10)
        self.ax.xaxis.label.set_visible(True)
        self.colorbar_nav = self.figure.colorbar(
            self.img_display, ax=self.ax, fraction=0.046, pad=0.04)

        self.img_display_mask = self.ax_mask_preview.imshow(
            np.zeros((512, 512), dtype='uint16'), cmap='inferno')
        self.img_display_mask.set_norm(SymLogNorm(linthresh=1))
        self.ax_mask_preview.set_title('Summed DP', fontsize=9)
        # ax_mask_preview keeps its x-axis label visible (for the
        # interaction hint below), so its ticks/spines are hidden
        # individually instead of via set_axis_off() - that sets
        # axison=False, which suppresses the *entire* axis decoration set at
        # draw time (including the xlabel) regardless of the label artist's
        # own set_visible(True).
        for spine in self.ax_mask_preview.spines.values():
            spine.set_visible(False)
        self.ax_mask_preview.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax_mask_preview.xaxis.label.set_visible(True)
        self.colorbar_mask = self.figure.colorbar(
            self.img_display_mask, ax=self.ax_mask_preview, fraction=0.046, pad=0.04)

        # The Ctrl+Scroll zoom hint applies to every axis on this canvas, so
        # it's a figure-wide supxlabel rather than repeated per-axis text.
        self.figure.supxlabel('Hold "Ctrl" + Scroll wheel to zoom either plot', fontsize=10)

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

        # keyboard shortcuts
        QShortcut(QKeySequence('Ctrl+Right'), self,
                  lambda: self.slider_imgNo.setValue(self.slider_imgNo.value() + 1))
        QShortcut(QKeySequence('Ctrl+Left'), self,
                  lambda: self.slider_imgNo.setValue(self.slider_imgNo.value() - 1))

        self.canvas.mpl_connect('scroll_event', self.on_scroll_mask)
        self.canvas.mpl_connect('button_press_event', self.on_press_mask)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_mask)
        self.canvas.mpl_connect('button_release_event', self.on_release_mask)
        # Separate handlers for the nav/test image's own Ctrl+drag ROI
        # (used by "Summed DP from ROI") - each checks event.inaxes for its own
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
                self.metadata_path_override = None  # new folder - re-derive comment.txt location
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
        if os.path.isdir(directory) and any(
                f.endswith('.tpx3') for f in os.listdir(directory)):
            self.load_metadata(silent=True)

    def browse_metadata_file(self):
        start_dir = self.lineEdit_dir_signal.text()
        path, _ = qtw.QFileDialog.getOpenFileName(
            self, "Select Metadata File", start_dir, "Text files (*.txt);;All Files (*)")
        if path:
            self.metadata_path_override = path
            self.load_metadata(silent=False)

    def load_metadata(self, silent=True):
        """Fill scan size / dwell time from a comment.txt in the 4D signals
        folder, if present (tpx3 acquisitions log scan metadata there).
        silent=True swallows a missing/unparsable comment.txt quietly (used
        for the automatic per-folder attempt); silent=False (the "Load
        Metadata" button, or a manually-browsed file) surfaces the failure
        to the user."""
        path_main = self.metadata_path_override or self.lineEdit_dir_signal.text()
        try:
            n_blocks = io.get_metadata_block_count(path_main)
        except Exception:
            n_blocks = 0
        self.spinbox_metadataCount.setEnabled(n_blocks > 1)
        if n_blocks <= 1:
            self.spinbox_metadataCount.setValue(0)
        count = self.spinbox_metadataCount.value()
        fn_used = path_main if os.path.isfile(path_main) else os.path.join(path_main, 'comment.txt')
        try:
            metadata = io.get_metadata(path_main, count=count)
            if not metadata:
                raise ValueError('comment.txt contained no parsable metadata')
            if 'scan size x' in metadata and 'scan size y' in metadata:
                self.lineEdit_scanSize_x.setText(str(int(metadata['scan size x'])))
                self.lineEdit_scanSize_y.setText(str(int(metadata['scan size y'])))
                self.checkbox_scanSize.setChecked(False)
            if 'dwelltime' in metadata:
                self.spinbox_dwellTime.setValue(int(metadata['dwelltime']))
            self.logger.info('Loaded scan metadata (block %d) from %s.', count, fn_used)
        except Exception as e:
            if silent:
                self.logger.debug('No comment.txt metadata loaded from %s (%s).', path_main, e)
            else:
                self.logger.warning('Could not load metadata from comment.txt in %s: %s',
                                     path_main, e)
                qtw.QMessageBox.warning(self, 'Metadata Not Loaded',
                    f'Could not read metadata from comment.txt in:\n{path_main}\n\n{e}')

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
        self.lineEdit_projectName.setText(os.path.basename(p.rstrip('/\\')))

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
        """Resolve the file to use for test/Summed DP actions: the item that
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
        # Re-seed the toolbar's view stack so its "Home" button resets to
        # *this* (correct, full-extent) view - otherwise Home falls back to
        # whatever the canvas showed before any data was loaded, since our
        # own Ctrl+Scroll zoom deliberately bypasses the toolbar's stack.
        self.toolbar.update()
        self.toolbar.push_current()
        self.canvas.draw_idle()
        self.logger.info('Test navigation image computed successfully for %s.', fn)

    def compute_sum_dp(self):
        """Sum all diffraction patterns of one representative file to get a
        Summed DP - the reference image used to place the virtual mask. Always
        uses the whole scan, regardless of any ROI currently drawn on the
        nav/test image - for a restricted Summed DP instead, use "Summed DP from
        Threshold..." or "Summed DP from ROI"."""
        fn = self._get_test_fn()
        if fn is None or not os.path.isfile(fn):
            qtw.QMessageBox.critical(self, 'No File',
                'Load a directory (and optionally select a file) before computing a Summed DP.')
            return
        dtype = os.path.splitext(fn)[-1]
        scanSize = self._get_current_scanSize()
        if dtype == '.tpx3' and scanSize is None:
            self.logger.warning('Cannot compute Summed DP for %s: scan size is required for .tpx3 files.', fn)
            self.message_box_tpx3()
            return

        self.logger.info('Computing Summed DP (summed diffraction pattern) from %s...', fn)
        self.button_computeSumDp.setDisabled(True)
        self._sum_dp_tic = perf_counter()
        worker = WorkerThread_General(io.get_sum_dp, 0, fn, dtype=dtype, scanSize=scanSize,
                                      dwellTime=self.spinbox_dwellTime.value(), roi=None,
                                      logger=self.logger)
        worker.signals.results.connect(self._on_sum_dp_computed)
        worker.signals.error.connect(self._on_sum_dp_failed)
        QThreadPool.globalInstance().start(worker)

    def compute_sum_dp_from_roi(self):
        """Sum diffraction patterns only over the scan-space rectangle drawn
        (Ctrl+drag) on the nav/test image - "Compute Summed DP" itself always
        ignores this ROI and sums the whole scan instead."""
        if not self.roi_navsig:
            qtw.QMessageBox.critical(self, 'No ROI',
                'Hold Ctrl and drag on the navigation/test image to draw a '
                'scan-space ROI first.')
            return
        fn = self._get_test_fn()
        if fn is None or not os.path.isfile(fn):
            qtw.QMessageBox.critical(self, 'No File',
                'Load a directory (and optionally select a file) before computing a Summed DP.')
            return
        dtype = os.path.splitext(fn)[-1]
        scanSize = self._get_current_scanSize()
        if dtype == '.tpx3' and scanSize is None:
            self.logger.warning('Cannot compute Summed DP for %s: scan size is required for .tpx3 files.', fn)
            self.message_box_tpx3()
            return

        self.logger.info('Computing Summed DP from ROI %s of %s...', self.roi_navsig, fn)
        self.button_computeSumDp.setDisabled(True)
        self.button_sumDpFromRoi.setDisabled(True)
        self._sum_dp_tic = perf_counter()
        worker = WorkerThread_General(io.get_sum_dp, 0, fn, dtype=dtype, scanSize=scanSize,
                                      dwellTime=self.spinbox_dwellTime.value(),
                                      roi=self.roi_navsig, logger=self.logger)
        worker.signals.results.connect(self._on_sum_dp_from_roi_computed)
        worker.signals.error.connect(self._on_sum_dp_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_sum_dp_from_roi_computed(self, result, index):
        self.button_sumDpFromRoi.setEnabled(True)
        self._on_sum_dp_computed(result, index)  # also logs completion + duration

    def _on_sum_dp_failed(self, traceback_text, index):
        # Without this, a failure in the background computation (which
        # WorkerThread_General now catches instead of letting it vanish
        # silently) would otherwise leave the triggering button disabled
        # forever, with no way to retry and no visible sign anything went wrong.
        self.button_computeSumDp.setEnabled(True)
        self.button_sumDpFromThreshold.setEnabled(True)
        self.button_sumDpFromRoi.setEnabled(True)
        self.logger.error('Failed to compute Summed DP:\n%s', traceback_text)
        qtw.QMessageBox.critical(self, 'Summed DP Failed',
            'Computing the Summed DP failed - see the log for details.')

    def _on_sum_dp_computed(self, result, index):
        self.button_computeSumDp.setEnabled(True)
        self.sum_dp = result
        det_y, det_x = self.sum_dp.shape
        for sb, val in ((self.spinbox_centerX, det_x), (self.spinbox_centerY, det_y),
                       (self.spinbox_rIn, max(det_x, det_y)), (self.spinbox_rOut, max(det_x, det_y))):
            sb.setMaximum(val)
        self.img_display_mask.set_data(self.sum_dp)
        self.img_display_mask.set_clim(vmin=1, vmax=self.sum_dp.max())
        self.img_display_mask.set_extent([0, det_x, det_y, 0])
        self.ax_mask_preview.set_xlim(0, det_x)
        self.ax_mask_preview.set_ylim(det_y, 0)
        # Re-seed the toolbar's view stack so its "Home" button resets to
        # *this* (correct, full-extent) view instead of a stale/placeholder
        # one - see the identical comment in _on_test_result.
        self.toolbar.update()
        self.toolbar.push_current()
        if self.checkbox_autoCenterDp.isChecked():
            self.auto_find_center(silent=True)
        self.update_mask_overlay()
        duration = perf_counter() - self._sum_dp_tic if hasattr(self, '_sum_dp_tic') else None
        if duration is not None:
            self.logger.info('Summed DP computed successfully (%d x %d px) in %s.',
                             det_x, det_y, io.format_duration_hms(duration))
        else:
            self.logger.info('Summed DP computed successfully (%d x %d px).', det_x, det_y)

    def update_mask_overlay(self):
        """Redraw the inner/outer radius circles and center marker on the
        Summed DP preview to match the current spinbox values."""
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
        """Draw concentric dashed 1/A rings on the Summed DP preview, centered
        on the found center (spinbox_centerX/Y) - in place of a conventional
        scale bar, which doesn't read naturally on a radially-symmetric
        diffraction pattern."""
        if not hasattr(self, 'sum_dp'):
            return
        center = (self.spinbox_centerX.value(), self.spinbox_centerY.value())
        self._sum_dp_recip_circles = io.draw_reciprocal_scale_circles(
            self.ax_mask_preview, self.lineEdit_scale_recip.text(), self.sum_dp.shape,
            center=center, old_artists=getattr(self, '_sum_dp_recip_circles', None))
        centering_mode = ('auto (large-sigma blur)' if self.checkbox_autoCenterDp.isChecked()
                          else 'manual - drag the "+" or use the spinboxes')
        self.ax_mask_preview.set_xlabel(
            f'Circle center: {centering_mode}\n'
            'Hold Ctrl and drag the center (+) or a circle edge to move/resize the virtual mask.',
            fontsize=10)
        self.canvas.draw_idle()

    def auto_find_center(self, silent=False):
        if not hasattr(self, 'sum_dp'):
            if not silent:
                qtw.QMessageBox.critical(self, 'No Summed DP',
                    'Click "Compute Summed DP" first so there is a summed diffraction '
                    'pattern to search for the center in.')
            return
        x, y = io.find_dp_center_blurred(self.sum_dp)
        self.spinbox_centerX.setValue(int(round(x)))
        self.spinbox_centerY.setValue(int(round(y)))
        self.logger.info('Auto-found diffraction pattern center at (%.0f, %.0f).', x, y)

    def open_threshold_dialog(self):
        """Open the ThresholdDialog popup to check/adjust the real-space
        threshold on the last-tested navigation image *before* committing
        to the actual (file-reading) Summed DP computation - kept off the main
        navigation-image plot, per user request."""
        if not hasattr(self, '_last_test_fn'):
            qtw.QMessageBox.critical(self, 'No Test Image',
                'Double-click a file in the list to compute a test navigation '
                'image first - the threshold is applied to that image.')
            return
        dlg = ThresholdDialog(self, self._last_test_img, self._last_test_fn)
        if dlg.exec_() == qtw.QDialog.Accepted:
            self.compute_sum_dp_from_threshold(dlg.fn, dlg.mask, dlg.combo_threshMethod.currentText())

    def compute_sum_dp_from_threshold(self, fn, mask, method):
        """Sum diffraction patterns only at the scan positions in `mask`
        (confirmed via the ThresholdDialog popup), instead of the whole
        scan - e.g. to exclude vacuum/background regions from the Summed DP
        used to find the diffraction center."""
        dtype = os.path.splitext(fn)[-1]
        scanSize = self._get_current_scanSize()
        self.logger.info(
            'Computing Summed DP from %s-thresholded scan positions of %s...', method, fn)
        self.button_sumDpFromThreshold.setDisabled(True)
        self._sum_dp_tic = perf_counter()
        worker = WorkerThread_General(self._sum_dp_from_mask_worker, 0, fn, dtype, scanSize,
                                      mask, self.logger)
        worker.signals.results.connect(self._on_sum_dp_from_threshold_computed)
        worker.signals.error.connect(self._on_sum_dp_failed)
        QThreadPool.globalInstance().start(worker)

    @staticmethod
    def _sum_dp_from_mask_worker(fn, dtype, scanSize, mask, logger=None):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        y0, y1 = int(y_idx[0]), int(y_idx[-1])
        x0, x1 = int(x_idx[0]), int(x_idx[-1])
        roi = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        dp = load_dp(fn, roi=roi, mask=mask, dtype=dtype, scanSize=scanSize, dwellTime=1)
        if hasattr(dp, 'compute'):
            with io.LoggingProgressBar(logger, 'Thresholded Summed DP'):
                dp = dp.compute()
        return np.asarray(dp)

    def _on_sum_dp_from_threshold_computed(self, result, index):
        self.button_sumDpFromThreshold.setEnabled(True)
        self._on_sum_dp_computed(result, index)  # also logs completion + duration

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
        this ROI only ever feeds "Summed DP from ROI", never "Compute Summed DP".
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
        self.button_sumDpFromRoi.setEnabled(True)
        self.logger.info('Nav. signal ROI: %s', self.roi_navsig)
        # Load the Summed DP for this region immediately, as a live preview -
        # the user can still re-click "Summed DP from ROI" later (e.g. after
        # switching files) to recompute it for the same region.
        self.compute_sum_dp_from_roi()

    def clear_navsig_roi(self):
        """Remove the drawn scan-space ROI (right-click on the nav/test
        image) so it stops feeding "Summed DP from ROI" - "Compute Summed DP"
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
        self.button_sumDpFromRoi.setDisabled(True)
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
            except ValueError:
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

        # Captured now (while these are all known) rather than re-derived at
        # save time, since "Save Results" can be clicked well after
        # "Calculate All" finishes - see _save_results_impl/metadata.json.
        self._analysis_metadata = {
            '4d_signals_directory': self.path_main,
            'n_files': len(fns),
            'files': [os.path.basename(fn) for fn in fns],
            'dtype': dtype,
            'scan_size': list(scanSize) if scanSize is not None else None,
            'dwell_time_us': dwellTime,
            'virtual_detector': {
                'used': mask_params is not None,
                'inner_radius': mask_params[0] if mask_params else None,
                'outer_radius': mask_params[1] if mask_params else None,
                'center': list(mask_params[2]) if mask_params else None,
            },
            'comment_txt_metadata_source': {
                'path': self.metadata_path_override or self.path_main,
                'block': self.spinbox_metadataCount.value(),
            } if dtype == '.tpx3' else None,
        }

        self.pathSave = self.lineEdit_dir_save.text()
        if not os.path.isdir(self.pathSave):
            os.mkdir(self.pathSave)

        self.button_cancel.setEnabled(True)
        self.button_save_results.setDisabled(True)
        self.create_navigation_signal(fns, dtype, scanSize, dwellTime, mask_params)

    def create_navigation_signal(self, fns, dtype, scanSize, dwellTime, mask_params=None):
        self.nav_imgs = [None] * len(fns)
        self.nav_counter = 0
        self.nav_counter_total = len(fns)
        self._nav_tic = perf_counter()
        self._nav_failed = False
        self._cancelling = False
        # Each worker saves its result to a .npy file in here and prints
        # just the path back, instead of a base64+pickle-encoded copy of
        # the array through stdout - see launch_next_nav_task/
        # handle_finished_nav and worker_nav_img.py's docstring for why.
        self._navimg_temp_dir = tempfile.mkdtemp(prefix='py5ded_navimg_')
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
        args = [fn, dtype, scanSize_str, str(dwellTime), str(i_index), self._navimg_temp_dir]
        if mask_params is not None:
            r_in, r_out, center = mask_params
            args += [str(r_in), str(r_out), str(center)]
        program, arguments = worker_command('nav_img', args)
        process = QProcess()
        process.setProgram(program)
        process.setArguments(arguments)
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
        """eventem's native tpx3-loading progress lands on stderr - not an
        error. With several worker processes running in parallel, live-
        logging every progress tick from all of them noticeably slowed the
        batch down, and a per-file progress bar isn't that useful anyway
        (the "Test File" in-thread path still shows live progress, and this
        tab's own progress bar widget already tracks overall file-by-file
        completion) - so this doesn't route it to the Qt log console. It's
        still echoed to the real console, though (like it always was,
        before any of this app's own logging existed) - it's just not
        going to the GUI, not vanishing entirely - and it's buffered so it
        can be surfaced if this worker turns out to have actually failed,
        in handle_finished_nav, which is the sole source of truth for
        success/failure (matching the same soft-info treatment already used
        for the SAM2 QProcess worker's stderr, tab_sam2.py's
        _handle_error_sam)."""
        self._stderr_buffer.append(process)

    def handle_finished_nav(self, process):
        if process in self.running_processes:
            self.running_processes.remove(process)
        i_index = self.process_task_map.pop(process, None)
        if self._cancelling:
            # cancel_running_work() already logged/showed one summary message for
            # the whole batch - a killed process's incomplete output would
            # otherwise trip the decode-failure path below and pop an error
            # dialog for every single worker that was still running.
            self.process_output_buffers.pop(process, None)
            self._stderr_buffer.discard(process)
            process.deleteLater()
            return
        # drain any remaining bytes not yet signalled
        self.process_output_buffers[process] += process.readAllStandardOutput().data()
        self._stderr_buffer.append(process)
        # worker_nav_img.py's final print(fn_out) is unconditionally the
        # last thing it ever writes to stdout - taking only the last
        # non-empty line (instead of the whole accumulated buffer) is
        # immune to any earlier noise landing on stdout too (e.g. eventem
        # writing some of its own progress there, not just to stderr).
        raw_all = bytes(self.process_output_buffers.pop(process, b'')).decode('utf-8', errors='replace')
        result_lines = [line.strip() for line in raw_all.splitlines() if line.strip()]
        raw = result_lines[-1] if result_lines else ''
        try:
            # raw is a path to the .npy file the worker saved its result to
            # (see worker_nav_img.py) - loading from disk instead of a
            # base64+pickle blob through the stdout pipe is what avoids
            # "Calculate All" getting progressively slower as more workers'
            # payloads compete for the same GUI-thread-driven pipe draining.
            nav_img = np.load(raw)
            self.nav_imgs[i_index] = nav_img
            try:
                os.remove(raw)
            except OSError:
                pass
        except Exception as e:
            stderr_text = self._stderr_buffer.pop_text(process)
            self.logger.error('Failed to load nav image for index %s: %s%s', i_index, e,
                              f'\nWorker stderr:\n{stderr_text}' if stderr_text else '')
            if stderr_text:
                qtw.QMessageBox.warning(self, 'Worker Error', stderr_text[:500])
        self._stderr_buffer.discard(process)
        process.deleteLater()
        self.nav_counter += 1
        self.update_progress_bar(self.nav_counter, self.nav_counter_total)
        if self.nav_counter >= self.nav_counter_total:
            duration = perf_counter() - self._nav_tic
            shutil.rmtree(self._navimg_temp_dir, ignore_errors=True)
            valid = [img for img in self.nav_imgs if img is not None]
            if not valid:
                self.logger.error(
                    'Navigation signal creation failed: all worker processes '
                    'failed to produce output (after %s).', io.format_duration_hms(duration))
                qtw.QMessageBox.critical(self, 'No Results', 'All worker processes failed to produce output.')
                self.button_cancel.setDisabled(True)
                return
            self.nav_imgs = np.stack(valid)
            self.update_canvas(0)
            self.slider_imgNo.setRange(0, len(self.nav_imgs) - 1)
            self.button_cancel.setDisabled(True)
            if self._nav_failed or len(valid) < self.nav_counter_total:
                self.logger.error(
                    'Navigation signal creation finished with errors: %d/%d file(s) '
                    'succeeded in %s.', len(valid), self.nav_counter_total, io.format_duration_hms(duration))
            else:
                self.logger.info(
                    'Navigation signal creation completed successfully for %d file(s) in %s.',
                    self.nav_counter_total, io.format_duration_hms(duration))
            self.button_save_results.setEnabled(True)
            if self.checkbox_autosave.isChecked():
                self.save_results()
        else:
            self.launch_next_nav_task()

    def process_failed_nav(self, error):
        if self._cancelling:
            return
        self._nav_failed = True
        self.logger.error("QProcess error occurred: %s", error)
        self.button_cancel.setDisabled(True)
        qtw.QMessageBox.critical(self, 'Process Error',
            f'A worker process failed to start (error code {error}).\n'
            'Check that Python is on PATH and worker_nav_img.py exists.')

    def cancel_running_work(self):
        # Killing several running processes at once each fires their own
        # finished/errorOccurred signals - without this flag,
        # handle_finished_nav/process_failed_nav would treat every one of
        # them as an independent failure and pop an error dialog each,
        # instead of the one summary message below.
        self._cancelling = True
        n_done = self.nav_counter
        n_total = getattr(self, 'nav_counter_total', n_done)
        self.tasks.clear()
        for p in list(self.running_processes):
            p.kill()
        self.running_processes.clear()
        self.process_task_map.clear()
        self.button_cancel.setDisabled(True)
        if hasattr(self, '_navimg_temp_dir'):
            shutil.rmtree(self._navimg_temp_dir, ignore_errors=True)
        self.logger.warning(
            'Cancelled by user (%d/%d file(s) already processed).',
            n_done, n_total)
        qtw.QMessageBox.information(self, 'Cancelled',
            f'Navigation signal creation cancelled - {n_done}/{n_total} file(s) '
            'were already processed.')

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
            self.logger.exception('Failed to save navigation signal after %s.',
                                   io.format_duration_hms(perf_counter() - tic))
            return
        self.logger.info(
            'Navigation signal saved successfully in %s (background frame/clip '
            'generation continues asynchronously).', io.format_duration_hms(perf_counter() - tic))

    def _save_results_impl(self):
        """Save into "<Save Path>/<Project>_5DED Analysis/navigator signal/
        <timestamp>/" - a dedicated, per-run subfolder, not directly in the
        Save Path - so every navigator run for a project lands under one
        shared "<Project>_5DED Analysis" tree (CV2/SAM2 default their own
        "Save Dir." into this same folder - see io.default_analysis_save_dir)
        and is directly loadable via "Load Saved Analysis" in those tabs.
        Everything this produces - the .hspy signal, per-frame TIFFs, clip,
        and metadata.json - lives under this one timestamped folder,
        nothing is left in the 4D signals folder."""
        threadpool = QThreadPool.globalInstance()
        if not os.path.isdir(self.pathSave):
            os.mkdir(self.pathSave)
        project_name = (self.lineEdit_projectName.text().strip()
                        or os.path.basename(self.path_main.rstrip('/\\')) or 'project')
        path_navigator = os.path.join(self.pathSave, f'{project_name}_5DED Analysis', 'navigator signal')
        os.makedirs(path_navigator, exist_ok=True)
        fld_1 = datetime.date.today()
        fld_2 = datetime.datetime.now().strftime("%H-%M-%S")
        path_save = os.path.join(path_navigator, f'{fld_1}__{fld_2}')
        os.mkdir(path_save)
        self.logger.info('Saving navigation signal to %s...', path_save)

        s = hs.signals.Signal2D(self.nav_imgs)
        save_name = self.lineEdit_saveName.text().strip() or 'navigation_signal'
        s.save(os.path.join(path_save, f'{save_name}.hspy'), overwrite=True)
        path_imgs = os.path.join(path_save, 'navigation_images')
        os.mkdir(path_imgs)
        worker_frames = WorkerThread_General(io.create_frames, 0, path_imgs, s.data)
        threadpool.start(worker_frames)
        fn_clip = os.path.join(path_save, 'navigation_images_clip')
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
        except ValueError:
            scale_real = None
        worker_clip = WorkerThread_General(io.create_clip_tracking, 0, fn_clip,
                                           s.data, None, scale_real,
                                           fps=self.spinbox_fps.value(), logger=self.logger)
        threadpool.start(worker_clip)

        scale_recip = self.lineEdit_scale_recip.text()
        try:
            scale_recip = float(scale_recip)
        except ValueError:
            scale_recip = None
        metadata = dict(getattr(self, '_analysis_metadata', {}))
        metadata['project_name'] = project_name
        metadata['saved_at'] = datetime.datetime.now().isoformat(timespec='seconds')
        metadata['nav_signal_shape'] = list(s.data.shape)
        metadata['scale_real_nm_per_px'] = scale_real
        metadata['scale_recip_invA_per_px'] = scale_recip
        with open(os.path.join(path_save, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

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
                io.add_readable_scalebar(self.ax, scale_real, 'nm')
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
