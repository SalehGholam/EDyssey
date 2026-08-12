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
import EDyssey.io_utils as io
import hyperspy.api as hs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from .logging_utils import LogConsole
from .base_tab import TabBase, compute_left_panel_width, get_existing_directory, build_left_panel
from .worker_thread import WorkerThread_General, ProcessStderrBuffer
from .worker_launch import worker_command
from .threshold_dialog import ThresholdDialog
from .smart_scan_dialog import SmartScanCheckDialog
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

        width_userInput = compute_left_panel_width()
        button_w = 90
        button_h_lrg = 50

        # layout top
        layout_userInput = build_left_panel(self._splitter, width_userInput)
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
            'Results are saved to "<Save Path>/<Project>_EDyssey Analysis/navigator signal/'
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

        self.double_validator = QDoubleValidator(0.0, 1e5, 5)

        #%% input parameters
        self.box_scanSize = qtw.QGroupBox('Input Parameters')
        layout_dir_scanSize.addWidget(self.box_scanSize)
        layout_scanSize = qtw.QVBoxLayout()
        self.box_scanSize.setLayout(layout_scanSize)

        layout_scanSize_row1 = qtw.QHBoxLayout()
        layout_scanSize.addLayout(layout_scanSize_row1)

        label_scanSize = qtw.QLabel('Scan Size')
        layout_scanSize_row1.addWidget(label_scanSize)

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

        # detector size (per side, in pixels) - Auto assumes 512x512.
        layout_detSize = qtw.QHBoxLayout()
        layout_scanSize.addLayout(layout_detSize)
        label_detSize = qtw.QLabel('Detector Size')
        label_detSize.setToolTip(
            'Detector (diffraction pattern) size in pixels - Auto assumes 512x512.')
        layout_detSize.addWidget(label_detSize)
        self.checkbox_detectorSizeAuto = qtw.QCheckBox('Auto')
        self.checkbox_detectorSizeAuto.setChecked(True)
        self.checkbox_detectorSizeAuto.setToolTip(label_detSize.toolTip())
        layout_detSize.addWidget(self.checkbox_detectorSizeAuto)
        self.spinbox_detectorSize_x = qtw.QSpinBox()
        self.spinbox_detectorSize_x.setFixedWidth(55)
        self.spinbox_detectorSize_x.setRange(1, 8192)
        self.spinbox_detectorSize_x.setValue(512)
        layout_detSize.addWidget(self.spinbox_detectorSize_x)
        label_detSize_cross = qtw.QLabel('X')
        layout_detSize.addWidget(label_detSize_cross)
        self.spinbox_detectorSize_y = qtw.QSpinBox()
        self.spinbox_detectorSize_y.setFixedWidth(55)
        self.spinbox_detectorSize_y.setRange(1, 8192)
        self.spinbox_detectorSize_y.setValue(512)
        layout_detSize.addWidget(self.spinbox_detectorSize_y)
        self.activate_detectorSize_spinboxes()
        self.checkbox_detectorSizeAuto.stateChanged.connect(self.activate_detectorSize_spinboxes)
        layout_detSize.addStretch(1)

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
        # Re-reads comment.txt (a cheap text-file parse, not the 4D data
        # file itself) for the newly-selected block as soon as the value
        # changes, instead of requiring an extra "Load" click every time.
        self.spinbox_metadataCount.valueChanged.connect(lambda: self.load_metadata(silent=True))
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

        # scale bars - moved out of Directories, real/reciprocal merged onto
        # one row; kept at the bottom of Input Parameters (below scan size/
        # detector size/metadata), since it's a display-only calibration
        # rather than an acquisition parameter.
        self.box_scale = qtw.QGroupBox('Scale bars')
        layout_box_scale = qtw.QVBoxLayout()
        self.box_scale.setLayout(layout_box_scale)
        layout_scanSize.addWidget(self.box_scale)

        layout_scale_row = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_row)
        label_scale_real = qtw.QLabel('Real (nm)')
        label_scale_real.setFixedWidth(55)
        layout_scale_row.addWidget(label_scale_real)
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        layout_scale_row.addWidget(self.lineEdit_scale_real)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        self.lineEdit_scale_real.textChanged.connect(lambda: self._update_nav_scalebar())
        label_scale_recip = qtw.QLabel('Recip. (Å<sup>-1</sup>)')
        label_scale_recip.setFixedWidth(55)
        layout_scale_row.addWidget(label_scale_recip)
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        layout_scale_row.addWidget(self.lineEdit_scale_recip)
        self.lineEdit_scale_recip.setValidator(self.double_validator)
        self.lineEdit_scale_recip.setToolTip(
            'Reciprocal-space calibration (1/A per pixel) for the Summed DP preview - '
            'drawn as concentric dashed rings every 1 1/A, centered on the found center')
        self.lineEdit_scale_recip.textChanged.connect(self.update_recip_scale_circles)

        #%% box smart scan (pattern-file) tomography support: each tilt angle
        # may have a "detection" file (dense, no pattern needed) and/or an
        # "acquisition" file (sparse, needs its own pattern file) - see
        # EDyssey/io_utils/smart_scan.py.
        self.box_smartScan = qtw.QGroupBox('Smart Scan')
        layout_box_smartScan = qtw.QVBoxLayout()
        self.box_smartScan.setLayout(layout_box_smartScan)
        layout_dir_scanSize.addWidget(self.box_smartScan)

        layout_scanSize_row3 = qtw.QHBoxLayout()
        layout_box_smartScan.addLayout(layout_scanSize_row3)

        self.checkbox_smartScan = qtw.QCheckBox('Smart Scanned')
        self.checkbox_smartScan.setToolTip(
            'This folder holds a smart-scanned tomography series (detection + '
            'acquisition files per tilt angle, each acquisition needing its own '
            'pattern file). See other_scripts/smart scanning guide/ for background.')
        layout_scanSize_row3.addWidget(self.checkbox_smartScan)
        self.checkbox_smartScan.stateChanged.connect(self.activate_smartScan_widgets)

        self.combo_smartScanRole = qtw.QComboBox()
        self.combo_smartScanRole.addItems(['Acquisition', 'Detection'])
        self.combo_smartScanRole.setToolTip(
            'Which file role to build the batch navigation signal from for each tilt '
            'angle - "Acquisition" (smart-scanned, uses the pattern file) or '
            '"Detection" (a plain dense raster, no pattern needed)')
        self.combo_smartScanRole.setDisabled(True)
        layout_scanSize_row3.addWidget(self.combo_smartScanRole)
        layout_scanSize_row3.addStretch(1)

        layout_scanSize_row4 = qtw.QHBoxLayout()
        layout_box_smartScan.addLayout(layout_scanSize_row4)

        label_patternDir = qtw.QLabel('Pattern Dir.')
        label_patternDir.setFixedWidth(70)
        layout_scanSize_row4.addWidget(label_patternDir)
        self.lineEdit_patternDir = qtw.QLineEdit()
        self.lineEdit_patternDir.setPlaceholderText('defaults to 4D Signals folder')
        self.lineEdit_patternDir.setDisabled(True)
        layout_scanSize_row4.addWidget(self.lineEdit_patternDir)
        self.button_browsePatternDir = qtw.QPushButton('...')
        self.button_browsePatternDir.setFixedWidth(30)
        self.button_browsePatternDir.setDisabled(True)
        self.button_browsePatternDir.clicked.connect(self.browse_pattern_dir)
        layout_scanSize_row4.addWidget(self.button_browsePatternDir)

        layout_scanSize_row4b = qtw.QHBoxLayout()
        layout_box_smartScan.addLayout(layout_scanSize_row4b)

        label_detectionDir = qtw.QLabel('Detect. Dir.')
        label_detectionDir.setFixedWidth(70)
        layout_scanSize_row4b.addWidget(label_detectionDir)
        self.lineEdit_detectionDir = qtw.QLineEdit()
        self.lineEdit_detectionDir.setPlaceholderText('defaults to 4D Signals folder')
        self.lineEdit_detectionDir.setDisabled(True)
        self.lineEdit_detectionDir.setToolTip(
            'Folder to look for detection files in, if they live somewhere other than the '
            '4D Signals folder (e.g. a separate folder of HAADF .tif/.tiff reference images '
            'for .mib/.hspy/.zspy, or a cleaner acquisition layout with detection/acquisition '
            'each in their own folder)')
        layout_scanSize_row4b.addWidget(self.lineEdit_detectionDir)
        self.button_browseDetectionDir = qtw.QPushButton('...')
        self.button_browseDetectionDir.setFixedWidth(30)
        self.button_browseDetectionDir.setDisabled(True)
        self.button_browseDetectionDir.clicked.connect(self.browse_detection_dir)
        layout_scanSize_row4b.addWidget(self.button_browseDetectionDir)

        # Detection/acquisition dwell times - a smart-scanned tilt series
        # logs two metadata blocks per angle (e.g. "Scan strategy: Raster"
        # for the dense detection pass, "Scan strategy: Custom" for the
        # sparse acquisition), each with its own dwelltime - editable here
        # in case they differ from (or aren't present in) comment.txt. Used
        # (instead of the generic Dwell T. above) for any single-file action
        # (Test File, Compute Summed DP, ...) once the file's role is known
        # via a confirmed Check Files match - see _get_dwell_time_for.
        # Own rows (rather than sharing one, too-wide-for-the-panel row) -
        # "Detection Dwell T. (μs)"/"Acquisition Dwell T. (μs)" are
        # both long labels.
        layout_scanSize_dwell1 = qtw.QHBoxLayout()
        layout_box_smartScan.addLayout(layout_scanSize_dwell1)
        label_dwellTime_detection = qtw.QLabel('Detection Dwell T. (μs)')
        layout_scanSize_dwell1.addWidget(label_dwellTime_detection)
        self.spinbox_dwellTime_detection = qtw.QSpinBox()
        self.spinbox_dwellTime_detection.setFixedWidth(70)
        self.spinbox_dwellTime_detection.setRange(1, 99999999)
        self.spinbox_dwellTime_detection.setDisabled(True)
        layout_scanSize_dwell1.addWidget(self.spinbox_dwellTime_detection)
        layout_scanSize_dwell1.addStretch(1)

        layout_scanSize_dwell2 = qtw.QHBoxLayout()
        layout_box_smartScan.addLayout(layout_scanSize_dwell2)
        label_dwellTime_acquisition = qtw.QLabel('Acquisition Dwell T. (μs)')
        layout_scanSize_dwell2.addWidget(label_dwellTime_acquisition)
        self.spinbox_dwellTime_acquisition = qtw.QSpinBox()
        self.spinbox_dwellTime_acquisition.setFixedWidth(70)
        self.spinbox_dwellTime_acquisition.setRange(1, 99999999)
        self.spinbox_dwellTime_acquisition.setDisabled(True)
        layout_scanSize_dwell2.addWidget(self.spinbox_dwellTime_acquisition)
        layout_scanSize_dwell2.addStretch(1)

        layout_scanSize_row5 = qtw.QHBoxLayout()
        layout_box_smartScan.addLayout(layout_scanSize_row5)
        self.button_checkSmartScanFiles = qtw.QPushButton('Check Files...')
        self.button_checkSmartScanFiles.setToolTip(
            'Review the automatic per-angle detection/acquisition/pattern-file match '
            'before calculating - fix or exclude any mismatched tilt angle by hand')
        self.button_checkSmartScanFiles.setDisabled(True)
        self.button_checkSmartScanFiles.clicked.connect(self.open_smart_scan_check_dialog)
        layout_scanSize_row5.addWidget(self.button_checkSmartScanFiles)
        self.label_smartScanSummary = qtw.QLabel('')
        layout_scanSize_row5.addWidget(self.label_smartScanSummary)
        layout_scanSize_row5.addStretch(1)

        self._smart_scan_rows = None  # set by open_smart_scan_check_dialog(); cleared on new 4D folder
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

        #%% calculate/save buttons + CPU cores/FPS/Autosave
        # Built here (widgets created/configured now, next to the settings
        # they act on) but not actually placed into layout_userInput until
        # the very end of init_widget - action buttons (Test File,
        # Calculate All, Save Results, Cancel) sit at the bottom of the
        # panel, below every configuration box, rather than interleaved
        # with them.
        layout_calculate_buttons = qtw.QHBoxLayout()
        layout_calculate_buttons.addStretch(1)

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

        self.button_save_results = qtw.QPushButton('Save Results')
        self.button_save_results.setFixedSize(button_w, button_h_lrg)
        layout_calculate_buttons.addWidget(self.button_save_results)
        self.button_save_results.clicked.connect(self.save_results)
        self.button_save_results.setToolTip(
            'Save the navigation signal, frames, and clip to the Save Path above')
        self.button_save_results.setDisabled(True)
        layout_calculate_buttons.addStretch(1)

        # Cancel itself is placed even later - the very last widget in the
        # whole panel (right after layout_calculate_buttons/layout_save near
        # the end of init_widget) - see the comment there.
        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.setFixedHeight(button_h_lrg)
        
        self.button_cancel.setStyleSheet("background-color: red; color: white;")
        self.button_cancel.setDisabled(True)
        self.button_cancel.clicked.connect(self.cancel_running_work)
        self.button_cancel.setToolTip(
            'Stop the running navigation signal creation. Already-running '
            'background computations finish silently; their results are discarded.')

        #%% CPU cores / Clip FPS / Autosave
        layout_save = qtw.QHBoxLayout()

        label_cores = qtw.QLabel('CPU Cores')
        label_cores.setFixedWidth(60)
        layout_save.addWidget(label_cores)
        self.spinbox_cpuCores = qtw.QSpinBox()
        self.spinbox_cpuCores.setFixedWidth(40)
        self.spinbox_cpuCores.setRange(1, os.cpu_count() or 1)
        self.spinbox_cpuCores.setValue(1)
        self.spinbox_cpuCores.setToolTip('Number of parallel worker processes for nav image computation')
        layout_save.addWidget(self.spinbox_cpuCores)

        label_fps = qtw.QLabel('Clip FPS')
        label_fps.setFixedWidth(60)
        layout_save.addWidget(label_fps)
        self.spinbox_fps = qtw.QSpinBox()
        self.spinbox_fps.setFixedWidth(40)
        self.spinbox_fps.setRange(1, 60)
        self.spinbox_fps.setValue(5)
        self.spinbox_fps.setToolTip('Frames per second for the navigation clip')
        layout_save.addWidget(self.spinbox_fps)

        self.checkbox_autosave = qtw.QCheckBox('Autosave')
        layout_save.addWidget(self.checkbox_autosave)
        self.checkbox_autosave.setToolTip(
            'Automatically save when "Calculate All" finishes, instead of needing '
            'to click "Save Results" manually')
        layout_save.addStretch(1)

        #%% list of files
        self.file_list_widget = qtw.QListWidget()
        layout_userInput.addWidget(self.file_list_widget)
        self.file_list_widget.setMinimumWidth(150)
        # A generous baseline height (rather than the few rows its bare
        # sizeHint would give) - the panel now lives in a QScrollArea, which
        # sizes its content to this natural/minimum size regardless of the
        # window's actual height, instead of stretching to fill whatever
        # space happens to be available the way a plain splitter pane would.
        self.file_list_widget.setMinimumHeight(300)
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

        # Several virtual detectors can be added to the list below and are
        # combined into a single navigation image at calculation time -
        # natively summed by eventem for .tpx3, OR-combined into one mask
        # for every other format (see create_virtual_detector_multi in
        # EDyssey/io_utils/nav_image.py). An empty list just keeps the
        # spinbox-defined detector above as the only one used, so nothing
        # changes for anyone who never touches this.
        self._extra_detectors = []

        layout_detector_list_buttons = qtw.QHBoxLayout()
        layout_mask.addLayout(layout_detector_list_buttons)
        self.button_addDetector = qtw.QPushButton('Add Detector')
        self.button_addDetector.setToolTip(
            'Add the center/radii above as another virtual detector - once one or '
            'more are added here, ALL of them (not the values above, until also '
            'added) are combined into the single navigation image used for '
            'calculations')
        layout_detector_list_buttons.addWidget(self.button_addDetector)
        self.button_addDetector.clicked.connect(self.add_extra_detector)
        self.button_removeDetector = qtw.QPushButton('Remove Selected')
        layout_detector_list_buttons.addWidget(self.button_removeDetector)
        self.button_removeDetector.clicked.connect(self.remove_extra_detector)
        layout_detector_list_buttons.addStretch(1)

        self.list_detectors = qtw.QListWidget()
        self.list_detectors.setToolTip(
            'Additional virtual detectors, combined with each other (but not with '
            'the center/radii spinboxes above unless also added here) into one '
            'navigation image at calculation time')
        self.list_detectors.setMaximumHeight(70)
        layout_mask.addWidget(self.list_detectors)
        self.list_detectors.itemSelectionChanged.connect(self._load_selected_detector)

        # Each row here holds at most one long-label button - a long label
        # can't shrink below its text's natural width (no eliding on
        # QPushButton), so packing several per row is what forced the whole
        # panel wider than intended; \n-wrapping the longer labels keeps
        # each button itself narrower too.
        layout_sum_dp = qtw.QHBoxLayout()
        layout_mask.addLayout(layout_sum_dp)
        layout_sum_dp.addStretch(1)

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
        layout_sum_dp.addStretch(1)

        # The Summed DP preview canvas itself lives in the main window, beside
        # the navigation image (see #%% canvas below), so it has more room
        # to work with and its own navigation toolbar.
        self._mask_artists = []
        self._mask_drag_mode = None
        # Blit state for on_press_mask/on_motion_mask/on_release_mask -
        # direct references to the 3 artists that get dragged (kept up to
        # date by update_mask_overlay), and the cached background snapshot
        # (everything else in ax_mask_preview) captured once a drag starts,
        # so on_motion_mask can move just those 3 without a full-figure
        # redraw on every mouse-move tick.
        self._current_circle_out = None
        self._current_circle_in = None
        self._current_center_marker = None
        self._mask_bg = None
        # Scan-space ROI drawn (Ctrl+drag) on the nav/test image, used only
        # by "Summed DP from ROI" - never by "Compute Summed DP", which always
        # sums the whole scan regardless of whether a ROI is currently drawn.
        self.roi_navsig = None
        self.rect_navsig = None
        self._navsig_press = None
        self._navsig_bg = None

        # Action buttons (built earlier, alongside the settings they act
        # on) go here, at the very bottom of the panel, below every
        # configuration box, directly one after another with no gap.
        layout_userInput.addLayout(layout_calculate_buttons)
        layout_userInput.addLayout(layout_save)
        layout_userInput.addWidget(self.button_cancel)
        # No trailing addStretch here - file_list_widget (given a stretch
        # factor above, and Expanding by default) already claims all
        # leftover vertical space, so Cancel naturally lands at the panel's
        # bottom edge instead of leaving a separate empty gap below it.
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

        # Display-only contrast for the navigation image (left plot) - a
        # set_clim() on the plotted image, purely cosmetic: never touches
        # self.nav_imgs/self._last_test_img or any downstream calculation
        # (center-finding, mask placement, batch results all still read the
        # untouched raw arrays). Mirrors the vmin/vmax slider pattern
        # already used for the DP axis in tab_roi_4d.py.
        layout_contrast_nav = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_contrast_nav)
        self.label_navVmin = qtw.QLabel('vmin')
        layout_contrast_nav.addWidget(self.label_navVmin)
        self.slider_navVmin = qtw.QSlider(Qt.Horizontal)
        self.slider_navVmin.setRange(0, 1)
        layout_contrast_nav.addWidget(self.slider_navVmin)
        self.label_navVmax = qtw.QLabel('vmax')
        layout_contrast_nav.addWidget(self.label_navVmax)
        self.slider_navVmax = qtw.QSlider(Qt.Horizontal)
        self.slider_navVmax.setRange(0, 1)
        self.slider_navVmax.setValue(1)
        layout_contrast_nav.addWidget(self.slider_navVmax)
        self.slider_navVmin.setToolTip(
            'Display contrast only - adjusts how the navigation image is shown, '
            'not the underlying data used for center-finding or calculations')
        self.slider_navVmax.setToolTip(self.slider_navVmin.toolTip())
        self.slider_navVmin.valueChanged.connect(self._update_nav_display_clim)
        self.slider_navVmax.valueChanged.connect(self._update_nav_display_clim)
        self.button_navContrastReset = qtw.QPushButton('Reset')
        self.button_navContrastReset.setToolTip('Reset display contrast to the full data range')
        self.button_navContrastReset.clicked.connect(self._reset_nav_contrast)
        layout_contrast_nav.addWidget(self.button_navContrastReset)
        self._nav_display_data_range = (0, 1)

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
        """Open a folder picker for whichever button (button_dir or
        button_dir_save) triggered it, and fill the corresponding line edit."""
        sender = self.sender()
        if sender == self.button_dir:
            path = get_existing_directory(self, "Select 4D Signals Folder")
            if path:
                self.metadata_path_override = None  # new folder - re-derive comment.txt location
                self.lineEdit_dir_signal.setText(path)
                # setText() above only fires textChanged (which
                # populate_file_list is connected to) when the path is
                # actually different from what's already there - re-picking
                # the *same* folder (e.g. after adding/removing files in it)
                # would otherwise silently leave the stale file list showing.
                self.populate_file_list()
        elif sender == self.button_dir_save:
            path = get_existing_directory(self, "Select Destination Folder")
            if path:
                self.lineEdit_dir_save.setText(path)

    def populate_file_list(self):
        """Refresh the file list for the current 4D signals folder, derive the
        save directory/project name from it, auto-load comment.txt metadata if
        present (any format - load_metadata(silent=True) no-ops quietly when
        comment.txt is missing/unparsable), and invalidate any
        previously-reviewed smart-scan match table."""
        directory = self.lineEdit_dir_signal.text()
        if os.path.isdir(directory):
            self.set_save_directory()
            self.load_metadata(silent=True)
        self.refresh_file_list()
        # A previously-reviewed smart-scan match table is only valid for the
        # folder it was built from - stale otherwise.
        self._smart_scan_rows = None
        self.label_smartScanSummary.setText('')

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

    def activate_smartScan_widgets(self):
        """Enable/disable the Smart Scan widgets to match the checkbox, and
        invalidate any previously-reviewed Check Files match table."""
        enable = self.checkbox_smartScan.isChecked()
        for wid in (self.combo_smartScanRole, self.lineEdit_patternDir,
                    self.button_browsePatternDir, self.lineEdit_detectionDir,
                    self.button_browseDetectionDir, self.button_checkSmartScanFiles,
                    self.spinbox_dwellTime_detection, self.spinbox_dwellTime_acquisition):
            wid.setEnabled(enable)
        self._smart_scan_rows = None
        self.label_smartScanSummary.setText('')

    def browse_pattern_dir(self):
        """Browse for the pattern-files folder; invalidates any previously-
        reviewed Check Files match table."""
        start_dir = self.lineEdit_patternDir.text() or self.lineEdit_dir_signal.text()
        path = get_existing_directory(self, "Select Pattern Files Folder", start_dir)
        if path:
            self.lineEdit_patternDir.setText(path)
            self._smart_scan_rows = None
            self.label_smartScanSummary.setText('')

    def get_pattern_dir(self):
        return self.lineEdit_patternDir.text() or self.lineEdit_dir_signal.text()

    def browse_detection_dir(self):
        """Browse for the detection-files folder; invalidates any previously-
        reviewed Check Files match table."""
        start_dir = self.lineEdit_detectionDir.text() or self.lineEdit_dir_signal.text()
        path = get_existing_directory(self, "Select Detection Files Folder", start_dir)
        if path:
            self.lineEdit_detectionDir.setText(path)
            self._smart_scan_rows = None
            self.label_smartScanSummary.setText('')

    def get_detection_dir(self):
        return self.lineEdit_detectionDir.text() or None

    def open_smart_scan_check_dialog(self):
        """Open the SmartScanCheckDialog for the current folder/data type and,
        if accepted, store the reviewed per-angle match table and update the
        summary label."""
        directory = self.lineEdit_dir_signal.text()
        if not os.path.isdir(directory):
            qtw.QMessageBox.critical(self, 'No Folder', 'Select the 4D signals folder first.')
            return
        if self.checkbox_selectAll.isChecked():
            dtype = None
            for ext in io.DATA_EXTENSIONS:
                if any(f.endswith(ext) for f in os.listdir(directory)):
                    dtype = ext
                    break
        else:
            dtype = self.combo_dtype.currentText()
        if dtype not in io.DATA_EXTENSIONS:
            qtw.QMessageBox.warning(self, 'Unsupported Format',
                f'Smart-scan file matching currently supports {", ".join(io.DATA_EXTENSIONS)} '
                'data only.')
            return
        dlg = SmartScanCheckDialog(self, directory, dtype, pattern_dir=self.get_pattern_dir(),
                                   detection_dir=self.get_detection_dir(),
                                   rows=self._smart_scan_rows)
        if dlg.exec_() == qtw.QDialog.Accepted:
            self._smart_scan_rows = dlg.rows
            n_ok = sum(1 for row in dlg.rows if not row['excluded'])
            self.label_smartScanSummary.setText(f'{n_ok} / {len(dlg.rows)} angle(s) included')
            self.logger.info('Smart-scan file check confirmed: %d / %d angle(s) included.',
                             n_ok, len(dlg.rows))

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
        """Default the save directory and project name from the 4D signals
        folder's parent/basename."""
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

    def activate_detectorSize_spinboxes(self):
        auto = self.checkbox_detectorSizeAuto.isChecked()
        self.spinbox_detectorSize_x.setDisabled(auto)
        self.spinbox_detectorSize_y.setDisabled(auto)

    def get_detector_shape(self, fn):
        """(shape_x, shape_y) of the detector/diffraction-pattern for `fn` -
        auto-detected from the file for formats that report it cheaply
        (io.get_det_size opens the file lazily, no full read), or the
        manual "Detector Size" X/Y spinboxes for .tpx3 (auto-detecting that
        would mean fully parsing the file - eventem has no cheaper
        metadata-only query - just to learn its shape; "Auto" here keeps
        the previous default of 512x512)."""
        dtype = os.path.splitext(fn)[-1]
        if dtype == '.tpx3':
            if self.checkbox_detectorSizeAuto.isChecked():
                return 512, 512
            return self.spinbox_detectorSize_x.value(), self.spinbox_detectorSize_y.value()
        return io.get_det_size(fn)

    def get_all_item_names(self):
        item_names = []
        for index in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(index)
            if item:
                item_names.append(item.text())
        return item_names

    def activate_combo_dtype(self, state):
        """Enable combo_dtype when "All files" is unchecked (state == 0,
        Qt.Unchecked) and refresh the file list either way."""
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

    def _get_fn_pattern_for(self, fn):
        """Pattern file to use for `fn` in the "Test File" single-file
        preview, when "Smart Scanned" is checked and the folder has already
        been reviewed via "Check Files..." (self._smart_scan_rows).

        Looks `fn` up by basename against *both* the detection and
        acquisition slots of every row - not just whichever role is
        currently selected in combo_smartScanRole - since the file the user
        actually double-clicked in the file list may be either one
        regardless of that dropdown (it only controls what the *batch*
        "Calculate All" run uses). Using the selected role instead of the
        clicked file's own role meant testing a smart-scanned acquisition
        file while "Detection" was selected silently loaded it without its
        pattern file and failed.

        Returns the pattern file for an acquisition-role match, None for a
        detection-role match (dense, no pattern needed) or when smart-scan
        isn't in use / hasn't been checked yet / `fn` isn't in the table.
        """
        if not self.checkbox_smartScan.isChecked() or not self._smart_scan_rows:
            return None
        name = os.path.basename(fn)
        for row in self._smart_scan_rows:
            if row.get('acquisition_file') and os.path.basename(row['acquisition_file']) == name:
                return row.get('pattern_file')
            if row.get('detection_file') and os.path.basename(row['detection_file']) == name:
                return None
        return None

    def _get_dwell_time_for(self, fn):
        """Dwell time (μs) to use for `fn`: the role-specific Detection/
        Acquisition Dwell T. spinbox when "Smart Scanned" is checked and
        `fn` matches a confirmed Check Files row (see _get_fn_pattern_for),
        else the generic Dwell T. spinbox."""
        if self.checkbox_smartScan.isChecked() and self._smart_scan_rows:
            name = os.path.basename(fn)
            for row in self._smart_scan_rows:
                if row.get('acquisition_file') and os.path.basename(row['acquisition_file']) == name:
                    return self.spinbox_dwellTime_acquisition.value()
                if row.get('detection_file') and os.path.basename(row['detection_file']) == name:
                    return self.spinbox_dwellTime_detection.value()
        return self.spinbox_dwellTime.value()

    def test_selected_file(self, item):
        """Compute and preview the navigation image for a single file
        (double-clicked in the file list), without touching the full batch
        result - lets the user check dwell time / scan size / virtual mask
        parameters before committing to the full "Calculate" run."""
        fn = self._get_test_fn(item)
        # os.path.exists, not isfile - .zspy stores are directories (Zarr),
        # not single files, so isfile() rejected every valid .zspy path here.
        if fn is None or not os.path.exists(fn):
            self.logger.error('Cannot test file: %s', fn)
            return
        dtype = os.path.splitext(fn)[-1]
        scanSize = self._get_current_scanSize()
        if self._scan_size_required(dtype) and scanSize is None:
            self.logger.warning('Cannot test %s: scan size is required.', fn)
            self.message_box_scan_size_required(dtype)
            return

        dwellTime = self._get_dwell_time_for(fn)
        fn_pattern = self._get_fn_pattern_for(fn)
        det_shape = self.get_detector_shape(fn)
        self.logger.info('Testing navigation image for %s...', fn)
        if self.checkbox_useMask.isChecked():
            worker = WorkerThread_General(
                io.calculate_nav_img_masked, 0, fn, dtype=dtype, scanSize=scanSize,
                dwellTime=dwellTime, detectors=self.get_active_detectors(),
                logger=self.logger, fn_pattern=fn_pattern, det_shape=det_shape)
        else:
            worker = WorkerThread_General(io.calculate_nav_img, 0, fn, dtype=dtype,
                                          scanSize=scanSize, dwellTime=dwellTime,
                                          logger=self.logger, fn_pattern=fn_pattern,
                                          det_shape=det_shape)
        worker.signals.results.connect(lambda result, idx, fn=fn: self._on_test_result(result, fn))
        QThreadPool.globalInstance().start(worker)

    def _on_test_result(self, result, fn):
        """Display a completed test navigation image (from test_selected_file)
        on the main canvas and reset the toolbar's Home view to match."""
        fig, ax = plt.subplots()
        ax.imshow(result)
        self._last_test_fn = fn
        self._last_test_img = result
        self.img_display.set_data(result)
        self._set_nav_contrast_range(result.min(), result.max())
        self._update_nav_display_clim(redraw=False)
        shape_x, shape_y = result.shape
        self.img_display.set_extent([0, shape_y, shape_x, 0])
        self.ax.set_xlim(0, shape_y)
        self.ax.set_ylim(shape_x, 0)
        self.ax.set_title(f'TEST: {os.path.basename(fn)}')
        self._update_nav_scalebar(redraw=False)
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
        # os.path.exists, not isfile - .zspy stores are directories (Zarr),
        # not single files, so isfile() rejected every valid .zspy path here.
        if fn is None or not os.path.exists(fn):
            qtw.QMessageBox.critical(self, 'No File',
                'Load a directory (and optionally select a file) before computing a Summed DP.')
            return
        dtype = os.path.splitext(fn)[-1]
        scanSize = self._get_current_scanSize()
        if self._scan_size_required(dtype) and scanSize is None:
            self.logger.warning('Cannot compute Summed DP for %s: scan size is required.', fn)
            self.message_box_scan_size_required(dtype)
            return

        self.logger.info('Computing Summed DP (summed diffraction pattern) from %s...', fn)
        self.button_computeSumDp.setDisabled(True)
        self._sum_dp_tic = perf_counter()
        worker = WorkerThread_General(io.get_dp, 0, fn, dtype=dtype, scanSize=scanSize,
                                      dwellTime=self._get_dwell_time_for(fn), roi=None,
                                      logger=self.logger, fn_pattern=self._get_fn_pattern_for(fn),
                                      det_shape=self.get_detector_shape(fn))
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
        # os.path.exists, not isfile - .zspy stores are directories (Zarr),
        # not single files, so isfile() rejected every valid .zspy path here.
        if fn is None or not os.path.exists(fn):
            qtw.QMessageBox.critical(self, 'No File',
                'Load a directory (and optionally select a file) before computing a Summed DP.')
            return
        dtype = os.path.splitext(fn)[-1]
        scanSize = self._get_current_scanSize()
        if self._scan_size_required(dtype) and scanSize is None:
            self.logger.warning('Cannot compute Summed DP for %s: scan size is required.', fn)
            self.message_box_scan_size_required(dtype)
            return

        self.logger.info('Computing Summed DP from ROI %s of %s...', self.roi_navsig, fn)
        self.button_computeSumDp.setDisabled(True)
        self.button_sumDpFromRoi.setDisabled(True)
        self._sum_dp_tic = perf_counter()
        worker = WorkerThread_General(io.get_dp, 0, fn, dtype=dtype, scanSize=scanSize,
                                      dwellTime=self._get_dwell_time_for(fn),
                                      roi=self.roi_navsig, logger=self.logger,
                                      fn_pattern=self._get_fn_pattern_for(fn),
                                      det_shape=self.get_detector_shape(fn))
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
        """Display a completed Summed DP on the mask-preview canvas, rescale
        the center/radius spinboxes to its size, auto-find the center if
        enabled, and redraw the mask overlay."""
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
        """Redraw the inner/outer radius circles and center marker for the
        current (spinbox-driven, drag-editable) detector, plus every
        detector already added via "Add Detector", on the Summed DP preview."""
        for artist in self._mask_artists:
            try:
                artist.remove()
            except Exception:
                self.logger.debug('Mask overlay artist already removed.', exc_info=True)
        self._mask_artists = []
        cx = self.spinbox_centerX.value()
        cy = self.spinbox_centerY.value()
        r_in = self.spinbox_rIn.value()
        r_out = self.spinbox_rOut.value()

        # Direct references (not just the _mask_artists list) so
        # on_motion_mask can reposition+blit just these 3 during a drag,
        # instead of a full update_mask_overlay()/canvas.draw_idle() on
        # every mouse-move tick.
        self._current_circle_out = patches.Circle(
            (cx, cy), r_out, fill=False, edgecolor='lime', linewidth=1.5)
        self.ax_mask_preview.add_patch(self._current_circle_out)
        self._mask_artists.append(self._current_circle_out)
        if r_in > 0:
            self._current_circle_in = patches.Circle(
                (cx, cy), r_in, fill=False, edgecolor='red', linewidth=1.5)
            self.ax_mask_preview.add_patch(self._current_circle_in)
            self._mask_artists.append(self._current_circle_in)
        else:
            self._current_circle_in = None
        # green stands out clearly against inferno (whose brightest values
        # are yellow/white, near the direct beam where the center usually is)
        self._current_center_marker = self.ax_mask_preview.scatter(
            [cx], [cy], color='lime', marker='+', s=80, linewidth=2)
        self._mask_artists.append(self._current_center_marker)

        # Already-added detectors (see add_extra_detector) - dashed and in a
        # different color so they stay visually distinct from the one still
        # being positioned (by drag or spinbox) above.
        for detector in self._extra_detectors:
            dcx, dcy = detector['center']
            circle = patches.Circle((dcx, dcy), detector['r_out'], fill=False,
                                    edgecolor='cyan', linewidth=1.2, linestyle='--')
            self.ax_mask_preview.add_patch(circle)
            self._mask_artists.append(circle)
            if detector['r_in'] > 0:
                circle_in2 = patches.Circle((dcx, dcy), detector['r_in'], fill=False,
                                            edgecolor='magenta', linewidth=1.2, linestyle='--')
                self.ax_mask_preview.add_patch(circle_in2)
                self._mask_artists.append(circle_in2)

        self.update_recip_scale_circles()
        self.canvas.draw_idle()

    def add_extra_detector(self):
        """Append the current center/radii spinbox values as another virtual
        detector, combined with every other added detector (and the spinbox
        values themselves, once anything has been added) at calculation
        time - see get_active_detectors/calculate_nav_img_masked."""
        r_in = self.spinbox_rIn.value()
        r_out = self.spinbox_rOut.value()
        if r_out <= r_in:
            qtw.QMessageBox.critical(self, 'Invalid Virtual Mask',
                'Outer radius must be greater than the inner radius.')
            return
        detector = {'center': (self.spinbox_centerX.value(), self.spinbox_centerY.value()),
                    'r_in': r_in, 'r_out': r_out}
        self._extra_detectors.append(detector)
        self.list_detectors.addItem(self._format_detector(detector))
        self.update_mask_overlay()

    def remove_extra_detector(self):
        """Remove the selected entry from the added-detectors list."""
        row = self.list_detectors.currentRow()
        if row < 0:
            return
        del self._extra_detectors[row]
        self.list_detectors.takeItem(row)
        self.update_mask_overlay()

    def _format_detector(self, detector):
        cx, cy = detector['center']
        return f"center=({cx:.0f}, {cy:.0f}), r={detector['r_in']:.0f}-{detector['r_out']:.0f}"

    def _load_selected_detector(self):
        """Load the selected list entry's values back into the center/radii
        spinboxes - lets it be inspected, or removed and re-added with edits."""
        row = self.list_detectors.currentRow()
        if row < 0 or row >= len(self._extra_detectors):
            return
        detector = self._extra_detectors[row]
        for sb, val in ((self.spinbox_centerX, detector['center'][0]),
                        (self.spinbox_centerY, detector['center'][1]),
                        (self.spinbox_rIn, detector['r_in']),
                        (self.spinbox_rOut, detector['r_out'])):
            sb.blockSignals(True)
            sb.setValue(int(val))
            sb.blockSignals(False)
        self.update_mask_overlay()

    def get_active_detectors(self):
        """The full set of virtual detectors to use for calculations right
        now: every entry added via "Add Detector", or - if none have been
        added - just the center/radii spinbox values alone. Passed as
        calculate_nav_img_masked's `detectors` argument."""
        if self._extra_detectors:
            return list(self._extra_detectors)
        return [{'center': (self.spinbox_centerX.value(), self.spinbox_centerY.value()),
                 'r_in': self.spinbox_rIn.value(), 'r_out': self.spinbox_rOut.value()}]

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
        if self.checkbox_autoCenterDp.isChecked():
            centering_mode = 'auto (large-sigma blur) - uncheck "Auto-center" to set manually'
        else:
            centering_mode = 'manual - Ctrl+Click, drag the "+", or use the spinboxes'
        self.ax_mask_preview.set_xlabel(
            f'Circle center: {centering_mode}\n'
            'Hold Ctrl and drag the center (+) or a circle edge to move/resize the virtual mask.',
            fontsize=10)
        self.canvas.draw_idle()

    def auto_find_center(self, silent=False):
        """Auto-detect the direct-beam center in self.sum_dp and update the
        center spinboxes. silent=True skips the "no Summed DP yet" dialog
        (used when called automatically right after a Summed DP is computed)."""
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
        fn_pattern = self._get_fn_pattern_for(fn)
        det_shape = self.get_detector_shape(fn)
        self.logger.info(
            'Computing Summed DP from %s-thresholded scan positions of %s...', method, fn)
        self.button_sumDpFromThreshold.setDisabled(True)
        self._sum_dp_tic = perf_counter()
        worker = WorkerThread_General(self._sum_dp_from_mask_worker, 0, fn, dtype, scanSize,
                                      mask, fn_pattern, det_shape, self.logger)
        worker.signals.results.connect(self._on_sum_dp_from_threshold_computed)
        worker.signals.error.connect(self._on_sum_dp_failed)
        QThreadPool.globalInstance().start(worker)

    @staticmethod
    def _sum_dp_from_mask_worker(fn, dtype, scanSize, mask, fn_pattern=None,
                                 det_shape=(512, 512), logger=None):
        """Worker for compute_sum_dp_from_threshold: crop to `mask`'s bounding
        box and sum diffraction patterns only at the masked scan positions."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        y0, y1 = int(y_idx[0]), int(y_idx[-1])
        x0, x1 = int(x_idx[0]), int(x_idx[-1])
        roi = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        dp = load_dp(fn, roi=roi, mask=mask, dtype=dtype, scanSize=scanSize, dwellTime=1,
                    fn_pattern=fn_pattern, det_shape=det_shape)
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
        Pan/Zoom tool, matching the rest of the app's convention. A
        Ctrl+Click that doesn't land on the marker/an edge instead
        re-centers directly to the click point (only while Auto-center is
        off), mirroring the click-anywhere DP recentering already present
        on the other three tabs."""
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

        if self._mask_drag_mode is None:
            # Not a drag start - re-center directly to the click point,
            # same gating as the other three tabs' click-anywhere DP
            # recentering (only while Auto-center is off, so it isn't
            # immediately overwritten by the next auto-centered Summed DP
            # computation).
            if not self.checkbox_autoCenterDp.isChecked():
                self.spinbox_centerX.setValue(int(round(event.xdata)))
                self.spinbox_centerY.setValue(int(round(event.ydata)))
                self.update_mask_overlay()
            return

        # Blit setup: snapshot everything in ax_mask_preview *except*
        # the 3 artists about to be dragged, so on_motion_mask can move
        # just those on every mouse-move tick instead of a full-figure
        # redraw - dragging the virtual detector was noticeably slow
        # before this, since self.ax and self.ax_mask_preview share one
        # figure/canvas and a plain draw_idle() redraws both.
        for artist in (self._current_circle_out, self._current_circle_in,
                       self._current_center_marker):
            if artist is not None:
                artist.set_visible(False)
        self.canvas.draw()
        self._mask_bg = self.canvas.copy_from_bbox(self.ax_mask_preview.bbox)
        for artist in (self._current_circle_out, self._current_circle_in,
                       self._current_center_marker):
            if artist is not None:
                artist.set_visible(True)

    def on_motion_mask(self, event):
        """Apply the drag started by on_press_mask (move the center, or
        resize r_in/r_out) to the current mouse position - repositions the
        dragged artists directly and blits, rather than going through the
        spinboxes' valueChanged -> update_mask_overlay -> full redraw path
        on every tick (still updates the spinboxes' displayed values, with
        their change signals blocked so that path isn't retriggered)."""
        if (self._mask_drag_mode is None or event.inaxes != self.ax_mask_preview
                or event.xdata is None or event.ydata is None):
            return
        if self._mask_drag_mode == 'center':
            for sb, val in ((self.spinbox_centerX, event.xdata), (self.spinbox_centerY, event.ydata)):
                sb.blockSignals(True)
                sb.setValue(int(round(val)))
                sb.blockSignals(False)
        else:
            cx = self.spinbox_centerX.value()
            cy = self.spinbox_centerY.value()
            r = int(round(np.hypot(event.xdata - cx, event.ydata - cy)))
            if self._mask_drag_mode == 'r_out':
                r = max(r, self.spinbox_rIn.value() + 1)
                self.spinbox_rOut.blockSignals(True)
                self.spinbox_rOut.setValue(r)
                self.spinbox_rOut.blockSignals(False)
            elif self._mask_drag_mode == 'r_in':
                r = min(r, self.spinbox_rOut.value() - 1)
                self.spinbox_rIn.blockSignals(True)
                self.spinbox_rIn.setValue(r)
                self.spinbox_rIn.blockSignals(False)

        cx = self.spinbox_centerX.value()
        cy = self.spinbox_centerY.value()
        self._current_circle_out.set_center((cx, cy))
        self._current_circle_out.set_radius(self.spinbox_rOut.value())
        if self._current_circle_in is not None:
            self._current_circle_in.set_center((cx, cy))
            self._current_circle_in.set_radius(self.spinbox_rIn.value())
        self._current_center_marker.set_offsets([[cx, cy]])

        self.canvas.restore_region(self._mask_bg)
        self.ax_mask_preview.draw_artist(self._current_circle_out)
        if self._current_circle_in is not None:
            self.ax_mask_preview.draw_artist(self._current_circle_in)
        self.ax_mask_preview.draw_artist(self._current_center_marker)
        self.canvas.blit(self.ax_mask_preview.bbox)

    def on_release_mask(self, event):
        """End the drag (if one was active) and do one full, correct redraw
        - update_mask_overlay()/update_recip_scale_circles() were skipped
        during the drag itself (see on_motion_mask), so the reciprocal-space
        rings in particular need this to catch up to the final center."""
        was_dragging = self._mask_drag_mode is not None
        self._mask_drag_mode = None
        self._mask_bg = None
        if was_dragging:
            self.update_mask_overlay()

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
                self.logger.debug('Nav-signal ROI rectangle already removed.', exc_info=True)
        self.rect_navsig = patches.Rectangle(self._navsig_press, 0, 0, linewidth=1,
                                             edgecolor='r', facecolor='none')
        self.ax.add_patch(self.rect_navsig)
        self.canvas.draw()
        self._navsig_bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def on_motion_navsig(self, event):
        """Update the in-progress ROI rectangle preview while dragging
        (started by on_press_navsig)."""
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
        """Finalize the dragged scan-space ROI (normalizing negative width/
        height), store it in self.roi_navsig, and trigger a preview "Summed
        DP from ROI" computation."""
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
                self.logger.debug('Nav-signal ROI rectangle already removed.', exc_info=True)
            self.rect_navsig = None
            self.canvas.draw_idle()
        self.button_sumDpFromRoi.setDisabled(True)
        if had_roi:
            self.logger.info('Cleared nav. signal ROI.')

    def calculate_button(self):
        """Handle "Calculate All": resolve the file list (plain or smart-scan),
        validate scan size/mask parameters, snapshot the run's settings into
        self._analysis_metadata (used later by save_results), and kick off
        create_navigation_signal."""
        self.path_main = self.lineEdit_dir_signal.text()
        fns_pattern = None  # parallel per-file pattern-file list, smart-scan only
        smart_scan_rows_used = None
        if self.checkbox_smartScan.isChecked():
            fns, dtype, fns_pattern, smart_scan_rows_used = self._resolve_smart_scan_fns()
            if fns is None:
                return
        else:
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
        if self._scan_size_required(dtype) and scanSize is None:
            self.logger.warning('Cannot start: scan size is required.')
            self.message_box_scan_size_required(dtype)
            return

        mask_params = None
        if self.checkbox_useMask.isChecked():
            detectors = self.get_active_detectors()
            for det in detectors:
                if det['r_out'] <= det['r_in']:
                    self.logger.warning(
                        'Cannot start: virtual mask outer radius (%d) must be greater than '
                        'the inner radius (%d).', det['r_out'], det['r_in'])
                    qtw.QMessageBox.critical(self, 'Invalid Virtual Mask',
                        'Outer radius must be greater than the inner radius.')
                    return
            mask_params = detectors
            self.logger.info('Using %d virtual detector(s): %s', len(detectors), detectors)

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
                'detectors': [{'center': list(d['center']), 'r_in': d['r_in'], 'r_out': d['r_out']}
                             for d in mask_params] if mask_params else None,
            },
            'comment_txt_metadata_source': {
                'path': self.metadata_path_override or self.path_main,
                'block': self.spinbox_metadataCount.value(),
            } if dtype == '.tpx3' else None,
            # Recorded (in this exact per-file order, matching `files` above)
            # whenever "Smart Scanned" was used to build this nav signal, so
            # CV2/SAM2's own extraction can reuse the *same* resolved
            # angle/file/pattern match via their own "Load Metadata" instead
            # of re-deriving (and potentially resolving differently) it from
            # the folder - see tab_tracking_cv2.py/tab_sam2.py's
            # apply_navigator_metadata().
            'smart_scan': {
                'role': self.combo_smartScanRole.currentText().lower(),
                'pattern_dir': self.get_pattern_dir(),
                'files': [{'angle': row['angle'], 'file': os.path.basename(row['file']),
                          'pattern_file': row['pattern_file']}
                         for row in smart_scan_rows_used],
            } if self.checkbox_smartScan.isChecked() else None,
        }

        self.pathSave = self.lineEdit_dir_save.text()
        if not os.path.isdir(self.pathSave):
            os.mkdir(self.pathSave)

        self.button_cancel.setEnabled(True)
        self.button_save_results.setDisabled(True)
        self.create_navigation_signal(fns, dtype, scanSize, dwellTime, mask_params, fns_pattern)

    def _resolve_smart_scan_fns(self):
        """Resolve the ordered (fns, dtype, fns_pattern, rows) to use for
        "Calculate All" when "Smart Scanned" is checked - opens the check
        dialog if the folder hasn't been reviewed yet (or the folder/role
        changed since), using `self.combo_smartScanRole`'s selected role.
        Returns (None, None, None, None) if the user cancels or there's
        nothing usable to run."""
        directory = self.lineEdit_dir_signal.text()
        if not os.path.isdir(directory):
            qtw.QMessageBox.critical(self, 'No Folder', 'Select the 4D signals folder first.')
            return None, None, None, None
        dtype = None
        if self.checkbox_selectAll.isChecked():
            for ext in io.DATA_EXTENSIONS:
                if any(f.endswith(ext) for f in os.listdir(directory)):
                    dtype = ext
                    break
        else:
            dtype = self.combo_dtype.currentText()
        if dtype not in io.DATA_EXTENSIONS:
            qtw.QMessageBox.warning(self, 'Unsupported Format',
                f'Smart-scan file matching currently supports {", ".join(io.DATA_EXTENSIONS)} '
                'data only.')
            return None, None, None, None

        if self._smart_scan_rows is None:
            self.open_smart_scan_check_dialog()
        if self._smart_scan_rows is None:  # still None: user cancelled the dialog
            return None, None, None, None

        role = self.combo_smartScanRole.currentText().lower()
        resolved = io.resolve_smart_scan_files(self._smart_scan_rows, role=role)
        if not resolved:
            qtw.QMessageBox.warning(self, 'No Files',
                f'No included tilt angle has a {role} file - check "Check Files..." above.')
            return None, None, None, None
        fns = [item['file'] for item in resolved]
        fns_pattern = [item['pattern_file'] if role == 'acquisition' else None for item in resolved]
        return fns, dtype, fns_pattern, resolved

    def create_navigation_signal(self, fns, dtype, scanSize, dwellTime, mask_params=None,
                                 fns_pattern=None):
        """Queue one nav-image task per file and launch the first batch of
        worker processes (up to spinbox_cpuCores), to be drained one at a
        time by launch_next_nav_task/handle_finished_nav."""
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
        self._navimg_temp_dir = tempfile.mkdtemp(prefix='edyssey_navimg_')
        self.logger.info('Starting navigation signal creation for %d file(s)...', len(fns))
        # Detector shape only actually varies by dtype (auto-detected per
        # file for non-.tpx3 formats, or the manual/Auto UI setting for
        # .tpx3) - dtype itself is already uniform across this whole batch,
        # so one lookup from the first file covers the batch.
        det_shape = self.get_detector_shape(fns[0])
        self.tasks = deque()
        for i, fn in enumerate(fns):
            fn_pattern = fns_pattern[i] if fns_pattern is not None else None
            self.tasks.append((fn, dtype, scanSize, dwellTime, i, mask_params, fn_pattern,
                              det_shape))
        self.running_processes = []
        self.process_task_map = {}
        self.process_output_buffers = {}
        self.max_processes = self.spinbox_cpuCores.value()
        self.update_progress_bar(0, self.nav_counter_total)
        for _ in range(min(self.max_processes, len(self.tasks))):
            self.launch_next_nav_task()

    def launch_next_nav_task(self):
        """Pop the next queued task and start it as a worker_nav_img QProcess,
        wiring up its output/finished/error signals."""
        if not self.tasks or len(self.running_processes) >= self.max_processes:
            return
        fn, dtype, scanSize, dwellTime, i_index, mask_params, fn_pattern, det_shape = \
            self.tasks.popleft()
        scanSize_str = str(scanSize) if scanSize is not None else 'None'
        args = [fn, dtype, scanSize_str, str(dwellTime), str(i_index), self._navimg_temp_dir]
        # detectors_json/fn_pattern/det_shape are always appended together
        # (even as 'None'/'' sentinels) so each lands in its correct
        # positional slot in calculate_nav_img_worker's signature in
        # worker_nav_img.py, regardless of which are actually in use.
        args += [json.dumps(mask_params) if mask_params is not None else 'None']
        args += [fn_pattern or '']
        args += [str(det_shape)]
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
        """Collect a finished nav-image worker process's result (a .npy path
        printed to stdout), store it, launch the next queued task, and once
        every file is done, stack the results and finalize the batch."""
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
            self._set_nav_contrast_range(self.nav_imgs.min(), self.nav_imgs.max())
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
        """Handle a QProcess.errorOccurred signal (e.g. the worker process
        failed to start at all) for a nav-image batch run."""
        if self._cancelling:
            return
        self._nav_failed = True
        self.logger.error("QProcess error occurred: %s", error)
        self.button_cancel.setDisabled(True)
        qtw.QMessageBox.critical(self, 'Process Error',
            f'A worker process failed to start (error code {error}).\n'
            'Check that Python is on PATH and worker_nav_img.py exists.')

    def cancel_running_work(self):
        """Kill all running nav-image worker processes, clear the remaining
        task queue, and report how many files had already finished."""
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
        """Save button handler: run _save_results_impl, logging success/
        failure with elapsed time."""
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
        """Save into "<Save Path>/<Project>_EDyssey Analysis/navigator signal/
        <timestamp>/" - a dedicated, per-run subfolder, not directly in the
        Save Path - so every navigator run for a project lands under one
        shared "<Project>_EDyssey Analysis" tree (CV2/SAM2 default their own
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
        path_navigator = os.path.join(self.pathSave, f'{project_name}_EDyssey Analysis', 'navigator signal')
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
        """Display frame `imgNo` of the computed nav-image stack (self.nav_imgs)
        on the main canvas, with the current display contrast and scale bar."""
        if hasattr(self, 'nav_imgs') and isinstance(self.nav_imgs, np.ndarray):
            self.img_display.set_data(self.nav_imgs[imgNo])
            # clim comes from the contrast sliders (display-only, user-
            # adjustable), not fresh min/max here - that would silently
            # override any manual contrast tweak on every single frame
            # scrub. The slider range itself is (re)anchored to the whole
            # stack's min/max once, right after "Calculate All" finishes -
            # see _set_nav_contrast_range().
            self._update_nav_display_clim(redraw=False)
            shape_x, shape_y = self.nav_imgs[imgNo].shape
            self.img_display.set_extent([0, shape_y, shape_x, 0])
            self.ax.set_title(f'Image No. {imgNo+1:d}')
            self._update_nav_scalebar(redraw=False)
            self.canvas.draw()

    def _update_nav_scalebar(self, redraw=True):
        """(Re)draw the scale bar on the nav/test image axis to match the
        current Real (nm) field, or remove it if the field is empty/invalid.
        Works regardless of whether the currently-displayed image came from
        the nav_imgs stack or a single-file test, and is wired to the field's
        textChanged so the scale bar updates live as it's edited."""
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
        except ValueError:
            io.remove_scalebar(self.ax)
        else:
            io.add_readable_scalebar(self.ax, scale_real, 'nm')
        if redraw:
            self.canvas.draw_idle()

    def _set_nav_contrast_range(self, data_min, data_max):
        """(Re)anchor the nav-image contrast sliders to freshly-displayed
        data's raw range, resetting them to "full range" (no manual
        adjustment) - called once whenever the underlying data actually
        changes (a new Test File result, or a freshly-completed batch), not
        on every frame scrub. Purely a display aid (drives set_clim on the
        plotted image) - never touches the data array itself."""
        data_min, data_max = int(data_min), int(data_max)
        if data_max <= data_min:
            data_max = data_min + 1
        self._nav_display_data_range = (data_min, data_max)
        for slider in (self.slider_navVmin, self.slider_navVmax):
            slider.blockSignals(True)
        self.slider_navVmin.setRange(data_min, data_max)
        self.slider_navVmax.setRange(data_min, data_max)
        self.slider_navVmin.setValue(data_min)
        self.slider_navVmax.setValue(data_max)
        for slider in (self.slider_navVmin, self.slider_navVmax):
            slider.blockSignals(False)
        self._update_nav_contrast_labels()

    def _update_nav_contrast_labels(self):
        self.label_navVmin.setText(f'vmin: {self.slider_navVmin.value():d}')
        self.label_navVmax.setText(f'vmax: {self.slider_navVmax.value():d}')

    def _update_nav_display_clim(self, redraw=True):
        """Apply the contrast sliders' current vmin/vmax to the displayed nav
        image, clamping vmax above vmin if the sliders were left equal."""
        vmin = self.slider_navVmin.value()
        vmax = self.slider_navVmax.value()
        if vmin >= vmax:
            vmax = vmin + 1
            self.slider_navVmax.blockSignals(True)
            self.slider_navVmax.setValue(vmax)
            self.slider_navVmax.blockSignals(False)
        self._update_nav_contrast_labels()
        self.img_display.set_clim(vmin, vmax)
        if redraw:
            self.canvas.draw_idle()

    def _reset_nav_contrast(self):
        data_min, data_max = self._nav_display_data_range
        self._set_nav_contrast_range(data_min, data_max)
        self._update_nav_display_clim()

    def message_box_tpx3(self):
       msg = qtw.QMessageBox()
       msg.setWindowTitle("Scan Size Error!")
       msg.setText("(Currently,) tpx3 conversion requires scan size input!")
       msg.setInformativeText("Enter scan size and try again.")
       msg.setStandardButtons(qtw.QMessageBox.Ok)
       msg.setIcon(qtw.QMessageBox.Critical)
       msg.exec_()

    def _scan_size_required(self, dtype):
        """Whether an explicit (non-"Auto") scan size is required for
        `dtype` - always true for .tpx3 (eventem has no other way to learn
        it), and also true for any format when "Smart Scanned" is checked:
        a smart-scanned acquisition's own per-file header (e.g. a .mib's
        .hdr "Frames in Acquisition"), when one exists at all, reflects the
        number of frames actually written to disk for that sparse
        acquisition, not the full scan grid - so "Auto" silently resolves
        to the wrong shape instead of raising an error. See
        loaders.get_scan_size_mib_hdr and _reconstruct_smart_scan."""
        return dtype == '.tpx3' or self.checkbox_smartScan.isChecked()

    def message_box_scan_size_required(self, dtype):
        reason = ('.tpx3 files' if dtype == '.tpx3' else 'smart-scanned acquisitions')
        msg = qtw.QMessageBox()
        msg.setWindowTitle("Scan Size Required")
        msg.setText(f"Scan size input is required for {reason}.")
        msg.setInformativeText(
            '"Auto" can\'t be relied on here - a per-file header, when present, reflects '
            'the number of frames actually written to disk, not the full scan grid. '
            'Uncheck "Auto" and enter the scan size (or click "Load" to read it from '
            'comment.txt), then try again.')
        msg.setStandardButtons(qtw.QMessageBox.Ok)
        msg.setIcon(qtw.QMessageBox.Critical)
        msg.exec_()
