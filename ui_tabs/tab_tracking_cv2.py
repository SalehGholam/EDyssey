# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:55:17 2024

@author: SGholam
"""

import ast
import os
import json
from glob import glob
from PyQt5.QtCore import Qt, QTimer
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
import numpy as np
import EDyssey.io_utils as io
import hyperspy.api as hs
from hyperspy.api import load
import EDyssey.tracking_utils as tr
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as patches
import datetime
from copy import deepcopy
from .worker_thread import WorkerThread_General, ProcessStderrBuffer
from worker_extract_frame import load_dp
from .worker_launch import worker_command
from .contrast_scaling import ContrastScalingBox
from .logging_utils import LogConsole
from .base_tab import TabBase, compute_left_panel_width, get_existing_directory, build_left_panel
from .pets2_dialog import Pets2ParamsDialog
from .smart_scan_dialog import SmartScanCheckDialog
from .mask_edit_dialog import MaskEditDialog
from skimage.filters import threshold_otsu, threshold_li, threshold_mean, threshold_yen
import gc
from time import perf_counter
# from dask.distributed import Client, LocalCluster, as_completed
import base64
import pickle
import threading
from collections import deque
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
import shutil
from .loading_label import LoadingSpinner
from .object_detection_widget import Object_Detector_Widget
import pandas as pd
#%% wdiget
class Tab_Tracking_CV2(TabBase):
    def __init__(self, parent=None):
        super().__init__('Tab_Tracking_CV2', parent)
        self.threadpool.setMaxThreadCount(max(1, os.cpu_count() - 2))
        self._tracking_lock = threading.Lock()
        self._stderr_buffer = ProcessStderrBuffer()

        # Recomputing constrained_layout's spacing solve on every redraw
        # (canvas.draw()'s default behavior) is one of the most expensive
        # parts of a redraw; update_canvas() freezes it after the first
        # real draw, once subplot spacing has settled.
        self._layout_frozen_nav = False
        self._layout_frozen_extract = False
        # Cached "clean" backgrounds (everything except the per-frame image/
        # ROI-box/title artists) used to blit slider-driven frame changes
        # instead of a full canvas redraw. None means "needs (re)capture" -
        # see _blit_canvas().
        self._bg_nav = None
        self._bg_extract = None

        self.init_widget()

        # cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='2GB')
        # client = Client(cluster)

    def init_widget(self):
        button_w = 110
        button_h_lrg = 50
        width_userInput = compute_left_panel_width()
        self.layout = qtw.QHBoxLayout(self)
        self.setLayout(self.layout)
        self._splitter = qtw.QSplitter(Qt.Horizontal)
        self.layout.addWidget(self._splitter)

        # layout top
        layout_userInput = build_left_panel(self._splitter, width_userInput)
        spacer = qtw.QSpacerItem(40, 20, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
        #%% directory
        self.box_dir = qtw.QGroupBox('Directories', self)
        self.box_dir.setFixedHeight(225)
        
        layout_dir = qtw.QVBoxLayout()
        layout_userInput.addWidget(self.box_dir)
        self.box_dir.setLayout(layout_dir)
        
        # nav signal dir
        layout_dir_entry = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_entry)
        label_dir = qtw.QLabel('Nav. Signal')
        layout_dir_entry.addWidget(label_dir)
        label_dir.setFixedWidth(55)
        
        self.lineEdit_dir_navSignal = qtw.QLineEdit()
        layout_dir_entry.addWidget(self.lineEdit_dir_navSignal)
        
        
        self.button_dir_navSignal = qtw.QPushButton('...')
        layout_dir_entry.addWidget(self.button_dir_navSignal)
        self.button_dir_navSignal.clicked.connect(lambda: self.show_dialog('file'))
        
        # 4d dir
        layout_dir_4dSignals = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_4dSignals)
        
        label_dir_4d = qtw.QLabel('4D Signals')
        layout_dir_4dSignals.addWidget(label_dir_4d)
        label_dir_4d.setFixedWidth(55)
        
        self.lineEdit_dir_4d = qtw.QLineEdit()
        layout_dir_4dSignals.addWidget(self.lineEdit_dir_4d)
        
        self.button_dir_4dSignals = qtw.QPushButton('...')
        layout_dir_4dSignals.addWidget(self.button_dir_4dSignals)
        self.button_dir_4dSignals.clicked.connect(lambda: self.show_dialog('folder'))

        self.combo_dtype_4d = qtw.QComboBox()
        self.combo_dtype_4d.setMaximumWidth(90)
        self.combo_dtype_4d.addItems(['.tpx3', '.hdf5', '.hspy', '.zspy', '.mib', 'All Files'])
        self.combo_dtype_4d.setToolTip(
            'Data type of the 4D signal files in the folder above - filters out any '
            "stray non-signal file (comment.txt, pattern .txt files, logs, ...) that "
            'would otherwise be picked up and cause a false frame-count mismatch. '
            "Ignored whenever the navigator tab's own recorded file list "
            '(metadata.json, loaded via "Load Signal") applies to this same folder.')
        layout_dir_4dSignals.addWidget(self.combo_dtype_4d)

        # save dir
        layout_dir_save = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_save)
        
        label_dir_save = qtw.QLabel('Save Dir.')
        layout_dir_save.addWidget(label_dir_save)
        label_dir_save.setFixedWidth(55)
        
        self.lineEdit_dir_save = qtw.QLineEdit()
        layout_dir_save.addWidget(self.lineEdit_dir_save)
        
        self.button_dir_save = qtw.QPushButton('...')
        layout_dir_save.addWidget(self.button_dir_save)
        self.button_dir_save.clicked.connect(lambda: self.show_dialog('folder'))
        #%% load buttons (scale bars now live in the Input Parameters box below)
        layout_loadSignal = qtw.QHBoxLayout()
        self.double_validator = QDoubleValidator(0.0, 1e5, 5)
        self.dp_center = None  # (x, y) - auto-found or last manually-clicked center
        # id(dp_array) at the time dp_center was last auto-found - lets
        # update_scalebar('reciprocal') skip re-running find_dp_center_blurred
        # (a real HyperSpy call) when the displayed DP hasn't actually
        # changed since, e.g. on every keystroke in the scale-recip field.
        # See _on_auto_center_toggled.
        self._dp_center_cache_key = None

        layout_dir.addLayout(layout_loadSignal)

        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        # self.button_loadNavigation.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        self.button_loadNavigation.setFixedSize(button_w, button_h_lrg)
        layout_loadSignal.addWidget(self.button_loadNavigation)
        
        self.button_loadNavigation.clicked.connect(self.load_navSignal)

        self.button_loadSavedAnalysis = qtw.QPushButton('Load Saved\nAnalysis')
        # self.button_loadSavedAnalysis.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        self.button_loadSavedAnalysis.setFixedSize(button_w, button_h_lrg)
        layout_loadSignal.addWidget(self.button_loadSavedAnalysis)
        self.button_loadSavedAnalysis.clicked.connect(self.load_saved_analysis)

        #%% box input parameters (scan dims, scale bars, detector size, dwell
        # time, metadata block - everything needed to extract DPs for 3DED,
        # except smart-scan-specific inputs, which get their own box below)
        self.box_scanSize = qtw.QGroupBox('Input Parameters')
        layout_box_scanSize = qtw.QVBoxLayout()
        self.box_scanSize.setLayout(layout_box_scanSize)
        layout_userInput.addWidget(self.box_scanSize)

        layout_scanSize_row1 = qtw.QHBoxLayout()
        layout_box_scanSize.addLayout(layout_scanSize_row1)

        label_scanSize = qtw.QLabel('Scan Size')
        layout_scanSize_row1.addWidget(label_scanSize)

        self.checkbox_scanSize = qtw.QCheckBox('Auto')
        self.checkbox_scanSize.setChecked(True)
        self.checkbox_scanSize.setToolTip(
            'When checked, scan size is taken from the loaded navigation '
            "signal's shape. Uncheck (or Load Metadata below) to override "
            'manually - needed when that shape does not match the raw 4D '
            'signal files being extracted from.')
        layout_scanSize_row1.addWidget(self.checkbox_scanSize)

        self.lineEdit_scanSize_x = qtw.QLineEdit()
        self.lineEdit_scanSize_x.setAlignment(Qt.AlignLeft)
        self.lineEdit_scanSize_x.setFixedWidth(40)
        self.lineEdit_scanSize_x.setValidator(QIntValidator(0, 99999))
        layout_scanSize_row1.addWidget(self.lineEdit_scanSize_x)

        label_cross = qtw.QLabel('X')
        layout_scanSize_row1.addWidget(label_cross)

        self.lineEdit_scanSize_y = qtw.QLineEdit()
        self.lineEdit_scanSize_y.setFixedWidth(40)
        self.lineEdit_scanSize_y.setValidator(QIntValidator(0, 99999))
        layout_scanSize_row1.addWidget(self.lineEdit_scanSize_y)

        self.activate_lineEdit_scanSize()
        self.checkbox_scanSize.stateChanged.connect(self.activate_lineEdit_scanSize)

        label_dwellTime = qtw.QLabel('Dwell T. (μs)')
        label_dwellTime.setToolTip('Dwell time in microseconds')
        self.spinbox_dwellTime = qtw.QSpinBox()
        self.spinbox_dwellTime.setFixedWidth(60)
        self.spinbox_dwellTime.setRange(1, 99999999)
        for wid in [label_dwellTime, self.spinbox_dwellTime]:
            layout_scanSize_row1.addWidget(wid)
        layout_scanSize_row1.addStretch(1)

        # detector size (per side, in pixels) - only truly needed for .tpx3
        # (see get_detector_shape); Auto keeps the 512x512 default other
        # formats auto-detect for free anyway.
        layout_detSize = qtw.QHBoxLayout()
        layout_box_scanSize.addLayout(layout_detSize)
        label_detSize = qtw.QLabel('Detector Size')
        label_detSize.setToolTip(
            'Detector (diffraction pattern) size in pixels - used to size the extracted '
            'DP array. Auto-detected from the 4D signal files for most formats; .tpx3 '
            'needs this set explicitly (auto-detecting it would mean fully parsing a '
            'file just to learn its shape) - Auto assumes 512x512.')
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
        layout_box_scanSize.addLayout(layout_scanSize_row2)

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

        # 4D signal file list recorded by the navigator tab's own
        # metadata.json (see apply_nav_signal_metadata/resolve_4d_files) -
        # None until a nav signal with a sibling metadata.json is loaded, or
        # invalidated by a manually-browsed 4D folder.
        self._nav_4d_files = None
        self._nav_4d_directory = None

        # scale bars - moved out of Directories, real/reciprocal merged onto
        # one row; kept at the bottom of Input Parameters (below scan size/
        # detector size/metadata), since it's a display-only calibration
        # rather than an acquisition parameter.
        self.box_scale = qtw.QGroupBox('Scale bars')
        layout_box_scale = qtw.QVBoxLayout()
        self.box_scale.setLayout(layout_box_scale)
        layout_box_scanSize.addWidget(self.box_scale)

        layout_scale_row = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_row)
        label_scale_real = qtw.QLabel('Real (nm)')
        label_scale_real.setFixedWidth(55)
        layout_scale_row.addWidget(label_scale_real)
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        layout_scale_row.addWidget(self.lineEdit_scale_real)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        label_scale_recip = qtw.QLabel('Recip. (Å<sup>-1</sup>)')
        label_scale_recip.setFixedWidth(55)
        layout_scale_row.addWidget(label_scale_recip)
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        layout_scale_row.addWidget(self.lineEdit_scale_recip)
        self.lineEdit_scale_recip.setValidator(self.double_validator)

        self.checkbox_autoCenterDp = qtw.QCheckBox('Auto-center')
        self.checkbox_autoCenterDp.setChecked(True)
        self.checkbox_autoCenterDp.setToolTip(
            'When checked, the reciprocal-space rings are re-centered on the '
            'direct beam automatically (found via a large-sigma blur) after '
            'every redraw. When unchecked, hold Ctrl and click on the DP '
            'plot to set the center manually.')
        layout_box_scale.addWidget(self.checkbox_autoCenterDp)
        self.checkbox_autoCenterDp.stateChanged.connect(self._on_auto_center_toggled)
        self.lineEdit_scale_recip.textChanged.connect(lambda: self.update_scalebar('reciprocal'))
        self.lineEdit_scale_real.textChanged.connect(lambda: self.update_scalebar('real'))

        # smart-scan (pattern-file) support: the 4D signals folder holds a
        # detection + acquisition tpx3/mib file pair per tracked frame -
        # extraction always reads the acquisition (smart-scanned) file, with
        # its matching pattern file - see EDyssey/io_utils/smart_scan.py.
        #%% box smart scan (pattern-file) support: the 4D signals folder holds a
        # detection + acquisition tpx3/mib file pair per tracked frame -
        # extraction always reads the acquisition (smart-scanned) file, with
        # its matching pattern file - see EDyssey/io_utils/smart_scan.py.
        self.box_smartScan = qtw.QGroupBox('Smart Scan')
        layout_box_smartScan = qtw.QVBoxLayout()
        self.box_smartScan.setLayout(layout_box_smartScan)
        layout_userInput.addWidget(self.box_smartScan)

        layout_scanSize_row3 = qtw.QHBoxLayout()
        layout_box_smartScan.addLayout(layout_scanSize_row3)
        self.checkbox_smartScan = qtw.QCheckBox('Smart Scanned')
        self.checkbox_smartScan.setToolTip(
            'The 4D signals folder holds a smart-scanned tomography series - 3DED '
            'extraction will read each frame\'s acquisition file with its matching '
            'pattern file, instead of every raw file in the folder.')
        layout_scanSize_row3.addWidget(self.checkbox_smartScan)
        self.checkbox_smartScan.stateChanged.connect(self.activate_smartScan_widgets)
        label_patternDir = qtw.QLabel('Pattern Dir.')
        layout_scanSize_row3.addWidget(label_patternDir)
        self.lineEdit_patternDir = qtw.QLineEdit()
        self.lineEdit_patternDir.setPlaceholderText('defaults to 4D Signals folder')
        self.lineEdit_patternDir.setDisabled(True)
        layout_scanSize_row3.addWidget(self.lineEdit_patternDir)
        self.button_browsePatternDir = qtw.QPushButton('...')
        self.button_browsePatternDir.setFixedWidth(30)
        self.button_browsePatternDir.setDisabled(True)
        self.button_browsePatternDir.clicked.connect(self.browse_pattern_dir)
        layout_scanSize_row3.addWidget(self.button_browsePatternDir)

        layout_scanSize_row3b = qtw.QHBoxLayout()
        layout_box_smartScan.addLayout(layout_scanSize_row3b)
        label_detectionDir = qtw.QLabel('Detect. Dir.')
        layout_scanSize_row3b.addWidget(label_detectionDir)
        self.lineEdit_detectionDir = qtw.QLineEdit()
        self.lineEdit_detectionDir.setPlaceholderText('defaults to 4D Signals folder')
        self.lineEdit_detectionDir.setDisabled(True)
        self.lineEdit_detectionDir.setToolTip(
            'Folder to look for detection files in, if they live somewhere other than the '
            '4D Signals folder (e.g. a separate folder of HAADF .tif/.tiff reference images '
            'for .mib/.hspy/.zspy, or a cleaner acquisition layout with detection/acquisition '
            'each in their own folder)')
        layout_scanSize_row3b.addWidget(self.lineEdit_detectionDir)
        self.button_browseDetectionDir = qtw.QPushButton('...')
        self.button_browseDetectionDir.setFixedWidth(30)
        self.button_browseDetectionDir.setDisabled(True)
        self.button_browseDetectionDir.clicked.connect(self.browse_detection_dir)
        layout_scanSize_row3b.addWidget(self.button_browseDetectionDir)

        # Detection/acquisition dwell times - a smart-scanned tilt series
        # logs two metadata blocks per angle (e.g. "Scan strategy: Raster"
        # for the dense detection pass, "Scan strategy: Custom" for the
        # sparse acquisition), each with its own dwelltime - editable here
        # in case they differ from (or aren't present in) comment.txt.
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

        layout_scanSize_row4 = qtw.QHBoxLayout()
        layout_box_smartScan.addLayout(layout_scanSize_row4)
        self.button_checkSmartScanFiles = qtw.QPushButton('Check Files...')
        self.button_checkSmartScanFiles.setToolTip(
            'Review the automatic per-frame detection/acquisition/pattern-file match '
            'before extracting - fix or exclude any mismatched frame by hand')
        self.button_checkSmartScanFiles.setDisabled(True)
        self.button_checkSmartScanFiles.clicked.connect(self.open_smart_scan_check_dialog)
        layout_scanSize_row4.addWidget(self.button_checkSmartScanFiles)
        self.label_smartScanSummary = qtw.QLabel('')
        layout_scanSize_row4.addWidget(self.label_smartScanSummary)
        layout_scanSize_row4.addStretch(1)

        self._smart_scan_rows = None  # set by open_smart_scan_check_dialog() or apply_nav_signal_metadata()

        #%% display contrast (8-bit conversion used for both display and tracking)
        self.box_contrast = ContrastScalingBox()
        layout_userInput.addWidget(self.box_contrast)
        self.box_contrast.settingsChanged.connect(self.rescale_nav_signal)

        #%% feature handling
        layout_userInput_2 = qtw.QVBoxLayout()
        self.box_buttons = qtw.QGroupBox('Feature Handling')
        layout_userInput.addWidget(self.box_buttons)
        self.box_buttons.setLayout(layout_userInput_2)
        
        # top
        layout_featureTop = qtw.QHBoxLayout()
        layout_userInput_2.addLayout(layout_featureTop)
        self.button_autoDetection = qtw.QPushButton('Auto Detector')
        layout_featureTop.addWidget(self.button_autoDetection)
        self.button_autoDetection.clicked.connect(self.launch_auto_detector)
        
        self.button_reset_rois = qtw.QPushButton('Reset ROIs')
        layout_featureTop.addWidget(self.button_reset_rois)
        self.button_reset_rois.clicked.connect(self.reset_rois)
        
        # tree
        self.tree_objects = qtw.QTreeWidget()
        layout_userInput_2.addWidget(self.tree_objects)
        self.tree_objects.setMinimumWidth(200)
        # A generous baseline height (rather than the few rows its bare
        # sizeHint would give) - the panel now lives in a QScrollArea, which
        # sizes its content to this natural/minimum size regardless of the
        # window's actual height, instead of stretching to fill whatever
        # space happens to be available the way a plain splitter pane would.
        self.tree_objects.setMinimumHeight(280)
        self.cols_tree = ["use", "idx", "init", "end", "ref", "trk", "ext", "dup", "del"]
        self.tree_objects.setHeaderLabels(["Use", "Idx", "Start", "End", "Ref", "Tracked", "Extracted", "Duplicate", "Delete"])
        self.tree_objects.setColumnCount(len(self.cols_tree))
        # Wide enough for their content: dup/del hold a 30px button, end
        # holds a QSpinBox with up/down arrows, trk/ext hold a status icon.
        col_widths = {'use': 35, 'idx': 30, 'init': 50, 'end': 60, 'ref': 45,
                      'trk': 60, 'ext': 65, 'dup': 60, 'del': 45}
        for i, col in enumerate(self.cols_tree):
            self.tree_objects.setColumnWidth(i, col_widths[col])
        self.tree_objects.setSelectionMode(qtw.QTreeWidget.SingleSelection)
        self.tree_objects.itemSelectionChanged.connect(self.update_canvas)
        
        self.patches_axNav = []
        self.patches_axTrack = []
        self.empty_main_dataframe()
        
        # bottom - blur/tracker selection on its own row, action buttons on
        # another, rather than all 6 widgets crammed into one (too wide for
        # the panel) row.
        layout_featureBottom = qtw.QHBoxLayout()
        layout_userInput_2.addLayout(layout_featureBottom)

        # self.checkbox_roiInRoi = qtw.QCheckBox('Select ROIinROI')
        # layout_featureBottom.addWidget(self.checkbox_roiInRoi)
        # self.checkbox_roiInRoi.setDisabled(True)

        label_blur_track = qtw.QLabel('Image Blur')
        layout_featureBottom.addWidget(label_blur_track)
        self.combo_blur_track = qtw.QComboBox()
        layout_featureBottom.addWidget(self.combo_blur_track)
        self.combo_blur_track.addItems([str(i) for i in range(1,23,2)])
        self.combo_blur_track.currentIndexChanged.connect(self.blur_navImages)

        spacer = qtw.QSpacerItem(40, 20, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
        # layout_featureBottom.addItem(spacer)

        label_track = qtw.QLabel('Tracker')
        layout_featureBottom.addWidget(label_track)
        self.combo_trackMethod = qtw.QComboBox()
        # Closed-state width tracks a fixed character count instead of the
        # longest item ('xcorr-template') - the popup itself still shows
        # full item text, only the always-visible closed box is capped.
        self.combo_trackMethod.setSizeAdjustPolicy(qtw.QComboBox.AdjustToMinimumContentsLength)
        self.combo_trackMethod.setMinimumContentsLength(8)
        layout_featureBottom.addWidget(self.combo_trackMethod)
        self.combo_trackMethod.addItems(['csrt', 'nano', 'mil', 'dasiamrpn', 'xcorr-phase', 'xcorr-template'])
        layout_featureBottom.addStretch(1)

        layout_featureButtons = qtw.QHBoxLayout()
        layout_userInput_2.addLayout(layout_featureButtons)
        layout_featureButtons.addStretch(1)

        self.button_track = qtw.QPushButton('Track!')
        layout_featureButtons.addWidget(self.button_track, alignment=Qt.AlignCenter)
        self.button_track.clicked.connect(self.track_rois)
        self.button_track.setDisabled(True)

        self.button_fineTuneMask = qtw.QPushButton('Fine-Tune Mask...')
        self.button_fineTuneMask.setToolTip(
            'Manually edit the selected ROI\'s mask, frame by frame - grow/shrink '
            'it directionally or apply edge detection to one frame')
        layout_featureButtons.addWidget(self.button_fineTuneMask, alignment=Qt.AlignCenter)
        self.button_fineTuneMask.clicked.connect(self.open_fine_tune_mask_dialog)
        self.button_fineTuneMask.setDisabled(True)
        layout_featureButtons.addStretch(1)
        #%% box thresh and 3DED
        layout_3ded = qtw.QVBoxLayout()
        layout_userInput.addLayout(layout_3ded)

        # Deliberately kept outside box_3ded (whose contents are disabled/
        # enabled together elsewhere) so it stays clickable regardless of
        # tracking/extraction state - it needs to work throughout both.
        # Added to layout_userInput as the panel's last widget (in
        # init_widget, right after box_3ded), directly below it with no
        # extra gap.
        self.button_cancel = qtw.QPushButton('Cancel')
        # Matches the height of the tab's other action buttons (Extract!,
        # Track!) rather than the shorter button_h_sml - a narrower width is
        # enough to read as "secondary", it doesn't need to be shorter too.
        # self.button_cancel.setFixedSize(button_w, button_h_lrg)
        self.button_cancel.setFixedHeight(button_h_lrg)
        self.button_cancel.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self.button_cancel.setStyleSheet("background-color: red; color: white;")
        self.button_cancel.setDisabled(True)
        self.button_cancel.setToolTip(
            'Stop the running tracking or 3DED extraction. Already-running '
            'background computations finish silently; their results are discarded.')
        self.button_cancel.clicked.connect(self.cancel_running_work)

        self.box_3ded = qtw.QGroupBox('Extract 3DED')
        layout_box_3ded = qtw.QVBoxLayout()
        self.box_3ded.setLayout(layout_box_3ded)
        layout_3ded.addWidget(self.box_3ded)
        
        layout_thresh_method = qtw.QHBoxLayout()
        layout_thresh_method.addItem(spacer)
        layout_box_3ded.addLayout(layout_thresh_method)
        label_thresh_method = qtw.QLabel('Threshold')
        label_thresh_method.setToolTip('Threshold method')
        layout_thresh_method.addWidget(label_thresh_method)
        
        
        self.combo_thresh_method = qtw.QComboBox()
        layout_thresh_method.addWidget(self.combo_thresh_method)
        self.combo_thresh_method.addItems(['li', 'otsu', 'yen', 'mean'])
        self.combo_thresh_method.currentIndexChanged.connect(lambda: self.update_canvas()) #TODO change to update mask
        
        label_blur = qtw.QLabel('ROI Blur')
        layout_thresh_method.addWidget(label_blur)
        self.combo_blur = qtw.QComboBox()
        layout_thresh_method.addWidget(self.combo_blur)
        self.combo_blur.addItems([str(i) for i in range(1,23,2)])
        self.combo_blur.currentIndexChanged.connect(lambda: self.update_canvas())

        layout_thresh_method.addItem(spacer)

        layout_deviation = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_deviation)
        label_thresh_dev = qtw.QLabel('Deviation')
        layout_deviation.addWidget(label_thresh_dev)
        
        self.slider_thresh = qtw.QSlider(1)
        layout_deviation.addWidget(self.slider_thresh)
        self.slider_thresh.setDisabled(True)
        self.slider_thresh.valueChanged.connect(lambda: self.update_canvas()) # TODO plot only mask ax
        self.slider_thresh.setRange(0, 200)
        
        self.button_thresh = qtw.QPushButton('Reset')
        layout_deviation.addWidget(self.button_thresh)
        self.button_thresh.clicked.connect(self.reset_thresh)

        self.box_edgeDetection = qtw.QGroupBox('Edge Detection')
        layout_box_3ded.addWidget(self.box_edgeDetection)
        layout_edgeDetection = qtw.QVBoxLayout()
        self.box_edgeDetection.setLayout(layout_edgeDetection)

        layout_edgeDetection_row1 = qtw.QHBoxLayout()
        layout_edgeDetection.addLayout(layout_edgeDetection_row1)
        self.checkbox_edgeOnly = qtw.QCheckBox('Edge Detection')
        self.checkbox_edgeOnly.setToolTip(
            'Reduce each frame\'s threshold mask to just its outline (via binary '
            'erosion) before it is previewed, extracted, or saved')
        layout_edgeDetection_row1.addWidget(self.checkbox_edgeOnly)
        self.checkbox_edgeOnly.stateChanged.connect(lambda: self.update_canvas())
        label_edgeKernel = qtw.QLabel('Kernel')
        layout_edgeDetection_row1.addWidget(label_edgeKernel)
        self.spinbox_edgeKernel = qtw.QSpinBox()
        self.spinbox_edgeKernel.setRange(1, 99)
        self.spinbox_edgeKernel.setValue(3)
        self.spinbox_edgeKernel.setToolTip('Erosion kernel size (pixels) - larger = wider edge band')
        layout_edgeDetection_row1.addWidget(self.spinbox_edgeKernel)
        self.spinbox_edgeKernel.valueChanged.connect(lambda: self.update_canvas())
        self.checkbox_revertMask = qtw.QCheckBox('Revert Mask')
        self.checkbox_revertMask.setToolTip(
            'Only applies together with Edge Detection: keep the mask\'s interior '
            '(and, with "Directional" on, its other sides) but cut out the detected '
            'edge band, instead of keeping only the edge band')
        layout_edgeDetection_row1.addWidget(self.checkbox_revertMask)
        self.checkbox_revertMask.stateChanged.connect(lambda: self.update_canvas())
        layout_edgeDetection_row1.addStretch(1)

        layout_edgeDetection_row2 = qtw.QHBoxLayout()
        layout_edgeDetection.addLayout(layout_edgeDetection_row2)
        self.checkbox_edgeDirectional = qtw.QCheckBox('Directional')
        self.checkbox_edgeDirectional.setToolTip(
            'Keep only the edge band facing one direction (e.g. just the mask\'s '
            'top edge) instead of the full outline - erosion becomes one-sided, '
            'along the angle below')
        layout_edgeDetection_row2.addWidget(self.checkbox_edgeDirectional)
        self.checkbox_edgeDirectional.stateChanged.connect(self._on_edge_directional_toggled)
        label_edgeDirection = qtw.QLabel('Angle (°)')
        layout_edgeDetection_row2.addWidget(label_edgeDirection)
        self.spinbox_edgeDirection = qtw.QDoubleSpinBox()
        self.spinbox_edgeDirection.setRange(0, 359.9)
        self.spinbox_edgeDirection.setDecimals(1)
        self.spinbox_edgeDirection.setSingleStep(5)
        self.spinbox_edgeDirection.setValue(0)
        self.spinbox_edgeDirection.setDisabled(True)
        self.spinbox_edgeDirection.setToolTip(
            '0° = right, increasing clockwise (90° = down/bottom edge, '
            '180° = left, 270° = up/top edge)')
        layout_edgeDetection_row2.addWidget(self.spinbox_edgeDirection)
        self.spinbox_edgeDirection.valueChanged.connect(lambda: self.update_canvas())
        layout_edgeDetection_row2.addStretch(1)
            #%% 3DED
        layout_threadNo = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_threadNo)
        layout_threadNo.addItem(spacer)
        
        label_threadNo = qtw.QLabel('CPU Cores')
        layout_threadNo.addWidget(label_threadNo)
        self.spinbox_threadNo = qtw.QSpinBox(self)
        self.spinbox_threadNo.setMaximumWidth(80)
        layout_threadNo.addWidget(self.spinbox_threadNo)
        self.spinbox_threadNo.setRange(1, os.cpu_count() or 1)
        self.spinbox_threadNo.setValue(max(1, (os.cpu_count() or 2) - 2))
        self.spinbox_threadNo.valueChanged.connect(self.set_threadNo)
        
        
        label_fps = qtw.QLabel('Clip FPS')
        layout_threadNo.addWidget(label_fps)
        self.spinbox_fps = qtw.QSpinBox(self)
        self.spinbox_fps.setMaximumWidth(80)
        layout_threadNo.addWidget(self.spinbox_fps)
        self.spinbox_fps.setRange(1, 60)
        self.spinbox_fps.setValue(5)
        self.spinbox_fps.setToolTip('Frames per second for saved video clips')

        self.checkbox_autosave = qtw.QCheckBox('Autosave')
        layout_threadNo.addWidget(self.checkbox_autosave)

        # Own row (rather than sharing layout_threadNo with the CPU/FPS/
        # Autosave row above) so the checkbox label has enough room and
        # doesn't get clipped by the left panel's fixed width.
        layout_saveOptions = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_saveOptions)

        self.checkbox_makePets2 = qtw.QCheckBox('Make *.pts2')
        layout_saveOptions.addWidget(self.checkbox_makePets2)
        self.pets2_params = None
        self.checkbox_makePets2.stateChanged.connect(self.on_makePets2_toggled)

        layout_saveOptions.addStretch()

        layout_extract = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_extract)

        # Natural (content-sized) width, not Expanding - lets the 3 buttons
        # form a compact cluster centered in the row via the stretches
        # below, instead of stretching edge-to-edge across the panel.
        layout_extract.addStretch(1)
        self.button_3ded = qtw.QPushButton('Extract All')
        layout_extract.addWidget(self.button_3ded)
        self.button_3ded.setFixedHeight(button_h_lrg)
        self.button_3ded.clicked.connect(self.extract_3ded)

        self.button_extractCurrentFrame = qtw.QPushButton('Extract DP\n(Current Frame)')
        layout_extract.addWidget(self.button_extractCurrentFrame)
        self.button_extractCurrentFrame.setFixedHeight(button_h_lrg)
        self.button_extractCurrentFrame.setToolTip(
            'Compute the diffraction pattern for just the selected ROI at the frame the '
            'slider is currently on - a quick one-off check, not saved as part of the series')
        self.button_extractCurrentFrame.clicked.connect(self.extract_dp_current_frame)

        self.button_save_results = qtw.QPushButton('Save Results')
        layout_extract.addWidget(self.button_save_results)
        self.button_save_results.setFixedHeight(button_h_lrg)
        self.button_save_results.clicked.connect(self.save_results)
        layout_extract.addStretch(1)

        # Same width for all three - sized to fit the widest label
        # ("Extract DP\n(Current Frame)") rather than a hardcoded guess.
        extract_btn_width = self.button_extractCurrentFrame.sizeHint().width()
        for btn in (self.button_3ded, self.button_extractCurrentFrame, self.button_save_results):
            btn.setFixedWidth(extract_btn_width)

        self.disable_3ded_widgets(True)
        layout_userInput.addWidget(self.button_cancel)
        # No trailing addStretch here - box_buttons (Feature Handling, given
        # a stretch factor above) already claims all leftover vertical
        # space, via its own Expanding size policy set alongside it, so
        # Cancel naturally lands at the panel's bottom edge instead of
        # leaving a separate empty gap below it.
        #%% canvas
        self._right_widget = qtw.QWidget()
        self._splitter.addWidget(self._right_widget)
        # self._splitter.setStretchFactor(0, 0)
        # self._splitter.setStretchFactor(1, 1)
        # self._splitter.setSizes([300, 900])
        layout_canvas = qtw.QVBoxLayout(self._right_widget)
        
        # self.figure_nav = Figure(constrained_layout=True)
        self.figure_nav = Figure(constrained_layout=True)
        self.canvas_nav = FigureCanvas(self.figure_nav)
        self.ax_nav = self.figure_nav.add_subplot(121)
        self.ax_track = self.figure_nav.add_subplot(122)

        self.figure_extract = Figure(constrained_layout=True)
        self.canvas_extract = FigureCanvas(self.figure_extract)
        self.ax_mask = self.figure_extract.add_subplot(121)
        self.ax_dp = self.figure_extract.add_subplot(122)

        # titles for axes
        self.ax_nav.set_title('(1) Nav. Signal')
        self.ax_track.set_title('(2) Tracking Results')
        self.ax_mask.set_title('(3) Roi with Threshold')
        self.ax_dp.set_title('(4) DP')
        self.img_display = {}
        self.img_zero = np.zeros((512,512), dtype='uint16')
        # One-off "Extract DP (Current Frame)" result - only shown while the
        # slider/selection still matches the (idx, i_fr) it was computed
        # for; update_canvas() clears it and falls back to the normal
        # per-series dp display as soon as either changes. See
        # extract_dp_current_frame()/_on_current_frame_dp().
        self._current_frame_dp_preview = None
        axes = ['nav', 'track', 'dp']

        for i, ax in enumerate([self.ax_nav, self.ax_track, self.ax_dp]):
            self.img_display[axes[i]] = ax.imshow(self.img_zero, cmap='viridis')
            # ax.set_axis_off()
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        self.ax_track.set_xlabel(
            'Select the reference ROI and Hold "ctrl" + Drag to draw ROIinROI', fontsize=10)
        self.ax_nav.set_xlabel(
            'Hold "ctrl" + Left Click+Drag => New ROI\n'
            'Hold "ctrl" + Right Click => Add init to existing ROI', fontsize=10)
        self.ax_nav.xaxis.label.set_visible(True)
        self.ax_track.xaxis.label.set_visible(True)
        self.ax_dp.xaxis.label.set_visible(True)

        # The Ctrl+Scroll zoom hint applies to every axis on its canvas, so
        # it's a figure-wide supxlabel (one per figure) rather than repeated
        # per-axis text.
        self.figure_nav.supxlabel('Hold "Ctrl" + Scroll wheel to zoom the axis under the cursor',
                                  fontsize=10)
        self.figure_extract.supxlabel('Hold "Ctrl" + Scroll wheel to zoom the axis under the cursor',
                                      fontsize=10)

        self.ax_mask.set_axis_off()
        self.img_display['img_mask'] = self.ax_mask.imshow(self.img_zero, cmap='gray')
        self.img_display['mask'] = self.ax_mask.imshow(self.img_zero, cmap='viridis', alpha=0.1)
        self.img_display['dp'].set_norm(SymLogNorm(linthresh=1))
        self.img_display['dp'].set_cmap('inferno')

        # Created once here (not per-frame) - update_canvas() only updates
        # the underlying image data, which keeps these in sync for free.
        # 'mask' is excluded: it's a low-alpha threshold overlay on top of
        # 'img_mask', not an independently meaningful scalar image.
        self.colorbars = {}
        self.colorbars['nav'] = self.figure_nav.colorbar(
            self.img_display['nav'], ax=self.ax_nav, fraction=0.046, pad=0.04)
        self.colorbars['track'] = self.figure_nav.colorbar(
            self.img_display['track'], ax=self.ax_track, fraction=0.046, pad=0.04)
        self.colorbars['img_mask'] = self.figure_extract.colorbar(
            self.img_display['img_mask'], ax=self.ax_mask, fraction=0.046, pad=0.04)
        self.colorbars['dp'] = self.figure_extract.colorbar(
            self.img_display['dp'], ax=self.ax_dp, fraction=0.046, pad=0.04)

        self.canvas_nav.setMinimumHeight(650)
        self.canvas_extract.setMinimumHeight(650)
        self.toolbar_nav = NavigationToolbar(self.canvas_nav, self)
        self.toolbar_extract = NavigationToolbar(self.canvas_extract, self)

        self._canvas_stack_widget = qtw.QWidget()
        layout_canvas_stack = qtw.QVBoxLayout(self._canvas_stack_widget)
        layout_canvas_stack.addWidget(self.canvas_nav)
        layout_canvas_stack.addWidget(self.toolbar_nav)
        layout_canvas_stack.addWidget(self.canvas_extract)
        layout_canvas_stack.addWidget(self.toolbar_extract)

        self._canvas_scroll = qtw.QScrollArea()
        self._canvas_scroll.setWidget(self._canvas_stack_widget)
        self._canvas_scroll.setWidgetResizable(True)
        self._canvas_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout_canvas.addWidget(self._canvas_scroll)

        # Connect mouse events - only the nav/track canvas has interactive
        # ROI drawing; the mask/dp canvas is display-only apart from zoom.
        self.rect = None            # Currently drawn rectangle
        self.rect_roiInRoi = None
        self.press = None           # Mouse press coordinates

        self.canvas_nav.mpl_connect('button_press_event', self.on_press)
        self.canvas_nav.mpl_connect('button_release_event', self.on_release)
        self.canvas_nav.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas_nav.mpl_connect('scroll_event', self.on_scroll)
        self.canvas_extract.mpl_connect('scroll_event', self.on_scroll)
        self.canvas_extract.mpl_connect('button_press_event', self.on_click_dp)
        # A resized canvas invalidates the cached blit backgrounds (wrong
        # pixel dimensions), so force a recapture on the next frame update.
        self.canvas_nav.mpl_connect('resize_event', lambda evt: setattr(self, '_bg_nav', None))
        self.canvas_extract.mpl_connect('resize_event', lambda evt: setattr(self, '_bg_extract', None))

        self.backgrounds = {}
        for ax_name, ax in (('nav', self.ax_nav), ('track', self.ax_track)):
            self.backgrounds[ax_name] = self.canvas_nav.copy_from_bbox(ax.bbox)

        #%% slider img num
        layout_slider = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_slider)

        self.label_imgCounter = qtw.QLabel('Img No.')
        layout_slider.addWidget(self.label_imgCounter)

        self.lineEdit_imgNo = qtw.QLineEdit()
        layout_slider.addWidget(self.lineEdit_imgNo)
        self.lineEdit_imgNo.setFixedWidth(35)
        self.lineEdit_imgNo.setValidator(QIntValidator(0, 0))
        self.lineEdit_imgNo.returnPressed.connect(self.jump_to_frame_no)

        self.slider_imgNo = qtw.QSlider(self)
        self.slider_imgNo.setOrientation(1)  # Horizontal slider
        self.slider_imgNo.setRange(0,0)
        layout_slider.addWidget(self.slider_imgNo)

        self.button_frame_start = qtw.QPushButton('Start')
        self.button_frame_start.setFixedWidth(45)
        self.button_frame_start.setToolTip('Jump to the first frame')
        self.button_frame_start.clicked.connect(
            lambda: self.slider_imgNo.setValue(self.slider_imgNo.minimum()))
        layout_slider.addWidget(self.button_frame_start)

        self.button_frame_middle = qtw.QPushButton('Mid')
        self.button_frame_middle.setFixedWidth(45)
        self.button_frame_middle.setToolTip('Jump to the middle frame')
        self.button_frame_middle.clicked.connect(
            lambda: self.slider_imgNo.setValue(
                (self.slider_imgNo.minimum() + self.slider_imgNo.maximum()) // 2))
        layout_slider.addWidget(self.button_frame_middle)

        self.button_frame_end = qtw.QPushButton('End')
        self.button_frame_end.setFixedWidth(45)
        self.button_frame_end.setToolTip('Jump to the last frame')
        self.button_frame_end.clicked.connect(
            lambda: self.slider_imgNo.setValue(self.slider_imgNo.maximum()))
        layout_slider.addWidget(self.button_frame_end)

        self.slider_imgNo.valueChanged.connect(self.update_canvas)
        #%% progress bar
        layout_progress_bar = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_progress_bar)
        
        self.progress_bar = qtw.QProgressBar()
        layout_progress_bar.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)

        # The app-wide log console lives here (below this tab's own plot
        # column) rather than under the whole window, so the left parameter
        # panel (a separate splitter pane) can span the full window height.
        self.log_console = LogConsole(self)
        layout_canvas.addWidget(self.log_console)

        # tooltips
        self.button_loadNavigation.setToolTip('Load navigation signal (Ctrl+O)')
        self.button_loadSavedAnalysis.setToolTip(
            'Load a previously saved analysis folder: navigation signal, '
            'tracked ROIs, and extracted diffraction patterns (Ctrl+Shift+O)')
        self.button_track.setToolTip('Track all enabled ROIs across frames (Ctrl+T)')
        self.button_3ded.setToolTip('Extract 3D electron diffraction patterns (Ctrl+E)')
        self.button_save_results.setToolTip('Save tracking and 3DED results to disk (Ctrl+S)')
        self.combo_trackMethod.setToolTip(
            'OpenCV tracking algorithm — CSRT is most accurate.\n'
            'xcorr-phase/xcorr-template track by frame-to-frame cross-correlation '
            'shift (fixed ROI size, best for drift-only motion)')
        self.combo_blur_track.setToolTip('Gaussian blur kernel applied before tracking (higher = more smoothing)')
        self.combo_thresh_method.setToolTip('Thresholding algorithm used to create the binary mask')
        self.combo_blur.setToolTip('Gaussian blur kernel applied before thresholding')
        self.spinbox_threadNo.setToolTip('Number of CPU cores used for parallel 4D extraction')
        self.checkbox_autosave.setToolTip('Automatically save results when extraction finishes')
        self.checkbox_makePets2.setToolTip(
            "Write a PETS2 project file (v1.pts2) into each ROI's pets/ folder on save, "
            'ready to open directly in PETS for further 3D ED processing')

        # keyboard shortcuts
        QShortcut(QKeySequence('Ctrl+O'), self, self.button_loadNavigation.click)
        QShortcut(QKeySequence('Ctrl+Shift+O'), self, self.button_loadSavedAnalysis.click)
        QShortcut(QKeySequence('Ctrl+T'), self, self.button_track.click)
        QShortcut(QKeySequence('Ctrl+E'), self, self.button_3ded.click)
        QShortcut(QKeySequence('Ctrl+S'), self, self.button_save_results.click)
        QShortcut(QKeySequence('Ctrl+Right'), self,
                  lambda: self.slider_imgNo.setValue(self.slider_imgNo.value() + 1))
        QShortcut(QKeySequence('Ctrl+Left'), self,
                  lambda: self.slider_imgNo.setValue(self.slider_imgNo.value() - 1))
    #%% load data
    def show_dialog(self, f):
        """Open a file/folder browser for whichever button triggered this
        (nav signal file, 4D signals folder, or save folder), dispatched on
        self.sender()."""
        sender = self.sender()
        if sender == self.button_dir_navSignal:
            file_filter = "supported signals (*.zspy *.hspy);;All Files (*)"
            # path = qtw.QFileDialog.getOpenFileNames(self, "Select 4D Signals Folder", '', file_filter)
            path = qtw.QFileDialog.getOpenFileName(self, "Select 4D Signals Folder", '', file_filter)
            # os.path.exists, not isfile - .zspy stores are directories
            # (Zarr), not single files.
            if path and os.path.exists(path[0]):
                self.lineEdit_dir_navSignal.setText(path[0])
                self.lineEdit_dir_save.setText(io.default_analysis_save_dir(path[0]))
                self.apply_nav_signal_metadata(path[0])


        elif sender == self.button_dir_4dSignals:
            path = get_existing_directory(self, "Select 4D Folder")
            # if path and os.path.isdir(path[0]):
            if path:
                self.metadata_path_override = None  # new folder - re-derive comment.txt location
                self.lineEdit_dir_4d.setText(path)
                self._smart_scan_rows = None  # stale for a different folder
                self._nav_4d_files = None  # stale navigator file list for a different folder
                self.label_smartScanSummary.setText('')
                # Attempted for every format, not just .tpx3 - comment.txt is
                # written for smart-scanned .mib/.hspy/.zspy acquisitions
                # too, and load_metadata(silent=True) already no-ops
                # quietly when comment.txt is missing or unparsable.
                self.load_metadata(silent=True)

        elif sender == self.button_dir_save:
            path = get_existing_directory(self, "Select Destination Folder")
            if path and os.path.isdir(path[0]):
                self.lineEdit_dir_save.setText(path)

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

    def get_scan_size(self):
        """Manual scan size override, or None to fall back to the loaded
        navigation signal's own shape (the "Auto" behavior)."""
        if self.checkbox_scanSize.isChecked():
            return None
        try:
            x = int(self.lineEdit_scanSize_x.text())
            y = int(self.lineEdit_scanSize_y.text())
            return (x, y)
        except Exception:
            return None

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

    def browse_metadata_file(self):
        start_dir = self.lineEdit_dir_4d.text()
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
        path_main = self.metadata_path_override or self.lineEdit_dir_4d.text()
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

    def apply_nav_signal_metadata(self, fn_nav):
        """Fill scan size, dwell time, real/reciprocal scale, and the 4D
        signals directory from the navigator tab's metadata.json, if one
        sits next to `fn_nav` (i.e. this signal was produced by that tab)."""
        metadata = io.load_analysis_metadata(fn_nav)
        if not metadata:
            return
        applied = []
        d4d = metadata.get('4d_signals_directory')
        if d4d:
            self.lineEdit_dir_4d.setText(d4d)
            applied.append('4D signals directory')
        dtype = metadata.get('dtype')
        if dtype:
            idx = self.combo_dtype_4d.findText(dtype)
            if idx >= 0:
                self.combo_dtype_4d.setCurrentIndex(idx)
        scan_size = metadata.get('scan_size')
        if scan_size:
            self.checkbox_scanSize.setChecked(False)
            self.lineEdit_scanSize_x.setText(str(int(scan_size[0])))
            self.lineEdit_scanSize_y.setText(str(int(scan_size[1])))
            applied.append('scan size')
        dwell = metadata.get('dwell_time_us')
        if dwell:
            self.spinbox_dwellTime.setValue(int(dwell))
            applied.append('dwell time')
        scale_real = metadata.get('scale_real_nm_per_px')
        if scale_real:
            self.lineEdit_scale_real.setText(str(scale_real))
            applied.append('real-space scale')
        scale_recip = metadata.get('scale_recip_invA_per_px')
        if scale_recip:
            self.lineEdit_scale_recip.setText(str(scale_recip))
            applied.append('reciprocal-space scale')
        smart_scan = metadata.get('smart_scan')
        if smart_scan and smart_scan.get('role') == 'acquisition':
            # Reuse the exact per-angle file/pattern match already confirmed
            # when this nav signal was built, instead of re-deriving (and
            # potentially resolving differently) it from the 4D signals
            # folder - avoids the detection/acquisition file-count mismatch
            # this tab used to need manual folder cleanup to work around.
            self.checkbox_smartScan.setChecked(True)
            self.lineEdit_patternDir.setText(smart_scan.get('pattern_dir') or '')
            d4d_for_join = d4d or self.lineEdit_dir_4d.text()
            self._smart_scan_rows = [{
                'angle': item['angle'], 'detection_file': None,
                'acquisition_file': os.path.join(d4d_for_join, item['file']),
                'pattern_file': item['pattern_file'],
                'extra_files': [], 'status': ['ok'], 'excluded': False,
            } for item in smart_scan.get('files', [])]
            self.label_smartScanSummary.setText(
                f"{len(self._smart_scan_rows)} angle(s) from navigator metadata")
            applied.append('smart-scan file match')
        elif d4d and metadata.get('files'):
            # Plain (non-smart-scan) run: the navigator recorded the exact
            # ordered file list it used to build this nav signal - reuse it
            # in resolve_4d_files() instead of re-globbing the folder, so
            # stray non-signal files (comment.txt, pattern files, logs, ...)
            # can't cause a false frame-count mismatch. Only valid for this
            # same folder - see resolve_4d_files.
            self._nav_4d_files = metadata['files']
            self._nav_4d_directory = d4d
            applied.append('4D signal file list')
        if applied:
            self.logger.info(
                'Applied metadata.json from the navigator tab (%s): %s.',
                fn_nav, ', '.join(applied))

    def activate_smartScan_widgets(self):
        enable = self.checkbox_smartScan.isChecked()
        for wid in (self.lineEdit_patternDir, self.button_browsePatternDir,
                    self.lineEdit_detectionDir, self.button_browseDetectionDir,
                    self.button_checkSmartScanFiles, self.spinbox_dwellTime_detection,
                    self.spinbox_dwellTime_acquisition):
            wid.setEnabled(enable)

    def browse_pattern_dir(self):
        start_dir = self.lineEdit_patternDir.text() or self.lineEdit_dir_4d.text()
        path = get_existing_directory(self, "Select Pattern Files Folder", start_dir)
        if path:
            self.lineEdit_patternDir.setText(path)
            self._smart_scan_rows = None
            self.label_smartScanSummary.setText('')

    def get_pattern_dir(self):
        return self.lineEdit_patternDir.text() or self.lineEdit_dir_4d.text()

    def browse_detection_dir(self):
        start_dir = self.lineEdit_detectionDir.text() or self.lineEdit_dir_4d.text()
        path = get_existing_directory(self, "Select Detection Files Folder", start_dir)
        if path:
            self.lineEdit_detectionDir.setText(path)
            self._smart_scan_rows = None
            self.label_smartScanSummary.setText('')

    def get_detection_dir(self):
        return self.lineEdit_detectionDir.text() or None

    def open_smart_scan_check_dialog(self):
        """Validate the 4D signals folder, detect its data format, and open
        SmartScanCheckDialog for the user to confirm/edit the per-angle
        detection/acquisition/pattern-file match."""
        directory = self.lineEdit_dir_4d.text()
        if not os.path.isdir(directory):
            qtw.QMessageBox.critical(self, 'No Folder', 'Select the 4D signals folder first.')
            return
        dtype = None
        for ext in io.DATA_EXTENSIONS:
            if any(f.endswith(ext) for f in os.listdir(directory)):
                dtype = ext
                break
        if dtype is None:
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

    def load_navSignal(self):
        """Load the navigation signal from lineEdit_dir_navSignal's path in
        a background worker thread, then hand it to initiate_processing()."""
        def get_signal(fn):
            return load(fn)

        fn = self.lineEdit_dir_navSignal.text()
        # os.path.exists, not isfile - .zspy stores are directories (Zarr),
        # not single files.
        if not os.path.exists(fn):
            self.logger.error('Navigation signal file not found: %s', fn)
            qtw.QMessageBox.critical(self, 'File Not Found',
                f'Cannot find navigation signal file:\n{fn}')
            return

        self.load_spinner()
        gc.collect()
        #TODO delete the previous batch
        if hasattr(self, 'rois_tracked'):
            self.reset_rois()
            self.disable_3ded_widgets(True)

        self.reset_data()
        self.logger.info('Loading navigation signal from %s...', fn)
        # self.s = load(fn)
        worker = WorkerThread_General(get_signal, 0, fn)
        worker.signals.results.connect(self.initiate_processing)
        self.threadpool.start(worker)
    
    def reset_data(self):
        """Clear all ROI patches from the nav/track axes, reset the ROI
        dataframe/tree and displayed images to blank placeholders."""
        # Artists may already be gone (e.g. axes cleared elsewhere since the
        # last reset) - that's expected, not an error.
        for p in self.ax_nav.patches:
            try:
                p.remove()
            except Exception: pass
        for p in self.ax_track.patches:
            try:
                p.remove()
            except Exception: pass
        for p in self.patches_axNav:
            try:
                p.remove()
            except Exception: pass
        for p in self.patches_axTrack:
            try:
                p.remove()
            except Exception: pass
        self.empty_main_dataframe()
        # New data may change image extents/overlays - force the blit
        # backgrounds to be recaptured on the next frame update.
        self._bg_nav = None
        self._bg_extract = None

        self.img_display['track'].set_data(self.img_zero)
        self.img_display['mask'].set_data(self.img_zero)
        self.img_display['img_mask'].set_data(self.img_zero)
        self.img_display['dp'].set_data(self.img_zero)
        self.img_display['nav'].set_data(self.img_zero)
        self.tree_objects.clear()
        # Cached PETS2 params (esp. the per-study center/alpha start/step)
        # were computed for the ROI set just wiped out above - don't let them
        # silently apply to whatever gets added next.
        self.pets2_params = None
        self.checkbox_makePets2.setChecked(False)
        
    
    def rescale_nav_signal(self):
        """Retune contrast without reloading the signal from disk: the
        currently-displayed frame is rescaled immediately (cheap, instant
        feedback), while the full stack (used for tracking, and to keep
        every other frame in sync) rescales in the background - a long
        stack no longer blocks/lags the GUI on every settings tweak.
        Rapid retuning cancels (i.e. discards the result of) any
        still-running previous background rescale - see
        ContrastScalingBox.rescale_async."""
        if not hasattr(self, 's'):
            return
        imgNo = self.slider_imgNo.value()
        frame_8bit = self.box_contrast.rescale_frame(self.nav_imgs_raw[imgNo])
        self.nav_imgs[imgNo] = frame_8bit
        self.img_display['nav'].set_clim(vmin=frame_8bit.min(), vmax=frame_8bit.max())
        self.img_display['track'].set_clim(vmin=frame_8bit.min(), vmax=frame_8bit.max())
        self.update_canvas()
        self.canvas_nav.draw_idle()
        self.canvas_extract.draw_idle()

        self.box_contrast.rescale_async(self.s, self.threadpool, self.logger,
                                        on_done=self._on_nav_signal_rescaled)

    def _on_nav_signal_rescaled(self, s_8bit):
        """Callback for box_contrast.rescale_async(): apply the freshly
        rescaled full image stack once the background rescale finishes."""
        self.s_8bit = s_8bit
        self.nav_imgs = deepcopy(s_8bit.data)
        self.img_display['nav'].set_clim(vmin=self.nav_imgs.min(), vmax=self.nav_imgs.max())
        self.img_display['track'].set_clim(vmin=self.nav_imgs.min(), vmax=self.nav_imgs.max())
        self.update_canvas()
        self.canvas_nav.draw_idle()
        self.canvas_extract.draw_idle()

    def initiate_processing(self, result, index):
        """WorkerThread_General callback for load_navSignal()/
        _on_saved_analysis_loaded(): `result` is the loaded hyperspy signal;
        sets up nav_imgs, axis extents/clims, scale bars, and the frame
        slider for it."""
        self.s = result
        # Anchor the clip-threshold sliders to this signal's raw range before
        # reading get_kwargs() below - a previous signal's clip values would
        # otherwise carry over onto a dataset with a different intensity scale.
        self.box_contrast.set_data_range(self.s.data.min(), self.s.data.max())
        self.s_8bit = io.convert_to_8bit(self.s, **self.box_contrast.get_kwargs())
        self.nav_imgs_raw = self.s.data
        self.nav_imgs = deepcopy(self.s_8bit.data)
        self.dp_center = None  # a new signal may have a different DP shape/center
        self._dp_center_cache_key = None
        self.spinner.stop()
        
        shape_x, shape_y = self.nav_imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['track'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['nav'].set_clim(vmin=self.nav_imgs.min(), vmax=self.nav_imgs.max())
        self.img_display['track'].set_clim(vmin=self.nav_imgs.min(), vmax=self.nav_imgs.max())
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.nav_imgs)))

        # Reset the view to the newly loaded data's full extent (in case the
        # user had already zoomed in on a previous signal, which disables
        # autoscale), then re-seed the toolbar's view stack so its "Home"
        # button resets to *this* view instead of doing nothing (it does
        # nothing until something pushes at least one view onto its stack,
        # which our own scroll-wheel zoom deliberately bypasses).
        # ax_mask is deliberately excluded here: unlike nav/track, it shows a
        # per-ROI *cropped* image whose size varies with each ROI, and whose
        # view is kept in sync with that in update_ax_mask() instead.
        for ax in (self.ax_nav, self.ax_track):
            ax.set_xlim(0, shape_y)
            ax.set_ylim(shape_x, 0)
        self.toolbar_nav.update()
        self.toolbar_nav.push_current()

        self.update_canvas(0)
        self.canvas_nav.draw()
        self.canvas_extract.draw()
        # Scale fields may already hold a value from a previous session/load -
        # update_scalebar() is otherwise only triggered by the fields' own
        # textChanged signal, so a fresh load wouldn't show it until touched.
        self.update_scalebar('real')
        self.update_scalebar('reciprocal')
        self.slider_imgNo.setRange(0, len(self.nav_imgs)-1)
        self.logger.info('No. of Images: %s', len(self.nav_imgs))
        
        self.button_reset_rois.setEnabled(True)
        # self.button_cur_roi.setEnabled(True)
        self.button_track.setEnabled(True)
        self.button_fineTuneMask.setEnabled(True)

    def load_saved_analysis(self):
        """Restore a previously saved analysis folder (produced by
        save_results): the navigation signal, per-ROI tracking (init points/
        out_rois/mask), and any extracted diffraction patterns."""
        path = get_existing_directory(
            self, "Select Saved Analysis Folder", self.lineEdit_dir_save.text())
        if not path:
            return
        # Newer saves only record the *path* the nav signal was loaded from
        # (see save_analysis_info); fall back to an in-folder copy for
        # analyses saved before that change.
        info = io.load_analysis_info(path)
        analysis_type = info.get('analysis_type') if info else None
        if analysis_type is not None and analysis_type != 'cv2':
            self.logger.warning(
                'Refusing to load %s: this analysis was saved from the %s tab, not CV2.',
                path, analysis_type)
            qtw.QMessageBox.warning(self, 'Wrong Analysis Type',
                f'This folder was saved from the {analysis_type.upper()} tab, not this '
                'ROI tracker - the two tabs save different per-object data (masks, points, '
                "columns) and this folder won't load correctly here.\n\n"
                f'Open it from the {analysis_type.upper()} tab instead.')
            return
        fn_nav = info.get('nav_signal_source') if info else None
        if not (fn_nav and os.path.isfile(fn_nav)):
            fn_nav_legacy = os.path.join(path, 'navigation_signal.hspy')
            if os.path.isfile(fn_nav_legacy):
                fn_nav = fn_nav_legacy
            else:
                missing = fn_nav or fn_nav_legacy
                self.logger.error('Cannot find the navigation signal for this analysis: %s', missing)
                qtw.QMessageBox.critical(self, 'Navigation Signal Not Found',
                    f'Cannot find the navigation signal for this analysis.\n\n'
                    f'Expected it at:\n{missing}\n\n'
                    'It may have been moved, renamed, or deleted since this analysis was saved.')
                return

        self.load_spinner()
        gc.collect()
        if hasattr(self, 'rois_tracked'):
            self.reset_rois()
            self.disable_3ded_widgets(True)
        self.reset_data()
        self.logger.info('Loading saved analysis from %s...', path)

        worker = WorkerThread_General(self._load_saved_analysis_worker, 0, path, fn_nav)
        worker.signals.results.connect(self._on_saved_analysis_loaded)
        self.threadpool.start(worker)

    @staticmethod
    def _load_npy_or_none(fn):
        """Load `fn` as a numpy array, or None. out_rois/mask are saved
        unconditionally even when still None (as a pickled 0-d object
        array), so both "file missing" and "file holds a pickled None" need
        to come back as None here."""
        if not os.path.isfile(fn):
            return None
        arr = np.load(fn, allow_pickle=True)
        if arr.ndim == 0 and arr.item() is None:
            return None
        return arr

    def _load_saved_analysis_worker(self, path, fn_nav):
        """Background-thread worker for load_saved_analysis(): loads the nav
        signal plus each ROI's json/out_rois/mask/dp from `path`, returning
        them for _on_saved_analysis_loaded() to apply on the GUI thread."""
        s = load(fn_nav)

        rois = []
        for name in sorted(os.listdir(path)):
            roi_dir = os.path.join(path, name)
            if not (os.path.isdir(roi_dir) and name.startswith('roi No ')):
                continue
            fn_json = os.path.join(roi_dir, f'{name}.json')
            if not os.path.isfile(fn_json):
                continue
            idx = int(name[len('roi No '):])
            with open(fn_json) as f:
                row = json.load(f)

            out_rois = self._load_npy_or_none(os.path.join(roi_dir, 'output_rois.npy'))
            mask = self._load_npy_or_none(os.path.join(roi_dir, 'output_mask.npy'))

            dp = None
            fn_dp_hspy = os.path.join(roi_dir, '3DED.hspy')
            fn_dp_npy = os.path.join(roi_dir, '3DED.npy')
            if os.path.isfile(fn_dp_hspy):
                dp = load(fn_dp_hspy).data
            elif os.path.isfile(fn_dp_npy):
                dp = np.load(fn_dp_npy)

            rois.append({
                'idx': idx, 'use': row['use'], 'init': row['init'],
                'in_rois': row['in_rois'], 'end': row['end'], 'ref': row['ref'],
                'out_rois': out_rois, 'mask': mask, 'dp': dp,
            })
        return s, rois, path, fn_nav

    def _on_saved_analysis_loaded(self, result, index):
        """Callback for _load_saved_analysis_worker(): populate df_rois and
        the ROI tree from the loaded data, then enable the 3DED controls."""
        s, rois, path, fn_nav = result
        self.lineEdit_dir_navSignal.setText(fn_nav)
        self.initiate_processing(s, index)

        for roi in rois:
            idx = roi['idx']
            self.df_rois.loc[idx] = [roi['use'], roi['init'], roi['in_rois'], roi['end'],
                                      roi['ref'], roi['out_rois'], roi['mask'], roi['dp']]
            self.add_item_tree(idx, roi['init'], roi['end'], roi['ref'], roi['use'])
            row_index = self.df_rois.index.get_loc(idx)
            if roi['out_rois'] is not None:
                self.toggle_tree_icon(row_index, 'trk', True)
            if roi['dp'] is not None:
                self.toggle_tree_icon(row_index, 'ext', True)

        self.disable_3ded_widgets(False)
        self.update_canvas(0)
        # Loaded DPs may have a different center than the placeholder - re-run
        # auto-centering now if enabled.
        self.update_scalebar('reciprocal')
        self.logger.info('Loaded saved analysis from %s (%d ROI(s)).', path, len(rois))

    def disable_3ded_widgets(self, state):
        for wid in self.box_3ded.findChildren(qtw.QWidget):
            if not isinstance(wid, qtw.QLabel):
                wid.setDisabled(state)
    
    def disable_roiInRoi_widgets(self, state):
        for wid in self.box_roiInRoi.findChildren(qtw.QWidget):
            if not isinstance(wid, qtw.QLabel):
                wid.setDisabled(state)
    
    def set_threadNo(self, value):
        self.threadpool.setMaxThreadCount(value)
    
    def empty_main_dataframe(self):
        """(Re)create df_rois as an empty dataframe with the expected
        columns/dtypes, and clear the cached ROI patch lists."""
        self.cols_df = ['use', 'init', 'in_rois', 'end',
                        'ref', 'out_rois', 'mask', 'dp']
        self.df_rois = pd.DataFrame([], columns=self.cols_df)
        self.df_rois = self.df_rois.astype({'use': int, 'init': object, 'in_rois': object, 'end': int, 
                                            'out_rois': object, 'dp': object, 'ref':str, 'mask':object})
        
        self.patches_axTrack.clear()
        self.patches_axNav.clear()

    def blur_navImages(self):
        kernelSize = int(self.combo_blur_track.currentText())
        new_images = np.zeros_like(self.nav_imgs)
        for i, img in enumerate(self.s_8bit.data):
            new_images[i] = io.gaussian_blur(img, kernelSize)
        self.nav_imgs = new_images
        self.update_canvas()
        self.logger.info('Applied blur (kernel size %d) to navigation images.', kernelSize)

    def reset_rois(self):
        self.tree_objects.clear()
        self.empty_main_dataframe()
        self.update_canvas()
        # Cached PETS2 params (esp. the per-study center/alpha start/step)
        # were computed for the ROI set just wiped out above - don't let them
        # silently apply to whatever gets added next.
        self.pets2_params = None
        self.checkbox_makePets2.setChecked(False)
        self.logger.info('Reset all ROIs.')
#%% canvas functions    
    def jump_to_frame_no(self):
        num = int(self.lineEdit_imgNo.text())
        self.slider_imgNo.setValue(num)
    
    def update_canvas(self, imgNo=None):
        """Redraw the nav/track/mask/dp panels for `imgNo` (or the slider's
        current value) and the selected ROI, via blit."""
        # Guards the edge-detection/threshold controls' live-preview wiring
        # (checkbox_edgeOnly, spinbox_edgeKernel, checkbox_edgeDirectional,
        # spinbox_edgeDirection) - they live in box_3ded, normally disabled
        # until a navigation signal is loaded, but a disabled QWidget still
        # emits its change signals when set programmatically, and an
        # uncaught exception inside a Qt slot aborts the whole process
        # rather than raising normally - so this can't just rely on
        # AttributeError being raised and caught somewhere.
        if not hasattr(self, 'nav_imgs') or len(self.nav_imgs) == 0:
            return
        if imgNo is None:
            imgNo = self.slider_imgNo.value()

        img = self.nav_imgs[imgNo]
        
        self.update_ax(img, 'nav', self.ax_nav, f'Nav Image No. {imgNo:d}')
        self.draw_rois_in(imgNo)
        
        selected_items = self.tree_objects.selectedItems()
        if selected_items:
            item = selected_items[0]
            idx = int(item.text(1))
            # track
            if not np.all(pd.isna(self.df_rois.loc[idx, 'out_rois'])):
                self.update_ax(img, 'track', self.ax_track, f'Nav Image No. {imgNo:d}')
                self.draw_rois_out(imgNo)

                # mask
                roi = self.df_rois.loc[idx, 'out_rois'][imgNo]
                if roi.any():
                    img_mask, img_roi = self.threshold_img(
                        img, self.df_rois.loc[idx, 'out_rois'][imgNo], 
                        self.combo_thresh_method.currentText(),
                        self.slider_thresh.value()) #TODO add thresholding mode to the GUI and function here
                    self.update_ax_mask(img_roi, img_mask)
                else:
                    self.update_ax(self.img_zero, 'track', self.ax_track)
                    self.update_ax_mask(self.img_zero, self.img_zero)
            else:
                self.update_ax(self.img_zero, 'track', self.ax_track)
                self.update_ax_mask(self.img_zero, self.img_zero)
            
            # dp
            preview = self._current_frame_dp_preview
            if preview is not None and preview['idx'] == idx and preview['i_fr'] == imgNo:
                self.update_ax(preview['dp'], 'dp', self.ax_dp,
                               f"DP (ROI {preview['idx']}, frame {preview['i_fr']} only)")
            else:
                self._current_frame_dp_preview = None
                if idx in self.df_rois.dp.dropna().index:
                    try:
                        self.update_ax(self.df_rois.loc[idx, 'dp'][imgNo], 'dp', self.ax_dp)
                        #TODO set the content size after getting the data
                    except Exception:
                        self.update_ax(self.img_zero, 'dp', self.ax_dp)
                else:
                    self.update_ax(self.img_zero, 'dp', self.ax_dp)

        # A single blit per canvas here (instead of a full canvas.draw()/
        # draw_idle() per frame) avoids re-rendering every artist in the
        # figure - scale bars, static titles/labels, axis chrome - on every
        # single slider tick; only the image data, ROI boxes, and the
        # per-frame title text actually change between frames, so only
        # those are redrawn. constrained_layout's spacing solve is also one
        # of the most expensive parts of a full draw and doesn't need to
        # repeat once subplot spacing has settled.
        nav_artists = ([self.img_display['nav'], self.img_display['track'],
                        self.ax_nav.title, self.ax_track.title]
                       + self.patches_axNav + self.patches_axTrack)
        self._blit_canvas(
            self.canvas_nav, self.figure_nav, '_bg_nav', nav_artists,
            hide_for_background=[self.img_display['nav'], self.img_display['track']],
            titles_for_background=(self.ax_nav, self.ax_track))
        if not self._layout_frozen_nav:
            self.figure_nav.set_layout_engine('none')
            self._layout_frozen_nav = True

        extract_artists = [self.img_display['img_mask'], self.img_display['mask'],
                           self.img_display['dp']]
        self._blit_canvas(
            self.canvas_extract, self.figure_extract, '_bg_extract', extract_artists,
            hide_for_background=extract_artists)
        if not self._layout_frozen_extract:
            self.figure_extract.set_layout_engine('none')
            self._layout_frozen_extract = True

    def _blit_canvas(self, canvas, figure, bg_attr, artists,
                     hide_for_background=(), titles_for_background=()):
        """Render `artists` onto `canvas` via blit instead of a full
        canvas.draw()/draw_idle().

        Reuses a cached "clean" background (everything in the figure except
        `artists`) stored in `self.<bg_attr>`. That background is captured
        lazily - whenever `self.<bg_attr>` is None (first use, or after
        being invalidated elsewhere e.g. on canvas resize, new data, or a
        scale bar being added/removed) - by briefly hiding
        `hide_for_background` and blanking `titles_for_background`, doing
        one full draw(), then restoring them before the real content is
        blitted on top.
        """
        if getattr(self, bg_attr) is None:
            prev_visible = [a.get_visible() for a in hide_for_background]
            for a in hide_for_background:
                a.set_visible(False)
            prev_titles = [ax.get_title() for ax in titles_for_background]
            for ax in titles_for_background:
                ax.set_title('')

            canvas.draw()
            setattr(self, bg_attr, canvas.copy_from_bbox(figure.bbox))

            for a, v in zip(hide_for_background, prev_visible):
                a.set_visible(v)
            for ax, t in zip(titles_for_background, prev_titles):
                ax.set_title(t)

        canvas.restore_region(getattr(self, bg_attr))
        for artist in artists:
            artist.axes.draw_artist(artist)
        canvas.blit(figure.bbox)

    def update_ax(self, img, img_disp, ax, title=None,):
        """Update the image data, title, and color limits for one display
        axis (nav/track/mask/dp)."""
        # Rendering is deferred to the single canvas.draw()/draw_idle() call
        # at the end of update_canvas(), rather than a redraw per axis here.
        self.img_display[img_disp].set_data(img)
        ax.set_title(title)
        self.img_display[img_disp].set_clim(vmin=img.min(), vmax=img.max())

    def draw_rois_in(self, imgNo):
        """Draw the input ROI rectangles (+ id labels) that start on frame
        `imgNo` onto ax_nav, replacing whatever was drawn there before."""
        if len(self.patches_axNav) > 0:
            for p in self.patches_axNav:
                p.remove()
            self.patches_axNav.clear()
        df = self.df_rois[self.df_rois.use == 1]
        if len(df) > 0:
            for i in df.index:
                if imgNo in df.loc[i, 'init']:
                    idx = df.loc[i, 'init'].index(imgNo)
                    roi = df.loc[i, 'in_rois'][idx]
                    x,y,w,h = roi
                    rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='r', 
                                             facecolor='none')
                    self.ax_nav.add_patch(rect)
                    self.patches_axNav.append(rect)
                    
                    # id
                    # pos = (x+w+15, y+h+15)
                    font_size = 8
                    pos = (x+w/2, y-15)
                    # font_size = 12
                    t = self.ax_nav.text(pos[0], pos[1], str(i), horizontalalignment='center', 
                                         verticalalignment='center', color='red', fontsize=font_size)
                    self.patches_axNav.append(t)
        # Rendering is deferred to the single canvas.draw()/draw_idle() call
        # at the end of update_canvas(), rather than a blit here.

    def draw_rois_out(self, imgNo):
        """Draw the tracked (output) ROI rectangles (+ id labels) for frame
        `imgNo` onto ax_track, replacing whatever was drawn there before."""
        if len(self.patches_axTrack) > 0:
            for p in self.patches_axTrack:
                p.remove()
            self.patches_axTrack.clear()
        df = self.df_rois[self.df_rois.use == 1]
        df = df.loc[df.out_rois.dropna().index]
        if len(df) > 0:
            for i in df.index:
                try:
                    roi = self.df_rois.loc[i, 'out_rois'][imgNo]
                    x,y,w,h = roi
                    if (w>0) and (h>0):
                        rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='tab:orange', 
                                                 facecolor='none')
                        self.ax_track.add_patch(rect)
                        self.patches_axTrack.append(rect)
                        
                        # id
                        # pos = (x+w+15, y+h+15)
                        font_size = 8
                        pos = (x+w/2, y-15)
                        # font_size = 12
                        t = self.ax_track.text(pos[0], pos[1], str(i), horizontalalignment='center', 
                                               verticalalignment='center', color='tab:orange', fontsize=font_size)
                        self.patches_axTrack.append(t)
                except Exception:
                    self.logger.debug('Skipped drawing ROI %s at frame %d.', i, imgNo, exc_info=True)
        # Rendering is deferred to the single canvas.draw()/draw_idle() call
        # at the end of update_canvas(), rather than a blit here.

    def update_ax_mask(self, img_roi, img_mask):
        """Update ax_mask's cropped ROI image and threshold-mask overlay,
        resetting the view to fit the (per-ROI-varying) crop size."""
        shape_x, shape_y = img_mask.shape
        self.img_display['img_mask'].set_data(img_roi)
        try:
            self.img_display['img_mask'].set_clim(vmin=img_roi.min(), vmax=img_roi.max())
        except ValueError:
            pass
        self.img_display['img_mask'].set_extent([0, shape_y, shape_x, 0])
        
        self.img_display['mask'].set_data(img_mask)
        self.img_display['mask'].set_clim(vmin=0, vmax=1)
        self.img_display['mask'].set_extent([0, shape_y, shape_x, 0])
        # The ROI crop's size varies between ROIs/selections, so (unlike
        # nav/track) the view here is reset to fit it every single time,
        # rather than only once at load — otherwise it stays at whatever
        # size an earlier, differently-sized ROI last used.
        self.ax_mask.set_xlim(0, shape_y)
        self.ax_mask.set_ylim(shape_x, 0)
        # Rendering is deferred to the single canvas.draw()/draw_idle() call
        # at the end of update_canvas(), rather than a blit here.
        # self.canvas.draw_idle()
        
    def update_scalebar(self, which):
        """Add/update the scale bar (which='real', on nav/track/mask) or
        the reciprocal-space rings (which='reciprocal', on the dp axis),
        based on the current scale line-edit text."""
        if which == 'real':
            try:
                scale_real = float(self.lineEdit_scale_real.text())

                for ax in [self.ax_nav, self.ax_track, self.ax_mask]:
                    io.add_readable_scalebar(ax, scale_real, 'nm')

                # ax_nav/ax_track live on canvas_nav, ax_mask on canvas_extract.
                # The scale bar itself is static across frames, so it needs
                # to be baked into the cached blit backgrounds - invalidate
                # them here so the next frame update recaptures it, while
                # still doing an immediate full draw for instant feedback now.
                self._bg_nav = None
                self._bg_extract = None
                self.canvas_nav.draw_idle()
                self.canvas_extract.draw_idle()

            except ValueError:

                for ax in [self.ax_nav, self.ax_track, self.ax_mask]:
                    for artist in ax.artists[:]:
                        if isinstance(artist, ScaleBar):
                            artist.remove()

                self._bg_nav = None
                self._bg_extract = None
                self.canvas_nav.draw_idle()
                self.canvas_extract.draw_idle()

        elif which == 'reciprocal':
            # A conventional linear scale bar doesn't read naturally on a
            # radially-symmetric diffraction pattern - concentric dashed
            # rings at every 1 1/A (centered on the DP) work better.
            dp_array = self.img_display['dp'].get_array()
            shape = dp_array.shape
            # Skip auto-centering on the all-zero placeholder shown before
            # any ROI has a diffraction pattern yet - the blurred-max search
            # degenerates to (0, 0) on blank data, which would otherwise
            # leave the reciprocal-space rings missing right after a fresh
            # load. Leaving dp_center at None falls back to the image's own
            # geometric center.
            # find_dp_center_blurred is a real HyperSpy call (Signal2D + a
            # gaussian-blur map) - only worth re-running when the displayed
            # DP has actually changed since the last time (id() is a cheap,
            # reliable proxy: set_data() always replaces the artist's array
            # wholesale, never mutates it in place), not on every redraw -
            # this is wired to per-keystroke text-field edits.
            dp_key = id(dp_array)
            if (self.checkbox_autoCenterDp.isChecked() and np.any(dp_array)
                    and self._dp_center_cache_key != dp_key):
                try:
                    self.dp_center = io.find_dp_center_blurred(dp_array)
                    self._dp_center_cache_key = dp_key
                except Exception:
                    self.logger.exception('Auto-centering the diffraction pattern failed.')
            self._dp_recip_circles = io.draw_reciprocal_scale_circles(
                self.ax_dp, self.lineEdit_scale_recip.text(), shape,
                center=self.dp_center, old_artists=getattr(self, '_dp_recip_circles', None))
            if self.checkbox_autoCenterDp.isChecked():
                self.ax_dp.set_xlabel(
                    'Circle center: auto (large-sigma blur) - uncheck "Auto-center" '
                    'to set manually via Ctrl+Click', fontsize=9)
            else:
                self.ax_dp.set_xlabel('Circle center: manual - Ctrl+Click DP plot to set', fontsize=9)
            # The circles are static across frames like the scale bars above.
            self._bg_extract = None
            self.canvas_extract.draw_idle()

    def _on_auto_center_toggled(self):
        """checkbox_autoCenterDp.stateChanged handler: force one fresh
        find_dp_center_blurred re-run on the next redraw, even though the
        displayed DP itself hasn't changed - otherwise re-enabling auto-
        center after a manual Ctrl+Click override would silently keep
        showing that stale manual center, since update_scalebar()'s cache
        only re-runs it when the DP's identity has changed."""
        self._dp_center_cache_key = None
        self.update_scalebar('reciprocal')

    def threshold_img(self, img, roi, thresh_method, thresh_offset, mode='full'):
        """Blur `img`, compute a threshold via `thresh_method` (over the
        whole image if mode='full', or just within `roi` if mode='roi'),
        binarize at `thresh_offset` (percent of the computed threshold),
        then crop both the mask and the raw image to `roi` and apply the
        current edge-detection settings to the mask. Returns
        (img_mask, img_cut)."""
        # thresh_method = self.combo_thresh_method.currentText()
        blur_kernel = int(self.combo_blur.currentText())
        threshold_methods = {'otsu': threshold_otsu, 'li': threshold_li, 
                             'yen': threshold_yen, 'mean': threshold_mean}
        threshold_func = threshold_methods[thresh_method]

        y,x,h,w = roi
        img_blur = io.gaussian_blur(img, blur_kernel)
        img_cut = img[x:x+w, y:y+h]
        if mode == 'full':
            th = io.threshold_ignore_zero(threshold_func, img_blur)
        elif mode == 'roi':
            th = io.threshold_ignore_zero(threshold_func, img_cut)
        thresh_offset = thresh_offset / 100
        thresh = thresh_offset * th
        img_mask = img_blur >= thresh
        img_mask = img_mask[x:x+w, y:y+h]
        img_mask = self.apply_edge_mask(img_mask)
        return img_mask, img_cut

    def apply_edge_mask(self, mask):
        """Reduce `mask` to just its edge/outline when "Edge Only" is
        checked (isotropic, or one-sided along "Directional"'s angle when
        that's also checked) - see io.erode_mask_edge. No-op otherwise."""
        if self.checkbox_edgeOnly.isChecked():
            direction = (self.spinbox_edgeDirection.value()
                        if self.checkbox_edgeDirectional.isChecked() else None)
            return io.erode_mask_edge(mask, self.spinbox_edgeKernel.value(), direction=direction,
                                       revert=self.checkbox_revertMask.isChecked())
        return mask

    def _on_edge_directional_toggled(self):
        self.spinbox_edgeDirection.setEnabled(self.checkbox_edgeDirectional.isChecked())
        self.update_canvas()

    def on_press(self, event):
        """Mouse-button-press handler for canvas_nav: with Ctrl held, start
        a new ROI rectangle on ax_nav, or a ROI-in-ROI rectangle on
        ax_track, at the click position."""
        if event.inaxes not in (self.ax_nav, self.ax_track) or 'ctrl' not in event.modifiers:
            # Plain click/drag is reserved for the navigation toolbar's
            # Pan/Zoom tool (and the scroll-wheel zoom) so images can be
            # zoomed into; hold "ctrl" to draw/edit a ROI instead.
            self.press = None
            return
        self.press = (event.xdata, event.ydata)
        if event.inaxes == self.ax_nav:
            if self.rect is not None:
                self.rect.remove()
            self.rect = patches.Rectangle(self.press, 0, 0, linewidth=1,
                                          edgecolor='r', facecolor='none')
            self.patches_axNav.append(self.rect)
            self.ax_nav.add_patch(self.rect)
            self.canvas_nav.draw()
            self.backgrounds['nav'] = self.canvas_nav.copy_from_bbox(self.ax_nav.bbox)

        elif (event.inaxes == self.ax_track):
            self.canvas_nav.restore_region(self.backgrounds['track'])
            if self.rect_roiInRoi is not None:
                self.rect_roiInRoi.remove()

            self.rect_roiInRoi = patches.Rectangle(self.press, 0, 0, linewidth=1,
                                                   edgecolor='r', facecolor='none')
            self.patches_axTrack.append(self.rect_roiInRoi)
            self.ax_track.add_patch(self.rect_roiInRoi)
            self.canvas_nav.draw()
            self.backgrounds['track'] = self.canvas_nav.copy_from_bbox(self.ax_track.bbox)
            
        else:
            self.press = None
        

    def on_motion(self, event):
        """Mouse-motion handler for canvas_nav: while a Ctrl+drag started by
        on_press is in progress, resize the in-progress ROI rectangle
        (clamped to the parent ROI's bounds for a ROI-in-ROI) and blit it."""
        if self.press is None or event.inaxes is None:
            return
        # if (event.inaxes == self.ax_track) and (not self.checkbox_roiInRoi.isChecked()):
            # return
        if event.inaxes == self.ax_nav:
            x0, y0 = self.press
            width = event.xdata - x0
            height = event.ydata - y0
            try:
                self.rect.set_width(width)
                self.rect.set_height(height)
                self.rect.set_xy((x0, y0))
            except AttributeError:
                self.press = None
            self.canvas_nav.restore_region(self.backgrounds['nav'])
            self.ax_nav.draw_artist(self.rect)
            self.canvas_nav.blit(self.ax_nav.bbox)
            
        elif (event.inaxes == self.ax_track):
            if event.xdata is None or event.ydata is None:
                return
        
            x0, y0 = self.press
            width = event.xdata - x0
            height = event.ydata - y0
        
            # Confine ROI
            try:
                selected_items = self.tree_objects.selectedItems()
                item = selected_items[0]
                ind = int(item.text(1))
            except (IndexError, ValueError):
                qtw.QMessageBox.critical(self, 'No Ref ROI', 'There is no reference ROI selected for ROI in ROI.')
                self.logger.warning('First select a reference ROI')
                self.press = None
                self.rect_roiInRoi = None
                return
            imgNo = self.slider_imgNo.value()
            xr, yr, wr, hr = self.df_rois.loc[ind, 'out_rois'][imgNo]
        
            # Clamp logic
            x1 = x0 + width
            y1 = y0 + height
            x1 = max(xr, min(x1, xr + wr))
            y1 = max(yr, min(y1, yr + hr))
            width = x1 - x0
            height = y1 - y0
        
            try:
                self.rect_roiInRoi.set_width(width)
                self.rect_roiInRoi.set_height(height)
                self.rect_roiInRoi.set_xy((x0, y0))
            except AttributeError:
                self.press = None
                return
        
            self.canvas_nav.restore_region(self.backgrounds['track'])
            self.ax_track.draw_artist(self.rect_roiInRoi)
            self.canvas_nav.blit(self.ax_track.bbox)


    def on_release(self, event):
        """Mouse-button-release handler for canvas_nav: finalize the
        rectangle started by on_press into a new ROI row (left click) or an
        additional init frame/box on the selected ROI (right click), and
        add/update its tree entry."""
        if self.press is None or event.inaxes is None:
            return
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
        roi = (int(x0), int(y0), int(width), int(height))
        
        # updating df roi
        # new roi or roi in roi
        imgNo = self.slider_imgNo.value()
        roiInRoi = False
        if (event.inaxes == self.ax_nav):
            ref = None
            # new roi or extension
            # self.add_item_tree(idx=idx, init=init, ref=ref)
        elif (event.inaxes == self.ax_track):
            selected_items = self.tree_objects.selectedItems()
            if selected_items:
                item = selected_items[0]
                ref = item.text(1)
            roiInRoi = True
        
        if event.button == 1: # left click
            new_row = True
            init = [imgNo]
            idx = 1
            while idx in self.df_rois.index:
                idx += 1
            
        elif event.button == 3: # right click
            new_row = False
            selected_items = self.tree_objects.selectedItems()
            if selected_items:
                item = selected_items[0]
            else:
                count = self.tree_objects.topLevelItemCount()
                item = self.tree_objects.topLevelItem(count - 1) # last one
            init = ast.literal_eval(item.text(2))
            init.append(imgNo)
            idx = int(item.text(1))
        
# =============================================================================
#         # addition of init to roiInRoi should be made with correct ref
#         if pd.isna(self.df_rois.loc[idx, 'ref']): 
#             self.press = None
#             return
# =============================================================================
        
        # plotting
        self.rect = None
        if roiInRoi:
            # self.patches_axTrack.append(t)
            # self.canvas.restore_region(self.backgrounds['track'])
            t = self.ax_track.text(x0, y0-15, str(idx), horizontalalignment='center',
                                   verticalalignment='center', color='red', fontsize=6)
            self.patches_axTrack.append(t)
            self.backgrounds['track'] = self.canvas_nav.copy_from_bbox(self.ax_track.bbox)
            self.canvas_nav.restore_region(self.backgrounds['track'])
            self.ax_track.draw_artist(t)
            self.canvas_nav.blit(self.ax_track.bbox)
            self.rect_roiInRoi = None
            
# =============================================================================
#         t = self.ax_nav.text(x0 + width/2, y0-15, str(idx), horizontalalignment='center', 
#                              verticalalignment='center', color='red', fontsize=12)
#         self.patches_axNav.append(t)
#         self.backgrounds['nav'] = self.canvas.copy_from_bbox(self.ax_nav.bbox)
#         self.canvas.restore_region(self.backgrounds['nav'])
#         self.ax_nav.draw_artist(t)
#         self.canvas.blit(self.ax_nav.bbox)
# =============================================================================
            
        if new_row:
            self.df_rois.loc[idx] = [1, init, [roi], len(self.nav_imgs),
                                                   ref, None, None, None]
            self.add_item_tree(idx=idx, init=init, end=None, ref=ref)

        else:
            # self.df_rois['init'] = self.df_rois['init'].astype(object)
            self.df_rois.at[idx, 'init'] = init
            self.df_rois.at[idx, 'in_rois'].append(roi)
            # print(self.df_rois.at[idx, 'in_rois'])
            item.setText(2, str(init))
        
        self.press = None
        self.update_canvas(imgNo)

    def on_scroll(self, event):
        """Zoom the axes under the cursor in/out on Ctrl+scroll wheel,
        centered on the cursor position."""
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
        # on_scroll is connected to both canvas_nav and canvas_extract, so
        # the event's own originating canvas (not a hardcoded one) must be
        # redrawn - the axis under the cursor could be on either.
        event.canvas.draw_idle()

    def on_click_dp(self, event):
        """Ctrl+Click on the DP plot sets the reciprocal-space rings' center
        manually - only takes effect when auto-centering is off, so it isn't
        immediately overwritten by the next auto-centered redraw."""
        if (event.inaxes == self.ax_dp and event.button == 1 and event.xdata is not None
                and 'ctrl' in event.modifiers and not self.checkbox_autoCenterDp.isChecked()):
            self.dp_center = (event.xdata, event.ydata)
            self.update_scalebar('reciprocal')
#%%
    def add_item_tree(self, idx, init=[0], end=None, ref=None, use=1):
        """Add one row to tree_objects for ROI `idx`, with its end-frame
        spinbox and Duplicate/Delete buttons wired up."""
        cols = {col: i for i,col in enumerate(self.cols_tree)}
        item = qtw.QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(cols['use'], Qt.Checked if use else Qt.Unchecked)
        item.setText(cols['idx'], f"{idx}")
        item.setText(cols['init'], f"{init}")
        self.tree_objects.itemChanged.connect(self.on_item_check_changed)

        self.tree_objects.addTopLevelItem(item)

        # end frame
        spinbox = qtw.QSpinBox()
        spinbox.setRange(0, len(self.nav_imgs))
        spinbox.setValue(end if end is not None else len(self.nav_imgs))
        self.tree_objects.setItemWidget(item, cols['end'], spinbox)
        spinbox.valueChanged.connect(lambda value: self.on_spinboxEnd_changed(idx, value))
        # spinbox.valueChanged.connect(partial(self.on_spinbox_changed, item, idx))
        
        # ref
        if ref is None:
            item.setText(cols['ref'], "Nav")
        else:
            item.setText(cols['ref'], f"{ref}")
        self.tree_objects.addTopLevelItem(item)
        
        # tracked
        cancel_icon = self.style().standardIcon(self.style().SP_DialogCancelButton)
        item.setIcon(cols['trk'], cancel_icon)
        item.setData(cols['trk'], Qt.UserRole, False)  # Store status boolean (False = not checked)
        
        # extracted
        item.setIcon(cols['ext'], cancel_icon)
        item.setData(cols['ext'], Qt.UserRole, False)  # Store status boolean (False = not checked)
        
        # duplicate
        duplicate_button = qtw.QPushButton('Dup')
        duplicate_button.setFixedSize(30, 30)
        duplicate_button.setToolTip(
            "Duplicate this item: deep-copies everything - use flag, start frame(s)/"
            "box(es), end frame, reference, and (if already done) tracked boxes, "
            "threshold mask, and extracted DPs - into a new, independent item")

        def duplicate_row():
            index = self.tree_objects.indexOfTopLevelItem(item)
            orig_idx = self.df_rois.index[index]
            new_idx = 1
            while new_idx in self.df_rois.index:
                new_idx += 1
            # Deep-copied wholesale (not just use/init/end/ref) so a
            # duplicate that was already tracked/extracted keeps that data -
            # editing the duplicate afterward (e.g. re-tracking) can't
            # silently mutate the original's arrays, since none are shared.
            # Per-column, not deepcopy(Series) - deepcopy() on a whole
            # pandas Series doesn't deep-copy object-dtype cell contents (a
            # well-known pandas gotcha, it only copies the references).
            src = self.df_rois.loc[orig_idx]
            orig = {col: deepcopy(src[col]) for col in self.cols_df}
            self.df_rois.loc[new_idx] = [orig[col] for col in self.cols_df]
            self.add_item_tree(idx=new_idx, init=deepcopy(orig['init']), end=orig['end'],
                               ref=orig['ref'], use=orig['use'])
            row_index = self.tree_objects.topLevelItemCount() - 1
            if orig['out_rois'] is not None:
                self.toggle_tree_icon(row_index, 'trk', True)
            if orig['dp'] is not None:
                self.toggle_tree_icon(row_index, 'ext', True)
            self.update_canvas()
            self.logger.info('Duplicated ROI %s as ROI %s (tracking/extraction data included).',
                             orig_idx, new_idx)

        duplicate_button.clicked.connect(duplicate_row)

        container_dup = qtw.QWidget()
        layout_dup = qtw.QHBoxLayout(container_dup)
        layout_dup.addWidget(duplicate_button)
        layout_dup.setContentsMargins(0, 0, 0, 0)
        layout_dup.setAlignment(Qt.AlignLeft)
        container_dup.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Preferred)
        container_dup.setLayout(layout_dup)
        self.tree_objects.setItemWidget(item, cols['dup'], container_dup)

        # delete
        delete_button = qtw.QPushButton()
        delete_button.setIcon(self.style().standardIcon(qtw.QStyle.SP_TrashIcon))
        delete_button.setFixedSize(30, 30)
        delete_button.setToolTip("Delete this item")

        def delete_row():
            index = self.tree_objects.indexOfTopLevelItem(item)
            # print(index)
            deleted_idx = self.df_rois.index[index]
            reply = qtw.QMessageBox.question(self, 'Delete ROI',
                f'Delete ROI {deleted_idx} and all its tracked masks/points?\n'
                'This cannot be undone.')
            if reply == qtw.QMessageBox.No:
                return
            self.tree_objects.takeTopLevelItem(index)
            self.df_rois = self.df_rois.drop(self.df_rois.index[index])
            # print(self.df_rois)
            self.update_canvas()
            self.logger.info('Deleted ROI %s.', deleted_idx)

        delete_button.clicked.connect(delete_row)

        # Wrap the button in a QWidget to add it to column 2
        container = qtw.QWidget()
        layout = qtw.QHBoxLayout(container)
        layout.addWidget(delete_button)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignLeft)
        container.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Preferred)

        container.setLayout(layout)

        self.tree_objects.setItemWidget(item, cols['del'], container)
    
    def on_item_check_changed(self, item, column):
        use_col = self.cols_tree.index('use')  # or `cols['use']` if accessible
        idx_col = self.cols_tree.index('idx')
        idx = int(item.text(idx_col))
        if item.checkState(use_col) == Qt.Checked:
            self.df_rois.at[idx, 'use'] = 1
        else:
            self.df_rois.at[idx, 'use'] = 0
            
    
    def on_spinboxEnd_changed(self, idx, value):
        self.df_rois.at[idx, 'end'] = value
        
    def toggle_tree_icon(self, row_index: int, col, status):
        item = self.tree_objects.topLevelItem(row_index)
        col = self.cols_tree.index(col)
        if item is None:
            return  # Invalid index
        # current_status = item.data(col, Qt.UserRole)
        icon = self.style().standardIcon(self.style().SP_DialogApplyButton if 
                                         status else self.style().SP_DialogCancelButton)
        item.setIcon(col, icon)
        item.setData(col, Qt.UserRole, status)
    
    def get_checked_items(self):
        """Log (rather than return) the tree indices of all checked ROI
        rows."""
        checked = []
        for i in range(self.tree_objects.topLevelItemCount()):
            item = self.tree_objects.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                checked.append(item.text(1))
        self.logger.info("Checked Items: %s", checked)
    
    def load_spinner(self,):
        """Show a centered, click-through loading spinner overlay on this tab."""
        self.spinner = LoadingSpinner(parent=self)
        self.spinner.setAttribute(Qt.WA_TransparentForMouseEvents)  # Optional: let clicks pass through
        self.spinner.setWindowFlags(Qt.SubWindow)  # Optional: prevent it from behaving like a popup

        x = (self.width() - self.spinner.width()) // 2
        y = (self.height() - self.spinner.height()) // 2
        self.spinner.move(x, y)
    
        self.spinner.raise_()
        self.spinner.start()
    
    def launch_auto_detector(self):
        self.imgNo_autoDet = self.slider_imgNo.value()
        self.object_detector = Object_Detector_Widget(self.nav_imgs[self.imgNo_autoDet])
        self.object_detector.final_objects.connect(self.receive_objects)
        self.object_detector.show()
    
    def receive_objects(self, objects):
        """Signal handler for Object_Detector_Widget.final_objects: add each
        detected bounding box as a new ROI, initialized on the
        auto-detector's frame."""
        try:
            idx_max = self.df_rois.index.to_numpy().max()
        except ValueError:
            idx_max = 0
        for i, obj in enumerate(objects):
            self.df_rois.loc[i+idx_max] = [1, [self.imgNo_autoDet], [obj], len(self.nav_imgs),
                                           'None', None, None, None]
            # self.df_rois.loc[idx] = [1, init, [roi], len(self.nav_imgs),
            #                                        ref, None, None, None]
            self.add_item_tree(idx=i+idx_max, init=[self.imgNo_autoDet], end=None, ref=None)
        # print(self.df_rois)
        self.update_canvas(self.imgNo_autoDet)
        self.canvas_nav.draw()
        self.logger.info('Auto-detector added %d object(s) on frame %d.',
                          len(objects), self.imgNo_autoDet)
            
    def track_rois(self):
        """Handler for the Track! button: launch a background CV2-tracking
        worker (tr.track_roi_cv2) for each enabled ROI, translating
        coordinates into the reference ROI's frame first if this ROI is a
        ROI-in-ROI."""
        if len(self.df_rois) == 0:
            self.logger.warning('Track requested but no ROIs have been drawn.')
            qtw.QMessageBox.warning(self, 'No ROIs',
                'Draw at least one ROI (hold Ctrl and drag on the navigation image) before tracking.')
            return
        df = self.df_rois[self.df_rois.use == 1]
        if len(df) == 0:
            self.logger.warning('Track requested but no ROIs are enabled ("Use" checkbox).')
            qtw.QMessageBox.warning(self, 'No ROIs Enabled',
                'Check the "Use" box for at least one ROI before tracking.')
            return

        self.load_spinner()
        self._cancelling = False
        self.button_cancel.setEnabled(True)
        self.tracking_counter = 0
        self.tracking_finished = False
        tracking_method = self.combo_trackMethod.currentText()

        self.tracking_counter_end = len(df.index)
        self._track_tic = perf_counter()
        self.logger.info('Starting CV2 tracking (%s) for %d ROI(s)...',
                          tracking_method, self.tracking_counter_end)
        for ind in df.index:
            init = np.array(df.loc[ind, 'init'])
            beg = min(init)
            end = df.loc[ind, 'end']
            imgs = self.nav_imgs[beg:end]
            
            rois_in = np.array(df.loc[ind, 'in_rois'])
            # shift frame number to the start
            rois_in -= beg
            init -= beg
            try:
                ref = int(df.loc[ind, 'ref'])
                try:
                    rois_ref = df[df.idx == ref].out_rois.to_numpy()
                except Exception:
                    raise ValueError(f'The reference roi for roi #{ind} is not available')
                imgs = tr.cut_imgs_by_roi(imgs, rois_ref)
                rois_in = tr.translate_roiInRoi(rois_in, rois_ref, fwd=True)
            except Exception:
                self.logger.warning(
                    'ROI-in-ROI translation for ROI %s failed - tracking it with '
                    'untranslated coordinates instead.', ind, exc_info=True)
            worker = WorkerThread_General(tr.track_roi_cv2, ind, imgs, rois_in,
                                          init, tracking_method)
            worker.signals.results.connect(self.get_tracking_results)  # Connect to result signal
            worker.signals.error.connect(self._on_track_roi_failed)
            # worker.signals.finished.connect(self.plot_tracking_result)
            self.threadpool.start(worker)
            
    def _on_track_roi_failed(self, error_msg, index):
        """WorkerThread_General error callback for a single ROI's tracking
        worker (e.g. an on-demand tracker-model download failing for
        'nano'/'dasiamrpn' - see asset_fetch.py). Counts toward completion
        the same as a successful result, so one failed ROI doesn't leave
        the spinner/Track button waiting forever for a result that will
        never arrive."""
        with self._tracking_lock:
            self.tracking_counter += 1
            counter_now = self.tracking_counter
        self.logger.error('Tracking failed for ROI %s:\n%s', index, error_msg)
        if counter_now == self.tracking_counter_end:
            self.spinner.stop()
            self.button_cancel.setDisabled(True)
        qtw.QMessageBox.warning(self, 'Tracking Failed',
            f'Tracking failed for ROI {index} - see log console for details.')

    def get_tracking_results(self, result, index):
        """WorkerThread_General callback for a single ROI's tracking result.
        Once every enabled ROI has reported in, re-derives ROI-in-ROI
        absolute coordinates from their reference ROI and enables the 3DED
        controls."""
        with self._tracking_lock:
            self.tracking_counter += 1
            counter_now = self.tracking_counter
        self.df_rois.at[index, 'out_rois'] = np.zeros((len(self.nav_imgs), 4), dtype=np.int16)
        st = min(self.df_rois.loc[index, 'init'])
        end = self.df_rois.loc[index, 'end']
        self.df_rois.at[index, 'out_rois'][st:end] = result
        self.toggle_tree_icon(self.df_rois.index.get_loc(index), 'trk', True)
        self.update_progress_bar(counter_now, self.tracking_counter_end)
        if counter_now == self.tracking_counter_end:
            self.tracking_finished = True
        
        if self.tracking_finished:
            # re-translate roi in roi coords
            df = self.df_rois[self.df_rois['use']==1]
            df = df.loc[df.ref.dropna().index.to_list()]
            for idx in df.index:
                try:
                    ref = int(self.df_rois.loc[idx].ref)
                    rois_ref = self.df_rois[self.df_rois.index == ref].out_rois.to_numpy()
                    rois_pre = self.df_rois.loc[idx, 'out_rois'].to_numpy()
                    self.df_rois.at[idx, 'out_rois'] = tr.translate_roiInRoi(rois_pre, rois_ref, fwd=False)
                except Exception:
                    self.logger.warning(
                        'Re-translating ROI-in-ROI coordinates for ROI %s failed; '
                        'its out_rois were left untranslated.', idx, exc_info=True)
                            
            # self.slider_imgNo.setValue(0)
            # activating widgets
            self.slider_thresh.setEnabled(True)
            self.slider_thresh.setValue(100)
            self.disable_3ded_widgets(False)
            # self.checkbox_roiInRoi.setEnabled(True)
            item = self.tree_objects.topLevelItem(0)
            item.setSelected(True)
            self.update_canvas(0)
            self.canvas_nav.draw()
            self.canvas_extract.draw()
            self.spinner.stop()
            self.button_cancel.setDisabled(True)

            duration = perf_counter() - self._track_tic
            self.logger.info(
                'CV2 tracking completed successfully for %d ROI(s) in %s.',
                self.tracking_counter_end, io.format_duration_hms(duration))

    def open_fine_tune_mask_dialog(self):
        """Open MaskEditDialog on the selected ROI's mask stack, seeded at
        the frame the slider is currently on; writes the edited stack back
        on Save & Close. If "Extract!" hasn't been run yet for this ROI
        (df_rois['mask'] not populated), builds a starting mask from
        out_rois + the current threshold/blur settings first, the same way
        extract_3ded does - kept raw (un-eroded), same as an already-
        extracted mask, since Edge Detection is only ever a live dialog
        preview now (see MaskEditDialog). A fresh raw re-threshold of
        out_rois (unaffected by anything edited in this dialog, this session
        or any previous one) is also always available as this ROI's "Reset
        to Tracking" target."""
        selected_items = self.tree_objects.selectedItems()
        if not selected_items:
            qtw.QMessageBox.warning(self, 'No ROI Selected',
                'Select a tracked ROI in the list to fine-tune its mask first.')
            return
        idx = int(selected_items[0].text(1))
        out_rois = self.df_rois.at[idx, 'out_rois']
        if np.all(pd.isna(out_rois)):
            qtw.QMessageBox.warning(self, 'Not Tracked Yet',
                'This ROI has not been tracked yet - run "Track!" first.')
            return
        thresh_method = self.combo_thresh_method.currentText()
        thresh_offset = self.slider_thresh.value() / 100
        blur_kernel = int(self.combo_blur.currentText())
        default_mask_stack = tr.create_masks(self.nav_imgs, out_rois, thresh_method,
                                             thresh_offset, blur_kernel)
        mask_stack = self.df_rois.at[idx, 'mask']
        if np.all(pd.isna(mask_stack)):
            mask_stack = default_mask_stack
        edge_settings = {
            'enabled': self.checkbox_edgeOnly.isChecked(),
            'kernel': self.spinbox_edgeKernel.value(),
            'revert': self.checkbox_revertMask.isChecked(),
            'directional': self.checkbox_edgeDirectional.isChecked(),
            'direction': self.spinbox_edgeDirection.value()}
        thresh_settings = {
            'method': thresh_method, 'offset_raw': self.slider_thresh.value(), 'blur': blur_kernel}
        dialog = MaskEditDialog(self, mask_stack, bg_stack=self.nav_imgs,
                                start_frame=self.slider_imgNo.value(), logger=self.logger,
                                default_mask_stack=default_mask_stack, edge_settings=edge_settings,
                                thresh_settings=thresh_settings,
                                recompute_thresh_fn=lambda method, offset, blur:
                                    tr.create_masks(self.nav_imgs, out_rois, method, offset, blur))
        if dialog.exec_() == qtw.QDialog.Accepted:
            self.df_rois.at[idx, 'mask'] = dialog.get_mask_stack()
            self._apply_dialog_settings_to_ui(dialog)
            self.logger.info('Fine-tuned mask saved for ROI %d.', idx)
            self.update_canvas()

    def _apply_dialog_settings_to_ui(self, dialog):
        """Sync MaskEditDialog's Edge Detection and Threshold box values
        back into this tab's own main controls on Save && Close, so
        whatever was left set there (not just the returned mask itself) is
        what "Extract!"/the live preview use next, instead of silently
        reverting to whatever was set before the dialog was opened."""
        edge = dialog.get_edge_settings()
        self.checkbox_edgeOnly.setChecked(edge['enabled'])
        self.spinbox_edgeKernel.setValue(edge['kernel'])
        self.checkbox_revertMask.setChecked(edge['revert'])
        self.checkbox_edgeDirectional.setChecked(edge['directional'])
        self.spinbox_edgeDirection.setValue(edge['direction'])
        thresh = dialog.get_thresh_settings()
        if thresh is not None:
            self.combo_thresh_method.setCurrentText(thresh['method'])
            self.combo_blur.setCurrentText(str(thresh['blur']))
            self.slider_thresh.setValue(thresh['offset_raw'])

    def resolve_4d_files(self, path_4d):
        """Plain (non-smart-scan) 4D signal file list for `path_4d`.

        Reuses the exact, ordered file list recorded in the navigator tab's
        metadata.json (self._nav_4d_files/_nav_4d_directory - see
        apply_nav_signal_metadata) whenever it was recorded for this same
        folder. Otherwise falls back to globbing the folder filtered to the
        "Data Type" combo (self.combo_dtype_4d) - fixes the old bare
        glob(path_4d, '*'), which picked up any stray non-signal file
        (comment.txt, pattern .txt files, logs, ...) alongside the real 4D
        signals and produced a false frame-count mismatch (or, for the
        single-frame preview, silently extracted the wrong file)."""
        if (self._nav_4d_files is not None and self._nav_4d_directory is not None
                and os.path.normcase(os.path.normpath(self._nav_4d_directory))
                    == os.path.normcase(os.path.normpath(path_4d))):
            return [os.path.join(path_4d, fn) for fn in self._nav_4d_files]
        ext = self.combo_dtype_4d.currentText()
        pattern = '*' if ext == 'All Files' else '*' + ext
        return sorted(glob(os.path.join(path_4d, pattern)))

    def extract_3ded(self):
        """Handler for the Extract! button: build each enabled ROI's
        threshold mask stack, resolve the (smart-scan-aware) list of 4D
        signal files, then queue and launch one extract_frame QProcess per
        (ROI, frame) pair to build the 3DED diffraction-pattern stacks."""
        self.load_spinner()
        
        path_4d = self.lineEdit_dir_4d.text()
        if path_4d == '': # no entry in 4D signals path
            self.spinner.stop()
            qtw.QMessageBox.critical(self, 'No Entry',
                'Enter the path to the folder containing 4D signal files '
                '(.hdf5, .tpx3, .zspy, etc.) before extraction.')
            return

        fns_pattern_4d = None  # parallel per-file pattern-file list, smart-scan only
        if self.checkbox_smartScan.isChecked():
            if self._smart_scan_rows is None:
                self.open_smart_scan_check_dialog()
            if self._smart_scan_rows is None:  # still None: user cancelled the dialog
                self.spinner.stop()
                return
            resolved = io.resolve_smart_scan_files(self._smart_scan_rows, role='acquisition')
            if not resolved:
                self.spinner.stop()
                qtw.QMessageBox.warning(self, 'No Files',
                    'No included tilt angle has an acquisition file - check "Check Files..." above.')
                return
            fns_4d = [item['file'] for item in resolved]
            fns_pattern_4d = [item['pattern_file'] for item in resolved]
        else:
            fns_4d = self.resolve_4d_files(path_4d)
        if len(fns_4d) == 0:
            self.spinner.stop()
            qtw.QMessageBox.critical(self, 'Wrong Path',
                f'No files found in:\n{path_4d}\n\nVerify the path and try again.')
            return
        if len(self.nav_imgs) != len(fns_4d):
            self.logger.warning(
                'No. of 4D signal files (%d) does not match no. of navigation '
                'images (%d).', len(fns_4d), len(self.nav_imgs))
            reply = qtw.QMessageBox.question(self, 'Mismatch',
                   'No. of 4D signals mismatches the number of images. Do you want to continue?',)
            if reply == qtw.QMessageBox.No:
                self.spinner.stop()
                self.logger.info('3DED extraction cancelled by user (frame-count mismatch).')
                return

        dtype = os.path.splitext(fns_4d[0])[-1]
        blur_kernel = int(self.combo_blur.currentText())
        thresh_method = self.combo_thresh_method.currentText()
        thresh_offset = self.slider_thresh.value() / 100
        edge_only = self.checkbox_edgeOnly.isChecked()
        for ind in self.df_rois[self.df_rois.use == 1].index:
            masks = tr.create_masks(
                self.nav_imgs, self.df_rois.loc[ind, 'out_rois'],
                thresh_method, thresh_offset, blur_kernel)
            if edge_only:
                # Applied per-frame - erode_mask_edge is a single-2D-mask
                # transform, and this stack is (N frames, H, W).
                masks = np.stack([self.apply_edge_mask(m) for m in masks])
            self.df_rois.at[ind, 'mask'] = masks

        shape_d_x, shape_d_y = self.get_detector_shape(fns_4d[0])
        scanSize = self.get_scan_size()
        if scanSize is None:  # "Auto": fall back to the loaded nav signal's own shape
            scanSize = tuple(self.nav_imgs.shape[1:])
        
        df = self.df_rois[self.df_rois['use'] == 1]

        self.tic = perf_counter()
        self._3ded_failed = False
        self._cancelling = False
        self.button_cancel.setEnabled(True)

        self.tomo_counter = 0
        self.tasks = deque()
        self.temp_dir = self.get_temp_dir()
        lengths = df.end - [min(df.init[idx]) for idx in df.index]
        self.tomo_counter_total = np.sum(lengths)
        self.update_progress_bar(0, self.tomo_counter_total)
        self.logger.info('Starting 3DED extraction for %d ROI(s), %d frame(s) total...',
                          len(df), self.tomo_counter_total)
        for idx in df.index:
            self.df_rois.at[idx, 'dp'] = np.zeros((len(self.nav_imgs), shape_d_x, 
                                                   shape_d_y), dtype='uint32')
            # beg = min(df.loc[idx].init)
            # end = df.loc[idx].end
            out_rois = self.df_rois.loc[idx, 'out_rois']
            for i_fr, fn in enumerate(fns_4d):
                if out_rois[i_fr].any():
                    fn_pattern = fns_pattern_4d[i_fr] if fns_pattern_4d is not None else None
                    self.tasks.append([fn, df.loc[idx, 'out_rois'][i_fr],
                                       os.path.join(self.temp_dir, f"mask_r{idx}_f{i_fr}.npy"),
                                       dtype, scanSize, (idx, i_fr), fn_pattern or '',
                                       f'({shape_d_x},{shape_d_y})'])
        
        
        # path_debug = r'C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\EDyssey\other_scripts\debug'
        # with open(os.path.join(path_debug, 'args.txt'), 'r') as f:
        #     f.writelines(self.tasks)
        
        
        self.max_processes = self.spinbox_threadNo.value()
        self.running_processes = []
        self.process_task_map = {}
        self.process_output_buffers = {}
        # self.launch_initial_tasks()
        for _ in range(min(self.max_processes, len(self.tasks))):
            self.launch_next_task()

    def get_temp_dir(self):
        """Return the shared temp directory used for per-frame extraction
        masks, creating it if it doesn't exist yet."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(os.path.dirname(script_dir), 'EDyssey', 'io_utils', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        self.logger.info('temp directory: %s', temp_dir)
        return temp_dir

    def extract_dp_current_frame(self):
        """Compute the diffraction pattern for just the selected ROI at the
        frame the slider currently points to - the single-frame equivalent
        of "Extract!" (extract_3ded), for quick data-checking. Runs inline
        (WorkerThread_General on self.threadpool) rather than as a QProcess,
        since it's a one-off single frame, not a whole series."""
        selected_items = self.tree_objects.selectedItems()
        if not selected_items:
            qtw.QMessageBox.warning(self, 'No ROI Selected', 'Select a tracked ROI first.')
            return
        idx = int(selected_items[0].text(1))
        i_fr = self.slider_imgNo.value()

        if np.all(pd.isna(self.df_rois.loc[idx, 'out_rois'])):
            qtw.QMessageBox.warning(self, 'Not Tracked', 'This ROI has not been tracked yet.')
            return
        out_rois = self.df_rois.loc[idx, 'out_rois']
        roi = out_rois[i_fr]
        if not roi.any():
            qtw.QMessageBox.warning(self, 'No ROI', 'No tracked ROI at the current frame.')
            return

        path_4d = self.lineEdit_dir_4d.text()
        if path_4d == '':
            qtw.QMessageBox.critical(self, 'No Entry',
                'Enter the path to the folder containing 4D signal files before extraction.')
            return

        fn_pattern = None
        if self.checkbox_smartScan.isChecked():
            if self._smart_scan_rows is None:
                self.open_smart_scan_check_dialog()
            if self._smart_scan_rows is None:
                return
            resolved = io.resolve_smart_scan_files(self._smart_scan_rows, role='acquisition')
            if i_fr >= len(resolved):
                qtw.QMessageBox.warning(self, 'Frame Out of Range',
                    'The current frame has no matching acquisition file in the smart-scan match.')
                return
            fn = resolved[i_fr]['file']
            fn_pattern = resolved[i_fr]['pattern_file']
        else:
            fns_4d = self.resolve_4d_files(path_4d)
            if i_fr >= len(fns_4d):
                qtw.QMessageBox.warning(self, 'Frame Out of Range',
                    'The current frame has no matching 4D signal file in the folder.')
                return
            fn = fns_4d[i_fr]
        dtype = os.path.splitext(fn)[-1]

        scanSize = self.get_scan_size()
        if scanSize is None:  # "Auto": fall back to the loaded nav signal's own shape
            scanSize = tuple(self.nav_imgs.shape[1:])

        thresh_method = self.combo_thresh_method.currentText()
        thresh_offset = self.slider_thresh.value() / 100
        blur_kernel = int(self.combo_blur.currentText())
        mask = tr.create_masks(self.nav_imgs[i_fr:i_fr + 1], out_rois[i_fr:i_fr + 1],
                               thresh_method, thresh_offset, blur_kernel)[0]
        mask = self.apply_edge_mask(mask)

        self.logger.info('Extracting DP for ROI %d, frame %d (current-frame check)...', idx, i_fr)
        self.button_extractCurrentFrame.setDisabled(True)
        worker = WorkerThread_General(load_dp, 0, fn, roi=roi, mask=mask, dtype=dtype,
                                      scanSize=scanSize, fn_pattern=fn_pattern,
                                      det_shape=self.get_detector_shape(fn))
        worker.signals.results.connect(
            lambda dp, _idx, idx=idx, i_fr=i_fr: self._on_current_frame_dp(dp, idx, i_fr))
        worker.signals.error.connect(self._on_current_frame_dp_failed)
        self.threadpool.start(worker)

    def _on_current_frame_dp(self, dp, idx, i_fr):
        """Callback for the "Extract DP (Current Frame)" worker: store the
        computed DP as a one-off preview and refresh the display."""
        self.button_extractCurrentFrame.setEnabled(True)
        if hasattr(dp, 'compute'):
            dp = dp.compute()
        # Routed through update_canvas() (rather than drawn directly here)
        # so it's shown/cleared by the exact same blit path as every other
        # frame - moving the slider (or changing the ROI selection) away
        # from (idx, i_fr) then correctly reverts to whatever update_canvas
        # would normally show, instead of this one-off result staying
        # plotted indefinitely.
        self._current_frame_dp_preview = {'idx': idx, 'i_fr': i_fr, 'dp': dp}
        self.update_canvas()
        # This is freshly-computed data the auto-centering circles have
        # never seen - re-run it now if enabled, same as after a full
        # "Extract!" (see handle_finished's identical pair of calls).
        self.update_scalebar('reciprocal')
        self.logger.info('Current-frame DP extraction complete (ROI %d, frame %d).', idx, i_fr)

    def _on_current_frame_dp_failed(self, traceback_text, _idx):
        self.button_extractCurrentFrame.setEnabled(True)
        self.logger.error('Current-frame DP extraction failed:\n%s', traceback_text)
        qtw.QMessageBox.warning(self, 'Extraction Failed',
            f'Could not extract the diffraction pattern:\n{traceback_text[-500:]}')

# =============================================================================
#     def launch_initial_tasks(self):
#         for _ in range(min(self.max_processes, len(self.tasks))):
#             self.launch_next_task()
# =============================================================================

    def launch_next_task(self):
        """Pop the next queued extract_frame task (if any, and if under
        max_processes) and launch it as a QProcess, saving its ROI mask to
        disk first so it can be passed to the subprocess by path."""
        if not self.tasks or len(self.running_processes) >= self.max_processes:
            return
    
        args = self.tasks.popleft()
        mask_path = args[2]
        # args = [fn, roi, mask_path, dtype, scanSize, (idx, i_fr), fn_pattern,
        # det_shape] - (idx, i_fr) is third-to-last (fn_pattern/det_shape,
        # always present since extract_3ded() appends them unconditionally,
        # must stay last two to land in extract_3ded_mask_single_frame's
        # matching positional slots).
        idx, i_fr = args[-3]
        np.save(mask_path, self.df_rois.loc[idx, 'mask'][i_fr])
        program, arguments = worker_command('extract_frame', args)
        process = QProcess()
        process.setProgram(program)
        process.setArguments(arguments)
        process.readyReadStandardOutput.connect(lambda: self._accumulate_output(process))
        process.readyReadStandardError.connect(lambda: self.handle_error(process))
        process.finished.connect(lambda: self.handle_finished(process))
        process.errorOccurred.connect(self.process_failed)


        self.running_processes.append(process)
        self.process_task_map[process] = args
        self.process_output_buffers[process] = bytearray()
        process.start()
        
    def process_failed(self, error):
        """Handler for QProcess.errorOccurred: abort extraction and report
        the failure to the user."""
        if self._cancelling:
            return
        self._3ded_failed = True
        self.logger.error("QProcess error occurred: %s", error)
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.spinner.stop()
        qtw.QMessageBox.critical(self, 'Process Error',
            f'A worker process failed to start (error code {error}).\n'
            'Check that Python is on PATH and worker_extract_frame.py exists.')

    def handle_error(self, process):
        """Handler for QProcess.readyReadStandardError."""
        # worker_extract_frame.py loads tpx3 via eventem, whose progress bar
        # (and any other routine diagnostics) writes straight to stderr on
        # every run, success or failure - this is not itself an error (see
        # ProcessStderrBuffer). A worker that genuinely fails to produce
        # output is instead caught in handle_output/handle_finished below.
        if self._cancelling:
            return
        self._stderr_buffer.log_info(process, self.logger, 'Worker')

    def _accumulate_output(self, process):
        """QProcess.readyReadStandardOutput handler: append this chunk to
        the process's buffer instead of decoding it directly - the
        base64+pickle payload for anything but the smallest diffraction
        pattern routinely spans more than one OS pipe delivery, so decoding
        straight out of a single readyReadStandardOutput signal (as this
        used to) intermittently truncated it and failed to unpickle (e.g.
        "invalid load key"). Actual decoding happens once, in
        handle_finished, from the complete accumulated buffer."""
        self.process_output_buffers[process] += process.readAllStandardOutput().data()

    def handle_finished(self, process):
        """Handler for QProcess.finished: decode the complete accumulated
        stdout (a base64-pickled (frame, mask coords) result - see
        _accumulate_output), store it into df_rois' dp array at the right
        (idx, frame) slot, update extraction progress, and once every
        queued (ROI, frame) task has finished, clean up the temp mask
        directory, mark extracted ROIs in the tree, and autosave if enabled
        - otherwise launch the next queued task."""
        if process in self.running_processes:
            self.running_processes.remove(process)
        task_info = self.process_task_map.pop(process, None)
        if self._cancelling:
            self.process_output_buffers.pop(process, None)
            process.deleteLater()
            return
        # drain any remaining bytes not yet signalled
        self.process_output_buffers[process] += process.readAllStandardOutput().data()
        raw_output = bytes(self.process_output_buffers.pop(process, b'')).decode(
            'utf-8', errors='replace').strip()
        process.deleteLater()

        try:
            result_array = pickle.loads(base64.b64decode(raw_output))
        except Exception:
            self.logger.exception('Failed to decode tracking extraction subprocess output.')
        else:
            if task_info is None:
                self.logger.warning("Unknown process")
            else:
                img, i_c = result_array
                idx, i_fr = ast.literal_eval(i_c)
                self.df_rois.loc[idx, 'dp'][i_fr] = img

        self.tomo_counter += 1
        self.update_progress_bar(self.tomo_counter, self.tomo_counter_total)
        
        if self.tomo_counter >= self.tomo_counter_total:
            self.toc = perf_counter()
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            duration = self.toc - self.tic
            if self._3ded_failed:
                self.logger.error(
                    '3DED extraction finished with errors after %s '
                    '(see log above for details).', io.format_duration_hms(duration))
            else:
                self.logger.info(
                    '3DED extraction completed successfully (%d frame(s)) in %s.',
                    self.tomo_counter_total, io.format_duration_hms(duration))
            self.update_canvas()
            # Freshly-extracted DPs may have a different center than whatever
            # was last found - re-run auto-centering now if enabled.
            self.update_scalebar('reciprocal')
            self.spinner.stop()
            self.button_cancel.setDisabled(True)
            for idx in self.df_rois[self.df_rois.use == 1].index:
                self.toggle_tree_icon(self.df_rois.index.get_loc(idx), 'ext', True)

            if self.checkbox_autosave.isChecked():
                self.save_results()
        else:
            self.launch_next_task()  # trigger next task if any left
        
    def update_progress_bar(self, value, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f'%v / {total}')

    def reset_thresh(self):
        self.slider_thresh.setValue(100)
        self.update_canvas()

    def on_makePets2_toggled(self, state):
        """Checkbox handler for 'Make *.pts2': when checked, open the PETS2
        params dialog (deferred - see comment below for why)."""
        if state != Qt.Checked:
            return
        # Opening a modal dialog synchronously from within the checkbox's own
        # stateChanged handler (as this used to do) leaves QCheckBox's
        # internal click/press state confused - after Cancel calls
        # setChecked(False) here, the very next check click wouldn't reopen
        # the dialog until the box had been toggled a few more times.
        # Deferring to the next event-loop iteration lets Qt finish handling
        # the click first, so the dialog opens cleanly every time.
        QTimer.singleShot(0, self._open_pets2_dialog)

    def _open_pets2_dialog(self):
        """Seed Pets2ParamsDialog from current metadata/settings (voltage,
        exposure, Å^-1/px scale, dp center) and open it; unchecks 'Make
        *.pts2' again if the user cancels."""
        voltage_kv = None
        try:
            path_main = self.metadata_path_override or self.lineEdit_dir_4d.text()
            metadata = io.get_metadata(path_main, count=self.spinbox_metadataCount.value())
            if 'Voltage' in metadata:
                voltage_kv = metadata['Voltage']
        except Exception:
            self.logger.debug('Could not auto-fill voltage from metadata; leaving it blank.',
                               exc_info=True)
        exposure_s = self.spinbox_dwellTime.value() / 1e6
        try:
            aperpixel = float(self.lineEdit_scale_recip.text())
        except ValueError:
            aperpixel = None
        dialog = Pets2ParamsDialog(self, voltage_kv=voltage_kv, exposure_s=exposure_s,
                                    aperpixel=aperpixel, center=self.dp_center)
        if dialog.exec_() == qtw.QDialog.Accepted:
            self.pets2_params = dialog.get_params()
        else:
            self.checkbox_makePets2.setChecked(False)

    def save_results(self):
        """Handler for the Save Results button: run _save_results_impl(),
        logging success/failure with timing."""
        tic = perf_counter()
        try:
            self._save_results_impl()
        except Exception as exc:
            self.logger.exception('Failed to save results after %s.',
                                   io.format_duration_hms(perf_counter() - tic))
            qtw.QMessageBox.critical(self, 'Save Failed',
                f'Failed to save results:\n{exc}\n\nSee the log for details.')
            return
        self.logger.info(
            'Results saved successfully in %s (background clip/frame '
            'generation for each ROI continues asynchronously).',
            io.format_duration_hms(perf_counter() - tic))

    def _save_results_impl(self):
        """Write the current tracking/extraction state (per-ROI json,
        out_rois/mask arrays, extracted DPs, PETS2 project files) to a new
        timestamped folder under the save directory. Per-ROI clip/frame
        generation is kicked off as background workers rather than done
        inline here."""
        path_save = self.lineEdit_dir_save.text()
        if not os.path.isdir(path_save):
            os.mkdir(path_save)

        fld_1 = datetime.date.today()
        fld_2 = datetime.datetime.now().strftime("%H-%M-%S")

        path_save = os.path.join(path_save, f'{fld_1}__{fld_2}')
        os.mkdir(path_save)
        self.logger.info('Saving results to %s...', path_save)

        # Navigation signal: rather than re-copying the (potentially large)
        # signal into every saved-analysis folder, just record the path it
        # was loaded from - "Load Saved Analysis" reloads from there.
        io.save_analysis_info(path_save, self.lineEdit_dir_navSignal.text(), analysis_type='cv2')

        # tracking results, rois, dp
        for idx in self.df_rois.index:
            path_save_roi = os.path.join(path_save, f'roi No {idx}')
            os.mkdir(path_save_roi)
            df = self.df_rois.loc[idx, ['use', 'init', 'in_rois', 'end', 'ref']]
            df['thresh'] = [('blur kernel', self.combo_blur.currentText()),
                            ('thresh method', self.combo_thresh_method.currentText()),
                            ('thresh offset', self.slider_thresh.value())]
            df['edge_detection'] = [('enabled', self.checkbox_edgeOnly.isChecked()),
                                    ('kernel_size', self.spinbox_edgeKernel.value()),
                                    ('directional', self.checkbox_edgeDirectional.isChecked()),
                                    ('direction_deg', self.spinbox_edgeDirection.value()),
                                    ('revert', self.checkbox_revertMask.isChecked())]
            df.to_json(os.path.join(path_save_roi, f'roi No {idx}.json'), orient='index', indent=4)
            np.save(os.path.join(path_save_roi, 'output_rois.npy'), self.df_rois.loc[idx, 'out_rois'])
            np.save(os.path.join(path_save_roi, 'output_mask.npy'), self.df_rois.loc[idx, 'mask'])

            # write frames (only if extraction was run for this roi)
            dp = self.df_rois.loc[idx, 'dp'] if 'dp' in self.df_rois.columns else None
            if isinstance(dp, np.ndarray):
                np.save(os.path.join(path_save_roi, '3DED.npy'), dp)
                # Also save as a hyperspy signal so "Load Saved Analysis" can
                # restore the diffraction patterns via hs.load(...).
                hs.signals.Signal2D(dp).save(
                    os.path.join(path_save_roi, '3DED.hspy'), overwrite=True)
                path_pets = os.path.join(path_save_roi, 'pets')
                os.mkdir(path_pets)
                fld_frames = os.path.join(path_pets, 'frames')
                worker_frames = WorkerThread_General(io.create_frames, 0, fld_frames, dp)
                self.threadpool.start(worker_frames)
                scale_recip = self.lineEdit_scale_recip.text()
                try:
                    scale_recip = float(scale_recip)
                except ValueError:
                    scale_recip = None

                if self.checkbox_makePets2.isChecked() and self.pets2_params is not None:
                    io.write_pts2(os.path.join(path_pets, f'Roi Num {idx}.pts2'), n_frames=dp.shape[0],
                                  frame_shape=dp.shape[1:], roi_id=idx, **self.pets2_params)

                fn_dp = os.path.join(path_save_roi, 'tomo clip')
                worker_clip_dp = WorkerThread_General(io.create_clip_dp, 0, fn_dp, dp,
                                                      scale_recip, center=self.dp_center,
                                                      fps=self.spinbox_fps.value(),
                                                      logger=self.logger)
                self.threadpool.start(worker_clip_dp)

            # clip for tracking
            scale_real = self.lineEdit_scale_real.text()
            try:
                scale_real = float(scale_real)
            except ValueError:
                scale_real = None

            fn = os.path.join(path_save_roi, 'tracking clip')
            worker_clip_tr_ref = WorkerThread_General(
                io.create_clip_tracking, 0, fn, self.nav_imgs,
                self.df_rois.loc[idx, 'out_rois'], scale_real,
                fps=self.spinbox_fps.value(), logger=self.logger)
            self.threadpool.start(worker_clip_tr_ref)
            
    def kill_running_process(self):
        """Forcefully kill and clear any still-running 3DED extraction
        subprocesses."""
        # self.running_processes is only created once extract_3ded() has
        # run at least once, so this must tolerate being called (e.g. from
        # cleanup() on window close) before that ever happened.
        if not hasattr(self, 'running_processes'):
            return
        for process in self.running_processes:
            process.kill()  # Forcefully terminates the subprocess
            process.deleteLater()
        self.running_processes.clear()

    def cancel_running_work(self):
        """Stop tracking or 3DED extraction and suppress the error popups
        that killing those workers would otherwise trigger.

        QThreadPool has no way to forcibly interrupt a runnable that has
        already started (only queued-but-not-started ones can be dropped),
        so an in-flight tracking job for a single ROI will still finish in
        the background - its result is applied harmlessly since it's still
        valid data, just not launched further. The 3DED extraction
        subprocesses are real OS processes though, so those are actually
        killed outright."""
        self._cancelling = True
        self.threadpool.clear()
        n_killed = len(getattr(self, 'running_processes', []))
        self.kill_running_process()
        if hasattr(self, 'tasks'):
            self.tasks.clear()
        if hasattr(self, 'process_task_map'):
            self.process_task_map.clear()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if hasattr(self, 'spinner'):
            self.spinner.stop()
        self.button_cancel.setDisabled(True)
        self.logger.warning(
            'Cancelled by user (%d running 3DED worker process(es) killed).', n_killed)
        qtw.QMessageBox.information(self, 'Cancelled',
            'Tracking/3DED extraction was cancelled.\n\n'
            'Any tracking job already running for a single ROI will still '
            'finish in the background (its result is kept) - only queued '
            'work and the 3DED extraction processes were stopped.')

    def cleanup(self):
        """Release resources held by this tab. Called by MainWindow.closeEvent
        so repeated runs of the app in the same console/kernel don't leave
        threadpools, running subprocesses, and matplotlib figures alive."""
        self.threadpool.clear()
        self.kill_running_process()
        self.log_console.disconnect_log()
        plt.close(self.figure_nav)
        plt.close(self.figure_extract)

# =============================================================================
# if __name__ == "__main__":
#     app = qtw.QApplication(sys.argv)
#     
#     # Create and show the main window
#     window = Tab_Tracking_CV2()
#     window.show()
#     
#     # Start the event loop
#     sys.exit(app.exec_())
# =============================================================================

