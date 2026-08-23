# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:43:00 2024

@author: SGholam
"""

import ast
import json
import sys
import hyperspy.api as hs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QProcess, QTimer
import pickle
import base64
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QKeySequence
from PyQt5.QtWidgets import QShortcut
import numpy as np
import os
import re
from PIL import Image
import gc
from copy import deepcopy
import datetime
from time import perf_counter
import EDyssey.io_utils as io
from EDyssey.tracking_utils import asset_fetch
from typing import Literal
from .worker_thread import WorkerThread_General, ProcessStderrBuffer
from .worker_launch import worker_command
from .contrast_scaling import ContrastScalingBox
from .logging_utils import LogConsole
from .base_tab import TabBase, get_existing_directory
from .clipping_thresholds import ClippingThresholdsWidget
from .pets2_dialog import Pets2ParamsDialog
from .smart_scan_dialog import SmartScanCheckDialog
from .mask_edit_dialog import MaskEditDialog
from .ribbon import RibbonPanel, RibbonTool
from worker_extract_frame import load_dp
from glob import glob
from matplotlib.colors import SymLogNorm
# import EDyssey.tracking_utils as tr
import shutil
import threading
from collections import deque
from hyperspy.api import signals as hsSignals
import pandas as pd
from .loading_label import LoadingSpinner
_ffmpeg = shutil.which('ffmpeg')
if _ffmpeg:
    plt.rcParams['animation.ffmpeg_path'] = _ffmpeg
#%% tab class
class Tab_SAM2(TabBase):
    def __init__(self, parent=None):
        super().__init__('Tab_SAM2', parent)

        # Recomputing constrained_layout's spacing solve on every redraw
        # (canvas.draw()'s default behavior) is one of the most expensive
        # parts of a redraw; update_canvas() freezes it after the first
        # real draw, once subplot spacing has settled.
        self._layout_frozen = False

        self._stderr_buffer = ProcessStderrBuffer()
        logical_processors = os.cpu_count()

        if logical_processors > 2:
            self.threadpool.setMaxThreadCount(logical_processors - 2)
        # self.threadpool.setMaxThreadCount(3)

        self.init_ui()
        # self.device = self.check_torch_device()
        
    def init_ui(self):
        spacer = qtw.QSpacerItem(40, 20, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
        # Set the window title and dimensions
        self.setWindowTitle("SAM2 Segmentation")
        
        self.central_widget = qtw.QWidget(self)
        self.layout = qtw.QVBoxLayout(self)

        button_w = 95
        button_h_lrg = 50

        #%% ribbon (top parameter ribbon, Word-style - see Tab_ROI_on_4D for
        # the original design, and TabBase for the shared helpers). Same
        # column shape as Tab_Tracking_CV2 (this tab's structure is nearly
        # identical), minus the Threshold/Deviation rows SAM2 doesn't need.
        ribbon_page = qtw.QWidget()
        self.ribbon_page = ribbon_page  # exposed for the Edit tab's ribbon-text-scale control
        layout_ribbon = qtw.QHBoxLayout(ribbon_page)
        layout_ribbon.setContentsMargins(4, 2, 4, 2)
        layout_ribbon.setSpacing(2)
        self.layout.addWidget(ribbon_page)

        #%% Files (ribbon column)
        self.box_dir, layout_dir = self._ribbon_group_start(layout_ribbon, stretch=1)

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
        layout_dir.addLayout(layout_loadSignal)
        self.double_validator = QDoubleValidator(0.0, 1e5, 5)

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

        # Smart-scan support - a titled box (matches the Navigator tab's
        # identical "Smart Scan" groupbox convention), merged into Files
        # rather than its own column. 3DED extraction always reads the
        # acquisition (smart-scanned) file - see EDyssey/io_utils/smart_scan.py -
        # so there's no role combo here (unlike Navigator, which can batch
        # over either role). 2x2: left column = activation controls
        # (checkbox, then Check Files button), right column = directory
        # pickers (pattern dir, then detection dir).
        groupbox_smartScan = qtw.QGroupBox('Smart Scan')
        layout_smartScan = qtw.QGridLayout(groupbox_smartScan)

        self.checkbox_smartScan = qtw.QCheckBox('Activate')
        self.checkbox_smartScan.setToolTip(
            'The 4D signals folder holds a smart-scanned tomography series - 3DED '
            'extraction will read each frame\'s acquisition file with its matching '
            'pattern file, instead of every raw file in the folder.')
        layout_smartScan.addWidget(self.checkbox_smartScan, 0, 0)
        self.checkbox_smartScan.stateChanged.connect(self.activate_smartScan_widgets)

        self.lineEdit_patternDir = qtw.QLineEdit()
        self.lineEdit_patternDir.setPlaceholderText('Pattern Dir. (defaults to 4D Signals folder)')
        self.lineEdit_patternDir.setDisabled(True)
        layout_smartScan.addWidget(self.lineEdit_patternDir, 0, 1)
        self.button_browsePatternDir = qtw.QPushButton('...')
        self.button_browsePatternDir.setFixedWidth(30)
        self.button_browsePatternDir.setDisabled(True)
        self.button_browsePatternDir.clicked.connect(self.browse_pattern_dir)
        layout_smartScan.addWidget(self.button_browsePatternDir, 0, 2)

        self.button_checkSmartScanFiles = qtw.QPushButton('Check Files...')
        self.button_checkSmartScanFiles.setToolTip(
            'Review the automatic per-frame detection/acquisition/pattern-file match '
            'before extracting - fix or exclude any mismatched frame by hand')
        self.button_checkSmartScanFiles.setDisabled(True)
        self.button_checkSmartScanFiles.clicked.connect(self.open_smart_scan_check_dialog)
        layout_smartScan.addWidget(self.button_checkSmartScanFiles, 1, 0)

        self.lineEdit_detectionDir = qtw.QLineEdit()
        self.lineEdit_detectionDir.setPlaceholderText('Detect. Dir. (defaults to 4D Signals folder)')
        self.lineEdit_detectionDir.setDisabled(True)
        self.lineEdit_detectionDir.setToolTip(
            'Folder to look for detection files in, if they live somewhere other than the '
            '4D Signals folder (e.g. a separate folder of HAADF .tif/.tiff reference images '
            'for .mib/.hspy/.zspy, or a cleaner acquisition layout with detection/acquisition '
            'each in their own folder)')
        layout_smartScan.addWidget(self.lineEdit_detectionDir, 1, 1)
        self.button_browseDetectionDir = qtw.QPushButton('...')
        self.button_browseDetectionDir.setFixedWidth(30)
        self.button_browseDetectionDir.setDisabled(True)
        self.button_browseDetectionDir.clicked.connect(self.browse_detection_dir)
        layout_smartScan.addWidget(self.button_browseDetectionDir, 1, 2)

        # Hidden until it actually has something to say (see
        # _set_smart_scan_summary) - an empty QLabel still reserves a full
        # text-line's height in the grid, which otherwise left a persistent
        # blank line at the bottom of this box before "Check Files..." was
        # ever run.
        self.label_smartScanSummary = qtw.QLabel('')
        self.label_smartScanSummary.setVisible(False)
        layout_smartScan.addWidget(self.label_smartScanSummary, 2, 0, 1, 3)

        layout_dir.addWidget(groupbox_smartScan)
        self._smart_scan_rows = None  # set by open_smart_scan_check_dialog() or apply_nav_signal_metadata()
        self._ribbon_group_end(layout_ribbon, layout_dir, 'Files')

        #%% Input Parameters (ribbon column) - scan dims, scale bars,
        # detector size, dwell time, metadata block - everything needed to
        # extract DPs for 3DED. Detector Size/Scan Size share one
        # QGridLayout (mirrors Tab_ROI_on_4D/Navigator's Input Parameters
        # column, including QSpinBox for Scan Size too - was a QLineEdit)
        # so their labels/X/Y cells line up row-to-row.
        self.box_scanSize, layout_box_scanSize = self._ribbon_group_start(layout_ribbon, stretch=1)

        layout_exp_items = qtw.QGridLayout()
        layout_exp_items.addWidget(qtw.QLabel('Detector Size'), 0, 0, 1, 2)
        detSize_tooltip = (
            'Detector (diffraction pattern) size in pixels - used to size the extracted '
            'DP array. Auto-detected from the 4D signal files for most formats; .tpx3 '
            'needs this set explicitly (auto-detecting it would mean fully parsing a '
            'file just to learn its shape) - Auto assumes 512x512.')
        self.checkbox_detectorSizeAuto = qtw.QCheckBox('Auto')
        self.checkbox_detectorSizeAuto.setChecked(True)
        self.checkbox_detectorSizeAuto.setToolTip(detSize_tooltip)
        layout_exp_items.addWidget(self.checkbox_detectorSizeAuto, 0, 2)
        layout_exp_items.addWidget(qtw.QLabel('X', alignment=Qt.AlignCenter), 0, 3)
        self.spinbox_detectorSize_x = qtw.QSpinBox()
        self.spinbox_detectorSize_x.setRange(1, 8192)
        self.spinbox_detectorSize_x.setValue(512)
        layout_exp_items.addWidget(self.spinbox_detectorSize_x, 0, 4, 1, 2)
        layout_exp_items.addWidget(qtw.QLabel('Y', alignment=Qt.AlignCenter), 0, 6)
        self.spinbox_detectorSize_y = qtw.QSpinBox()
        self.spinbox_detectorSize_y.setRange(1, 8192)
        self.spinbox_detectorSize_y.setValue(512)
        layout_exp_items.addWidget(self.spinbox_detectorSize_y, 0, 7, 1, 2)
        self.activate_detectorSize_spinboxes()
        self.checkbox_detectorSizeAuto.stateChanged.connect(self.activate_detectorSize_spinboxes)

        layout_exp_items.addWidget(qtw.QLabel('Scan Size'), 1, 0, 1, 2)
        self.checkbox_scanSize = qtw.QCheckBox('Auto')
        self.checkbox_scanSize.setChecked(True)
        self.checkbox_scanSize.setToolTip(
            'When checked, scan size is taken from the loaded navigation '
            "signal's shape. Uncheck (or Load Metadata below) to override "
            'manually - needed when that shape does not match the raw 4D '
            'signal files being extracted from.')
        layout_exp_items.addWidget(self.checkbox_scanSize, 1, 2)
        layout_exp_items.addWidget(qtw.QLabel('X', alignment=Qt.AlignCenter), 1, 3)
        self.spinbox_scanSize_x = qtw.QSpinBox()
        self.spinbox_scanSize_x.setRange(1, 99999)
        layout_exp_items.addWidget(self.spinbox_scanSize_x, 1, 4, 1, 2)
        layout_exp_items.addWidget(qtw.QLabel('Y', alignment=Qt.AlignCenter), 1, 6)
        self.spinbox_scanSize_y = qtw.QSpinBox()
        self.spinbox_scanSize_y.setRange(1, 99999)
        layout_exp_items.addWidget(self.spinbox_scanSize_y, 1, 7, 1, 2)
        self.activate_lineEdit_scanSize()
        self.checkbox_scanSize.stateChanged.connect(self.activate_lineEdit_scanSize)
        layout_box_scanSize.addLayout(layout_exp_items)

        self.dp_center = None  # (x, y) - auto-found or last manually-clicked center
        # id(dp_array) at the time dp_center was last auto-found - lets
        # add_scalebar() skip re-running find_dp_center_blurred (a real
        # HyperSpy call) when the displayed DP hasn't actually changed
        # since, e.g. on every keystroke in the scale-recip field. See
        # _on_auto_center_toggled.
        self._dp_center_cache_key = None

        # Acquisition Dwell T. - 3DED extraction always reads the
        # acquisition (smart-scanned) file when Smart Scanned is checked
        # (see the Smart Scan groupbox in Files), so a single dwell time
        # covers both the smart-scan and plain cases - unlike Navigator,
        # there's no separate "Detection"-role extraction path here that
        # would need its own dwell spinbox.
        layout_dwell_row = qtw.QHBoxLayout()
        label_dwellTime = qtw.QLabel('Acquisition Dwell T. (μs)')
        label_dwellTime.setToolTip('Dwell time in microseconds')
        self.spinbox_dwellTime_acquisition = qtw.QSpinBox()
        self.spinbox_dwellTime_acquisition.setFixedWidth(70)
        self.spinbox_dwellTime_acquisition.setRange(1, 99999999)
        layout_dwell_row.addWidget(label_dwellTime)
        layout_dwell_row.addWidget(self.spinbox_dwellTime_acquisition)
        layout_box_scanSize.addLayout(layout_dwell_row)

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

        self.button_viewMetadata = qtw.QPushButton('View...')
        self.button_viewMetadata.setToolTip(
            'Show the full raw comment.txt content in a read-only window - '
            'everything actually logged there, not just Scan Size/Dwell Time above')
        self.button_viewMetadata.clicked.connect(self.show_metadata_dialog)
        layout_scanSize_row2.addWidget(self.button_viewMetadata)
        layout_scanSize_row2.addStretch(1)

        self.metadata_path_override = None  # set by browse_metadata_file(); cleared on new 4D folder

        # 4D signal file list recorded by the navigator tab's own
        # metadata.json (see apply_nav_signal_metadata/resolve_4d_files) -
        # None until a nav signal with a sibling metadata.json is loaded, or
        # invalidated by a manually-browsed 4D folder.
        self._nav_4d_files = None
        self._nav_4d_directory = None

        # Scale bars - Real | Recip. share one row, below everything else
        # in this column, matching the other 3 tabs' identical merge.
        layout_scale_row = qtw.QHBoxLayout()
        layout_scale_row.addWidget(qtw.QLabel('Real Space Scale (nm)'))
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        layout_scale_row.addWidget(self.lineEdit_scale_real)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        self._ribbon_inline_separator(layout_scale_row)
        layout_scale_row.addWidget(qtw.QLabel('Reciprocal Space Scale (Å<sup>-1</sup>)'))
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        layout_scale_row.addWidget(self.lineEdit_scale_recip)
        self.lineEdit_scale_recip.setValidator(self.double_validator)
        self.button_centerRecip = qtw.QPushButton('Center')
        self.button_centerRecip.setToolTip(
            'Find the beam center now (large-sigma blur) and re-center the '
            'reciprocal-space rings there. Hold Ctrl and click on the DP plot to '
            'set the center manually instead.')
        self.button_centerRecip.clicked.connect(self.find_and_center_recip)
        layout_scale_row.addWidget(self.button_centerRecip)
        layout_box_scanSize.addLayout(layout_scale_row)

        self.lineEdit_scale_recip.textChanged.connect(self.add_scalebar)
        self.lineEdit_scale_real.textChanged.connect(self.add_scalebar)
        self._ribbon_group_end(layout_ribbon, layout_box_scanSize, 'Input Parameters')

        #%% Display Contrast (ribbon column) - see the identical treatment
        # in Tab_Tracking_CV2 for why this is added directly rather than
        # via _ribbon_group_start/_end.
        self.box_contrast = ContrastScalingBox()
        layout_ribbon.addWidget(self.box_contrast, 1)
        self.box_contrast.settingsChanged.connect(self.rescale_nav_signal)
        sep_contrast = qtw.QFrame()
        sep_contrast.setFrameShape(qtw.QFrame.VLine)
        sep_contrast.setFrameShadow(qtw.QFrame.Sunken)
        layout_ribbon.addWidget(sep_contrast)

        #%% Feature Handling (ribbon column) - tree_objects is height-capped
        # to fit the ribbon (same treatment as the other 2 tabs' tall lists).
        self.box_table, layout_features = self._ribbon_group_start(layout_ribbon, stretch=1)

        # tree
        self.tree_objects = qtw.QTreeWidget()
        layout_features.addWidget(self.tree_objects)
        self.cols_tree = ["use", "idx", "fr_idx", "end", "trk", "ext", "dup", "del"]
        self.tree_objects.setColumnCount(len(self.cols_tree))
        self.tree_objects.setHeaderLabels(
            ["Use", "Idx", "Frame", "End", "Tracked", "Extracted", "Duplicate", "Delete"])
        # Wide enough for their content: dup/del hold a 30px button, end
        # holds a QSpinBox with up/down arrows, trk/ext hold a status icon.
        col_widths = {'use': 35, 'idx': 30, 'fr_idx': 50, 'end': 60,
                      'trk': 60, 'ext': 65, 'dup': 45, 'del': 45}
        for i, col in enumerate(self.cols_tree):
            self.tree_objects.setColumnWidth(i, col_widths[col])
        self.tree_objects.setMinimumWidth(200)
        # Height-capped (rather than the generous min-height=280 this had
        # in the old left panel) to fit the ribbon's height budget.
        self.tree_objects.setMaximumHeight(110)
        self.tree_objects.setSelectionMode(qtw.QTreeWidget.SingleSelection)
        self.tree_objects.itemSelectionChanged.connect(self.update_canvas)
        
# =============================================================================
#         header = self.tree_objects.header()
#         # Prevent the last column from auto-stretching
#         header.setStretchLastSection(False)
#         # Let all columns resize to contents
#         header.setSectionResizeMode(qtw.QHeaderView.ResizeToContents)
# =============================================================================
        #%% run sam2
        layout_sam_buttons_1 = qtw.QHBoxLayout()
        layout_features.addLayout(layout_sam_buttons_1)
        layout_sam_buttons_2 = qtw.QHBoxLayout()
        layout_features.addLayout(layout_sam_buttons_2)
        
        # image
        self.button_runSeg_img = qtw.QPushButton('Seg Image', self)
        # self.button_runSeg_img.setFixedSize(button_w, button_h_lrg)
        layout_sam_buttons_1.addWidget(self.button_runSeg_img)
        # self.button_runSeg_img.clicked.connect(self.SAM2_image_predictor)
        self.button_runSeg_img.clicked.connect(self.initiate_image_segmentation)
        self.button_runSeg_img.setDisabled(True)
        
        # layout_sam_buttons.addItem(spacer)
        
        # num
        layout_stack = qtw.QVBoxLayout()
        layout_sam_buttons_2.addLayout(layout_stack)
        layout_stack_top = qtw.QHBoxLayout()
        layout_stack.addLayout(layout_stack_top)
        
        label_stackNum = qtw.QLabel('Stack Num')
        label_stackNum.setToolTip('Number of frames per SAM2 processing stack.\nLower values use less GPU memory but run more sequential processes.')
        layout_stack_top.addWidget(label_stackNum)
        self.spinbox_stackNum = qtw.QSpinBox()
        self.spinbox_stackNum.setMaximumWidth(80)
        self.spinbox_stackNum.setToolTip('Frames per SAM2 stack (e.g. 25 = process 25 frames at a time)')
        layout_stack_top.addWidget(self.spinbox_stackNum)
        self.spinbox_stackNum.setSingleStep(25)
        
        self.label_stack = qtw.QLabel('')
        # layout_stack.addWidget(self.label_stack)
        layout_features.addWidget(self.label_stack)
        self.spinbox_stackNum.valueChanged.connect(self.update_stack_guide)
        
        # clip
        self.button_runSeg_clip = qtw.QPushButton('Track', self)
        # self.button_runSeg_clip.setFixedSize(button_w, button_h_lrg)
        layout_sam_buttons_1.addWidget(self.button_runSeg_clip)
        self.button_runSeg_clip.clicked.connect(self.initiate_video_segmentation)
        self.button_runSeg_clip.setEnabled(False)

        self.button_fineTuneMask = qtw.QPushButton('Fine-Tune Mask...', self)
        self.button_fineTuneMask.setToolTip(
            'Manually edit the selected object\'s tracked mask, frame by frame - '
            'grow/shrink it directionally or apply edge detection to one frame')
        layout_sam_buttons_1.addWidget(self.button_fineTuneMask)
        self.button_fineTuneMask.clicked.connect(self.open_fine_tune_mask_dialog)
        self.button_fineTuneMask.setDisabled(True)

        self.button_stop_tr = qtw.QPushButton('Stop')
        layout_sam_buttons_2.addWidget(self.button_stop_tr)
        self.button_stop_tr.clicked.connect(self.stop_processes)
        # self.button_stop_tr.setEnabled(False)
        
# =============================================================================
#         self.button_reset_state = qtw.QPushButton('Reset State', self)
#         self.button_reset_state.setFixedSize(button_w, button_h)
#         layout_sam.addWidget(self.button_reset_state)
#         self.button_reset_state.clicked.connect(self.reset_state)
# =============================================================================
        
        for wid in layout_sam_buttons_1.findChildren(qtw.QWidget):
            wid.setDisabled(True)
        for wid in layout_sam_buttons_2.findChildren(qtw.QWidget):
            wid.setDisabled(True)
        self._ribbon_group_end(layout_ribbon, layout_features, 'Feature Handling')

        #%% Edge Detection / Extract (combined ribbon column, stacked like
        # Tab_Tracking_CV2's Threshold/Edge Detection column) - SAM2
        # doesn't need a Threshold/Deviation section (segmentation comes
        # from SAM2 itself), so Edge Detection pairs with Extract directly
        # instead of needing a 3rd column.
        self.box_3ded, layout_box_3ded = self._ribbon_group_start(layout_ribbon, stretch=1)

        # Cancel needs to stay clickable regardless of tracking/segmentation/
        # extraction state (unlike the rest of this column, whose widgets
        # are enabled/disabled together elsewhere) - built here, before
        # disable_3ded_widgets() runs, since that sweep checks
        # `wid is self.button_cancel`.
        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.setFixedHeight(button_h_lrg)
        self.button_cancel.setStyleSheet("background-color: red; color: white;")
        self.button_cancel.setDisabled(True)
        self.button_cancel.setToolTip(
            'Stop the running SAM2 tracking/segmentation or 3DED extraction. '
            'Already-running background computations finish silently; their results are discarded.')
        self.button_cancel.clicked.connect(self.cancel_running_work)

        layout_edgeDetection_row1 = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_edgeDetection_row1)
        self.checkbox_edgeOnly = qtw.QCheckBox('Edge Detection')
        self.checkbox_edgeOnly.setToolTip(
            'Reduce each frame\'s SAM2 mask to just its outline (via binary erosion) '
            'before it is displayed, extracted, or saved')
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
        layout_box_3ded.addLayout(layout_edgeDetection_row2)
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
        self._ribbon_group_end(layout_ribbon, layout_box_3ded, 'Edge Detection', stretch=False)

        sep_3ded = qtw.QFrame()
        sep_3ded.setFrameShape(qtw.QFrame.HLine)
        sep_3ded.setFrameShadow(qtw.QFrame.Sunken)
        layout_box_3ded.addWidget(sep_3ded)

        layout_threadNum = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_threadNum)

        label_threadNo = qtw.QLabel('CPU Cores')
        layout_threadNum.addWidget(label_threadNo)
        self.spinbox_threadNum = qtw.QSpinBox(self)
        self.spinbox_threadNum.setMaximumWidth(80)
        layout_threadNum.addWidget(self.spinbox_threadNum)
        self.spinbox_threadNum.setRange(1, os.cpu_count() or 1)
        self.spinbox_threadNum.setValue(max(1, (os.cpu_count() or 2) - 2))
        self.spinbox_threadNum.valueChanged.connect(self.set_threadNo)
        
        label_fps = qtw.QLabel('Clip FPS')
        layout_threadNum.addWidget(label_fps)
        self.spinbox_fps = qtw.QSpinBox(self)
        self.spinbox_fps.setMaximumWidth(80)
        layout_threadNum.addWidget(self.spinbox_fps)
        self.spinbox_fps.setRange(1, 60)
        self.spinbox_fps.setValue(5)
        self.spinbox_fps.setToolTip('Frames per second for saved video clips')

        self.checkbox_autosave = qtw.QCheckBox('Autosave')
        layout_threadNum.addWidget(self.checkbox_autosave)

        layout_threadNum.addSpacerItem(spacer)

        # Own row (rather than sharing layout_threadNum with the CPU/FPS/
        # Autosave row above) so the checkbox label has enough room and
        # doesn't get clipped by the left panel's fixed width.
        layout_saveOptions = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_saveOptions)

        self.checkbox_makePets2 = qtw.QCheckBox('Make *.pts2')
        layout_saveOptions.addWidget(self.checkbox_makePets2)
        self.pets2_params = None
        self.checkbox_makePets2.stateChanged.connect(self.on_makePets2_toggled)

        layout_saveOptions.addStretch()

        layout_extract_button = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_extract_button)
        # Natural (content-sized) width, not Expanding - lets the 3 buttons
        # form a compact cluster centered in the row via the stretches
        # below, instead of stretching edge-to-edge across the panel.
        layout_extract_button.addStretch(1)
        self.button_3ded = qtw.QPushButton('Extract All')
        self.button_3ded.setFixedHeight(button_h_lrg)
        layout_extract_button.addWidget(self.button_3ded)
        self.button_3ded.clicked.connect(self.extract_3ded)

        self.button_extractCurrentFrame = qtw.QPushButton('Extract DP\n(Current Frame)')
        self.button_extractCurrentFrame.setFixedHeight(button_h_lrg)
        layout_extract_button.addWidget(self.button_extractCurrentFrame)
        self.button_extractCurrentFrame.setToolTip(
            'Compute the diffraction pattern for just the selected object at the frame the '
            'slider is currently on - a quick one-off check, not saved as part of the series')
        self.button_extractCurrentFrame.clicked.connect(self.extract_dp_current_frame)

        self.button_save_results = qtw.QPushButton('Save Results')
        self.button_save_results.setFixedHeight(button_h_lrg)
        layout_extract_button.addWidget(self.button_save_results)
        self.button_save_results.clicked.connect(self.save_results)
        layout_extract_button.addStretch(1)

        # Same width for all three - sized to fit the widest label
        # ("Extract DP\n(Current Frame)") rather than a hardcoded guess.
        extract_btn_width = self.button_extractCurrentFrame.sizeHint().width()
        for btn in (self.button_3ded, self.button_extractCurrentFrame, self.button_save_results):
            btn.setFixedWidth(extract_btn_width)

        layout_box_3ded.addWidget(self.button_cancel)
        self.disable_3ded_widgets(True)
        self._ribbon_group_end(layout_ribbon, layout_box_3ded, 'Extract', separator=False)
        layout_ribbon.addStretch(1)

        #%% canvas (below the ribbon, using the tab's full width)
        self._right_widget = qtw.QWidget()
        self.layout.addWidget(self._right_widget, 1)
        layout_right_outer = qtw.QHBoxLayout(self._right_widget)
        layout_right_outer.setContentsMargins(0, 0, 0, 0)
        layout_right_outer.setSpacing(0)
        self._canvas_container = qtw.QWidget()
        layout_right_outer.addWidget(self._canvas_container, 1)
        layout_canvas = qtw.QVBoxLayout(self._canvas_container)
        
        self.figure = Figure(constrained_layout=True)
        # self.figure = Figure(figsize=(16,8)) # with figsize
        self.canvas = FigureCanvas(self.figure)
        self.ax_nav = self.figure.add_subplot(131)
        self.ax_seg = self.figure.add_subplot(132)
        self.ax_dp = self.figure.add_subplot(133)
        self.img_zero = np.zeros((512,512), dtype='int16')
        # One-off "Extract DP (Current Frame)" result - only shown while the
        # slider/selection still matches the (obj_id, imgNo) it was computed
        # for; update_canvas() clears it and falls back to the normal
        # per-object dp display as soon as either changes. See
        # extract_dp_current_frame()/_on_current_frame_dp().
        self._current_frame_dp_preview = None
        self.img_display = {}
        self.img_display['nav'] = self.ax_nav.imshow(self.img_zero, cmap='gray')
        self.ax_nav.set_title('Navigation')
        self.img_display['seg'] = self.ax_seg.imshow(self.img_zero, cmap='gray')
        self.img_display['seg_mask'] = self.ax_seg.imshow(self.img_zero, cmap='gray')
        self.ax_seg.set_title('Segmented')
        self.img_display['dp'] = self.ax_dp.imshow(self.img_zero, cmap='inferno',
                                                    norm=SymLogNorm(linthresh=1))
        self.ax_dp.set_title('Extracted DP')

        # Created once here (not per-frame) - update_canvas() only updates
        # the underlying image data, which keeps these in sync for free.
        # seg_mask is excluded: it's an RGBA overlay (see show_mask), not a
        # scalar-valued image, so a colorbar wouldn't mean anything for it.
        self.colorbars = {}
        self.colorbars['nav'] = self.figure.colorbar(
            self.img_display['nav'], ax=self.ax_nav, fraction=0.046, pad=0.04)
        self.colorbars['seg'] = self.figure.colorbar(
            self.img_display['seg'], ax=self.ax_seg, fraction=0.046, pad=0.04)
        self.colorbars['dp'] = self.figure.colorbar(
            self.img_display['dp'], ax=self.ax_dp, fraction=0.046, pad=0.04)

        for ax in [self.ax_dp, self.ax_nav, self.ax_seg]:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # self.figure.tight_layout()
        layout_canvas.addWidget(self.wrap_canvas_in_scroll(self.canvas))
        self.ax_nav.set_xlabel(
            'Hold "ctrl" + Left Click => Positive Point\n'
            'Hold "ctrl" + Right Click => Negative Point\n'
            'Add "shift" to add Points to an Existing Object\n'
            'Middle Click => Delete Last Point', fontsize=10)
        self.ax_nav.xaxis.label.set_visible(True)
        self.ax_dp.xaxis.label.set_visible(True)
        # The Ctrl+Scroll zoom hint applies to every axis on this canvas, so
        # it's a figure-wide supxlabel rather than repeated per-axis text.
        self.figure.supxlabel('Hold "Ctrl" + Scroll wheel to zoom the axis under the cursor',
                              fontsize=10)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        # self.masks_plotted = []
        self.create_main_dataframe()
        self.imgs = deepcopy([self.img_zero])
        self.imgs_8bit = deepcopy([self.img_zero])
        self.scatter_plots = []
        #%% slider
        layout_slider = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_slider)
        
        self.label_imgCounter = qtw.QLabel('Img No.')
        layout_slider.addWidget(self.label_imgCounter)
        
        self.lineEdit_imgNo = qtw.QLineEdit()
        layout_slider.addWidget(self.lineEdit_imgNo)
        self.lineEdit_imgNo.setFixedWidth(35)
        self.lineEdit_imgNo.setValidator(QIntValidator(0, 0))
        self.lineEdit_imgNo.returnPressed.connect(self.jump_to_frame_no)
        
        self.label_imgCounter = qtw.QLabel('Img No.')
        layout_slider.addWidget(self.label_imgCounter)
        
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

        # self.update_canvas(0)
        self.slider_imgNo.valueChanged.connect(self.update_canvas)
        #%% stack navigation buttons
        layout_stacks = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_stacks)

        label_stacks_nav = qtw.QLabel('Stacks:')
        label_stacks_nav.setFixedWidth(45)
        label_stacks_nav.setToolTip('Click a button to jump to the first frame of that stack.\n'
                                    'You must provide at least one point per stack.')
        layout_stacks.addWidget(label_stacks_nav)

        self._stack_scroll = qtw.QScrollArea()
        self._stack_scroll.setWidgetResizable(True)
        self._stack_scroll.setFixedHeight(36)
        self._stack_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._stack_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._stack_scroll.setFrameShape(qtw.QFrame.NoFrame)
        layout_stacks.addWidget(self._stack_scroll)

        self._stack_buttons_widget = qtw.QWidget()
        self._stack_buttons_layout = qtw.QHBoxLayout(self._stack_buttons_widget)
        self._stack_buttons_layout.setContentsMargins(2, 2, 2, 2)
        self._stack_buttons_layout.setSpacing(3)
        self._stack_scroll.setWidget(self._stack_buttons_widget)

        self.tree_objects.itemSelectionChanged.connect(self.update_stack_guide)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout_canvas.addWidget(self.toolbar)

        #%% ribbon
        # Docked along the right edge - an additional way to reach the same
        # canvas interaction already available via Ctrl-click (see on_click)
        # and matplotlib's own toolbar (below); deliberately does NOT
        # duplicate the left panel's buttons (Seg Image, Track, Extract All,
        # Save Results, ...), only actions that act directly on the plot
        # itself. 'add_point' is the only tool mode on_click actually checks
        # (see RibbonPanel.active_tool there) - left/right click still
        # choose positive/negative, and Shift still chooses
        # new-object-vs-append, exactly as before.
        self.ribbon = RibbonPanel([
            RibbonTool('add_point', 'add_point', 'Add point: click on Navigation '
                      '(left = positive, right = negative)\nAdd Shift to add to the '
                      'selected object instead of starting a new one\n'
                      '(same as holding Ctrl and clicking)', 'tool'),
            RibbonTool('remove_point', 'remove_point', 'Remove last point (same as middle-click)',
                      'action', self.delete_last_point),
            RibbonTool('sep1', kind='separator'),
            RibbonTool('pan', 'pan', 'Toggle pan mode (same as the toolbar below)',
                      'action', self.toolbar.pan),
            RibbonTool('zoom', 'zoom', 'Toggle rectangle-zoom mode (same as the toolbar below)',
                      'action', self.toolbar.zoom),
            RibbonTool('home', 'home', 'Reset the view (same as the toolbar below)',
                      'action', self.toolbar.home),
        ], parent=self)
        self.ribbon.toolChanged.connect(self._on_ribbon_tool_changed)
        # Deferred (see _apply_ribbon_cursor's docstring) - reapplies the
        # ribbon cursor after mpl's own NavigationToolbar2 cursor-restore
        # logic (wrapped around every canvas.draw()) has already run.
        self.canvas.mpl_connect(
            'draw_event', lambda evt: QTimer.singleShot(0, self._apply_ribbon_cursor))

        # Clipping Thresholds beside ax_dp (the rightmost of the 3
        # subplots) - only the DP axis, per the decision that Display
        # Contrast already covers the nav image on this tab.
        self.clip_dp = ClippingThresholdsWidget()
        layout_right_outer.addWidget(self.clip_dp)
        self._dp_clip_initialized = False
        self.clip_dp.valueChanged.connect(self._update_dp_clip)

        layout_right_outer.addWidget(self.ribbon)

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
        self.button_loadNavigation.setToolTip('Load navigation signal (.hspy or .zspy)  [Ctrl+O]')
        self.button_loadSavedAnalysis.setToolTip(
            'Load a previously saved analysis folder: navigation signal, '
            'tracked objects, and extracted diffraction patterns  [Ctrl+Shift+O]')
        self.button_runSeg_clip.setToolTip('Track objects across all frames using SAM2  [Ctrl+T]')
        self.button_runSeg_img.setToolTip(
            "Segment the selected object on the current frame only, using its "
            "points on this frame (no tracking across other frames)")
        self.button_3ded.setToolTip('Extract 3D electron diffraction patterns  [Ctrl+E]')
        self.button_save_results.setToolTip('Save segmentation and 3DED results to disk  [Ctrl+S]')
        self.spinbox_threadNum.setToolTip('Number of CPU cores used for parallel 4D extraction')
        self.checkbox_autosave.setToolTip('Automatically save results when extraction finishes')
        self.checkbox_makePets2.setToolTip(
            "Write a PETS2 project file (v1.pts2) into each object's pets/ folder on save, "
            'ready to open directly in PETS for further 3D ED processing')

        # keyboard shortcuts
        QShortcut(QKeySequence('Ctrl+O'), self, self.button_loadNavigation.click)
        QShortcut(QKeySequence('Ctrl+Shift+O'), self, self.button_loadSavedAnalysis.click)
        QShortcut(QKeySequence('Ctrl+T'), self, self.button_runSeg_clip.click)
        QShortcut(QKeySequence('Ctrl+E'), self, self.button_3ded.click)
        QShortcut(QKeySequence('Ctrl+S'), self, self.button_save_results.click)
        QShortcut(QKeySequence('Ctrl+Right'), self,
                  lambda: self.slider_imgNo.setValue(self.slider_imgNo.value() + 1))
        QShortcut(QKeySequence('Ctrl+Left'), self,
                  lambda: self.slider_imgNo.setValue(self.slider_imgNo.value() - 1))

        # Picks up any non-default DisplaySettings already set by the Edit
        # tab (e.g. this instance is a duplicate opened after adjusting
        # sizes) - see TabBase.apply_display_settings.
        self.apply_display_settings()
#%% load data
    def show_dialog(self, f):
        """Open the file/folder dialog matching whichever of the three
        directory buttons was clicked (identified via self.sender()) and
        fill in the corresponding line edit."""
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
            if path:
                self.metadata_path_override = None  # new folder - re-derive comment.txt location
                self.lineEdit_dir_4d.setText(path)
                self._smart_scan_rows = None  # stale for a different folder
                self._nav_4d_files = None  # stale navigator file list for a different folder
                self._set_smart_scan_summary('')
                # Attempted for every format, not just .tpx3 - comment.txt is
                # written for smart-scanned .mib/.hspy/.zspy acquisitions
                # too, and load_metadata(silent=True) already no-ops
                # quietly when comment.txt is missing or unparsable.
                self.load_metadata(silent=True)

        elif sender == self.button_dir_save:
            path = get_existing_directory(self, "Select Destination Folder")
            if path:
                self.lineEdit_dir_save.setText(path)

    def activate_lineEdit_scanSize(self):
        auto = self.checkbox_scanSize.isChecked()
        self.spinbox_scanSize_x.setDisabled(auto)
        self.spinbox_scanSize_y.setDisabled(auto)

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
            x = int(self.spinbox_scanSize_x.text())
            y = int(self.spinbox_scanSize_y.text())
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

    def apply_edge_mask(self, mask):
        """Reduce a single 2-D mask to just its edge/outline when "Edge
        Only" is checked (see io.erode_mask_edge) - a no-op otherwise.
        SAM2 masks are always kept raw in self.df_obj (see
        handle_finished_sam/handle_finished_image_sam) so this can be
        applied fresh - and re-applied live whenever "Edge Only"/the kernel
        size changes - as a view at display/extraction/save time, instead
        of destructively baking erosion into the stored mask (which would
        make it impossible to undo by unchecking the box again)."""
        if self.checkbox_edgeOnly.isChecked():
            direction = (self.spinbox_edgeDirection.value()
                        if self.checkbox_edgeDirectional.isChecked() else None)
            return io.erode_mask_edge(mask, self.spinbox_edgeKernel.value(), direction=direction,
                                       revert=self.checkbox_revertMask.isChecked())
        return mask

    def apply_edge_mask_stack(self, mask_stack):
        """`apply_edge_mask`, applied per-frame to a (N, H, W) mask stack."""
        if self.checkbox_edgeOnly.isChecked():
            kernel = self.spinbox_edgeKernel.value()
            direction = (self.spinbox_edgeDirection.value()
                        if self.checkbox_edgeDirectional.isChecked() else None)
            revert = self.checkbox_revertMask.isChecked()
            return np.stack([io.erode_mask_edge(m, kernel, direction=direction, revert=revert)
                             for m in mask_stack])
        return mask_stack

    def _on_edge_directional_toggled(self):
        """Enable the edge-angle spinbox only while "Directional" is checked, then redraw."""
        self.spinbox_edgeDirection.setEnabled(self.checkbox_edgeDirectional.isChecked())
        self.update_canvas()

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
                self.spinbox_scanSize_x.setValue(int(metadata['scan size x']))
                self.spinbox_scanSize_y.setValue(int(metadata['scan size y']))
                self.checkbox_scanSize.setChecked(False)
            if 'dwelltime' in metadata:
                self.spinbox_dwellTime_acquisition.setValue(int(metadata['dwelltime']))
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
            self.spinbox_scanSize_x.setValue(int(scan_size[0]))
            self.spinbox_scanSize_y.setValue(int(scan_size[1]))
            applied.append('scan size')
        dwell = metadata.get('dwell_time_us')
        if dwell:
            self.spinbox_dwellTime_acquisition.setValue(int(dwell))
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
            self._set_smart_scan_summary(
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

    def _set_smart_scan_summary(self, text):
        """Set label_smartScanSummary's text and keep it hidden while
        empty (see its own setVisible(False) at construction) - avoids a
        permanent blank line at the bottom of the Smart Scan box before/
        between "Check Files..." runs."""
        self.label_smartScanSummary.setText(text)
        self.label_smartScanSummary.setVisible(bool(text))

    def activate_smartScan_widgets(self):
        enable = self.checkbox_smartScan.isChecked()
        for wid in (self.lineEdit_patternDir, self.button_browsePatternDir,
                    self.lineEdit_detectionDir, self.button_browseDetectionDir,
                    self.button_checkSmartScanFiles):
            wid.setEnabled(enable)

    def browse_pattern_dir(self):
        """Browse for the smart-scan pattern-files folder; picking a new one
        invalidates the cached file match (_smart_scan_rows)."""
        start_dir = self.lineEdit_patternDir.text() or self.lineEdit_dir_4d.text()
        path = get_existing_directory(self, "Select Pattern Files Folder", start_dir)
        if path:
            self.lineEdit_patternDir.setText(path)
            self._smart_scan_rows = None
            self._set_smart_scan_summary('')

    def get_pattern_dir(self):
        """Pattern-files directory override, or the 4D signals folder if unset."""
        return self.lineEdit_patternDir.text() or self.lineEdit_dir_4d.text()

    def browse_detection_dir(self):
        """Browse for the smart-scan detection-files folder; picking a new one
        invalidates the cached file match (_smart_scan_rows)."""
        start_dir = self.lineEdit_detectionDir.text() or self.lineEdit_dir_4d.text()
        path = get_existing_directory(self, "Select Detection Files Folder", start_dir)
        if path:
            self.lineEdit_detectionDir.setText(path)
            self._smart_scan_rows = None
            self._set_smart_scan_summary('')

    def get_detection_dir(self):
        """Detection-files directory override, or None if unset (unlike
        get_pattern_dir, this does not fall back to the 4D signals folder)."""
        return self.lineEdit_detectionDir.text() or None

    def open_smart_scan_check_dialog(self):
        """Open SmartScanCheckDialog to review/edit the per-angle detection/
        acquisition/pattern file match for the 4D signals folder; the
        confirmed rows are cached in _smart_scan_rows."""
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
            self._set_smart_scan_summary(f'{n_ok} / {len(dlg.rows)} angle(s) included')
            self.logger.info('Smart-scan file check confirmed: %d / %d angle(s) included.',
                             n_ok, len(dlg.rows))

# =============================================================================
#     def check_torch_device(self):
#         import torch
#         # check device (cuda or cpu)
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#         elif torch.backends.mps.is_available():
#             device = torch.device("mps")
#         else:
#             device = torch.device("cpu")
#         print(f"using device: {device}")
#         
#         if device.type == "cuda":
#             # use bfloat16 for the entire notebook
#             torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#             # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#             if torch.cuda.get_device_properties(0).major >= 8:
#                 torch.backends.cuda.matmul.allow_tf32 = True
#                 torch.backends.cudnn.allow_tf32 = True
#         elif device.type == "mps":
#             print(
#                 "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#                 "give numerically different outputs and sometimes degraded performance on MPS. "
#                 "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#             )
#         
#         return device
# =============================================================================
    
    def _load_spinner(self):
        self.spinner = LoadingSpinner(parent=self)
        self.spinner.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.spinner.setWindowFlags(Qt.SubWindow)
        x = (self.width() - self.spinner.width()) // 2
        y = (self.height() - self.spinner.height()) // 2
        self.spinner.move(x, y)
        self.spinner.raise_()
        self.spinner.start()

    def _ensure_sam2_ready(self, on_ready, on_failed):
        """Ensure the SAM2 checkpoint is present (downloading it first if
        not - not bundled in the installer, see asset_fetch.py) before
        calling `on_ready()`. Runs the check/download in a background
        worker with the loading spinner up, so a first-use ~898MB download
        doesn't freeze the GUI. Calls `on_failed(error_msg)` instead if the
        download fails, e.g. no internet connection."""
        self._load_spinner()

        def _ready(_path, _idx):
            self.spinner.stop()
            on_ready()

        def _failed(error_msg, _idx):
            self.spinner.stop()
            self.logger.error('SAM2 checkpoint download failed:\n%s', error_msg)
            on_failed(error_msg)

        worker = WorkerThread_General(asset_fetch.ensure_sam2_checkpoint, 0)
        worker.signals.results.connect(_ready)
        worker.signals.error.connect(_failed)
        self.threadpool.start(worker)

    def load_navSignal(self):
        """Validate the nav-signal path, reset any existing analysis, and load
        it (hs.load()) in a background worker; _on_navSignal_loaded applies the result."""
        fn = self.lineEdit_dir_navSignal.text()
        # os.path.exists, not isfile - .zspy stores are directories (Zarr),
        # not single files.
        if not os.path.exists(fn):
            self.logger.error('Cannot find navigation signal at: %s', fn)
            qtw.QMessageBox.critical(self, 'File Not Found',
                f'Cannot find navigation signal at:\n{fn}')
            return
        self.logger.info('Loading navigation signal from %s...', fn)
        self.reset_data()
        self._load_spinner()

        def _load(fn):
            s = hs.load(fn)
            return s, s.data.copy()

        worker = WorkerThread_General(_load, 0, fn)
        worker.signals.results.connect(self._on_navSignal_loaded)
        self.threadpool.start(worker)

    def _on_navSignal_loaded(self, result, index):
        """WorkerThread_General callback for load_navSignal(): apply the
        loaded signal and its 8-bit render to the UI."""
        s, imgs = result
        self.spinner.stop()
        self.fn_navSignal = self.lineEdit_dir_navSignal.text()
        self.create_main_dataframe()
        # Anchor the clip-threshold sliders to this signal's raw range before
        # reading get_kwargs() below - a previous signal's clip values would
        # otherwise carry over onto a dataset with a different intensity scale.
        self.box_contrast.set_data_range(imgs.min(), imgs.max())
        # 8-bit conversion happens here (main thread), not inside the
        # background worker above, since it reads the contrast method/
        # parameters live from box_contrast's widgets - not safe to touch
        # from a non-GUI thread.
        imgs_8bit = io.convert_to_8bit(s, **self.box_contrast.get_kwargs()).data
        self._apply_loaded_nav_signal(s, imgs, imgs_8bit)
        self.logger.info('Navigation signal loaded: %d frame(s), %d x %d px.',
                          len(self.imgs), self.imgs[0].shape[0], self.imgs[0].shape[1])

    def _apply_loaded_nav_signal(self, s, imgs, imgs_8bit):
        """Wire up a freshly-loaded (or restored) navigation signal: store it
        and its derived arrays, and reset the canvas/widgets to match. Shared
        by both load_navSignal (fresh load) and load_saved_analysis (restore)."""
        self.s_navSignal = s
        self.imgs = imgs
        self.imgs_8bit = imgs_8bit
        self.dp_center = None  # a new signal may have a different DP shape/center
        self._dp_center_cache_key = None

        self.spinbox_stackNum.setMaximum(len(s))
        shape_x, shape_y = self.imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        # Displayed (and SAM2-fed) from imgs_8bit, not the raw imgs - so the
        # "Display Contrast" method/parameters actually take visible effect.
        self.img_display['nav'].set_clim(vmin=self.imgs_8bit.min(), vmax=self.imgs_8bit.max())
        self.img_display['seg'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['seg_mask'].set_extent([0, shape_y, shape_x, 0])
        # Reset the view to the newly loaded data's full extent (in case the
        # user had already zoomed in on a previous signal, which disables
        # autoscale), then re-seed the toolbar's view stack so its "Home"
        # button resets to *this* view instead of doing nothing (it does
        # nothing until something pushes at least one view onto its stack,
        # which our own scroll-wheel zoom deliberately bypasses).
        for ax in (self.ax_nav, self.ax_seg):
            ax.set_xlim(0, shape_y)
            ax.set_ylim(shape_x, 0)
        self.toolbar.update()
        self.toolbar.push_current()
        self.update_canvas(0)
        self.slider_imgNo.setRange(0, len(self.imgs) - 1)
        self.button_runSeg_clip.setEnabled(True)
        self.button_runSeg_img.setEnabled(True)
        self.button_fineTuneMask.setEnabled(True)
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.imgs)))
        self.spinbox_stackNum.setValue(len(self.imgs))
        # Scale fields may already hold a value from a previous session/load -
        # add_scalebar() is otherwise only triggered by the fields' own
        # textChanged signal, so a fresh load wouldn't show it until touched.
        self.add_scalebar()

    def rescale_nav_signal(self):
        """Retune contrast without reloading the signal from disk: the
        currently-displayed frame is rescaled immediately (cheap, instant
        feedback), while the full stack (used for SAM2/tracking, and to
        keep every other frame in sync) rescales in the background - a
        long stack no longer blocks/lags the GUI on every settings tweak.
        Rapid retuning cancels (i.e. discards the result of) any
        still-running previous background rescale - see
        ContrastScalingBox.rescale_async."""
        if not hasattr(self, 's_navSignal'):
            return
        imgNo = self.slider_imgNo.value()
        frame_8bit = self.box_contrast.rescale_frame(self.imgs[imgNo])
        self.imgs_8bit[imgNo] = frame_8bit
        self.img_display['nav'].set_clim(vmin=frame_8bit.min(), vmax=frame_8bit.max())
        self.update_canvas()
        self.canvas.draw_idle()

        self.box_contrast.rescale_async(self.s_navSignal, self.threadpool, self.logger,
                                        on_done=self._on_nav_signal_rescaled)

    def _on_nav_signal_rescaled(self, s_8bit):
        """ContrastScalingBox.rescale_async callback: apply the fully-rescaled
        8-bit stack once the background recompute finishes."""
        self.imgs_8bit = s_8bit.data
        self.img_display['nav'].set_clim(vmin=self.imgs_8bit.min(), vmax=self.imgs_8bit.max())
        self.update_canvas()
        self.canvas.draw_idle()

    def load_saved_analysis(self):
        """Restore a previously saved analysis folder (produced by
        save_results): the navigation signal, per-object tracking (points/
        labels/rois/masks), and any extracted diffraction patterns."""
        path = get_existing_directory(
            self, "Select Saved Analysis Folder", self.lineEdit_dir_save.text())
        if not path:
            return
        # Newer saves only record the *path* the nav signal was loaded from
        # (see save_analysis_info); fall back to an in-folder copy for
        # analyses saved before that change.
        info = io.load_analysis_info(path)
        analysis_type = info.get('analysis_type') if info else None
        if analysis_type is not None and analysis_type != 'sam2':
            self.logger.warning(
                'Refusing to load %s: this analysis was saved from the %s tab, not SAM2.',
                path, analysis_type)
            qtw.QMessageBox.warning(self, 'Wrong Analysis Type',
                f'This folder was saved from the {analysis_type.upper()} tab, not SAM2 - '
                'the two tabs save different per-object data (masks, points, columns) and '
                "this folder won't load correctly here.\n\n"
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

        self.logger.info('Loading saved analysis from %s...', path)
        self.reset_data()
        self._load_spinner()

        worker = WorkerThread_General(self._load_saved_analysis_worker, 0, path, fn_nav)
        worker.signals.results.connect(self._on_saved_analysis_loaded)
        self.threadpool.start(worker)

    def _load_saved_analysis_worker(self, path, fn_nav):
        """Background-thread body for load_saved_analysis(): load the nav
        signal and every object's saved points/rois/mask/dp from `path`,
        returning them for _on_saved_analysis_loaded to apply on the main thread."""
        s = hs.load(fn_nav)
        imgs = s.data.copy()

        objects = []
        for name in sorted(os.listdir(path)):
            obj_dir = os.path.join(path, name)
            if not (os.path.isdir(obj_dir) and name.startswith('roi No ')):
                continue
            fn_json = os.path.join(obj_dir, f'{name}.json')
            if not os.path.isfile(fn_json):
                continue
            idx = int(name[len('roi No '):])
            with open(fn_json) as f:
                row = json.load(f)

            fn_rois = os.path.join(obj_dir, 'rois.npy')
            rois = np.load(fn_rois) if os.path.isfile(fn_rois) else None

            fn_mask = os.path.join(obj_dir, f'segmentation masks_ obj ID {idx}.npy')
            mask = np.load(fn_mask) if os.path.isfile(fn_mask) else None

            dp = None
            fn_dp_hspy = os.path.join(obj_dir, '3DED.hspy')
            fn_dp_npy = os.path.join(obj_dir, '3DED.npy')
            if os.path.isfile(fn_dp_hspy):
                dp = hs.load(fn_dp_hspy).data
            elif os.path.isfile(fn_dp_npy):
                dp = np.load(fn_dp_npy)

            objects.append({
                'idx': idx, 'use': row['use'], 'frame_idx': row['frame_idx'],
                'points': row['points'], 'labels': row['labels'], 'end': row['end'],
                'rois': rois, 'mask': mask, 'dp': dp,
            })
        return s, imgs, objects, path, fn_nav

    def _on_saved_analysis_loaded(self, result, index):
        """WorkerThread_General callback for load_saved_analysis(): apply the
        loaded signal and repopulate df_obj/tree_objects from each restored object."""
        s, imgs, objects, path, fn_nav = result
        self.spinner.stop()
        self.fn_navSignal = fn_nav
        self.create_main_dataframe()
        # 8-bit conversion happens here (main thread) - see _on_navSignal_loaded.
        self.box_contrast.set_data_range(imgs.min(), imgs.max())
        imgs_8bit = io.convert_to_8bit(s, **self.box_contrast.get_kwargs()).data
        self._apply_loaded_nav_signal(s, imgs, imgs_8bit)

        for obj in objects:
            idx = obj['idx']
            # mask_default (the tracking-derived "Reset to Tracking" target -
            # see MaskEditDialog) was never itself persisted to disk, so a
            # restored object's just-loaded mask is the best available
            # stand-in for it.
            self.df_obj.loc[idx] = [obj['use'], idx, obj['frame_idx'], obj['points'],
                                     obj['labels'], obj['end'], None, obj['mask'],
                                     obj['mask'], obj['rois'], obj['dp']]
            self.add_item_tree(idx, obj['frame_idx'], obj['end'], obj['use'])
            row_index = self.df_obj.index.get_loc(idx)
            if obj['mask'] is not None:
                self.toggle_tree_icon(row_index, 'trk', True)
            if obj['dp'] is not None:
                self.toggle_tree_icon(row_index, 'ext', True)

        self.activate_3ded_widgets(True)
        self.update_canvas(0)
        # Loaded DPs may have a different center than the placeholder - re-run
        # auto-centering now if enabled.
        self.add_scalebar()
        self.logger.info('Loaded saved analysis from %s (%d object(s)).', path, len(objects))

    def create_main_dataframe(self):
        """(Re)create the empty per-object dataframe (df_obj) with its column
        schema, and reset the added-points history."""
        self.cols_df = ['use', 'idx', 'frame_idx', 'points', 'labels', 'end',
                        'single_mask', 'mask', 'mask_default', 'rois', 'dp']
        self.df_obj = pd.DataFrame([], columns=self.cols_df)
        self.df_obj = self.df_obj.astype({'use': int, 'idx': int,'frame_idx': object,
                                          'points': object, 'labels': object,
                                          'end': int, 'single_mask': object,
                                          'dp': object,'mask':object, 'mask_default': object,
                                          'rois':object})
        self.initiate_adding_points()
        
    def reset_data(self):
        """Clear all objects, points, and plotted markers, and drop any
        cached PETS2 params - called before loading a new signal or saved analysis."""
        for p in self.scatter_plots:
            p.remove()
        self.scatter_plots.clear()
        self.tree_objects.clear()
        self.create_main_dataframe()
        self.label_stack.setText('')
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.imgs)))
        self.update_canvas()
        # self.button_runSeg_clip.setEnabled(False)
        # Cached PETS2 params (esp. the per-study center/alpha start/step)
        # were computed for the object set just wiped out above - don't let
        # them silently apply to whatever gets added next.
        self.pets2_params = None
        self.checkbox_makePets2.setChecked(False)
    
    def initiate_adding_points(self):
        """Reset the point-addition history (df_added_points) used by
        delete_last_point() to undo the most recent click."""
        cols = ['new', 'idx', 'point', 'frame']
        self.df_added_points = pd.DataFrame(data=[], columns=cols)
        self.df_added_points = self.df_added_points.astype({'point': object})
    
    def delete_tree_item(self, col:str):
        """Remove the tree row whose "Idx" column matches `col` (an object id
        string, despite the parameter name)."""
        for i in reversed(range(self.tree_objects.topLevelItemCount())):
            item = self.tree_objects.topLevelItem(i)
            if item.text(1) == col:
                self.tree_objects.takeTopLevelItem(i)
#%% object tree and funcs
    def add_item_tree(self, idx, fr_idx=[0], end=None, use=1):
        """Add a row to tree_objects for object `idx`: use checkbox, idx/frame
        labels, an end-frame spinbox, tracked/extracted status icons, and a
        delete button; selects the new row."""
        cols = {col: i for i,col in enumerate(self.cols_tree)}
        item = qtw.QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(cols['use'], Qt.Checked if use else Qt.Unchecked)
        item.setText(cols['idx'], f"{idx}")
        item.setText(cols['fr_idx'], f"{fr_idx}")
        self.tree_objects.itemChanged.connect(self.on_item_check_changed)
        self.tree_objects.addTopLevelItem(item)

        spinbox = qtw.QSpinBox()
        spinbox.setRange(0, len(self.imgs))
        spinbox.setValue(end if end is not None else len(self.imgs))
        self.tree_objects.setItemWidget(item, cols['end'], spinbox)
        spinbox.valueChanged.connect(lambda value: self.on_spinboxEnd_changed(idx, value))
        # spinbox.valueChanged.connect(partial(self.on_spinbox_changed, item, idx))
        
        # self.tree_objects.addTopLevelItem(item)
        
        cancel_icon = self.style().standardIcon(self.style().SP_DialogCancelButton)
        item.setIcon(cols['trk'], cancel_icon)
        item.setData(cols['trk'], Qt.UserRole, False)  # Store status boolean (False = not checked)

        item.setIcon(cols['ext'], cancel_icon)
        item.setData(cols['ext'], Qt.UserRole, False)  # Store status boolean (False = not checked)

        duplicate_button = qtw.QPushButton('Dup')
        duplicate_button.setFixedSize(30, 30)
        duplicate_button.setToolTip(
            'Duplicate this object into a new row - points, tracked masks, '
            'extracted DPs, everything - so you can branch off it (e.g. tweak the '
            'ROI/points and re-track) without losing the original')
        duplicate_button.clicked.connect(
            lambda: self.duplicate_object(self.df_obj.index[self.tree_objects.indexOfTopLevelItem(item)]))

        container_dup = qtw.QWidget()
        layout_dup = qtw.QHBoxLayout(container_dup)
        layout_dup.addWidget(duplicate_button)
        layout_dup.setContentsMargins(0, 0, 0, 0)
        layout_dup.setAlignment(Qt.AlignLeft)
        container_dup.setLayout(layout_dup)
        container_dup.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Preferred)
        self.tree_objects.setItemWidget(item, cols['dup'], container_dup)

        delete_button = qtw.QPushButton()
        delete_button.setIcon(self.style().standardIcon(qtw.QStyle.SP_TrashIcon))
        delete_button.setFixedSize(30, 30)
        delete_button.setToolTip("Delete this item")
        
        def delete_row():
            index = self.tree_objects.indexOfTopLevelItem(item)
            obj_id = self.df_obj.index[index]
            reply = qtw.QMessageBox.question(self, 'Delete Object',
                f'Delete object {obj_id} and all its tracked masks/points?\n'
                'This cannot be undone.')
            if reply == qtw.QMessageBox.No:
                return
            self.logger.info('Deleted object %s.', obj_id)
            self.tree_objects.takeTopLevelItem(index)
            self.df_obj = self.df_obj.drop(self.df_obj.index[index])
            # print(self.df_obj)
            self.update_canvas()
    
        delete_button.clicked.connect(delete_row)
    
        # Wrap the button in a QWidget to add it to a column
        container = qtw.QWidget()
        layout = qtw.QHBoxLayout(container)
        layout.addWidget(delete_button)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignLeft)
        container.setLayout(layout)
        container.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Preferred)
        self.tree_objects.setItemWidget(item, cols['del'], container)

        self.tree_objects.setCurrentItem(item)
        item.setSelected(True)  # optional: highlight

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
    
    def on_spinboxEnd_changed(self, idx, value):
        self.df_obj.at[idx, 'end'] = value

    def duplicate_object(self, old_idx):
        """Clone object `old_idx` into a new row - deep-copying every column
        (points, tracked masks, extracted DPs, ROIs) rather than just the
        object reference, so editing the duplicate (e.g. adding a point and
        re-tracking) can never silently mutate the original too. If the
        original was already tracked/extracted, the duplicate starts out
        fully tracked/extracted as well, with its "Tracked"/"Extracted" tree
        icons set to match - re-running Track/Extract! isn't needed unless
        the duplicate is then changed. Called from each row's own "Dup"
        button - see add_item_tree()."""
        old_idx = int(old_idx)
        new_idx = 1
        while new_idx in self.df_obj.index:
            new_idx += 1

        # deepcopy() on a whole pandas Series does NOT deep-copy
        # object-dtype cell contents (a well-known pandas gotcha - it only
        # copies the references) - each column has to be deep-copied
        # individually instead, or the "duplicate" would still share the
        # exact same mask/dp arrays as the original underneath.
        row = self.df_obj.loc[old_idx]
        self.df_obj.loc[new_idx] = [deepcopy(row[col]) for col in self.cols_df]
        self.df_obj.at[new_idx, 'idx'] = new_idx

        fr_idx = deepcopy(self.df_obj.at[old_idx, 'frame_idx'])
        end = int(self.df_obj.at[old_idx, 'end'])
        use = int(self.df_obj.at[old_idx, 'use'])
        self.add_item_tree(new_idx, fr_idx, end=end, use=use)

        row_index = self.tree_objects.topLevelItemCount() - 1
        if not np.all(pd.isna(self.df_obj.at[new_idx, 'mask'])):
            self.toggle_tree_icon(row_index, 'trk', True)
        if not np.all(pd.isna(self.df_obj.at[new_idx, 'dp'])):
            self.toggle_tree_icon(row_index, 'ext', True)

        self.logger.info('Duplicated object %d as new object %d.', old_idx, new_idx)
        self.update_canvas()

    def open_fine_tune_mask_dialog(self):
        """Open MaskEditDialog on the selected object's tracked mask stack,
        seeded at the frame the slider is currently on; writes the edited
        stack back on Save & Close."""
        selected_items = self.tree_objects.selectedItems()
        if not selected_items:
            qtw.QMessageBox.warning(self, 'No Object Selected',
                'Select an object in the list to fine-tune its mask first.')
            return
        obj_id = int(selected_items[0].text(1))
        mask_stack = self.df_obj.at[obj_id, 'mask']
        if np.all(pd.isna(mask_stack)):
            qtw.QMessageBox.warning(self, 'Not Tracked Yet',
                'This object has no tracked mask yet - run "Track" or "Seg Image" first.')
            return
        default_mask_stack = self.df_obj.at[obj_id, 'mask_default']
        if np.all(pd.isna(default_mask_stack)):
            default_mask_stack = None
        edge_settings = {
            'enabled': self.checkbox_edgeOnly.isChecked(),
            'kernel': self.spinbox_edgeKernel.value(),
            'revert': self.checkbox_revertMask.isChecked(),
            'directional': self.checkbox_edgeDirectional.isChecked(),
            'direction': self.spinbox_edgeDirection.value()}
        dialog = MaskEditDialog(self, mask_stack, bg_stack=self.imgs_8bit,
                                start_frame=self.slider_imgNo.value(), logger=self.logger,
                                default_mask_stack=default_mask_stack, edge_settings=edge_settings)
        if dialog.exec_() == qtw.QDialog.Accepted:
            self.df_obj.at[obj_id, 'mask'] = dialog.get_mask_stack()
            self._apply_dialog_settings_to_ui(dialog)
            self.logger.info('Fine-tuned mask saved for object %d.', obj_id)
            self.update_canvas()

    def _apply_dialog_settings_to_ui(self, dialog):
        """Sync MaskEditDialog's Edge Detection box values back into this
        tab's own main controls on Save && Close, so whatever was left set
        there (not just the returned mask itself) is what the live preview
        uses next, instead of silently reverting to whatever was set before
        the dialog was opened. SAM2 masks aren't threshold-derived, so no
        Threshold box is ever built here (get_thresh_settings() is None) -
        see MaskEditDialog."""
        edge = dialog.get_edge_settings()
        self.checkbox_edgeOnly.setChecked(edge['enabled'])
        self.spinbox_edgeKernel.setValue(edge['kernel'])
        self.checkbox_revertMask.setChecked(edge['revert'])
        self.checkbox_edgeDirectional.setChecked(edge['directional'])
        self.spinbox_edgeDirection.setValue(edge['direction'])

    def on_item_check_changed(self, item, column):
        use_col = self.cols_tree.index('use')  # or `cols['use']` if accessible
        idx_col = self.cols_tree.index('idx')
        idx = int(item.text(idx_col))
        if item.checkState(use_col) == Qt.Checked:
            self.df_obj.at[idx, 'use'] = 1
        else:
            self.df_obj.at[idx, 'use'] = 0
#%% canvas
    def _on_ribbon_tool_changed(self, tool_id):
        self.logger.debug('Ribbon tool changed to %s', tool_id)
        self._apply_ribbon_cursor()

    def _apply_ribbon_cursor(self):
        """Set the canvas cursor to match the ribbon's active tool - besides
        the ribbon button's own highlighted (QToolButton:checked) style,
        this gives the active tool a distinct cursor too, since which mode
        is armed wasn't obvious enough from the ribbon alone. Also called
        (deferred - see the 'draw_event' connection in init_widget) on
        every canvas redraw: NavigationToolbar2's _wait_cursor_for_draw_cm()
        wraps every canvas.draw() call and restores its own internally-
        tracked cursor afterward, which would otherwise silently undo this
        on the very next on_click/etc. redraw."""
        cursor = {'add_point': Qt.PointingHandCursor}.get(self.ribbon.active_tool)
        self.canvas.setCursor(cursor if cursor is not None else Qt.ArrowCursor)

    def on_click(self, event):
        """Canvas mouse-click handler: middle-click deletes the last added
        point; Ctrl+click on the DP plot (only while auto-centering is off)
        sets a manual diffraction-pattern center; Ctrl+click on the nav
        image (or a plain click while the ribbon's "Add point" tool is
        active) adds a positive (left) or negative (right) point to the
        selected object, starting a new object unless Shift is held."""
        if event.button == 2: # middle click:
            self.delete_last_point()
            return
        if (event.inaxes == self.ax_dp and event.button == 1 and event.xdata is not None
                and 'ctrl' in event.modifiers):
            # Manual re-centering of the reciprocal-space rings.
            self.dp_center = (event.xdata, event.ydata)
            self.add_scalebar()
            return
        if (event.inaxes != self.ax_nav) or (event.button not in [1,3]):
            return
        if self.ribbon.active_tool != 'add_point' and 'ctrl' not in event.modifiers:
            # Plain click/drag is reserved for the navigation toolbar's
            # Pan/Zoom tool (and the scroll-wheel zoom below) so images can
            # be zoomed into; hold "ctrl" (or activate the ribbon's "Add
            # point" tool) to add a point instead.
            return

        imgNo = self.slider_imgNo.value()
        p = [event.xdata, event.ydata]
        # left click is positive and right click negative
        # click = 'pos' if event.button() == Qt.LeftButton else 'neg' # if event.button == 3 else False
        new_item = not ('shift' in event.modifiers) # if shift is held, it is NOT a new object
        if event.button == 1:
            label = 1
        elif event.button == 3:
            label = 0
        if new_item:
            idx = 1
            while idx in self.df_obj.index:
                idx += 1
            fr_idx = [imgNo]
            self.df_obj.loc[idx] = [1, idx, fr_idx, [p], [label], len(self.imgs),
                                    None, None, None, None, None]
            self.add_item_tree(idx, fr_idx)
        else:
            selected_items = self.tree_objects.selectedItems()
            if selected_items:
                item = selected_items[0]
            else: # the last item, if nothing is selected
                count = self.tree_objects.topLevelItemCount()
                item = self.tree_objects.topLevelItem(count - 1) # last one
            idx = int(item.text(1))
            self.df_obj.at[idx, 'frame_idx'].append(imgNo)
            self.df_obj.at[idx, 'points'].append(p)
            self.df_obj.at[idx, 'labels'].append(label)
            item.setText(2, str(self.df_obj.at[idx, 'frame_idx']))
        # print(self.df_obj.loc[:,['idx', 'frame_idx', 'points', 'labels', 'end']])
        added_point = [new_item, idx, p, imgNo]
        i_ap = self.df_added_points.index.max()
        i_ap = 1 if pd.isna(i_ap) else i_ap+1
        self.df_added_points.loc[i_ap] = added_point
        self.update_canvas(imgNo) # TODO fix

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
        self.canvas.draw_idle()

    def delete_last_point(self):
        """Undo the most recently added point (see on_click): drops the whole
        object if it was a new one, otherwise just pops its last point/label/frame."""
        try:
            i = self.df_added_points.index[-1]
        except IndexError: # no point to delete
            return
        if self.df_added_points.loc[i, 'new']:
            self.delete_tree_item(str(self.df_added_points.loc[i, 'idx']))
            self.df_obj = self.df_obj.drop(self.df_added_points.loc[i, 'idx'])
        else:
            idx = self.df_added_points.loc[i, 'idx']
            _ = self.df_obj.at[idx, 'frame_idx'].pop()
            _ = self.df_obj.at[idx, 'points'].pop()
            _ = self.df_obj.at[idx, 'labels'].pop()
            count = self.tree_objects.topLevelItemCount()
            item = self.tree_objects.topLevelItem(count - 1) # last one
            item.setText(2, str(self.df_obj.at[idx, 'frame_idx']))
        self.df_added_points = self.df_added_points.drop(i)
        self.update_canvas()
    
    def jump_to_frame_no(self):
        num = int(self.lineEdit_imgNo.text())
        self.slider_imgNo.setValue(num)
    
    def update_canvas(self, imgNo=None, obj_id=None):
        """Redraw the nav/segmentation/DP panels for `imgNo` (default: slider
        value) and `obj_id` (default: selected object): shows the nav frame,
        the object's mask (tracked or single-frame) and diffraction pattern
        if present, then draws the canvas."""
        if imgNo is None:
            imgNo = self.slider_imgNo.value()
        if obj_id is None:
            try:
                item_selected = self.tree_objects.currentItem()
                obj_id = int(item_selected.text(1))
            except Exception:
                obj_id = None
        self.remove_plotted_points()
        # Displayed from imgs_8bit, not the raw imgs - see _apply_loaded_nav_signal.
        self.img_display['nav'].set_data(self.imgs_8bit[imgNo])
        self.ax_nav.set(title=f'Nav. Image No: {imgNo}')

        if obj_id is not None:
            self.plot_points(imgNo, obj_id)
            # plot segmentation masks for video
            if (not np.all(pd.isna(self.df_obj.loc[obj_id, 'mask']))):
                self.img_display['seg'].set_data(self.imgs_8bit[imgNo])
                self.img_display['seg'].set_clim(vmin=self.imgs_8bit[imgNo].min(), vmax=self.imgs_8bit[imgNo].max())

                try:
                    self.show_mask(self.apply_edge_mask(self.df_obj.loc[obj_id, 'mask'][imgNo]), obj_id)
                except Exception:
                    self.show_mask(self.img_zero)

            # plot segmentation masks for single images
            else:
                try:
                    self.img_display['seg'].set_data(self.imgs_8bit[imgNo])
                    self.img_display['seg'].set_clim(vmin=self.imgs_8bit[imgNo].min(), vmax=self.imgs_8bit[imgNo].max())
                    mask = self.apply_edge_mask(self.df_obj.loc[obj_id, 'single_mask'][imgNo])
                    self.show_mask(mask, 0)
                except Exception:
                    self.img_display['seg'].set_data(self.img_zero)
            
            # diffraction pattern
            preview = self._current_frame_dp_preview
            if preview is not None and preview['obj_id'] == obj_id and preview['imgNo'] == imgNo:
                dp = preview['dp']
                self.img_display['dp'].set_data(dp)
                self._apply_dp_clip(dp)
                shape_x, shape_y = dp.shape
                self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])
            else:
                self._current_frame_dp_preview = None
                if (not np.all(pd.isna(self.df_obj.loc[obj_id, 'dp']))):
                    try:
                        self.plot_dp(obj_id=obj_id, imgNo=imgNo)
                    except Exception:
                        self.img_display['dp'].set_data(self.img_zero)
                else:
                    self.img_display['dp'].set_data(self.img_zero)

        if not self._layout_frozen:
            # Let constrained_layout solve spacing once with real content,
            # then freeze it so later redraws (including draw_idle() below)
            # don't repeat that expensive solve every single frame/point.
            self.canvas.draw()
            self.figure.set_layout_engine('none')
            self._layout_frozen = True
        else:
            self.canvas.draw_idle()

    def add_scalebar(self):
        """Redraw the real-space scale bar (nav/seg axes) and the
        reciprocal-space calibration rings (DP axis), re-finding the
        auto-center first if enabled."""
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
            for ax in [self.ax_nav, self.ax_seg]:
                io.add_readable_scalebar(ax, scale_real, 'nm')
        except Exception:
            pass

        # A conventional linear scale bar doesn't read naturally on a
        # radially-symmetric diffraction pattern - concentric dashed rings
        # at every 1 1/A (centered on the DP) work better.
        dp_array = self.img_display['dp'].get_array()
        shape = dp_array.shape
        # Centering is purely manual now (see find_and_center_recip and
        # Ctrl+Click) - self.dp_center just persists across redraws until
        # one of those changes it, no more continuous auto-re-finding here.
        self._dp_recip_circles = io.draw_reciprocal_scale_circles(
            self.ax_dp, self.lineEdit_scale_recip.text(), shape,
            center=self.dp_center, old_artists=getattr(self, '_dp_recip_circles', None))
        self.ax_dp.set_xlabel(
            'Circle center: click "Center" (Input Parameters) to find it, or hold '
            'Ctrl and click the DP plot to set it manually', fontsize=9)

        self.canvas.draw()

    def find_and_center_recip(self):
        """Find the beam center now and jump the reciprocal-space rings
        there - the "Center" button's slot."""
        dp_array = self.img_display['dp'].get_array()
        if not np.any(dp_array):
            qtw.QMessageBox.warning(self, 'No Diffraction Pattern',
                'Load/segment an object first, so a beam center can be found.')
            return
        try:
            self.dp_center = io.find_dp_center_blurred(dp_array)
        except Exception:
            self.logger.exception('Auto-centering failed.')
            return
        self.add_scalebar()

    def show_mask(self, mask, cmap_idx=0):
        """Render `mask` as a translucent RGBA overlay on the segmentation
        axis, colored by `cmap_idx` (tab10)."""
        cmap = plt.get_cmap("tab10")
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        # mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        self.img_display['seg_mask'].set_data(mask_image)
    
    def remove_plotted_points(self):
        for p in self.scatter_plots:
            try:
                p.remove()
            except Exception:
                self.logger.debug('Point scatter artist already removed.', exc_info=True)
        self.scatter_plots.clear()
        
    def plot_points(self, imgNo, obj_id):
        """Scatter object `obj_id`'s annotated points on frame `imgNo` onto
        the nav axis (green = positive, red = negative)."""
        if imgNo not in self.df_obj.loc[obj_id, 'frame_idx']: # no point for this image and object id
            return
        frames = np.array(self.df_obj.loc[obj_id, 'frame_idx'])
        toPlot = np.where(frames == imgNo)
        points = np.array(self.df_obj.loc[obj_id, 'points'])[toPlot]
        labels = np.array(self.df_obj.loc[obj_id, 'labels'])[toPlot]
        for l, p in zip(labels, points):
            if l: # positive point
                scatter_p = self.ax_nav.scatter(p[0], p[1], color='green', 
                                                marker='o', s=20, linewidth=1.25)
            else: # negative point
                scatter_p = self.ax_nav.scatter(p[0], p[1], color='red', 
                                                marker='o', s=20, linewidth=1.25)
            self.scatter_plots.append(scatter_p)
    
    def plot_dp(self, obj_id=None, imgNo=None):
        if not imgNo:
            imgNo = self.slider_imgNo.value()
        img = self.df_obj.loc[obj_id, 'dp'][imgNo]
        self.img_display['dp'].set_data(img)
        self._apply_dp_clip(img)
        shape_x, shape_y = img.shape
        self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])

    def _apply_dp_clip(self, img):
        """Anchor clip_dp's Clipping Thresholds to a newly-displayed DP's
        range and apply them as set_clim. Like Tab_Tracking_CV2, DP changes
        every frame here, so the range only resets to "no clipping" once,
        the first time real data appears - after that the user's chosen
        thresholds persist across frame/object changes."""
        self.clip_dp.set_range(img.min(), img.max(), reset=not self._dp_clip_initialized)
        self._dp_clip_initialized = True
        vmin, vmax = self.clip_dp.values()
        self.img_display['dp'].set_clim(vmin=vmin, vmax=vmax)

    def _update_dp_clip(self):
        """clip_dp.valueChanged slot: re-apply its current vmin/vmax to the
        already-displayed DP image (no new data) and redraw."""
        if 'dp' not in self.img_display:
            return
        vmin, vmax = self.clip_dp.values()
        self.img_display['dp'].set_clim(vmin=vmin, vmax=vmax)
        self.canvas.draw_idle()
#%% SAM2 video segmentation
    def update_stack_guide(self):
        """Rebuild the stack-navigation button strip for the currently selected object.

        Each button is labelled with the stack index and, when clicked, moves the slider
        to the global frame number where that stack begins.  The start of each stack is
        shifted by the object's earliest frame_idx so objects that don't start at frame 0
        are handled correctly.
        """
        while self._stack_buttons_layout.count():
            item = self._stack_buttons_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.label_stack.setText('')

        try:
            stack = self.spinbox_stackNum.value()
            if stack <= 0:
                return
            items = self.tree_objects.selectedItems()
            if not items or not hasattr(self, 'imgs_8bit'):
                return
            idx = int(items[0].text(1))
            beg = min(self.df_obj.loc[idx, 'frame_idx'])
            end = int(self.df_obj.loc[idx, 'end'])
            end = min(end, len(self.imgs_8bit))

            # Global frame index at the start of each stack for this object
            arr = np.arange(beg, end, stack)

            self.label_stack.setText(f'{len(arr)} stack(s)')

            for i, frame_no in enumerate(arr):
                frame_no = int(frame_no)
                btn = qtw.QPushButton(str(i + 1))
                btn.setFixedSize(26, 26)
                btn.setToolTip(f'Stack {i + 1} — jump to frame {frame_no}')
                btn.clicked.connect(lambda checked, f=frame_no: self.slider_imgNo.setValue(f))
                self._stack_buttons_layout.addWidget(btn)
            self._stack_buttons_layout.addStretch(1)
        except Exception:
            self.label_stack.setText('')

    def initiate_video_segmentation(self):
        """Kick off SAM2 tracking for every "used" object: split each
        object's frame range into stack_num-sized chunks, export each
        chunk's frames as JPGs in the background, and record the per-chunk
        points/labels needed to seed segmentation (df_toSegment).
        run_video_segmentation() launches the actual SAM2 subprocesses once
        JPG export finishes."""
        # self.button_runSeg_img.setDisabled(True)
        self.button_runSeg_clip.setDisabled(True)
        self._track_tic = perf_counter()
        self._track_failed = False
        self._cancelling = False
        self.button_cancel.setEnabled(True)

        pathSave = self.lineEdit_dir_save.text()
        if not (os.path.isdir(pathSave)):
            os.mkdir(pathSave)
        self.path_jpg = os.path.join(pathSave, 'JPG Images')
        if os.path.isdir(self.path_jpg):
            shutil.rmtree(self.path_jpg)
            # os.rmdir(self.path_jpg)
        os.mkdir(self.path_jpg)
        
        df = self.df_obj[self.df_obj.use == 1]
        self.stack_num = self.spinbox_stackNum.value()
        if self.stack_num == 0:
            self.stack_num = len(self.imgs)
            self.spinbox_stackNum.setValue(self.stack_num)

        self.logger.info('Starting SAM2 tracking for %d object(s) (stack size %d frames)...',
                          len(df), self.stack_num)

        self.total_threads_jpg = 0
        for idx in df.index:
            st = min(df.loc[idx, 'frame_idx'])
            end = df.loc[idx, 'end']
            imgs = self.imgs_8bit[st:end]
            arr_stack = np.arange(0, len(imgs), self.stack_num)
            self.total_threads_jpg += len(arr_stack)
        
        self.df_toSegment = pd.DataFrame(data=[], columns=[
            'path_jpg', 'idx_ref', 'stack_num', 'frame_idx', 'points', 
            'labels', 'mask'])
        self.df_toSegment = self.df_toSegment.astype({
            'path_jpg': str,
            'idx_ref': int,
            'stack_num': int,
            'frame_idx': object, 
            'points': object, 
            'labels': object,
            'mask': object})
        
        self.worker_count_jpg = 0
        i_c = 0
        for idx in df.index:
            st = min(df.loc[idx, 'frame_idx'])
            end = df.loc[idx, 'end']
            imgs = self.imgs_8bit[st:end]
            arr_stack = np.arange(0, len(imgs), self.stack_num)
            arr_stack = np.append(arr_stack, len(imgs))
            # Allocate 'mask' at full dataset length (not just this object's
            # st:end range) so every consumer that indexes it by the GLOBAL
            # frame number (update_canvas, extract_3ded, create_clip_tracking_
            # with_mask, ...) lines up correctly, instead of needing a local/
            # global offset conversion that was easy to get wrong in one place
            # and miss in another.
            self.df_obj.at[idx, 'mask'] = np.zeros(self.imgs_8bit.shape, dtype=bool)
            path_obj = os.path.join(self.path_jpg, f'{idx}')
            if os.path.isdir(path_obj):
                shutil.rmtree(path_obj)
            os.mkdir(path_obj)
            
            for i_fld, _ in enumerate(arr_stack[:-1]):
                path_stack = os.path.join(path_obj, f'{i_fld}')
                if os.path.isdir(path_stack):
                    shutil.rmtree(path_stack)
                os.mkdir(path_stack)
                
                st_2 = arr_stack[i_fld]
                end_2 = arr_stack[i_fld+1]
                imgs_stack = imgs[st_2:end_2]
                frame_idx, points, labels = df.loc[idx, 
                   ['frame_idx', 'points', 'labels']]
                frame_idx = np.array(frame_idx) - st - st_2
                cond = np.where((frame_idx>=0) & (frame_idx<self.stack_num))
                frame_idx = frame_idx[cond]
                points = np.array(points)[cond]
                labels = np.array(labels)[cond]
                self.df_toSegment.loc[i_c] = [path_stack, idx, i_fld, frame_idx,
                                                points, labels, None]
                fn = os.path.join(path_stack, 'seg_input.pkl')
                self.df_toSegment.loc[i_c][:-1].to_pickle(fn)
                i_c += 1
                worker_make_jpg = WorkerThread_General(self.make_jpg_imgs, 0, 
                                           path_stack, imgs_stack)
                self.threadpool.start(worker_make_jpg)
                worker_make_jpg.signals.finished.connect(self.check_jpg_completion)
    
    def make_jpg_imgs(self, path, imgs):
        for i_img, img in enumerate(imgs):
            img = Image.fromarray(img)
            # img = img.convert('L')  # Convert to grayscale
            img.save(os.path.join(path, f'{i_img:04d}.jpg'))

    def check_jpg_completion(self):
        self.worker_count_jpg += 1
        if self.total_threads_jpg == self.worker_count_jpg:
            self.run_video_segmentation()
        
    def run_video_segmentation(self):
         """Ensure the SAM2 checkpoint is available, then launch the first
         stack's SAM2 subprocess; handle_finished_sam() chains the rest
         sequentially as each one finishes."""
         self.running_processes_sam = {}
         self.running_processes_sam_total = len(self.df_toSegment.index)
         idx = self.df_toSegment.index.sort_values()[0]
         path = self.df_toSegment.loc[idx, 'path_jpg']

         def _on_checkpoint_failed(error_msg):
             self.button_runSeg_clip.setEnabled(True)
             self.button_cancel.setDisabled(True)
             qtw.QMessageBox.warning(self, 'SAM2 Checkpoint Download Failed',
                 'Could not download the SAM2 model checkpoint - check your '
                 'internet connection and see the log console for details.')

         self._ensure_sam2_ready(lambda: self.launch_next_video_seg(path, idx),
                                  _on_checkpoint_failed)
                
    def launch_next_video_seg(self, path, idx):
        """Start one stack's SAM2 video-tracking subprocess for `idx` and
        wire up its signal handlers."""
        self.logger.info("Next project: %s %s", idx, path)
        program, arguments = worker_command('sam', ['video', path, str(idx)])
        process_sam = QProcess(self)
        process_sam.setProgram(program)
        process_sam.setArguments(arguments)

        # process_sam.setProcessChannelMode(QProcess.MergedChannels)  # Combine stdout+stderr
        
        # process_sam.readyReadStandardOutput.connect(lambda: 
        #                     self.handle_output_sam(process_sam, idx))
            
        process_sam.readyReadStandardError.connect(lambda: 
                            self.handle_error_sam(process_sam, idx))
        process_sam.finished.connect(lambda exit_code, exit_status: 
                     self.handle_finished_sam(
                     process_sam, idx, exit_code, exit_status))
        
        process_sam.errorOccurred.connect(lambda error: self.process_failed_sam(error, idx))

        self.running_processes_sam[idx] = process_sam
        process_sam.start()
        
    def process_failed_sam(self, error, idx):
        self._track_failed = True
        self.logger.error("[%s] QProcess error occurred: %s", idx, error)

    def handle_error_sam(self, process, idx):
        # SAM2/PyTorch routinely write progress bars, warnings, and other
        # non-fatal diagnostics to stderr even on a fully successful run,
        # so this is just informational rather than an error — a genuine
        # failure is already caught via the JSON-decode check in
        # handle_finished_sam() and via process_failed_sam() below (the
        # QProcess itself failing to launch).
        self._stderr_buffer.log_info(process, self.logger, str(idx))
        # self.spinner.stop()
    
# =============================================================================
#     def handle_output_sam(self, process, idx): #TODO
#         data = process.readAllStandardOutput()
#         text = bytes(data).decode("utf-8")
#         # self.output_box.append(f"[{idx}] {text}")
#     
#         match = re.search(r"(\d+)%\|", text)
#         if match:
#             percent = int(match.group(1))
#             self.progress_bar.setValue(percent, 100)
# =============================================================================

    def _show_missing_dependency_dialog(self, message):
        """worker_sam.py reports this when torch/sam2 aren't importable -
        deliberately not bundled in a frozen build (huge, CUDA-version-
        specific, see INSTALL.md). Point at the fix instead of just failing."""
        self.logger.error('SAM2 worker reported a missing dependency: %s', message)
        qtw.QMessageBox.warning(self, 'SAM2 Dependencies Not Installed',
            f'{message}\n\n'
            'SAM2 needs torch and the sam2 package installed, which this '
            "app doesn't bundle. Run, from a command prompt with pip "
            'available:\n\n'
            'pip install --target "<install_dir>\\_internal" torch '
            '--index-url https://download.pytorch.org/whl/cu121\n'
            'pip install --target "<install_dir>\\_internal" '
            'git+https://github.com/facebookresearch/sam2.git\n\n'
            '(swap the --index-url per pytorch.org/get-started/locally for '
            'your GPU, or omit it for CPU-only; "<install_dir>" is where '
            'EDyssey.exe is installed). See INSTALL.md for details.')

    def handle_finished_sam(self, process, idx, exit_code, exit_status):
        """SAM2 video-tracking subprocess completion handler: load the
        stack's output mask, launch the next queued stack if any remain, and
        once every stack for every object has returned, stitch each object's
        per-stack masks into its full-length mask array and mark it tracked."""
        if self._cancelling:
            return
        self.logger.info("[%s] Process finished with exit code %s, status %s",
                          idx, exit_code, exit_status)

        data = process.readAllStandardOutput()
        text = bytes(data).decode("utf-8")
        self.logger.info('text: %s', text)
        # process.kill()
        # sleep(3)
        try:
            result = json.loads(text.strip())
            if result.get('error') == 'missing_dependency':
                self._track_failed = True
                self._show_missing_dependency_dialog(result['message'])
                self.button_runSeg_clip.setEnabled(True)
                self.button_cancel.setDisabled(True)
                return
            fn_output = result["path"]
            idx = int(result["idx"])

            with np.load(fn_output) as f:
                mask_stack = f['masks']
            self.df_toSegment.at[idx, 'mask'] = mask_stack
            # launch next segmentation
            if len(self.running_processes_sam) != self.running_processes_sam_total:
                for idx in self.df_toSegment.index.sort_values():
                    #TODO already put the masks in df_obj
                    if idx not in self.running_processes_sam:
                        path = self.df_toSegment.loc[idx, 'path_jpg']
                        self.launch_next_video_seg(path, idx)
                        break
            else: # finished
                for i_ref in np.unique(self.df_toSegment.idx_ref):
                    df = self.df_toSegment[self.df_toSegment.idx_ref==i_ref]
                    for idx in df.index:
                        i_c = df.loc[idx, 'stack_num']
                        beg = min(self.df_obj.loc[i_ref, 'frame_idx'])

                        # 'mask' is now allocated at full dataset length (global
                        # frame numbers), so each per-stack chunk is placed at
                        # its own global offset: beg (object's start frame) +
                        # i_c full stacks in. frame_num (rather than assuming a
                        # full self.stack_num) makes this correct even for the
                        # last chunk of an object, which is often shorter than
                        # a full stack.
                        frame_num = len(df.loc[idx, 'mask'])
                        start = beg + i_c * self.stack_num
                        # Kept raw (un-eroded) here - "Edge Only" is applied
                        # as a view at display time (update_canvas) and at
                        # extraction/save time instead, so toggling it later
                        # doesn't require re-tracking to see the effect, and
                        # can be turned back off without losing the original
                        # SAM2 result. See apply_edge_mask()/apply_edge_mask_stack().
                        self.df_obj.at[i_ref, 'mask'][
                            start : start + frame_num] = df.loc[idx, 'mask']
                    # Snapshot the freshly-tracked (still un-eroded, un-edited)
                    # result as this object's permanent "Reset to Tracking"
                    # target for MaskEditDialog - taken here, before any
                    # fine-tune-dialog edits can ever touch 'mask'.
                    self.df_obj.at[i_ref, 'mask_default'] = self.df_obj.at[i_ref, 'mask'].copy()
                    row_index = self.df_obj.index.get_loc(i_ref)
                    self.toggle_tree_icon(row_index, 'trk', True)

                n_objects = len(np.unique(self.df_toSegment.idx_ref))
                _ = gc.collect()
                del self.df_toSegment
                self.activate_3ded_widgets(True)
                self.update_canvas()

                duration = perf_counter() - self._track_tic
                if self._track_failed:
                    self.logger.error(
                        'SAM2 tracking finished with errors for %d object(s) '
                        'after %s (see log above for details).',
                        n_objects, io.format_duration_hms(duration))
                else:
                    self.logger.info(
                        'SAM2 tracking completed successfully for %d object(s) in %s.',
                        n_objects, io.format_duration_hms(duration))
                self.button_cancel.setDisabled(True)
        except json.JSONDecodeError:
            self._track_failed = True
            self.logger.error("Could not decode result: %s", text)
            qtw.QMessageBox.warning(self, 'SAM2 Error',
                f'Could not decode SAM2 output. Check console for details.\n'
                f'Raw output (first 200 chars): {text[:200]}')
        self.button_runSeg_clip.setEnabled(True)
    
    def stop_processes(self):
        if hasattr(self, 'running_processes_sam'):
            while len(self.running_processes_sam) > 0:
                idx, pr = self.running_processes_sam.popitem()
                pr.kill()
#%% image segmentation
    def initiate_image_segmentation(self):
        """Segment the currently-selected object on the current frame only,
        using SAM2's single-image predictor (no cross-frame propagation) —
        unlike Track, which runs SAM2's video predictor across every frame."""
        try:
            item_selected = self.tree_objects.currentItem()
            obj_id = int(item_selected.text(1))
        except Exception:
            self.logger.warning('Single-image segmentation requested but no object is selected.')
            qtw.QMessageBox.critical(self, 'No Object Selected',
                'Select an object in the list before running single-image segmentation.')
            return

        imgNo = self.slider_imgNo.value()
        frames = np.array(self.df_obj.loc[obj_id, 'frame_idx'])
        toPlot = np.where(frames == imgNo)
        points = np.array(self.df_obj.loc[obj_id, 'points'])[toPlot]
        labels = np.array(self.df_obj.loc[obj_id, 'labels'])[toPlot]
        if len(points) == 0:
            self.logger.warning(
                'Single-image segmentation requested for object %s but it has '
                'no points on frame %d.', obj_id, imgNo)
            qtw.QMessageBox.critical(self, 'No Points on This Frame',
                f'Object {obj_id} has no annotated points on frame {imgNo}.\n'
                'Hold Ctrl and click on the image to add at least one point '
                'on this frame first.')
            return

        self.logger.info(
            'Starting SAM2 single-image segmentation for object %s on frame '
            '%d (%d point(s))...', obj_id, imgNo, len(points))
        self.button_runSeg_img.setDisabled(True)

        def _on_checkpoint_failed(error_msg):
            self.button_runSeg_img.setEnabled(True)
            qtw.QMessageBox.warning(self, 'SAM2 Checkpoint Download Failed',
                'Could not download the SAM2 model checkpoint - check your '
                'internet connection and see the log console for details.')

        self._ensure_sam2_ready(
            lambda: self._launch_image_segmentation(obj_id, imgNo, points, labels),
            _on_checkpoint_failed)

    def _launch_image_segmentation(self, obj_id, imgNo, points, labels):
        """Build the single-image seg_input.pkl and launch worker_sam.py -
        split out from initiate_image_segmentation() so it can run only
        after _ensure_sam2_ready() confirms the checkpoint is present."""
        pathSave = self.lineEdit_dir_save.text()
        if not os.path.isdir(pathSave):
            os.mkdir(pathSave)
        path_seg = os.path.join(pathSave, 'JPG Images', 'single_image_seg')
        if os.path.isdir(path_seg):
            shutil.rmtree(path_seg)
        os.makedirs(path_seg)

        seg_input = pd.Series({'image': self.imgs_8bit[imgNo], 'points': points, 'labels': labels})
        seg_input.to_pickle(os.path.join(path_seg, 'seg_input.pkl'))

        program, arguments = worker_command('sam', ['image', path_seg, str(obj_id)])
        process_sam = QProcess(self)
        process_sam.setProgram(program)
        process_sam.setArguments(arguments)
        process_sam.readyReadStandardError.connect(lambda:
                            self.handle_error_sam(process_sam, obj_id))
        process_sam.finished.connect(lambda exit_code, exit_status:
                     self.handle_finished_image_sam(
                     process_sam, obj_id, imgNo, exit_code, exit_status))
        process_sam.errorOccurred.connect(lambda error:
                     self.process_failed_image_sam(error, obj_id, imgNo))

        if not hasattr(self, 'running_processes_sam'):
            self.running_processes_sam = {}
        key = f'img_{obj_id}_{imgNo}'
        self.running_processes_sam[key] = process_sam
        process_sam.start()

    def process_failed_image_sam(self, error, obj_id, imgNo):
        self.running_processes_sam.pop(f'img_{obj_id}_{imgNo}', None)
        self.button_runSeg_img.setEnabled(True)
        self.logger.error("[%s] Single-image segmentation QProcess error occurred: %s",
                           obj_id, error)

    def handle_finished_image_sam(self, process, obj_id, imgNo, exit_code, exit_status):
        """SAM2 single-image subprocess completion handler: load the
        resulting mask into `single_mask` at `imgNo` for `obj_id` and
        refresh the canvas."""
        self.running_processes_sam.pop(f'img_{obj_id}_{imgNo}', None)
        self.button_runSeg_img.setEnabled(True)
        self.logger.info(
            "[%s] Single-image segmentation process finished with exit code %s, status %s",
            obj_id, exit_code, exit_status)

        data = process.readAllStandardOutput()
        text = bytes(data).decode("utf-8")
        self.logger.info('text: %s', text)
        try:
            result = json.loads(text.strip())
            if result.get('error') == 'missing_dependency':
                self._show_missing_dependency_dialog(result['message'])
                return
            fn_output = result["path"]
            with np.load(fn_output) as f:
                mask = f['mask']
            # Kept raw (un-eroded) - see apply_edge_mask()/update_canvas().
            if not isinstance(self.df_obj.at[obj_id, 'single_mask'], np.ndarray):
                self.df_obj.at[obj_id, 'single_mask'] = np.zeros(self.imgs_8bit.shape, dtype=bool)
            self.df_obj.loc[obj_id, 'single_mask'][imgNo] = mask
            self.logger.info(
                'SAM2 single-image segmentation completed successfully for '
                'object %s, frame %d.', obj_id, imgNo)
            self.update_canvas(imgNo)
        except json.JSONDecodeError:
            self.logger.error("Could not decode SAM2 single-image segmentation result: %s", text)
            qtw.QMessageBox.warning(self, 'SAM2 Error',
                f'Could not decode SAM2 output. Check console for details.\n'
                f'Raw output (first 200 chars): {text[:200]}')

#%% 3DED
    def activate_3ded_widgets(self, state):
        for wid in self.box_3ded.findChildren(qtw.QWidget):
            if not isinstance(wid, qtw.QLabel):
                wid.setEnabled(state)
    
    def make_rois(self):
        """Compute a bounding-box ROI (x, y, w, h) per frame from each
        object's tracked mask, storing them in df_obj['rois']; frames with no
        True pixels get a (0, 0, 0, 0) placeholder."""
        for obj_id in self.df_obj.index:
            rois = []
            for i_img, mask in enumerate(self.df_obj.loc[obj_id, 'mask']):
                temp = np.where(mask==True)
                try:
                    if temp[0].shape != 0: # no pixel found
                        ymin = temp[0].min()
                        ymax = temp[0].max() +1
                        xmin = temp[1].min()
                        xmax = temp[1].max() +1
                        w = xmax - xmin
                        h = ymax - ymin
                        r = [xmin, ymin, w, h]
                        r = tuple([int(item) for item in r])
                        # rois[i_obj][i_img] = r
                        rois.append(r)
                except ValueError:
                    rois.append((0,0,0,0))
            rois = np.array(rois)
            self.df_obj.at[obj_id, 'rois'] = rois

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
        """Kick off background 3DED extraction for every "used" object:
        resolve the 4D signal files (plain folder listing, or the smart-scan
        match), build one extraction task per (object, frame), and launch
        them via launch_initial_tasks()."""
        self.make_rois()
        
        path_4d = self.lineEdit_dir_4d.text()
        if path_4d == '':
            self.logger.error('3DED extraction cancelled: no 4D signals path entered.')
            qtw.QMessageBox.critical(self, 'No 4D path', 'Please enter a valid path for 4D signals.')
            return
        fns_pattern_4d = None  # parallel per-file pattern-file list, smart-scan only
        if self.checkbox_smartScan.isChecked():
            if self._smart_scan_rows is None:
                self.open_smart_scan_check_dialog()
            if self._smart_scan_rows is None:  # still None: user cancelled the dialog
                return
            resolved = io.resolve_smart_scan_files(self._smart_scan_rows, role='acquisition')
            if not resolved:
                qtw.QMessageBox.warning(self, 'No Files',
                    'No included tilt angle has an acquisition file - check "Check Files..." above.')
                return
            fns_4d = [item['file'] for item in resolved]
            fns_pattern_4d = [item['pattern_file'] for item in resolved]
        else:
            fns_4d = self.resolve_4d_files(path_4d)
        if len(fns_4d) == 0:
            self.logger.error('3DED extraction cancelled: no files found in %s', path_4d)
            qtw.QMessageBox.critical(self, 'Wrong Path', 'No files was found in the path for 4D signals!')
            return
        dtype = os.path.splitext(fns_4d[0])[1]

        if len(self.imgs) != len(fns_4d):
            self.logger.warning(
                'Number of 4D signal files (%d) does not match the number of '
                'navigation images (%d).', len(fns_4d), len(self.imgs))
            reply = qtw.QMessageBox.question(self, 'Mismatch',
                   'No of 4D signals mismatches the number of images. Do you want to continue?',)
            if reply == qtw.QMessageBox.No:
                self.logger.info('3DED extraction cancelled by user after mismatch warning.')
                return
        shape_d_x, shape_d_y = self.get_detector_shape(fns_4d[0])
        scanSize = self.get_scan_size()
        if scanSize is None:  # "Auto": fall back to the loaded nav signal's own shape
            scanSize = tuple(self.imgs.shape[1:])
        
        df = self.df_obj[self.df_obj.use == 1]
        self.tomo_counter = 0
        
        lengths = df.end - [min(df.frame_idx[idx]) for idx in df.index]
        self.tomo_counter_total = np.sum(lengths)
        self.update_progress_bar(0, self.tomo_counter_total)
        self.tic = perf_counter()
        self._3ded_failed = False
        self._cancelling = False
        self.button_cancel.setEnabled(True)
        self.logger.info('Starting 3DED extraction for %d object(s), %d frame(s) total...',
                          len(df), self.tomo_counter_total)

        self.tasks = deque()
        self.temp_dir = self.get_temp_dir()
        for idx in df.index:
            self.df_obj.at[idx, 'dp'] = np.zeros((len(self.imgs), shape_d_x, 
                                                  shape_d_y), dtype='uint32')
            beg = min(df.loc[idx].frame_idx)
            end = df.loc[idx].end
            for i_fr, fn in enumerate(fns_4d[beg:end]):
                i_fr += beg
                fn_pattern = fns_pattern_4d[i_fr] if fns_pattern_4d is not None else None
                self.tasks.append([fn, df.loc[idx, 'rois'][i_fr],
                                   os.path.join(self.temp_dir, f"mask_r{idx}_f{i_fr}.npy"),
                                   dtype, scanSize, (idx, i_fr), fn_pattern or '',
                                   f'({shape_d_x},{shape_d_y})'])
        
        self.max_processes = self.spinbox_threadNum.value()
        self.running_processes = []
        self.process_sam_task_map = {}
        self.process_output_buffers = {}
        self.launch_initial_tasks()
        
# =============================================================================
#         for i_obj in self.masks_video.keys():
#             self.tomo_ds[i_obj] = np.zeros((len(fns_4d), shape_d_x, shape_d_y), dtype='uint32')
#             for i_fr, fn in enumerate(fns_4d):
#                 worker = WorkerThread_General(tr.extract_3ded_mask_single_frame, (i_obj, i_fr),
#                                               fn, self.masks_video[i_obj][i_fr], dtype, scanSize,
#                                               self.rois[i_obj][i_fr])
#                 worker.signals.results.connect(get_tomo_ds)
#                 self.threadpool.start(worker)
# =============================================================================

    def extract_dp_current_frame(self):
        """Compute the diffraction pattern for just the selected object's
        mask at the frame the slider currently points to - the single-frame
        equivalent of "Extract!" (extract_3ded), for quick data-checking.
        Runs inline (WorkerThread_General on self.threadpool) rather than as
        a QProcess, since it's a one-off single frame, not a whole series."""
        try:
            item_selected = self.tree_objects.currentItem()
            obj_id = int(item_selected.text(1))
        except Exception:
            qtw.QMessageBox.warning(self, 'No Object Selected', 'Select an object first.')
            return
        imgNo = self.slider_imgNo.value()

        mask = None
        if not np.all(pd.isna(self.df_obj.loc[obj_id, 'mask'])):
            mask = self.df_obj.loc[obj_id, 'mask'][imgNo]
        elif isinstance(self.df_obj.at[obj_id, 'single_mask'], np.ndarray):
            mask = self.df_obj.loc[obj_id, 'single_mask'][imgNo]
        if mask is None or not mask.any():
            qtw.QMessageBox.warning(self, 'No Mask',
                'No SAM2 mask at the current frame for this object - run "Track" or '
                '"Seg Image" first.')
            return

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        y0, y1 = int(y_idx[0]), int(y_idx[-1])
        x0, x1 = int(x_idx[0]), int(x_idx[-1])
        roi = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

        path_4d = self.lineEdit_dir_4d.text()
        if path_4d == '':
            qtw.QMessageBox.critical(self, 'No 4D path', 'Please enter a valid path for 4D signals.')
            return

        fn_pattern = None
        if self.checkbox_smartScan.isChecked():
            if self._smart_scan_rows is None:
                self.open_smart_scan_check_dialog()
            if self._smart_scan_rows is None:
                return
            resolved = io.resolve_smart_scan_files(self._smart_scan_rows, role='acquisition')
            if imgNo >= len(resolved):
                qtw.QMessageBox.warning(self, 'Frame Out of Range',
                    'The current frame has no matching acquisition file in the smart-scan match.')
                return
            fn = resolved[imgNo]['file']
            fn_pattern = resolved[imgNo]['pattern_file']
        else:
            fns_4d = self.resolve_4d_files(path_4d)
            if imgNo >= len(fns_4d):
                qtw.QMessageBox.warning(self, 'Frame Out of Range',
                    'The current frame has no matching 4D signal file in the folder.')
                return
            fn = fns_4d[imgNo]
        dtype = os.path.splitext(fn)[-1]

        scanSize = self.get_scan_size()
        if scanSize is None:  # "Auto": fall back to the loaded nav signal's own shape
            scanSize = tuple(self.imgs.shape[1:])

        mask = self.apply_edge_mask(mask)

        self.logger.info('Extracting DP for object %d, frame %d (current-frame check)...',
                         obj_id, imgNo)
        self.button_extractCurrentFrame.setDisabled(True)
        worker = WorkerThread_General(load_dp, 0, fn, roi=roi, mask=mask, dtype=dtype,
                                      scanSize=scanSize, fn_pattern=fn_pattern,
                                      det_shape=self.get_detector_shape(fn))
        worker.signals.results.connect(
            lambda dp, _idx, obj_id=obj_id, imgNo=imgNo: self._on_current_frame_dp(dp, obj_id, imgNo))
        worker.signals.error.connect(self._on_current_frame_dp_failed)
        self.threadpool.start(worker)

    def _on_current_frame_dp(self, dp, obj_id, imgNo):
        """WorkerThread_General callback for extract_dp_current_frame(): show
        the one-off DP via update_canvas(), then re-run auto-centering."""
        self.button_extractCurrentFrame.setEnabled(True)
        if hasattr(dp, 'compute'):
            dp = dp.compute()
        # Routed through update_canvas() (rather than drawn directly here)
        # so it's shown/cleared the exact same way as every other frame -
        # moving the slider (or changing the object selection) away from
        # (obj_id, imgNo) then correctly reverts to whatever update_canvas
        # would normally show, instead of this one-off result staying
        # plotted indefinitely.
        self._current_frame_dp_preview = {'obj_id': obj_id, 'imgNo': imgNo, 'dp': dp}
        self.update_canvas(imgNo=imgNo, obj_id=obj_id)
        # This is freshly-computed data the auto-centering circles have
        # never seen - re-run it now if enabled, same as after a full
        # "Extract!" (see handle_finished_3ded's identical pair of calls).
        self.add_scalebar()
        self.logger.info('Current-frame DP extraction complete (object %d, frame %d).',
                         obj_id, imgNo)

    def _on_current_frame_dp_failed(self, traceback_text, _idx):
        self.button_extractCurrentFrame.setEnabled(True)
        self.logger.error('Current-frame DP extraction failed:\n%s', traceback_text)
        qtw.QMessageBox.warning(self, 'Extraction Failed',
            f'Could not extract the diffraction pattern:\n{traceback_text[-500:]}')

    def get_temp_dir(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(os.path.dirname(script_dir), 'EDyssey', 'io_utils', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        self.logger.info('temp dir: %s', temp_dir)
        return temp_dir

    def save_mask_to_temp(self, temp_dir, mask_array, r_id, i_fr):
        filename = f"mask_r{r_id}_f{i_fr}.npy"
        filepath = os.path.join(temp_dir, filename)
        np.save(filepath, mask_array)
        return filepath

    def launch_initial_tasks(self):
        for _ in range(min(self.max_processes, len(self.tasks))):
            self.launch_next_task()
    
    def launch_next_task(self):
        """Pop the next queued extraction task (if under max_processes) and
        launch it as an extract_frame subprocess."""
        if not self.tasks or len(self.running_processes) >= self.max_processes:
            return
    
        args = self.tasks.popleft()
        # args = [fn, roi, mask_path, dtype, scanSize, (idx, i_fr), fn_pattern,
        # det_shape] - (idx, i_fr) is third-to-last (fn_pattern/det_shape,
        # always present since extract_3ded() appends them unconditionally,
        # must stay last two to land in extract_3ded_mask_single_frame's
        # matching positional slots).
        *_, (idx, i_fr), _fn_pattern, _det_shape = args
        mask = self.apply_edge_mask(self.df_obj.loc[idx, 'mask'][i_fr])
        _ = self.save_mask_to_temp(self.temp_dir, mask, idx, i_fr)
        
        program, arguments = worker_command('extract_frame', args)
        process = QProcess()
        process.setProgram(program)
        process.setArguments(arguments)
        process.readyReadStandardOutput.connect(lambda: self._accumulate_output_3ded(process))
        process.readyReadStandardError.connect(lambda: self.handle_error_3ded(process))
        process.finished.connect(lambda: self.handle_finished_3ded(process, idx))
        process.errorOccurred.connect(self.process_failed_3ded)


        self.running_processes.append(process)
        self.process_sam_task_map[process] = args
        self.process_output_buffers[process] = bytearray()
        process.start()
        
    def process_failed_3ded(self, error):
        if self._cancelling:
            return
        self._3ded_failed = True
        self.logger.error("QProcess error occurred: %s", error)
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        qtw.QMessageBox.critical(self, 'Process Error',
            f'A worker process failed to start (error code {error}).\n'
            'Check that Python is on PATH and worker_extract_frame.py exists.')

    def handle_error_3ded(self, process):
        # worker_extract_frame.py loads tpx3 via eventem, whose progress bar
        # (and any other routine diagnostics) writes straight to stderr on
        # every run, success or failure - this is not itself an error (see
        # ProcessStderrBuffer).
        if self._cancelling:
            return
        self._stderr_buffer.log_info(process, self.logger, 'Worker')

    def _accumulate_output_3ded(self, process):
        """QProcess.readyReadStandardOutput handler: append this chunk to
        the process's buffer instead of decoding it directly - the
        base64+pickle payload for anything but the smallest diffraction
        pattern routinely spans more than one OS pipe delivery, so decoding
        straight out of a single readyReadStandardOutput signal (as this
        used to) intermittently truncated it and failed to unpickle (e.g.
        "invalid load key"). Actual decoding happens once, in
        handle_finished_3ded, from the complete accumulated buffer."""
        self.process_output_buffers[process] += process.readAllStandardOutput().data()

    def handle_finished_3ded(self, process, idx):
        """extract_frame subprocess completion handler: decode the complete
        accumulated stdout (a base64-pickled (image, "idx,i_fr") tuple - see
        _accumulate_output_3ded), store the extracted DP into df_obj,
        advance the progress bar, and once every task has returned, clean up
        the temp dir, mark all objects as extracted, refresh the canvas, and
        autosave if enabled."""
        if process in self.running_processes:
            self.running_processes.remove(process)
        task_info = self.process_sam_task_map.pop(process, None)
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
            self.logger.exception('Failed to decode 3DED extraction subprocess output.')
        else:
            if task_info is None:
                self.logger.warning("Unknown process")
            else:
                img, r_id = result_array
                idx_r, i_fr = ast.literal_eval(r_id)
                self.df_obj.at[idx_r, 'dp'][i_fr] = img

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
            for idx in self.df_obj[self.df_obj.use == 1].index:
                self.toggle_tree_icon(self.df_obj.index.get_loc(idx), 'ext', True)
            self.update_canvas()
            # Freshly-extracted DPs may have a different center than whatever
            # was last found - re-run auto-centering now if enabled.
            self.add_scalebar()
            self.button_cancel.setDisabled(True)
            if self.checkbox_autosave.isChecked():
                self.save_results()
        else:
            self.launch_next_task()  # trigger next task if any left

    def set_threadNo(self, value):
        self.threadpool.setMaxThreadCount(value)
        
    def disable_3ded_widgets(self, state):
        # button_cancel now lives inside box_3ded too (see init_ui) but
        # must stay independent of this sweep - it needs to stay clickable
        # regardless of tracking/segmentation/extraction state, managed by
        # its own enable/disable calls elsewhere.
        for wid in self.box_3ded.findChildren(qtw.QWidget):
            if isinstance(wid, qtw.QLabel) or wid is self.button_cancel:
                continue
            wid.setDisabled(state)
    
    def update_progress_bar(self, value, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f'%v / {total}')
    
#%% Save Data
    def on_makePets2_toggled(self, state):
        """When "Make *.pts2" is checked (unchecking is a no-op here), open
        the PETS2 params dialog to collect the values needed to write a
        .pts2 project file on save."""
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
        """Open Pets2ParamsDialog pre-filled with voltage/exposure/pixel-size
        read from the current metadata and UI fields; unchecks "Make *.pts2"
        again if the user cancels."""
        voltage_kv = None
        try:
            path_main = self.metadata_path_override or self.lineEdit_dir_4d.text()
            metadata = io.get_metadata(path_main, count=self.spinbox_metadataCount.value())
            if 'Voltage' in metadata:
                voltage_kv = metadata['Voltage']
        except Exception:
            self.logger.debug('Could not auto-fill voltage from metadata; leaving it blank.',
                               exc_info=True)
        exposure_s = self.spinbox_dwellTime_acquisition.value() / 1e6
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
        """Save results via _save_results_impl, logging success/failure with elapsed time."""
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
            'generation for each object continues asynchronously).',
            io.format_duration_hms(perf_counter() - tic))

    def _save_results_impl(self):
        """Write a timestamped analysis folder: per-object tracking data
        (points/labels/ROIs/masks), extracted diffraction patterns (as .npy
        and .hspy, plus an optional PETS2 project file), and
        background-rendered preview clips for each object's DP series and
        tracked mask."""
        path_save = self.lineEdit_dir_save.text()
        if not os.path.isdir(path_save):
            os.mkdir(path_save)

        date = datetime.date.today()
        tim = datetime.datetime.now().strftime("%H-%M-%S")

        path_save = os.path.join(path_save, f'{date}__{tim}')
        os.mkdir(path_save)
        self.logger.info('Saving results to %s...', path_save)

        # Navigation signal: rather than re-copying the (potentially large)
        # signal into every saved-analysis folder, just record the path it
        # was loaded from - "Load Saved Analysis" reloads from there.
        io.save_analysis_info(path_save, getattr(self, 'fn_navSignal', None), analysis_type='sam2')

        # tracking results, rois, dp
        for idx in self.df_obj.index:
            path_save_objID = os.path.join(path_save, f'roi No {idx}')
            os.mkdir(path_save_objID)

            df = self.df_obj.loc[idx, ['use', 'idx', 'frame_idx', 'points', 'labels',
                                       'end']]
            df['edge_detection'] = [('enabled', self.checkbox_edgeOnly.isChecked()),
                                    ('kernel_size', self.spinbox_edgeKernel.value()),
                                    ('directional', self.checkbox_edgeDirectional.isChecked()),
                                    ('direction_deg', self.spinbox_edgeDirection.value()),
                                    ('revert', self.checkbox_revertMask.isChecked())]
            df.to_json(os.path.join(path_save_objID, f'roi No {idx}.json'), orient='index', indent=4)
            if not (np.all(pd.isna(self.df_obj.loc[idx, 'rois']))):
                np.save(os.path.join(path_save_objID, 'rois.npy'),
                    self.df_obj.loc[idx, 'rois'])
            if not (np.all(pd.isna(self.df_obj.loc[idx, 'dp']))):
                # Saved as actually used for extraction (edge-view applied,
                # when "Edge Only" is checked) - see apply_edge_mask_stack().
                np.save(os.path.join(path_save_objID, 'output_mask.npy'),
                        self.apply_edge_mask_stack(self.df_obj.loc[idx, 'mask']))

            # write frames
            if not (np.all(pd.isna(self.df_obj.loc[idx, 'dp']))):
                dp = self.df_obj.loc[idx, 'dp']
                np.save(os.path.join(path_save_objID, '3DED.npy'), dp)
                # Also save as a hyperspy signal so "Load Saved Analysis" can
                # restore the diffraction patterns via hs.load(...).
                hs.signals.Signal2D(dp).save(
                    os.path.join(path_save_objID, '3DED.hspy'), overwrite=True)
                path_pets = os.path.join(path_save_objID, 'pets')
                os.mkdir(path_pets)
                fld_frames = os.path.join(path_pets, 'frames')
                worker_frames = WorkerThread_General(io.create_frames, 0,
                                 fld_frames, self.df_obj.loc[idx, 'dp'])
                self.threadpool.start(worker_frames)

                # clip dp
                scale_recip = self.lineEdit_scale_recip.text()
                try:
                    scale_recip = float(scale_recip)
                except ValueError:
                    scale_recip = None

                if self.checkbox_makePets2.isChecked() and self.pets2_params is not None:
                    io.write_pts2(os.path.join(path_pets, f'Roi Num {idx}.pts2'), n_frames=dp.shape[0],
                                  frame_shape=dp.shape[1:], roi_id=idx, **self.pets2_params)

                fn_clip_dp = os.path.join(path_save_objID, 'tomo clip')
                worker_clip_dp = WorkerThread_General(io.create_clip_dp, 0, fn_clip_dp,
                                self.df_obj.loc[idx, 'dp'], scale_recip, center=self.dp_center,
                                fps=self.spinbox_fps.value(), logger=self.logger)
                self.threadpool.start(worker_clip_dp)

            # clip tracking
            if not (np.all(pd.isna(self.df_obj.loc[idx, 'mask']))):
                mask_effective = self.apply_edge_mask_stack(self.df_obj.loc[idx, 'mask'])
                np.save(os.path.join(path_save_objID, f'segmentation masks_ obj ID {idx}.npy'),
                        mask_effective)
                scale_real = self.lineEdit_scale_real.text()
                try:
                    scale_real = float(scale_real)
                except ValueError:
                    scale_real = None
                fn_clip_tracking = os.path.join(path_save_objID, 'tracking clip')
                worker_tracking = WorkerThread_General(
                    io.create_clip_tracking_with_mask, 0,
                    fn_clip_tracking, self.imgs,
                    mask_effective, idx, scale_real,
                    fps=self.spinbox_fps.value(), cmap='Grays_r', logger=self.logger)
                self.threadpool.start(worker_tracking)
    
    def cancel_running_work(self):
        """Stop SAM2 tracking/segmentation or 3DED extraction and suppress
        the error popups that killing those workers would otherwise
        trigger.

        QThreadPool has no way to forcibly interrupt a runnable that has
        already started (only queued-but-not-started ones can be dropped),
        so an in-flight helper job will still finish in the background and
        is simply ignored when it does. The SAM2/3DED subprocesses are real
        OS processes though, so those are actually killed outright."""
        self._cancelling = True
        self.threadpool.clear()
        n_killed = len(getattr(self, 'running_processes_sam', {}))
        n_killed += len(getattr(self, 'running_processes', []))
        self.stop_processes()  # kills any running_processes_sam (SAM2 tracking/segmentation)
        for process in getattr(self, 'running_processes', []):
            process.kill()
        if hasattr(self, 'running_processes'):
            self.running_processes.clear()
        if hasattr(self, 'tasks'):
            self.tasks.clear()
        if hasattr(self, 'process_sam_task_map'):
            self.process_sam_task_map.clear()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.button_cancel.setDisabled(True)
        self.button_runSeg_clip.setEnabled(True)
        self.button_runSeg_img.setEnabled(True)
        self.logger.warning('Cancelled by user (%d running worker process(es) killed).', n_killed)
        qtw.QMessageBox.information(self, 'Cancelled',
            'Tracking/segmentation/3DED extraction was cancelled.\n\n'
            'Any helper job already running in the background will still '
            'finish silently - only queued work and the worker processes were stopped.')

    def cleanup(self):
        """Release resources held by this tab. Called by MainWindow.closeEvent
        so repeated runs of the app in the same console/kernel don't leave
        threadpools, running subprocesses, and matplotlib figures alive."""
        self.threadpool.clear()
        self.stop_processes()  # kills any running_processes_sam
        for process in getattr(self, 'running_processes', []):
            process.kill()
        self.log_console.disconnect_log()
        plt.close(self.figure)

    def closeEvent(self,event):
        # empty_cache()
        self.cleanup()
        gc.collect()
        event.accept()
        # app.exit()

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    
    # Create the main window and show it
    window = Tab_SAM2()
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec_())
