# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:43:00 2024

@author: SGholam
"""

import json
import sys
import tempfile
import hyperspy.api as hs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QProcess, QTimer
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QKeySequence
from PyQt5.QtWidgets import QShortcut
import numpy as np
import os
import re
from PIL import Image
import gc
import pickle
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
from .base_tab import (TabBase, get_existing_directory, resolve_hdf5_dtype, glob_ext_for_dtype,
                       HDF5_EVENTEM_LABEL)
from .clipping_thresholds import ClippingThresholdsWidget
from .pets2_dialog import Pets2ParamsDialog
from .transposed_object_table import TransposedObjectTable
from .frame_flag_bar import FrameFlagBar
from .smart_scan_dialog import SmartScanCheckDialog
from .mask_edit_dialog import MaskEditDialog
from .sam2_auto_detector_widget import SAM2AutoDetectorWidget
from .ribbon import RibbonPanel, RibbonTool
from worker_extract_frame import load_dp
import worker_pool_utils as wpu
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
        # Cached "clean" background (everything except the nav/seg image
        # data and their point/mask overlays) used to blit the cheap
        # denoise/contrast preview refresh instead of a full canvas redraw -
        # see _blit_current_frame_display(). None means "needs (re)capture";
        # invalidated by update_canvas() (covers everything this fast path
        # doesn't itself touch) and the couple of other full-figure redraws
        # outside it (on_scroll's zoom, add_scalebar).
        self._denoise_bg = None

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

        button_w = 100
        button_h_lrg = 25

        #%% ribbon (top parameter ribbon, Word-style - see Tab_ROI_on_4D for
        # the original design, and TabBase for the shared helpers). Same
        # column shape as Tab_Tracking_CV2 (this tab's structure is nearly
        # identical), minus the Threshold/Deviation rows SAM2 doesn't need.
        ribbon_page = qtw.QWidget()
        self.ribbon_page = ribbon_page  # exposed for the Edit tab's ribbon-text-scale control
        layout_ribbon = qtw.QHBoxLayout(ribbon_page)
        layout_ribbon.setContentsMargins(4, 2, 4, 2)
        layout_ribbon.setSpacing(2)
        self._main_splitter = qtw.QSplitter(Qt.Vertical)
        self._main_splitter.addWidget(ribbon_page)
        self.layout.addWidget(self._main_splitter, 1)

        #%% Files (ribbon column)
        self.box_dir, layout_dir = self._ribbon_group_start(layout_ribbon, stretch=0)

        # nav signal dir
        layout_dir_entry = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_entry)
        label_dir = qtw.QLabel('Nav. Signal')
        layout_dir_entry.addWidget(label_dir)
        label_dir.setFixedWidth(60)
        
        self.lineEdit_dir_navSignal = qtw.QLineEdit()
        self.lineEdit_dir_navSignal.setFixedWidth(340)
        layout_dir_entry.addWidget(self.lineEdit_dir_navSignal)
        
        self.button_dir_navSignal = qtw.QPushButton('...')
        self.button_dir_navSignal.setFixedWidth(30)
        layout_dir_entry.addWidget(self.button_dir_navSignal)
        self.button_dir_navSignal.clicked.connect(lambda: self.show_dialog('file'))
        
        # 4d dir
        layout_dir_4dSignals = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_4dSignals)
        
        label_dir_4d = qtw.QLabel('4D Signals')
        layout_dir_4dSignals.addWidget(label_dir_4d)
        label_dir_4d.setFixedWidth(60)
        
        self.lineEdit_dir_4d = qtw.QLineEdit()
        layout_dir_4dSignals.addWidget(self.lineEdit_dir_4d)
        
        self.button_dir_4dSignals = qtw.QPushButton('...')
        self.button_dir_4dSignals.setFixedWidth(30)
        layout_dir_4dSignals.addWidget(self.button_dir_4dSignals)
        self.button_dir_4dSignals.clicked.connect(lambda: self.show_dialog('folder'))

        self.combo_dtype_4d = qtw.QComboBox()
        self.combo_dtype_4d.setMaximumWidth(110)
        self.combo_dtype_4d.addItems(['.tpx3', HDF5_EVENTEM_LABEL, '.hdf5', '.hspy', '.zspy',
                                      '.mib', '.blo', 'All Files'])
        self.combo_dtype_4d.setToolTip(
            'Data type of the 4D signal files - filters out stray non-signal files '
            '(comment.txt, pattern files, logs), AND (for a .hdf5 file specifically) '
            f'selects which of the two loaders to use - "{HDF5_EVENTEM_LABEL}" (eventem\'s '
            'own raw export layout) or plain ".hdf5" (a conventional/third-party '
            'HDF5 file, loaded via HyperSpy - both commonly share the same on-disk '
            '.hdf5 extension, so this choice is otherwise ambiguous). Ignored if the '
            'navigator\'s own recorded file list applies to this folder.')
        layout_dir_4dSignals.addWidget(self.combo_dtype_4d)

        # save dir
        layout_dir_save = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_save)
        
        label_dir_save = qtw.QLabel('Save Dir.')
        layout_dir_save.addWidget(label_dir_save)
        label_dir_save.setFixedWidth(60)
        
        self.lineEdit_dir_save = qtw.QLineEdit()
        layout_dir_save.addWidget(self.lineEdit_dir_save)
        
        self.button_dir_save = qtw.QPushButton('...')
        layout_dir_save.addWidget(self.button_dir_save)
        self.button_dir_save.clicked.connect(lambda: self.show_dialog('folder'))
        #%% load buttons (scale bars now live in the Input Parameters box below)
        self.double_validator = QDoubleValidator(0.0, 1e5, 5)

        # Acquisition Dwell T. - moved here from Input Parameters (mirrors
        # Navigator's convention: dwell time lives with the File/Smart Scan
        # controls, not with Detector/Scan Size). 3DED extraction always
        # reads the acquisition (smart-scanned) file when Smart Scanned is
        # checked (see the Smart Scan groupbox below), so a single dwell
        # time covers both the smart-scan and plain cases.
        layout_dwell_row = qtw.QHBoxLayout()
        label_dwellTime = qtw.QLabel('Acquisition Dwell T. (μs)')
        label_dwellTime.setToolTip('Dwell time in microseconds')
        self.spinbox_dwellTime_acquisition = qtw.QSpinBox()
        self.spinbox_dwellTime_acquisition.setFixedWidth(70)
        self.spinbox_dwellTime_acquisition.setRange(1, 99999999)
        layout_dwell_row.addWidget(label_dwellTime)
        layout_dwell_row.addWidget(self.spinbox_dwellTime_acquisition)
        layout_dwell_row.addStretch(1)
        layout_dir.addLayout(layout_dwell_row)

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
            'Smart-scanned series - reads each frame\'s acquisition + pattern file, '
            'not every raw file in the folder.')
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

        self.button_checkSmartScanFiles = qtw.QPushButton('Check Files')
        self.button_checkSmartScanFiles.setToolTip(
            'Review/fix the automatic per-frame file match before extracting')
        self.button_checkSmartScanFiles.setDisabled(True)
        self.button_checkSmartScanFiles.clicked.connect(self.open_smart_scan_check_dialog)
        layout_smartScan.addWidget(self.button_checkSmartScanFiles, 1, 0)

        self.lineEdit_detectionDir = qtw.QLineEdit()
        self.lineEdit_detectionDir.setPlaceholderText('Detect. Dir. (defaults to 4D Signals folder)')
        self.lineEdit_detectionDir.setDisabled(True)
        self.lineEdit_detectionDir.setToolTip(
            'Folder to look for detection files in, if different from the 4D Signals folder')
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
        self.box_scanSize, layout_box_experiment = self._ribbon_group_start(layout_ribbon, stretch=0)

        self.dp_center = None  # (x, y) - auto-found or last manually-clicked center
        # id(dp_array) at the time dp_center was last auto-found - lets
        # add_scalebar() skip re-running find_dp_center_blurred (a real
        # HyperSpy call) when the displayed DP hasn't actually changed
        # since, e.g. on every keystroke in the scale-recip field. See
        # _on_auto_center_toggled.
        self._dp_center_cache_key = None

        # Detector Size/Scan Size/Metadata/Scales each get their own
        # QGroupBox, arranged 2x2 (mirrors Tab_ROI_on_4D/Navigator's Input
        # Parameters column exactly): Detector Size beside Scan Size,
        # Metadata beside Scales below them.
        layout_exp_groups = qtw.QGridLayout()
        spin_w_size = 55  # detector/scan size spinboxes only hold a handful of digits

        #### Detector Size
        self.groupbox_detectorSize = qtw.QGroupBox('Detector Size')
        layout_detSize = qtw.QHBoxLayout(self.groupbox_detectorSize)
        detSize_tooltip = (
            'Detector size in pixels - auto-detected for most formats; .tpx3 needs it '
            'set explicitly (Auto assumes 512x512).')
        self.checkbox_detectorSizeAuto = qtw.QCheckBox('Auto')
        self.checkbox_detectorSizeAuto.setChecked(True)
        self.checkbox_detectorSizeAuto.setToolTip(detSize_tooltip)
        layout_detSize.addWidget(self.checkbox_detectorSizeAuto)
        self.spinbox_detectorSize_x = qtw.QSpinBox()
        self.spinbox_detectorSize_x.setRange(1, 8192)
        self.spinbox_detectorSize_x.setValue(512)
        self.spinbox_detectorSize_x.setFixedWidth(spin_w_size)
        layout_detSize.addWidget(self.spinbox_detectorSize_x)
        layout_detSize.addWidget(qtw.QLabel('×', alignment=Qt.AlignCenter))
        self.spinbox_detectorSize_y = qtw.QSpinBox()
        self.spinbox_detectorSize_y.setRange(1, 8192)
        self.spinbox_detectorSize_y.setValue(512)
        self.spinbox_detectorSize_y.setFixedWidth(spin_w_size)
        layout_detSize.addWidget(self.spinbox_detectorSize_y)
        layout_exp_groups.addWidget(self.groupbox_detectorSize, 0, 0)
        self.activate_detectorSize_spinboxes()
        self.checkbox_detectorSizeAuto.stateChanged.connect(self.activate_detectorSize_spinboxes)

        #### Scan Size
        self.groupbox_scanSize = qtw.QGroupBox('Scan Size')
        layout_scanSizeBox = qtw.QHBoxLayout(self.groupbox_scanSize)
        self.checkbox_scanSize = qtw.QCheckBox('Auto')
        self.checkbox_scanSize.setChecked(True)
        self.checkbox_scanSize.setToolTip(
            'Scan size from the loaded navigation signal - uncheck to override manually')
        layout_scanSizeBox.addWidget(self.checkbox_scanSize)
        self.spinbox_scanSize_x = qtw.QSpinBox()
        self.spinbox_scanSize_x.setFixedWidth(spin_w_size)
        self.spinbox_scanSize_x.setRange(1, 99999)
        layout_scanSizeBox.addWidget(self.spinbox_scanSize_x)
        layout_scanSizeBox.addWidget(qtw.QLabel('×', alignment=Qt.AlignCenter))
        self.spinbox_scanSize_y = qtw.QSpinBox()
        self.spinbox_scanSize_y.setFixedWidth(spin_w_size)
        self.spinbox_scanSize_y.setRange(1, 99999)
        layout_scanSizeBox.addWidget(self.spinbox_scanSize_y)
        layout_exp_groups.addWidget(self.groupbox_scanSize, 0, 1)
        self.activate_lineEdit_scanSize()
        self.checkbox_scanSize.stateChanged.connect(self.activate_lineEdit_scanSize)

        #### Metadata (comment.txt) auto-fill - Load/Browse/View on row 0,
        # Block # on row 1 (keeps this box narrow, matching Navigator/
        # Tab_ROI_on_4D).
        self.groupbox_metadata = qtw.QGroupBox('Metadata')
        layout_metadata = qtw.QGridLayout(self.groupbox_metadata)
        self.button_loadMetadata = qtw.QPushButton('Load')
        self.button_loadMetadata.setToolTip(
            'Fill scan size/dwell time from comment.txt (tpx3 only)')
        layout_metadata.addWidget(self.button_loadMetadata, 0, 0)
        self.button_loadMetadata.clicked.connect(lambda: self.load_metadata(silent=False))

        self.button_browseMetadata = qtw.QPushButton('...')
        self.button_browseMetadata.setFixedWidth(30)
        self.button_browseMetadata.setToolTip('Browse for the metadata file')
        layout_metadata.addWidget(self.button_browseMetadata, 0, 1)
        self.button_browseMetadata.clicked.connect(self.browse_metadata_file)

        self.button_viewMetadata = qtw.QPushButton('View...')
        self.button_viewMetadata.setToolTip('View the full raw comment.txt content')
        layout_metadata.addWidget(self.button_viewMetadata, 0, 2)
        self.button_viewMetadata.clicked.connect(self.show_metadata_dialog)

        label_metadataCount = qtw.QLabel('Block #')
        label_metadataCount.setToolTip(
            'Which metadata block to read (enabled if comment.txt logs more than one)')
        layout_metadata.addWidget(label_metadataCount, 1, 0)
        self.spinbox_metadataCount = qtw.QSpinBox()
        self.spinbox_metadataCount.setFixedWidth(50)
        self.spinbox_metadataCount.setRange(0, 99999)
        self.spinbox_metadataCount.setValue(0)
        self.spinbox_metadataCount.setDisabled(True)  # re-enabled once >1 block is found
        # Re-reads comment.txt (a cheap text-file parse, not the 4D data
        # file itself) for the newly-selected block as soon as the value
        # changes, instead of requiring an extra "Load" click every time.
        self.spinbox_metadataCount.valueChanged.connect(lambda: self.load_metadata(silent=True))
        layout_metadata.addWidget(self.spinbox_metadataCount, 1, 1)
        layout_exp_groups.addWidget(self.groupbox_metadata, 1, 0)

        self.metadata_path_override = None  # set by browse_metadata_file(); cleared on new 4D folder

        # 4D signal file list recorded by the navigator tab's own
        # metadata.json (see apply_nav_signal_metadata/resolve_4d_files) -
        # None until a nav signal with a sibling metadata.json is loaded, or
        # invalidated by a manually-browsed 4D folder.
        self._nav_4d_files = None
        self._nav_4d_directory = None

        #### Scales - Real (row 0), then Recip. + "Center" (row 1) - moved
        # into a groupbox, matching Tab_ROI_on_4D/Navigator.
        self.groupbox_scales = qtw.QGroupBox('Scales')
        layout_scales = qtw.QGridLayout(self.groupbox_scales)
        layout_scales.addWidget(qtw.QLabel('Real (nm)'), 0, 0)
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        self.lineEdit_scale_real.setMaximumWidth(70)
        layout_scales.addWidget(self.lineEdit_scale_real, 0, 1)

        layout_scales.addWidget(qtw.QLabel('Recip. (Å<sup>-1</sup>)'), 1, 0)
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        self.lineEdit_scale_recip.setValidator(self.double_validator)
        self.lineEdit_scale_recip.setMaximumWidth(70)
        layout_scales.addWidget(self.lineEdit_scale_recip, 1, 1)

        self.button_centerRecip = qtw.QPushButton('Center')
        self.button_centerRecip.setToolTip(
            'Find the beam center now, or Ctrl+Click the DP plot to set it manually')
        self.button_centerRecip.clicked.connect(self.find_and_center_recip)
        layout_scales.addWidget(self.button_centerRecip, 1, 2)
        layout_exp_groups.addWidget(self.groupbox_scales, 1, 1)
        self.lineEdit_scale_recip.textChanged.connect(self.add_scalebar)
        self.lineEdit_scale_real.textChanged.connect(self.add_scalebar)

        layout_box_experiment.addLayout(layout_exp_groups)

        #### Load buttons - added BEFORE _ribbon_group_end() below (not
        # after) - that call adds this column's "Input Parameters" caption
        # label right where it's called, so calling it right after the
        # Input Parameters grid but before these buttons would sandwich the
        # caption between the grid and the buttons, reading as unwanted
        # blank space/separation between them instead of one clean column
        # with its caption at the very bottom, like every other ribbon
        # column (see the identical fix on ROI Tracker's Input Parameters
        # column).
        layout_loadSignal = qtw.QHBoxLayout()
        layout_box_experiment.addLayout(layout_loadSignal)

        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        # self.button_loadNavigation.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        self.button_loadNavigation.setFixedSize(button_w, button_h_lrg*2)
        layout_loadSignal.addWidget(self.button_loadNavigation, alignment=Qt.AlignCenter)
        self.button_loadNavigation.clicked.connect(self.load_navSignal)

        self.button_loadSavedAnalysis = qtw.QPushButton('Load Saved\nAnalysis')
        # self.button_loadSavedAnalysis.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        self.button_loadSavedAnalysis.setFixedSize(button_w, button_h_lrg*2)
        layout_loadSignal.addWidget(self.button_loadSavedAnalysis, alignment=Qt.AlignCenter)
        self.button_loadSavedAnalysis.clicked.connect(self.load_saved_analysis)

        self._ribbon_group_end(layout_ribbon, layout_box_experiment, 'Input Parameters', stretch=True)
        
        #%% Tracking / Extract
        # Edge Detection used to live here as a tab-wide control - it's now
        # per-object, set only from the Fine-Tune Mask dialog's Segments
        # feature (same as Mesh/Dilate-Erode already were - see
        # apply_edge_mask/_edge_settings_for). Tracking - moved here from
        # the bottom of the left object-list panel (below tree_objects),
        # directly above Extract in the same stacked ribbon column, per
        # user request.
        # Fixed width (rather than sizing to content) - matches ROI
        # Tracker's own Threshold/Tracking/Extract column so the two tabs'
        # ribbons don't visibly shift width against each other.
        self.box_3ded, layout_box_3ded = self._ribbon_group_start(layout_ribbon, stretch=0, width=320)

        #%% Tracking (first sub-section in this combined column)
        layout_box_tracking = layout_box_3ded
        layout_sam_buttons_1 = qtw.QHBoxLayout()
        layout_box_tracking.addLayout(layout_sam_buttons_1)
        layout_sam_buttons_2 = qtw.QHBoxLayout()
        layout_box_tracking.addLayout(layout_sam_buttons_2)

        # image
        self.button_runSeg_img = qtw.QPushButton('Seg Image', self)
        # self.button_runSeg_img.setFixedSize(button_w, button_h_lrg)
        layout_sam_buttons_1.addWidget(self.button_runSeg_img)
        # self.button_runSeg_img.clicked.connect(self.SAM2_image_predictor)
        self.button_runSeg_img.clicked.connect(self.initiate_image_segmentation)
        self.button_runSeg_img.setDisabled(True)

        # num
        layout_stack = qtw.QVBoxLayout()
        layout_sam_buttons_2.addLayout(layout_stack)
        layout_stack_top = qtw.QHBoxLayout()
        layout_stack.addLayout(layout_stack_top)

        label_stackNum = qtw.QLabel('Stack Num')
        label_stackNum.setToolTip('Frames per SAM2 stack - lower = less GPU memory, more processes')
        layout_stack_top.addWidget(label_stackNum)
        self.spinbox_stackNum = qtw.QSpinBox()
        self.spinbox_stackNum.setMaximumWidth(80)
        self.spinbox_stackNum.setToolTip('Frames per SAM2 stack')
        layout_stack_top.addWidget(self.spinbox_stackNum)
        self.spinbox_stackNum.setSingleStep(25)

        # Stack navigation buttons (jump to a stack's first frame) - stacked
        # below the Stack Num row itself, in the same column, rather than
        # under the slider beside the canvas (moved here per user request).
        layout_stack_nav = qtw.QHBoxLayout()
        layout_stack.addLayout(layout_stack_nav)
        label_stacks_nav = qtw.QLabel('Stacks:')
        label_stacks_nav.setToolTip('Jump to a stack\'s first frame - each stack needs at least one point')
        layout_stack_nav.addWidget(label_stacks_nav)

        self._stack_scroll = qtw.QScrollArea()
        self._stack_scroll.setWidgetResizable(True)
        self._stack_scroll.setFixedHeight(36)
        self._stack_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._stack_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._stack_scroll.setFrameShape(qtw.QFrame.NoFrame)
        layout_stack_nav.addWidget(self._stack_scroll)

        self._stack_buttons_widget = qtw.QWidget()
        self._stack_buttons_layout = qtw.QHBoxLayout(self._stack_buttons_widget)
        self._stack_buttons_layout.setContentsMargins(2, 2, 2, 2)
        self._stack_buttons_layout.setSpacing(3)
        self._stack_scroll.setWidget(self._stack_buttons_widget)

        self.label_stack = qtw.QLabel('')
        layout_box_tracking.addWidget(self.label_stack)
        self.spinbox_stackNum.valueChanged.connect(self.update_stack_guide)

        # clip
        self.button_runSeg_clip = qtw.QPushButton('Track', self)
        # self.button_runSeg_clip.setFixedSize(button_w, button_h_lrg)
        layout_sam_buttons_1.addWidget(self.button_runSeg_clip)
        self.button_runSeg_clip.clicked.connect(self.initiate_video_segmentation)
        self.button_runSeg_clip.setEnabled(False)

        for wid in layout_sam_buttons_1.findChildren(qtw.QWidget):
            wid.setDisabled(True)
        for wid in layout_sam_buttons_2.findChildren(qtw.QWidget):
            wid.setDisabled(True)
        self._ribbon_group_end(layout_ribbon, layout_box_tracking, 'Tracking', separator=False, stretch=False)

        sep_extract = qtw.QFrame()
        sep_extract.setFrameShape(qtw.QFrame.HLine)
        sep_extract.setFrameShadow(qtw.QFrame.Sunken)
        layout_box_3ded.addWidget(sep_extract)

        #%% Extract
        layout_threadNum = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_threadNum)

        label_threadNo = qtw.QLabel('CPU Cores')
        layout_threadNum.addWidget(label_threadNo)
        self.spinbox_threadNum = qtw.QSpinBox(self)
        self.spinbox_threadNum.setMaximumWidth(80)
        layout_threadNum.addWidget(self.spinbox_threadNum)
        self.spinbox_threadNum.setRange(1, os.cpu_count() or 1)
        self.spinbox_threadNum.setValue(2)
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

        # Lets the user preview/adjust the PETS2 export parameters at any
        # time - not just the one moment the checkbox is first checked
        # (on_makePets2_toggled's own trigger) - e.g. to double-check them
        # ahead of a run, or tweak something after the fact without having
        # to uncheck-then-recheck the box to reopen the dialog.
        self.button_checkPets2Options = qtw.QPushButton('Check Options...')
        self.button_checkPets2Options.setToolTip(
            'Open the PETS2 export parameters dialog to review/edit them, '
            'without needing to uncheck and recheck "Make *.pts2"')
        self.button_checkPets2Options.clicked.connect(
            lambda: self._open_pets2_dialog(uncheck_on_cancel=False))
        layout_saveOptions.addWidget(self.button_checkPets2Options)

        layout_saveOptions.addStretch()

        #### Buttons
        layout_extract_button = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_extract_button)
        self.button_3ded = qtw.QPushButton('Extract All')
        self.button_3ded.setFixedHeight(button_h_lrg)
        layout_extract_button.addWidget(self.button_3ded)
        self.button_3ded.clicked.connect(self.extract_3ded)

        self.button_extractCurrentFrame = qtw.QPushButton('Extract Frame')
        self.button_extractCurrentFrame.setFixedHeight(button_h_lrg)
        layout_extract_button.addWidget(self.button_extractCurrentFrame)
        self.button_extractCurrentFrame.setToolTip(
            'Compute the DP for the current frame only (not saved)')
        self.button_extractCurrentFrame.clicked.connect(self.extract_dp_current_frame)
        
        layout_ribbon_final = qtw.QVBoxLayout()
        layout_ribbon.addLayout(layout_ribbon_final)
        self.button_save_results = qtw.QPushButton('Save Results')
        self.button_save_results.setFixedSize(button_w, button_h_lrg*3)
        layout_ribbon_final.addWidget(self.button_save_results)
        self.button_save_results.clicked.connect(self.save_results)
        
        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.setFixedSize(button_w, button_h_lrg*3)
        self.button_cancel.setStyleSheet("background-color: red; color: white;")
        self.button_cancel.setDisabled(True)
        self.button_cancel.setToolTip('Stop the running tracking/segmentation/extraction')
        self.button_cancel.clicked.connect(self.cancel_running_work)
        layout_ribbon_final.addWidget(self.button_cancel)
        # disable_3ded_widgets(True) is called further down instead, right
        # after button_fineTuneMask exists (see there) - it explicitly
        # toggles that button too, which isn't built yet at this point.
        self._ribbon_group_end(layout_ribbon, layout_box_3ded, 'Extract')
        layout_ribbon.addStretch(1)

        #%% Adjust Contrast (top) + Feature Handling (below it) - moved out
        # of the ribbon into one stacked column beside the canvas, same
        # position as the Navigator tab's file list (see the #%% canvas
        # section below for where this widget is actually placed).
        widget_featurePanel = qtw.QWidget()
        # Fixed (not just an initial splitter size) - a child widget's own
        # minimum-size floor (e.g. tree_objects.setMinimumWidth below) would
        # otherwise let this pane grow past 220px on a squeezed window,
        # independently of whatever floor the other 3 tabs' own left panes
        # happen to have, so the 4 tabs' panes could drift to different
        # actual widths even though every tab starts from the same 220.
        widget_featurePanel.setFixedWidth(220)
        layout_featurePanel = qtw.QVBoxLayout(widget_featurePanel)
        layout_featurePanel.setContentsMargins(2, 2, 2, 2)

        self.box_contrast = ContrastScalingBox()
        self.box_contrast.settingsChanged.connect(self.rescale_nav_signal)
        # Denoise-only changes are decoupled from the full-stack rescale
        # above (some methods are too slow to re-run on every frame for
        # every parameter tweak) - see _on_denoise_preview_changed/
        # _apply_denoise_to_all_frames, and box_contrast's own docstring.
        self.box_contrast.denoisePreviewChanged.connect(self._on_denoise_preview_changed)
        self.box_contrast.denoiseApplyAllRequested.connect(self._apply_denoise_to_all_frames)
        self.box_contrast.checkMethodsRequested.connect(self._show_denoise_check_methods)
        # True once a Denoise change has been previewed on the current frame
        # only, but not yet (re)applied to the rest of the stack - see
        # _on_denoise_preview_changed/_on_slider_imgNo_changed.
        self._denoise_dirty = False
        layout_featurePanel.addWidget(self.box_contrast)

        # Auto Detector / Reset Objects - sit above the object list, same
        # position/pairing as ROI Tracker's Auto Detector/Reset ROIs row
        # above its own tree_objects.
        layout_sam_top = qtw.QHBoxLayout()
        layout_featurePanel.addLayout(layout_sam_top)

        self.button_autoDetector = qtw.QPushButton('Auto Detector', self)
        self.button_autoDetector.setToolTip(
            "Run SAM2's automatic mask generator on the current frame to find "
            'candidate objects, then pick which ones to add to the object list')
        layout_sam_top.addWidget(self.button_autoDetector)
        self.button_autoDetector.clicked.connect(self.launch_auto_detector)
        self.button_autoDetector.setDisabled(True)

        self.button_reset_objects = qtw.QPushButton('Reset Objects')
        self.button_reset_objects.setToolTip(
            'Clear every object, point, and cached PETS2 parameter (same as '
            'ROI Tracker\'s "Reset ROIs")')
        layout_sam_top.addWidget(self.button_reset_objects)
        self.button_reset_objects.clicked.connect(self.reset_data)

        # Own row, directly below Auto Detector/Reset Objects - acts on the
        # selected object below (tree_objects), not on the tracker controls
        # up in the ribbon, so it moved down here from the ribbon's
        # Tracking group for that reason (mirrors ROI Tracker's identical
        # Fine-Tune Mask.../Blob Settings... row in the same position -
        # SAM2 has no Blob Selection feature of its own to pair it with).
        # Still governed by disable_3ded_widgets/activate_3ded_widgets (see
        # their own docstrings) even though it's no longer one of box_3ded's
        # own children.
        row_maskActions = qtw.QHBoxLayout()
        layout_featurePanel.addLayout(row_maskActions)
        self.button_fineTuneMask = qtw.QPushButton('Fine-Tune Mask...')
        self.button_fineTuneMask.setToolTip('Manually edit the tracked mask, frame by frame')
        row_maskActions.addWidget(self.button_fineTuneMask)
        self.button_fineTuneMask.clicked.connect(self.open_fine_tune_mask_dialog)
        self.button_fineTuneMask.setDisabled(True)

        # Moved here (from right after the Extract ribbon group) - needs
        # button_fineTuneMask to already exist, since disable_3ded_widgets
        # toggles it explicitly too (see its own docstring).
        self.disable_3ded_widgets(True)

        # tree - stretches to fill the rest of this column's height now that
        # it sits beside the (tall) canvas, rather than being capped to fit
        # inside a short ribbon column.
        # Transposed (see TransposedObjectTable): property names run down
        # the fixed first column instead of across the top, and each
        # tracked object is one column instead of one row, so adding an
        # object adds a column - still called tree_objects (not literally a
        # QTreeWidget anymore) since renaming the many existing references
        # below wasn't worth it.
        self.cols_tree = ["use", "idx", "fr_idx", "end", "trk", "ext", "qlty", "dup", "del"]
        # Kept at or under "Start"'s own length (see TransposedObjectTable.
        # _HEADER_WIDTH_REF, matching ROI Tracker's identical object list) -
        # the ones that don't fit unabbreviated get a row_tooltips entry
        # with their full word instead.
        row_labels = ["Use", "Idx", "Frame", "End", "Track", "Extr", "Qlty", "Dup", "Del"]
        row_tooltips = [None, None, None, None, "Tracked", "Extracted",
                        "Tracking Quality - flagged (see the frame-flag bar under the "
                        "slider) if any frame's mask area looks anomalous after tracking, "
                        "e.g. the tracker may have lost the object",
                        "Duplicate", "Delete"]
        # Selected Object / All Active Objects - governs the Segmented
        # panel's mask overlay only: the DP panel still follows whichever
        # object is actually selected in the table below, regardless of
        # this choice (see update_canvas) - "All Active Objects" only
        # changes what the mask overlay itself shows, from one object's
        # own mask to every active ("Use" checked) object's mask
        # composited at once, each in its own color with its index label
        # at its centroid (see _draw_all_object_masks).
        row_maskMode = qtw.QHBoxLayout()
        layout_featurePanel.addLayout(row_maskMode)
        self.radio_maskSelected = qtw.QRadioButton('Selected Object')
        self.radio_maskSelected.setChecked(True)
        self.radio_maskAll = qtw.QRadioButton('All Active Objects')
        self._group_maskMode = qtw.QButtonGroup(self)
        self._group_maskMode.addButton(self.radio_maskSelected)
        self._group_maskMode.addButton(self.radio_maskAll)
        row_maskMode.addWidget(self.radio_maskSelected)
        row_maskMode.addWidget(self.radio_maskAll)
        row_maskMode.addStretch(1)
        self.radio_maskSelected.toggled.connect(self._on_mask_mode_changed)

        self.tree_objects = TransposedObjectTable(self.cols_tree, row_labels, row_tooltips)
        layout_featurePanel.addWidget(self.tree_objects, 1)
        # Tall enough for their content: dup/del hold a 30px button, end
        # holds a QSpinBox with up/down arrows, trk/ext hold a status icon.
        row_heights = {'use': 24, 'idx': 24, 'fr_idx': 24, 'end': 28,
                       'trk': 24, 'ext': 24, 'qlty': 24, 'dup': 34, 'del': 34}
        for i, col in enumerate(self.cols_tree):
            self.tree_objects.setRowHeight(i, row_heights[col])
        self.tree_objects.setMinimumWidth(200)
        self.tree_objects.itemSelectionChanged.connect(self.update_canvas)
        self.tree_objects.itemChanged.connect(self.on_item_check_changed)

        #%% canvas (below the ribbon, using the tab's full width)
        self._right_widget = qtw.QWidget()
        self._main_splitter.addWidget(self._right_widget)
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        _right_widget_outer_layout = qtw.QVBoxLayout(self._right_widget)
        _right_widget_outer_layout.setContentsMargins(0, 0, 0, 0)
        _right_widget_outer_layout.setSpacing(0)
        layout_right_outer = qtw.QSplitter(Qt.Horizontal)
        _right_widget_outer_layout.addWidget(layout_right_outer)
        layout_right_outer.addWidget(widget_featurePanel)
        self._canvas_container = qtw.QWidget()
        layout_right_outer.addWidget(self._canvas_container)
        _canvas_container_outer_layout = qtw.QVBoxLayout(self._canvas_container)
        _canvas_container_outer_layout.setContentsMargins(0, 0, 0, 0)
        _canvas_container_outer_layout.setSpacing(0)
        layout_canvas_splitter = qtw.QSplitter(Qt.Vertical)
        _canvas_container_outer_layout.addWidget(layout_canvas_splitter)

        # Canvas + slider + progress bar share one pane of the vertical
        # splitter above (the log console is the other pane, see below).
        _canvas_pane = qtw.QWidget()
        layout_canvas = qtw.QVBoxLayout(_canvas_pane)
        layout_canvas.setContentsMargins(0, 0, 0, 0)
        layout_canvas_splitter.addWidget(_canvas_pane)
        
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
        # Clipping Thresholds beside ax_dp (the rightmost of the 3
        # subplots) - only the DP axis, per the decision that Display
        # Contrast already covers the nav image on this tab. Sits directly
        # beside the canvas (in the same row, not a sibling pane of the
        # whole canvas+console splitter), so its height matches the
        # canvas's own height, not canvas+console combined - matches ROI
        # Tracker's identical canvas-row arrangement.
        self.clip_dp = ClippingThresholdsWidget(title='DP Clipping\nThresh.')
        _canvas_row_widget = qtw.QWidget()
        layout_canvas_row = qtw.QHBoxLayout(_canvas_row_widget)
        layout_canvas_row.setContentsMargins(0, 0, 0, 0)
        layout_canvas_row.addWidget(self.wrap_canvas_in_scroll(self.canvas), 1)
        layout_canvas_row.addWidget(self.clip_dp)
        layout_canvas.addWidget(_canvas_row_widget)
        # The Ctrl+Scroll zoom hint applies to every axis on this canvas, so
        # it's a figure-wide supxlabel rather than repeated per-axis text.
        self.figure.supxlabel('Hold "Ctrl" + Scroll wheel to zoom the axis under the cursor',
                              fontsize=10)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        # self.masks_plotted = []
        self.create_main_dataframe()
        self.imgs = deepcopy([self.img_zero])
        self.imgs_8bit = deepcopy([self.img_zero])
        self.scatter_plots = []
        self.press = None            # Mouse press coords, for the ribbon's "Remove points" box-select
        self._remove_points_rect = None
        self._remove_points_bg = None
        # {obj_id: [flagged_frame_idx, ...]} - tracking-quality flags (see
        # _compute_tracking_quality/frame_flag_bar.FrameFlagBar), recomputed
        # fresh after every tracking run - deliberately NOT a df_obj column
        # (or persisted in Save Results/Load Saved Analysis): purely a
        # derived diagnostic, cheap to recompute, not part of this object's
        # actual tracked/extracted data.
        self._quality_flags = {}
        # Index-label artists for the "All Active Objects" mask-overlay
        # mode (see _draw_all_object_masks) - cleared/rebuilt every call.
        self._all_mask_artists = []
        #%% slider
        layout_slider = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_slider)

        # Frame-flag bar (see frame_flag_bar.FrameFlagBar/
        # _compute_tracking_quality) - same as ROI Tracker's identical one.
        # frameClicked wired to slider_imgNo further below, once that
        # widget actually exists (constructed later in this same row).
        self.frame_flag_bar = FrameFlagBar()
        layout_canvas.addWidget(self.frame_flag_bar)

        self.label_imgCounter = qtw.QLabel('Img No.')
        layout_slider.addWidget(self.label_imgCounter)
        
        self.lineEdit_imgNo = qtw.QLineEdit()
        layout_slider.addWidget(self.lineEdit_imgNo)
        self.lineEdit_imgNo.setFixedWidth(35)
        self.lineEdit_imgNo.setValidator(QIntValidator(0, 0))
        self.lineEdit_imgNo.returnPressed.connect(self.jump_to_frame_no)
        
        self.label_imgCounter = qtw.QLabel('Img No.')
        layout_slider.addWidget(self.label_imgCounter)

        # Prev/Next sit together, right before the slider itself, rather
        # than flanking it on both sides.
        self.button_prevFrame = qtw.QPushButton('◀')
        self.button_prevFrame.setFixedWidth(28)
        self.button_prevFrame.setToolTip('Previous frame')
        self.button_prevFrame.clicked.connect(lambda: self._step_frame(-1))
        layout_slider.addWidget(self.button_prevFrame)

        self.button_nextFrame = qtw.QPushButton('▶')
        self.button_nextFrame.setFixedWidth(28)
        self.button_nextFrame.setToolTip('Next frame')
        self.button_nextFrame.clicked.connect(lambda: self._step_frame(1))
        layout_slider.addWidget(self.button_nextFrame)

        self.slider_imgNo = qtw.QSlider(self)
        self.slider_imgNo.setOrientation(1)  # Horizontal slider
        self.slider_imgNo.setRange(0,0)
        layout_slider.addWidget(self.slider_imgNo)
        self.frame_flag_bar.frameClicked.connect(self.slider_imgNo.setValue)

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
        self.slider_imgNo.valueChanged.connect(self._on_slider_imgNo_changed)

        self.tree_objects.itemSelectionChanged.connect(self.update_stack_guide)
        # Kept alive (not shown) purely for its view-stack bookkeeping
        # (.update()/.push_current(), used to seed the ribbon's Home button)
        # and as the target of the ribbon's own Pan/Zoom/Home actions below -
        # the toolbar strip itself is no longer shown under the canvas.
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()

        #%% ribbon
        # Docked along the right edge - an additional way to reach the same
        # canvas interaction already available via Ctrl-click (see on_click);
        # deliberately does NOT duplicate the left panel's buttons (Seg
        # Image, Track, Extract All, Save Results, ...), only actions that
        # act directly on the plot itself. 'add_point' is the only tool mode
        # on_click actually checks (see RibbonPanel.active_tool there) -
        # left/right click still choose positive/negative, and Shift still
        # chooses new-object-vs-append, exactly as before. Pan/Zoom/Home
        # drive matplotlib's own toolbar (kept alive but hidden - see
        # self.toolbar above), which no longer has a visible strip of its
        # own under the canvas.
        self.ribbon = RibbonPanel([
            RibbonTool('add_point', 'add_point', 'Add point (left=+/right=-); +Shift to append '
                      'to the selected object - same as Ctrl+click', 'tool'),
            RibbonTool('remove_point', 'remove_point', 'Remove last point (same as middle-click)',
                      'action', self.delete_last_point),
            RibbonTool('remove_points_roi', 'select_roi', 'Remove points: click+drag a box on '
                      'the Nav. Image to delete every point (any object) it contains on this frame',
                      'tool'),
            RibbonTool('sep1', kind='separator'),
            RibbonTool('pan', 'pan', 'Toggle pan mode',
                      'action', self.toolbar.pan),
            RibbonTool('zoom', 'zoom', 'Toggle rectangle-zoom mode',
                      'action', self.toolbar.zoom),
            RibbonTool('home', 'home', 'Reset the view',
                      'action', self.toolbar.home),
            RibbonTool('sep2', kind='separator'),
            RibbonTool('help', 'help', 'Shortcuts & mouse controls for this tab',
                      'action', self.show_help_dialog),
        ], parent=self)
        self.ribbon.toolChanged.connect(self._on_ribbon_tool_changed)
        # Deferred (see _apply_ribbon_cursor's docstring) - reapplies the
        # ribbon cursor after mpl's own NavigationToolbar2 cursor-restore
        # logic (wrapped around every canvas.draw()) has already run.
        self.canvas.mpl_connect(
            'draw_event', lambda evt: QTimer.singleShot(0, self._apply_ribbon_cursor))

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
        layout_canvas_splitter.addWidget(self.log_console)
        layout_canvas_splitter.setStretchFactor(0, 1)
        layout_canvas_splitter.setStretchFactor(1, 0)

        # Only the canvas column claims extra horizontal space by default;
        # the left panel opens at a fixed default width shared across all 4
        # tabs (tab_roi_4d.py, tab_create_navSignal.py, tab_tracking_cv2.py,
        # tab_sam2.py).
        for _i in range(layout_right_outer.count()):
            layout_right_outer.setStretchFactor(_i, 0)
        layout_right_outer.setStretchFactor(1, 1)

        def _apply_initial_splitter_sizes():
            """(Re-)apply every splitter's default pane sizes, and disable
            collapsing on all of them. The ribbon icon strip (last pane,
            fixed-width; clip_dp is no longer a pane of this splitter - see
            the canvas-row widget above) is really just "as small as it's
            allowed to be" below - but a QSplitter.setSizes() call made
            before the window has ever actually been shown (i.e. still has
            no real geometry, as here - this runs during __init__, well
            before MainWindow.show()) only stores those sizes
            proportionally against whatever placeholder width Qt reports at
            that moment, not real pixels - so calling it only once, here,
            left that pane rendered collapsed to nothing until the user
            manually dragged it open. Re-running the exact same calls
            once more via QTimer.singleShot(0, ...) - after the event loop
            has actually processed the window's first show/resize, so
            every widget's real minimum size is now known - fixes that.
            setCollapsible(False) on every pane is kept too, as a static
            safety net against the same collapse happening later from a
            user drag."""
            layout_canvas_splitter.setSizes([2000, 150])
            layout_right_outer.setSizes([220, 3000, 0])
            for _i in range(layout_canvas_splitter.count()):
                layout_canvas_splitter.setCollapsible(_i, False)
            for _i in range(layout_right_outer.count()):
                layout_right_outer.setCollapsible(_i, False)
            for _i in range(self._main_splitter.count()):
                self._main_splitter.setCollapsible(_i, False)
            # Sizes _main_splitter's ribbon pane too, respecting the current
            # Ribbon Height display setting (rather than hardcoding its
            # natural sizeHint here) - see apply_display_settings.
            self.apply_display_settings()

        _apply_initial_splitter_sizes()
        QTimer.singleShot(0, _apply_initial_splitter_sizes)
        # tooltips
        self.button_loadNavigation.setToolTip('Load navigation signal (.hspy or .zspy)  [Ctrl+O]')
        self.button_loadSavedAnalysis.setToolTip('Load a saved analysis folder  [Ctrl+Shift+O]')
        self.button_runSeg_clip.setToolTip('Track objects across all frames using SAM2  [Ctrl+T]')
        self.button_runSeg_img.setToolTip('Segment the current frame only (no tracking)')
        self.button_3ded.setToolTip('Extract 3D electron diffraction patterns  [Ctrl+E]')
        self.button_save_results.setToolTip('Save segmentation and 3DED results to disk  [Ctrl+S]')
        self.spinbox_threadNum.setToolTip('Number of CPU cores used for parallel 4D extraction')
        self.checkbox_autosave.setToolTip('Automatically save results when extraction finishes')
        self.checkbox_makePets2.setToolTip(
            "Write a PETS2 project file (.pts2) into each object's folder on save")

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
            path = get_existing_directory(self, "Select 4D Folder", self.lineEdit_dir_4d.text())
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
            path = get_existing_directory(self, "Select Destination Folder", self.lineEdit_dir_save.text())
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
        dtype = resolve_hdf5_dtype(fn, self.combo_dtype_4d.currentText())
        if dtype == '.tpx3':
            if self.checkbox_detectorSizeAuto.isChecked():
                return 512, 512
            return self.spinbox_detectorSize_x.value(), self.spinbox_detectorSize_y.value()
        return io.get_det_size(fn, dtype)

    def _mesh_settings_for(self, obj_id):
        """This object's Mesh settings (see MaskEditDialog/get_mesh_settings),
        or None if it has none set / obj_id is None."""
        if obj_id is None:
            return None
        mesh = self.df_obj.at[obj_id, 'mesh']
        return mesh if isinstance(mesh, dict) else None

    def _dilate_erode_settings_for(self, obj_id):
        """This object's Dilate/Erode segments (see MaskEditDialog/
        get_dilate_erode_settings - `{'segments': [...]}`), or None if it
        has none set / obj_id is None. Per-object, set only from the
        Fine-Tune Mask dialog - no main-tab control (same as Mesh, and,
        since the Segments feature, Edge Detection too - see
        _edge_settings_for)."""
        if obj_id is None:
            return None
        dilate_erode = self.df_obj.at[obj_id, 'dilate_erode']
        return dilate_erode if isinstance(dilate_erode, dict) else None

    def _edge_settings_for(self, obj_id):
        """This object's Edge Detection segments (see MaskEditDialog/
        get_edge_settings - `{'segments': [...]}`), or None if it has none
        set / obj_id is None. Per-object (like Mesh/Dilate-Erode) - Edge
        Detection used to be a single tab-wide setting before the Fine-Tune
        Mask dialog's Segments feature; now it's only ever set there, one
        range at a time, same as the other two."""
        if obj_id is None:
            return None
        edge = self.df_obj.at[obj_id, 'edge']
        return edge if isinstance(edge, dict) else None

    def _has_active_postprocessing(self, obj_id):
        """Whether object `obj_id` has ANY segment (see MaskEditDialog's
        Segments feature) with Dilate/Erode, Edge Detection, or Mesh
        actually enabled - lets apply_edge_mask_stack skip its per-frame
        work entirely for an object with nothing to do there."""
        def _any_enabled(settings, extra=lambda s: True):
            segs = (settings or {}).get('segments') or []
            return any(s.get('enabled') and extra(s) for s in segs)
        return (_any_enabled(self._edge_settings_for(obj_id))
               or _any_enabled(self._dilate_erode_settings_for(obj_id), lambda s: (
                   s.get('kernel', 0) != 0 or s.get('open_kernel', 0) != 0 or s.get('close_kernel', 0) != 0))
               or _any_enabled(self._mesh_settings_for(obj_id), lambda s: s.get('cells')))

    def apply_edge_mask(self, mask, obj_id=None, frame_idx=None):
        """Grow/shrink a single 2-D mask uniformly when `obj_id`'s
        Dilate/Erode setting for `frame_idx` (see MaskEditDialog's
        Segments - resolved via io.segment_for_frame) is enabled (see
        io.dilate_erode_mask), then reduce it to just its edge/outline when
        that frame's Edge Detection segment is enabled (see
        io.erode_mask_edge), then - if `obj_id` is given and that frame's
        Mesh segment has a restriction set - restrict it to the selected
        mesh cell(s), relative to the object's own position on THIS frame
        (see io.mesh_restrict_mask/io.mask_centroid, and
        MaskEditDialog._effective_mask's identical convention) so a
        tracked object's motion across frames doesn't throw off which part
        of it the selection actually covers. A no-op otherwise. SAM2 masks
        are always kept raw in self.df_obj (see handle_finished_sam/
        handle_finished_image_sam) so this can be applied fresh - and
        re-applied live whenever any segment's settings change - as a view
        at display/extraction/save time, instead of destructively baking
        any of them into the stored mask (which would make it impossible
        to undo by unchecking/re-editing it)."""
        mesh_segments = (self._mesh_settings_for(obj_id) or {}).get('segments')
        mesh = io.segment_for_frame(mesh_segments, frame_idx) if mesh_segments else None
        mesh_on = bool(mesh and mesh.get('enabled') and mesh.get('cells'))
        origin = io.mask_centroid(mask) if mesh_on else None

        de_segments = (self._dilate_erode_settings_for(obj_id) or {}).get('segments')
        de = io.segment_for_frame(de_segments, frame_idx) if de_segments else None
        if de and de.get('enabled'):
            if de.get('kernel', 0) != 0:
                mask = io.dilate_erode_mask(mask, de['kernel'])
            if de.get('open_kernel', 0) != 0:
                mask = io.open_mask(mask, de['open_kernel'])
            if de.get('close_kernel', 0) != 0:
                mask = io.close_mask(mask, de['close_kernel'])

        edge_segments = (self._edge_settings_for(obj_id) or {}).get('segments')
        edge = io.segment_for_frame(edge_segments, frame_idx) if edge_segments else None
        if edge and edge.get('enabled'):
            direction = edge.get('direction') if edge.get('directional') else None
            mask = io.erode_mask_edge(mask, edge.get('kernel', 3), direction=direction,
                                      revert=edge.get('revert', False))

        if mesh_on:
            mask = io.mesh_restrict_mask(mask, mesh.get('angle', 0), mesh.get('cell_size', 20),
                                         [tuple(c) for c in mesh['cells']], origin=origin,
                                         lines_only=mesh.get('lines_only', False))
        return mask

    def apply_edge_mask_stack(self, mask_stack, obj_id=None):
        """`apply_edge_mask`, applied per-frame to a (N, H, W) mask stack -
        each frame resolves its own segment (see io.segment_for_frame) for
        Dilate/Erode, Edge Detection, and Mesh independently, so a stack
        whose settings vary partway through gets each frame's own range
        applied correctly."""
        if not self._has_active_postprocessing(obj_id):
            return mask_stack
        return np.stack([self.apply_edge_mask(m, obj_id, i) for i, m in enumerate(mask_stack)])

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
            if dtype in ('.hdf5', '.hdf5_eventem'):
                # metadata.json stores the internal dtype identifier (see
                # resolve_hdf5_dtype), not this combo's own display text,
                # which for the eventem entry is now HDF5_EVENTEM_LABEL
                # rather than the identifier itself - so both a legacy
                # '.hdf5' value (written before the format split into
                # eventem/conventional; back then a bare '.hdf5' always
                # meant eventem, the only one that existed) and a current
                # '.hdf5_eventem' value need remapping here for findText()
                # below to find the right combo item, instead of silently
                # matching nothing (old projects) or the wrong, unrelated
                # "conventional HDF5" combo entry (current ones).
                dtype = HDF5_EVENTEM_LABEL
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
        # "Adjust Contrast" method/parameters actually take visible effect.
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
        self.frame_flag_bar.set_range(len(self.imgs))
        self._quality_flags = {}
        self.button_runSeg_clip.setEnabled(True)
        self.button_runSeg_img.setEnabled(True)
        self.button_fineTuneMask.setEnabled(True)
        self.button_autoDetector.setEnabled(True)
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.imgs)))
        self.spinbox_stackNum.setValue(len(self.imgs))
        # Scale fields may already hold a value from a previous session/load -
        # add_scalebar() is otherwise only triggered by the fields' own
        # textChanged signal, so a fresh load wouldn't show it until touched.
        self.add_scalebar()

    def rescale_nav_signal(self):
        """Contrast settings changed: the currently-displayed frame is
        rescaled immediately (cheap, instant feedback), while the full
        stack (used for SAM2/tracking, and to keep every other frame in
        sync) rescales in the background - a long stack no longer blocks/
        lags the GUI on every settings tweak. Rapid retuning cancels (i.e.
        discards the result of) any still-running previous background
        rescale - see ContrastScalingBox.rescale_async. Also used as
        _apply_denoise_to_all_frames's own worker - both end up wanting
        exactly this same full-stack-refresh-from-current-settings."""
        if not hasattr(self, 's_navSignal'):
            return
        self._refresh_current_frame_display()
        self.box_contrast.set_denoise_apply_all_busy(True)
        self.progress_bar.setRange(0, len(self.imgs))
        self.progress_bar.setValue(0)
        self.box_contrast.rescale_async(self.s_navSignal, self.threadpool, self.logger,
                                        on_progress=self.update_progress_bar,
                                        on_done=self._on_nav_signal_rescaled,
                                        on_error=self._on_nav_signal_rescale_failed)

    def _refresh_current_frame_display(self, imgNo=None):
        """Rescale (contrast + denoise, current settings) and redraw just
        the currently-displayed frame - cheap, so safe to call on every
        Denoise parameter tweak (see _on_denoise_preview_changed) or frame
        navigation (see _on_slider_imgNo_changed) without waiting for a full
        background stack rescale.

        Blits just the nav/seg image data (_blit_current_frame_display)
        instead of routing through update_canvas's full draw - update_canvas
        resolves the selected object, masks, points, and DP fresh every
        call, none of which a Denoise parameter tweak can actually change,
        so running its full (non-blitted) draw on every single spinbox
        nudge made retuning a slow method noticeably laggy."""
        if not hasattr(self, 's_navSignal'):
            return
        if imgNo is None:
            imgNo = self.slider_imgNo.value()
        frame_8bit = self.box_contrast.rescale_frame(self.imgs[imgNo])
        self.imgs_8bit[imgNo] = frame_8bit
        self.img_display['nav'].set_data(frame_8bit)
        self.img_display['nav'].set_clim(vmin=frame_8bit.min(), vmax=frame_8bit.max())
        self.img_display['seg'].set_data(frame_8bit)
        self.img_display['seg'].set_clim(vmin=frame_8bit.min(), vmax=frame_8bit.max())
        # The mask overlay is per-frame too (a tracked object's mask stack
        # has one entry per frame) - without refreshing it here it stayed
        # whichever frame's mask was last drawn by update_canvas, showing
        # the WRONG frame's mask on top of this now-current frame's image
        # (the image itself updated above, correctly, since it's a plain
        # per-frame array; the mask needs the same apply_edge_mask
        # resolution update_canvas uses, just without its full (non-
        # blitted) redraw - see this method's own docstring on why).
        self._refresh_current_frame_mask(imgNo)
        self._blit_current_frame_display()

    def _refresh_current_frame_mask(self, imgNo):
        """Recompute and set (but don't draw/blit) the seg_mask overlay for
        `imgNo` - the mask-resolution half of update_canvas's own obj_id/
        mask branch, factored out so _refresh_current_frame_display can
        keep the mask in sync with the frame it just switched to without
        paying for update_canvas's full redraw (points, DP, titles, ...) on
        every slider tick. Respects the Selected Object/All Active Objects
        toggle, same as update_canvas itself."""
        if self.radio_maskAll.isChecked():
            self._draw_all_object_masks(imgNo)
            return
        try:
            item_selected = self.tree_objects.currentItem()
            obj_id = int(item_selected.text(1))
        except Exception:
            return
        try:
            if not np.all(pd.isna(self.df_obj.loc[obj_id, 'mask'])):
                self.show_mask(self.apply_edge_mask(self.df_obj.loc[obj_id, 'mask'][imgNo], obj_id, imgNo), obj_id)
            else:
                mask = self.apply_edge_mask(self.df_obj.loc[obj_id, 'single_mask'][imgNo], obj_id, imgNo)
                self.show_mask(mask, 0)
        except Exception:
            self.show_mask(self.img_zero)

    def _blit_current_frame_display(self):
        """Blit just the nav/seg image artists (plus their existing point/
        mask overlays, redrawn on top so repainting the image doesn't erase
        them - draw_artist() overwrites raw pixels in its axes' region
        regardless of the other artists baked into the restored background)
        onto the canvas - see TabBase._blit_canvas. The point/mask overlays
        themselves are untouched by a denoise/contrast tweak, but still need
        to be included here (not just left baked into the cached
        background) since they're drawn on top of the very same axes the
        image artists just overwrote."""
        artists = ([self.img_display['nav'], self.img_display['seg'],
                    self.img_display['seg_mask']] + self.scatter_plots
                   + self._all_mask_artists)
        self._blit_canvas(
            self.canvas, self.figure, '_denoise_bg', artists,
            hide_for_background=[self.img_display['nav'], self.img_display['seg']])

    def _on_denoise_preview_changed(self):
        """box_contrast's Denoise method/parameter changed: refresh just
        the current frame (cheap) - the rest of the stack is intentionally
        left as-is (some denoise methods are too slow to re-run on every
        frame for every tweak) until "Apply to All Images" is clicked (see
        _apply_denoise_to_all_frames) or a contrast change triggers a full
        refresh anyway (rescale_nav_signal). Marks the stack "dirty" so
        navigating to a different frame in the meantime also gets a fresh
        preview instead of showing that frame's old, differently-denoised
        pixels - see _on_slider_imgNo_changed."""
        self._denoise_dirty = True
        self._refresh_current_frame_display()

    def _apply_denoise_to_all_frames(self):
        """box_contrast's "Apply to All Images" button: run the full
        contrast+denoise pipeline across the whole stack now, on demand -
        exactly what rescale_nav_signal already does for a contrast change,
        just triggered explicitly instead.

        Clears "dirty" right away, rather than waiting for the background
        rescale to actually finish: the whole point of this button is to
        make frame navigation fast again immediately, not just once a
        possibly slow (large stack, slow method) background job eventually
        completes - during that window _on_slider_imgNo_changed no longer
        recomputes per-frame (which would otherwise keep contending with
        the background job for CPU, defeating the point), so a frame
        visited in that window may briefly show its pre-rescale pixels
        until _on_nav_signal_rescaled's own refresh catches it up."""
        self._denoise_dirty = False
        self.rescale_nav_signal()

    def _show_denoise_check_methods(self):
        """box_contrast's "Check Methods..." button: compare every
        denoising method on the currently-displayed raw frame."""
        if not hasattr(self, 's_navSignal'):
            qtw.QMessageBox.warning(self, 'No Signal Loaded',
                'Load a signal first to compare denoising methods on it.')
            return
        imgNo = self.slider_imgNo.value()
        self._check_methods_dlg = self.box_contrast.open_check_methods_dialog(
            self.imgs[imgNo], parent=self)

    def _on_nav_signal_rescaled(self, s_8bit):
        """ContrastScalingBox.rescale_async callback: apply the fully-rescaled
        8-bit stack once the background recompute finishes - every frame now
        reflects the current Denoise settings too, so the stack is no
        longer "dirty" (see _on_denoise_preview_changed)."""
        self.box_contrast.set_denoise_apply_all_busy(False)
        self.imgs_8bit = s_8bit.data
        self._denoise_dirty = False
        self.img_display['nav'].set_clim(vmin=self.imgs_8bit.min(), vmax=self.imgs_8bit.max())
        self.update_canvas()
        self.canvas.draw_idle()

    def _on_nav_signal_rescale_failed(self, traceback_text):
        """ContrastScalingBox.rescale_async's on_error callback: without
        this, a failed full-stack rescale (e.g. Apply to All Images hitting
        a denoise-method error) would vanish silently - the "dirty" flag
        would stay False (cleared optimistically by
        _apply_denoise_to_all_frames) forever, leaving every frame but the
        one on screen at that moment permanently stuck showing pre-rescale
        pixels with no way to tell it had failed. Re-dirtying falls back to
        per-frame recompute on navigation (see _on_slider_imgNo_changed)
        until the user retries."""
        self.box_contrast.set_denoise_apply_all_busy(False)
        self._denoise_dirty = True
        self.logger.error('Full-stack contrast/denoise rescale failed:\n%s', traceback_text)
        qtw.QMessageBox.warning(self, 'Rescale Failed',
            f'Could not apply the current contrast/denoise settings to the '
            f'full image stack:\n{traceback_text[-500:]}')

    def _on_slider_imgNo_changed(self, imgNo):
        """Frame slider moved: if the currently-configured Denoise settings
        haven't been applied to the whole stack yet (see
        _on_denoise_preview_changed/_apply_denoise_to_all_frames), refresh
        just this now-visible frame first, so navigating around always
        reflects the live-configured settings without eagerly recomputing
        every other frame too."""
        if self._denoise_dirty:
            self._refresh_current_frame_display(imgNo)
        else:
            self.update_canvas(imgNo)

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
                                     obj['mask'], obj['rois'], obj['dp'], None, None, None]
            self.add_item_tree(idx, obj['frame_idx'], obj['end'], obj['use'])
            row_index = self.df_obj.index.get_loc(idx)
            if obj['mask'] is not None:
                self.toggle_tree_icon(row_index, 'trk', True)
            if obj['dp'] is not None:
                self.toggle_tree_icon(row_index, 'ext', True)

        self.activate_3ded_widgets(True)
        # Select the first restored object, if any, so its tracking/mask/DP
        # actually show up right away - update_canvas() only draws the
        # currently-selected object's own results, and nothing in the tree
        # is selected by default just from populating it above (add_item_tree
        # doesn't select what it adds), so without this the loaded results
        # sat in df_obj unseen until the user clicked a row themselves.
        if objects:
            self.tree_objects.setCurrentItem(self.tree_objects.topLevelItem(0))
        self.update_canvas(0)
        # Loaded DPs may have a different center than the placeholder - re-run
        # auto-centering now if enabled.
        self.add_scalebar()
        self.logger.info('Loaded saved analysis from %s (%d object(s)).', path, len(objects))

    def create_main_dataframe(self):
        """(Re)create the empty per-object dataframe (df_obj) with its column
        schema, and reset the added-points history."""
        self.cols_df = ['use', 'idx', 'frame_idx', 'points', 'labels', 'end',
                        'single_mask', 'mask', 'mask_default', 'rois', 'dp', 'mesh',
                        'dilate_erode', 'edge']
        self.df_obj = pd.DataFrame([], columns=self.cols_df)
        self.df_obj = self.df_obj.astype({'use': int, 'idx': int,'frame_idx': object,
                                          'points': object, 'labels': object,
                                          'end': int, 'single_mask': object,
                                          'dp': object,'mask':object, 'mask_default': object,
                                          'rois':object, 'mesh': object, 'dilate_erode': object,
                                          'edge': object})
        self.initiate_adding_points()
        
    def reset_data(self):
        """Clear all objects, points, and plotted markers, and drop any
        cached PETS2 params - called before loading a new signal or saved analysis."""
        for p in self.scatter_plots:
            p.remove()
        self.scatter_plots.clear()
        self.tree_objects.clear()
        self.create_main_dataframe()
        self._quality_flags = {}
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
        """Add a column to tree_objects for object `idx`: use checkbox, idx/frame
        labels, an end-frame spinbox, tracked/extracted status icons, and a
        delete button; selects the new column."""
        cols = {col: i for i,col in enumerate(self.cols_tree)}
        item = self.tree_objects.addTopLevelItem()
        item.setCheckState(cols['use'], Qt.Checked if use else Qt.Unchecked)
        item.setText(cols['idx'], f"{idx}")
        item.setText(cols['fr_idx'], f"{fr_idx}")

        spinbox = qtw.QSpinBox()
        spinbox.setRange(0, len(self.imgs))
        spinbox.setValue(end if end is not None else len(self.imgs))
        self.tree_objects.setItemWidget(item, cols['end'], spinbox)
        spinbox.valueChanged.connect(lambda value: self.on_spinboxEnd_changed(idx, value))
        # spinbox.valueChanged.connect(partial(self.on_spinbox_changed, item, idx))

        cancel_icon = self.style().standardIcon(self.style().SP_DialogCancelButton)
        item.setIcon(cols['trk'], cancel_icon)
        item.setData(cols['trk'], Qt.UserRole, False)  # Store status boolean (False = not checked)

        item.setIcon(cols['ext'], cancel_icon)
        item.setData(cols['ext'], Qt.UserRole, False)  # Store status boolean (False = not checked)

        # quality - left blank (no icon at all) rather than a false-looking
        # "bad" cancel icon, until an actual quality check has run for this
        # object (see _refresh_quality_icon/_compute_tracking_quality) -
        # there's nothing to report yet before that.

        duplicate_button = qtw.QPushButton('Dup')
        duplicate_button.setFixedSize(48, 30)
        duplicate_button.setToolTip('Duplicate this object (points, masks, DPs) into a new row')
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
            self._quality_flags.pop(obj_id, None)
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

    def _compute_tracking_quality(self, obj_id):
        """Per-frame mask-area quality check for object `obj_id`, run right
        after tracking (see handle_finished_sam) - applies the same
        Dilate/Erode/Edge Detection/Mesh post-processing extraction itself
        would (see apply_edge_mask), so flagged frames reflect what would
        actually get extracted, not just SAM2's own raw per-frame mask.
        See io.flag_anomalous_mask_areas for the flagging heuristic
        itself. Returns [] if this object isn't tracked at all yet.
        Frame indices returned are global (this object's own `frame_idx`/
        `end` range only - the mask array itself is allocated at full
        dataset length, so frames outside that range are just unused
        padding, not something a quality check has anything to say about)."""
        mask_stack = self.df_obj.at[obj_id, 'mask']
        if mask_stack is None or not isinstance(mask_stack, np.ndarray):
            return []
        beg = min(self.df_obj.at[obj_id, 'frame_idx'])
        end = min(int(self.df_obj.at[obj_id, 'end']), len(mask_stack))
        if end <= beg:
            return []
        areas = np.array([self.apply_edge_mask(mask_stack[i], obj_id, i).sum()
                          for i in range(beg, end)], dtype=float)
        return [beg + f for f in io.flag_anomalous_mask_areas(areas)]

    def _refresh_quality_icon(self, obj_id):
        """Set object `obj_id`'s "Qlty" column icon from self._quality_flags
        - a warning icon if any frame is flagged, a plain check if none are
        (still distinguishable from the blank/no-icon-yet state an object
        starts in before its first quality check - see add_item_tree)."""
        row_index = self.df_obj.index.get_loc(obj_id)
        item = self.tree_objects.topLevelItem(row_index)
        if item is None:
            return
        col = self.cols_tree.index('qlty')
        flagged = self._quality_flags.get(obj_id) or []
        icon = self.style().standardIcon(
            self.style().SP_MessageBoxWarning if flagged else self.style().SP_DialogApplyButton)
        item.setIcon(col, icon)
        tip = (f'{len(flagged)} possibly mistracked frame(s): {", ".join(map(str, flagged[:20]))}'
              + (f' (+{len(flagged) - 20} more)' if len(flagged) > 20 else '')) if flagged \
            else 'No flagged frames'
        item.setToolTip(col, tip)

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
        edge_settings = self._edge_settings_for(obj_id)
        mesh_settings = self._mesh_settings_for(obj_id)
        dilate_erode_settings = self._dilate_erode_settings_for(obj_id)
        # Contrast-only (no denoise) - MaskEditDialog applies its own
        # Denoise box fresh on top of this, seeded from box_contrast's own
        # current state below, so its preview starts out looking the same
        # as self.imgs_8bit (which already has that denoise baked in)
        # without double-applying it - see MaskEditDialog's class docstring.
        bg_stack_contrast_only = io.convert_to_8bit(self.s_navSignal, **self.box_contrast.get_kwargs()).data
        dialog = MaskEditDialog(self, mask_stack, bg_stack=bg_stack_contrast_only,
                                start_frame=self.slider_imgNo.value(), logger=self.logger,
                                default_mask_stack=default_mask_stack, edge_settings=edge_settings,
                                mesh_settings=mesh_settings, dilate_erode_settings=dilate_erode_settings,
                                denoise_state=self.box_contrast.box_denoise.get_state())
        if dialog.exec_() == qtw.QDialog.Accepted:
            self.df_obj.at[obj_id, 'mask'] = dialog.get_mask_stack()
            # Mesh/Dilate-Erode/Edge Detection are all per-object (no
            # main-tab equivalent to sync back to anymore - SAM2 masks
            # aren't threshold-derived either, so unlike ROI Tracker's
            # MaskEditDialog there's nothing left here that needs a
            # _apply_dialog_settings_to_ui-style sync at all) - they
            # round-trip straight into this object's own columns instead.
            self.df_obj.at[obj_id, 'mesh'] = dialog.get_mesh_settings()
            self.df_obj.at[obj_id, 'dilate_erode'] = dialog.get_dilate_erode_settings()
            self.df_obj.at[obj_id, 'edge'] = dialog.get_edge_settings()
            self.logger.info('Fine-tuned mask saved for object %d.', obj_id)
            self.update_canvas()

    def on_item_check_changed(self, item):
        """tree_objects.itemChanged handler - unlike QTreeWidget's own
        itemChanged(item, column), a real QTableWidgetItem's own signal
        only carries the cell itself; its row/column give which property
        (self.cols_tree[item.row()]) and which object-column this cell
        belongs to."""
        if item.row() != self.cols_tree.index('use'):
            return  # some other cell changed, not the 'use' checkbox row
        idx_item = self.tree_objects.item(self.cols_tree.index('idx'), item.column())
        if idx_item is None or not idx_item.text():
            return
        idx = int(idx_item.text())
        self.df_obj.at[idx, 'use'] = 1 if item.checkState() == Qt.Checked else 0
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
        cursor = {'add_point': Qt.PointingHandCursor,
                  'remove_points_roi': Qt.CrossCursor}.get(self.ribbon.active_tool)
        self.canvas.setCursor(cursor if cursor is not None else Qt.ArrowCursor)

    def on_click(self, event):
        """Canvas mouse-press handler: middle-click deletes the last added
        point; Ctrl+click on the DP plot (only while auto-centering is off)
        sets a manual diffraction-pattern center; Ctrl+click on the nav
        image (or a plain click while the ribbon's "Add point" tool is
        active) adds a positive (left) or negative (right) point to the
        selected object, starting a new object unless Shift is held; a
        click+drag while the ribbon's "Remove points" tool is active starts
        a box-select instead (finished in on_release) that deletes every
        point it contains."""
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
        if self.ribbon.active_tool == 'remove_points_roi' and event.button == 1:
            self.press = (event.xdata, event.ydata)
            if self._remove_points_rect is not None:
                self._remove_points_rect.remove()
            self._remove_points_rect = patches.Rectangle(
                self.press, 0, 0, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
            self.ax_nav.add_patch(self._remove_points_rect)
            # One full draw() "bakes in" the current state (nav image, the
            # just-added zero-size rect) into a cached background snapshot,
            # so on_motion can cheaply blit just the growing rectangle on
            # top of it instead of redrawing the whole figure on every
            # mouse-move (same pattern as Tab_ROI_on_4D's on_press/on_motion).
            self.canvas.draw()
            self._remove_points_bg = self.canvas.copy_from_bbox(self.ax_nav.bbox)
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
                                    None, None, None, None, None, None, None, None]
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

    def on_motion(self, event):
        """Resize the in-progress "Remove points" box as the mouse moves,
        blitting just the rectangle onto the cached background for speed
        (see on_click/on_release)."""
        if self.press is None or self._remove_points_rect is None or event.inaxes is None:
            return
        x0, y0 = self.press
        width = event.xdata - x0
        height = event.ydata - y0
        self._remove_points_rect.set_width(width)
        self._remove_points_rect.set_height(height)
        self._remove_points_rect.set_xy((x0, y0))
        self.canvas.restore_region(self._remove_points_bg)
        self.ax_nav.draw_artist(self._remove_points_rect)
        self.canvas.blit(self.ax_nav.bbox)

    def on_release(self, event):
        """Finalize the "Remove points" box on mouse-up (see on_click) and
        delete every point (any object) on the currently-displayed frame
        that falls inside it."""
        if self.press is None or self._remove_points_rect is None:
            return
        x0, y0 = self.press
        self.press = None
        self._remove_points_rect.remove()
        self._remove_points_rect = None
        if event.xdata is None or event.ydata is None:
            self.canvas.draw_idle()
            return
        xlo, xhi = sorted((x0, event.xdata))
        ylo, yhi = sorted((y0, event.ydata))
        imgNo = self.slider_imgNo.value()
        n_removed = self._remove_points_in_box(xlo, xhi, ylo, yhi, imgNo)
        self.update_canvas(imgNo)
        if n_removed:
            self.logger.info('Removed %d point(s) within the drawn box on frame %d.',
                              n_removed, imgNo)

    def _remove_points_in_box(self, xlo, xhi, ylo, yhi, imgNo):
        """Delete every point of every object on frame `imgNo` whose (x, y)
        falls within [xlo, xhi] x [ylo, yhi] - used by the ribbon's "Remove
        points" box-select tool (on_click/on_motion/on_release). Drops an
        object entirely if this removes its last remaining point anywhere.
        Returns the number of points removed."""
        n_removed = 0
        for idx in list(self.df_obj.index):
            frame_idx = self.df_obj.at[idx, 'frame_idx']
            points = self.df_obj.at[idx, 'points']
            labels = self.df_obj.at[idx, 'labels']
            keep = [i for i, (fr, p) in enumerate(zip(frame_idx, points))
                    if not (fr == imgNo and xlo <= p[0] <= xhi and ylo <= p[1] <= yhi)]
            if len(keep) == len(frame_idx):
                continue
            n_removed += len(frame_idx) - len(keep)
            if not keep:
                self.delete_tree_item(str(idx))
                self.df_obj = self.df_obj.drop(idx)
                continue
            self.df_obj.at[idx, 'frame_idx'] = [frame_idx[i] for i in keep]
            self.df_obj.at[idx, 'points'] = [points[i] for i in keep]
            self.df_obj.at[idx, 'labels'] = [labels[i] for i in keep]
            # Keep the tree row's frame-list text in sync (mirrors on_click's
            # own update after appending a point).
            for i in range(self.tree_objects.topLevelItemCount()):
                item = self.tree_objects.topLevelItem(i)
                if item.text(1) == str(idx):
                    item.setText(2, str(self.df_obj.at[idx, 'frame_idx']))
                    break
        return n_removed

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
        self._denoise_bg = None  # view changed - see _blit_current_frame_display
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

    def _step_frame(self, delta):
        """Previous/Next Frame buttons: move the slider by one frame,
        clamped to its range - mirrors MaskEditDialog's own _step_frame."""
        self.slider_imgNo.setValue(int(np.clip(
            self.slider_imgNo.value() + delta,
            self.slider_imgNo.minimum(), self.slider_imgNo.maximum())))

    def update_canvas(self, imgNo=None, obj_id=None):
        """Redraw the nav/segmentation/DP panels for `imgNo` (default: slider
        value) and `obj_id` (default: selected object): shows the nav frame,
        the object's mask (tracked or single-frame) and diffraction pattern
        if present, then draws the canvas."""
        # Invalidates the cheap denoise-preview blit's cached background
        # (see _blit_current_frame_display) - this full draw is about to
        # change things (selected object, masks, points, DP, titles) that
        # fast path doesn't itself redraw and would otherwise leave stale
        # underneath freshly blitted image data on the next parameter tweak.
        self._denoise_bg = None
        if imgNo is None:
            imgNo = self.slider_imgNo.value()
        if obj_id is None:
            try:
                item_selected = self.tree_objects.currentItem()
                obj_id = int(item_selected.text(1))
            except Exception:
                obj_id = None
        self.frame_flag_bar.set_flags(self._quality_flags.get(obj_id) if obj_id is not None else None)
        self.remove_plotted_points()
        # Displayed from imgs_8bit, not the raw imgs - see _apply_loaded_nav_signal.
        self.img_display['nav'].set_data(self.imgs_8bit[imgNo])
        self.ax_nav.set(title=f'Nav. Image No: {imgNo}')

        if obj_id is not None:
            self.plot_points(imgNo, obj_id)

        # Segmented panel mask overlay: "All Active Objects" ignores obj_id
        # entirely - every use==1 object's own mask composited at once,
        # regardless of selection (see _draw_all_object_masks). "Selected
        # Object" needs obj_id to actually point at a tracked/segmented
        # object, same conditions this used to gate directly on
        # `if obj_id is not None:`.
        if self.radio_maskAll.isChecked():
            self.img_display['seg'].set_data(self.imgs_8bit[imgNo])
            self.img_display['seg'].set_clim(vmin=self.imgs_8bit[imgNo].min(), vmax=self.imgs_8bit[imgNo].max())
            self._draw_all_object_masks(imgNo)
        elif obj_id is not None:
            # plot segmentation masks for video
            if (not np.all(pd.isna(self.df_obj.loc[obj_id, 'mask']))):
                self.img_display['seg'].set_data(self.imgs_8bit[imgNo])
                self.img_display['seg'].set_clim(vmin=self.imgs_8bit[imgNo].min(), vmax=self.imgs_8bit[imgNo].max())

                try:
                    self.show_mask(self.apply_edge_mask(self.df_obj.loc[obj_id, 'mask'][imgNo], obj_id, imgNo), obj_id)
                except Exception:
                    self.show_mask(self.img_zero)

            # plot segmentation masks for single images
            else:
                try:
                    self.img_display['seg'].set_data(self.imgs_8bit[imgNo])
                    self.img_display['seg'].set_clim(vmin=self.imgs_8bit[imgNo].min(), vmax=self.imgs_8bit[imgNo].max())
                    mask = self.apply_edge_mask(self.df_obj.loc[obj_id, 'single_mask'][imgNo], obj_id, imgNo)
                    self.show_mask(mask, 0)
                except Exception:
                    self.img_display['seg'].set_data(self.img_zero)

        if obj_id is not None:
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

        # The scale bar/rings are static across frames like update_canvas's
        # own titles/masks/etc. - see _blit_current_frame_display.
        self._denoise_bg = None
        self.canvas.draw()

    def show_help_dialog(self):
        """Ribbon "?" tool: shortcuts/mouse controls for this tab, moved
        here from each subplot's own xlabel (see the canvas-setup history) -
        crowded, and on a narrow window two adjacent subplots' multi-line
        hints could visibly run into each other."""
        self.show_shortcuts_dialog(
            'Nav. Image:\n'
            '  Hold "Ctrl" + Left Click  ->  Positive point\n'
            '  Hold "Ctrl" + Right Click  ->  Negative point\n'
            '  Add "Shift"  ->  Add points to the selected/existing object\n'
            '  Middle Click  ->  Delete last point\n'
            '\n'
            'Diffraction Pattern:\n'
            '  Click "Center" (Input Parameters)  ->  Find the beam center\n'
            '  Hold "Ctrl" + Click  ->  Set the beam center manually\n'
            '\n'
            'Every axis:\n'
            '  Hold "Ctrl" + Scroll wheel  ->  Zoom the axis under the cursor\n'
            '\n'
            'The ribbon (right of the canvas) offers the same actions as icons - '
            'hover any icon for its own tooltip.')

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

    def _on_mask_mode_changed(self):
        self.update_canvas()

    def _draw_all_object_masks(self, imgNo):
        """"All Active Objects" mask-overlay mode (see the radio_maskAll/
        radio_maskSelected toggle above tree_objects): composite every
        active ("Use" checked) object's own mask for this frame at once -
        unlike show_mask's normal single-object view - each in its own
        tab10 color, with its index labeled at its own mask centroid.
        Ignores the table's own selection entirely - every active object
        is shown regardless of which one, if any, is currently selected."""
        for artist in self._all_mask_artists:
            try:
                artist.remove()
            except Exception:
                self.logger.debug('All-objects mask artist already removed.', exc_info=True)
        self._all_mask_artists = []

        h, w = self.imgs_8bit[imgNo].shape[-2:]
        composite = np.zeros((h, w, 4))
        cmap = plt.get_cmap('tab10')
        labels = []
        for obj_id2 in self.df_obj[self.df_obj['use'] == 1].index:
            mask_stack = self.df_obj.at[obj_id2, 'mask']
            single_stack = self.df_obj.at[obj_id2, 'single_mask']
            try:
                if isinstance(mask_stack, np.ndarray) and not np.all(pd.isna(mask_stack)):
                    mask = self.apply_edge_mask(mask_stack[imgNo], obj_id2, imgNo)
                elif isinstance(single_stack, np.ndarray) and not np.all(pd.isna(single_stack)):
                    mask = self.apply_edge_mask(single_stack[imgNo], obj_id2, imgNo)
                else:
                    continue
            except Exception:
                continue
            mask = np.asarray(mask, dtype=bool)
            if not mask.any():
                continue
            color = np.array([*cmap(obj_id2 % 10)[:3], 0.28])
            composite[mask] = color
            cx, cy = io.mask_centroid(mask)
            labels.append((obj_id2, cx, cy))

        self.img_display['seg_mask'].set_data(composite)
        for obj_id2, cx, cy in labels:
            text = self.ax_seg.text(cx, cy, str(obj_id2), color='white', fontsize=9,
                                    fontweight='bold', horizontalalignment='center',
                                    verticalalignment='center')
            self._all_mask_artists.append(text)

    def show_mask(self, mask, cmap_idx=0):
        """Render `mask` as a translucent RGBA overlay on the segmentation
        axis, colored by `cmap_idx` (tab10)."""
        # Stale index-label artists from a previous "All Active Objects"
        # mode (see _draw_all_object_masks) don't mean anything once back
        # in this single-object view.
        if self._all_mask_artists:
            for artist in self._all_mask_artists:
                try:
                    artist.remove()
                except Exception:
                    self.logger.debug('All-objects mask artist already removed.', exc_info=True)
            self._all_mask_artists = []
        cmap = plt.get_cmap("tab10")
        color = np.array([*cmap(cmap_idx)[:3], 0.2])
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

    def _group_objects_by_frame_range(self, df):
        """Group `df`'s object ids by their exact (start, end) frame range -
        SAM2's video predictor natively seeds any number of object ids
        against one encoded video, so objects sharing a range don't need a
        separate SAM2 run each (see initiate_video_segmentation). Returns an
        ordered {(start, end): [obj_id, ...]} dict."""
        groups = {}
        for idx in df.index:
            st = min(df.loc[idx, 'frame_idx'])
            end = df.loc[idx, 'end']
            groups.setdefault((st, end), []).append(idx)
        return groups

    def initiate_video_segmentation(self):
        """Kick off SAM2 tracking for every "used" object: objects sharing
        the exact same start/end frame range are batched into ONE SAM2
        video-predictor run (different object ids seeded against the same
        encoded video) instead of a separate run each - see
        _group_objects_by_frame_range(). Each group's frame range is split
        into stack_num-sized chunks, each chunk's frames exported as JPGs
        ONCE PER GROUP (not once per object) in the background, with every
        member object's points/labels needed to seed segmentation recorded
        in that chunk's seg_input.pkl (df_toSegment). run_video_segmentation()
        launches the actual SAM2 subprocesses once JPG export finishes."""
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

        groups = self._group_objects_by_frame_range(df)
        self.logger.info(
            'Starting SAM2 tracking for %d object(s) in %d batch(es) '
            '(stack size %d frames)...', len(df), len(groups), self.stack_num)

        self.total_threads_jpg = 0
        for (st, end), obj_ids in groups.items():
            imgs = self.imgs_8bit[st:end]
            arr_stack = np.arange(0, len(imgs), self.stack_num)
            self.total_threads_jpg += len(arr_stack)

        self.df_toSegment = pd.DataFrame(data=[], columns=[
            'path_jpg', 'obj_ids', 'stack_num', 'mask'])
        self.df_toSegment = self.df_toSegment.astype({
            'path_jpg': str,
            'obj_ids': object,
            'stack_num': int,
            'mask': object})

        self.worker_count_jpg = 0
        i_c = 0
        for g_i, ((st, end), obj_ids) in enumerate(groups.items()):
            imgs = self.imgs_8bit[st:end]
            arr_stack = np.arange(0, len(imgs), self.stack_num)
            arr_stack = np.append(arr_stack, len(imgs))
            # Allocate 'mask' at full dataset length (not just this object's
            # st:end range) so every consumer that indexes it by the GLOBAL
            # frame number (update_canvas, extract_3ded, create_clip_tracking_
            # with_mask, ...) lines up correctly, instead of needing a local/
            # global offset conversion that was easy to get wrong in one place
            # and miss in another.
            for obj_id in obj_ids:
                self.df_obj.at[obj_id, 'mask'] = np.zeros(self.imgs_8bit.shape, dtype=bool)
            path_group = os.path.join(self.path_jpg, f'group_{g_i}')
            if os.path.isdir(path_group):
                shutil.rmtree(path_group)
            os.mkdir(path_group)

            for i_fld, _ in enumerate(arr_stack[:-1]):
                path_stack = os.path.join(path_group, f'{i_fld}')
                if os.path.isdir(path_stack):
                    shutil.rmtree(path_stack)
                os.mkdir(path_stack)

                st_2 = arr_stack[i_fld]
                end_2 = arr_stack[i_fld+1]
                imgs_stack = imgs[st_2:end_2]

                objects = {}
                for obj_id in obj_ids:
                    frame_idx, points, labels = df.loc[obj_id,
                       ['frame_idx', 'points', 'labels']]
                    frame_idx = np.array(frame_idx) - st - st_2
                    cond = np.where((frame_idx>=0) & (frame_idx<self.stack_num))
                    objects[int(obj_id)] = {
                        'frame_idx': frame_idx[cond],
                        'points': np.array(points)[cond],
                        'labels': np.array(labels)[cond]}

                self.df_toSegment.loc[i_c] = [path_stack, list(obj_ids), i_fld, None]
                fn = os.path.join(path_stack, 'seg_input.pkl')
                with open(fn, 'wb') as f:
                    pickle.dump({'path_jpg': path_stack, 'objects': objects}, f)
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
            "app doesn't bundle. Use Help > Set Up SAM2... in the menu bar "
            'to install them - it runs the required pip commands for you '
            'and shows the progress. See INSTALL.md for the manual steps if '
            "you'd rather run them yourself.")

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
                # One key per batched object in this chunk (see
                # _group_objects_by_frame_range/initiate_video_segmentation) -
                # f'obj_{obj_id}' for every id in this row's obj_ids.
                mask_by_obj = {int(k.split('_', 1)[1]): f[k] for k in f.files}
            self.df_toSegment.at[idx, 'mask'] = mask_by_obj
            # launch next segmentation
            if len(self.running_processes_sam) != self.running_processes_sam_total:
                for idx in self.df_toSegment.index.sort_values():
                    #TODO already put the masks in df_obj
                    if idx not in self.running_processes_sam:
                        path = self.df_toSegment.loc[idx, 'path_jpg']
                        self.launch_next_video_seg(path, idx)
                        break
            else: # finished
                all_obj_ids = set()
                for idx in self.df_toSegment.index:
                    i_c = self.df_toSegment.loc[idx, 'stack_num']
                    mask_by_obj = self.df_toSegment.loc[idx, 'mask']
                    for obj_id in self.df_toSegment.loc[idx, 'obj_ids']:
                        all_obj_ids.add(obj_id)
                        beg = min(self.df_obj.loc[obj_id, 'frame_idx'])
                        obj_mask = mask_by_obj[int(obj_id)]

                        # 'mask' is now allocated at full dataset length (global
                        # frame numbers), so each per-stack chunk is placed at
                        # its own global offset: beg (object's start frame) +
                        # i_c full stacks in. frame_num (rather than assuming a
                        # full self.stack_num) makes this correct even for the
                        # last chunk of an object, which is often shorter than
                        # a full stack.
                        frame_num = len(obj_mask)
                        start = beg + i_c * self.stack_num
                        # Kept raw (un-eroded) here - "Edge Only" is applied
                        # as a view at display time (update_canvas) and at
                        # extraction/save time instead, so toggling it later
                        # doesn't require re-tracking to see the effect, and
                        # can be turned back off without losing the original
                        # SAM2 result. See apply_edge_mask()/apply_edge_mask_stack().
                        self.df_obj.at[obj_id, 'mask'][
                            start : start + frame_num] = obj_mask
                flagged_summary = {}
                for obj_id in all_obj_ids:
                    # Snapshot the freshly-tracked (still un-eroded, un-edited)
                    # result as this object's permanent "Reset to Tracking"
                    # target for MaskEditDialog - taken here, before any
                    # fine-tune-dialog edits can ever touch 'mask'.
                    self.df_obj.at[obj_id, 'mask_default'] = self.df_obj.at[obj_id, 'mask'].copy()
                    row_index = self.df_obj.index.get_loc(obj_id)
                    self.toggle_tree_icon(row_index, 'trk', True)
                    # Tracking-quality check (see _compute_tracking_quality).
                    flags = self._compute_tracking_quality(obj_id)
                    self._quality_flags[obj_id] = flags
                    self._refresh_quality_icon(obj_id)
                    if flags:
                        flagged_summary[obj_id] = flags

                n_objects = len(all_obj_ids)
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

                if flagged_summary:
                    lines = [f'  Object {obj_id}: {len(flags)} frame(s) '
                            f'({", ".join(map(str, flags[:10]))}{", ..." if len(flags) > 10 else ""})'
                            for obj_id, flags in flagged_summary.items()]
                    qtw.QMessageBox.warning(self, 'Tracking Quality Check',
                        f'{len(flagged_summary)} of {n_objects} object(s) have possibly '
                        'mistracked frames (an abrupt mask-area change, or the mask '
                        'vanishing entirely):\n\n' + '\n'.join(lines) +
                        '\n\nCheck them via the red marks on the frame-flag bar under the '
                        'slider (click one to jump there), or the "Qlty" column in the '
                        'object list.')
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
#%% auto detector
    def launch_auto_detector(self):
        """Launch the SAM2 Auto Detector popup (see
        ui_tabs/sam2_auto_detector_widget.py) on the currently-displayed
        frame - mirrors ROI Tracker's launch_auto_detector."""
        imgNo = self.slider_imgNo.value()
        pathSave = self.lineEdit_dir_save.text()
        self.auto_detector = SAM2AutoDetectorWidget(
            self.imgs_8bit[imgNo], imgNo, pathSave, self._ensure_sam2_ready, self.logger)
        self.auto_detector.final_objects.connect(self.receive_auto_detected_objects)
        self.auto_detector.show()

    def receive_auto_detected_objects(self, objects):
        """Signal handler for SAM2AutoDetectorWidget.final_objects: add each
        accepted candidate as a new object, seeded with the points/labels
        the widget computed (center of mass + optional background points)
        on the frame it was run on - same df_obj row shape as a manually
        Ctrl+clicked new object (see on_click)."""
        if not objects:
            return
        for obj in objects:
            idx = 1
            while idx in self.df_obj.index:
                idx += 1
            fr_idx = [obj['frame_idx']] * len(obj['points'])
            self.df_obj.loc[idx] = [1, idx, fr_idx, obj['points'], obj['labels'],
                                    len(self.imgs), None, None, None, None, None, None, None, None]
            self.add_item_tree(idx, fr_idx)
        self.update_canvas()
        self.canvas.draw()
        self.logger.info('Auto Detector added %d object(s) on frame %d.',
                          len(objects), objects[0]['frame_idx'])
#%% image segmentation
    def initiate_image_segmentation(self):
        """Segment every "used" object that has at least one point on the
        current frame, using SAM2's single-image predictor (no cross-frame
        propagation) — unlike Track, which runs SAM2's video predictor
        across every frame. Objects applicable to this frame are batched
        into ONE SAM2 run (the image is only embedded once - see
        worker_sam.py's 'image' branch) instead of a separate run each."""
        imgNo = self.slider_imgNo.value()
        df = self.df_obj[self.df_obj.use == 1]
        objects = {}
        for obj_id in df.index:
            frames = np.array(df.loc[obj_id, 'frame_idx'])
            toPlot = np.where(frames == imgNo)
            points = np.array(df.loc[obj_id, 'points'])[toPlot]
            if len(points) == 0:
                continue
            labels = np.array(df.loc[obj_id, 'labels'])[toPlot]
            objects[int(obj_id)] = {'points': points, 'labels': labels}

        if not objects:
            self.logger.warning(
                'Single-image segmentation requested but no "used" object has '
                'points on frame %d.', imgNo)
            qtw.QMessageBox.critical(self, 'No Points on This Frame',
                f'No object has annotated points on frame {imgNo}.\n'
                'Hold Ctrl and click on the image to add at least one point '
                'on this frame first.')
            return

        self.logger.info(
            'Starting SAM2 single-image segmentation for %d object(s) on '
            'frame %d...', len(objects), imgNo)
        self.button_runSeg_img.setDisabled(True)

        def _on_checkpoint_failed(error_msg):
            self.button_runSeg_img.setEnabled(True)
            qtw.QMessageBox.warning(self, 'SAM2 Checkpoint Download Failed',
                'Could not download the SAM2 model checkpoint - check your '
                'internet connection and see the log console for details.')

        self._ensure_sam2_ready(
            lambda: self._launch_image_segmentation(objects, imgNo),
            _on_checkpoint_failed)

    def _launch_image_segmentation(self, objects, imgNo):
        """Build the single-image seg_input.pkl (every batched object's
        points/labels) and launch worker_sam.py - split out from
        initiate_image_segmentation() so it can run only after
        _ensure_sam2_ready() confirms the checkpoint is present."""
        pathSave = self.lineEdit_dir_save.text()
        if not os.path.isdir(pathSave):
            os.mkdir(pathSave)
        path_seg = os.path.join(pathSave, 'JPG Images', 'single_image_seg')
        if os.path.isdir(path_seg):
            shutil.rmtree(path_seg)
        os.makedirs(path_seg)

        seg_input = {'image': self.imgs_8bit[imgNo], 'objects': objects}
        with open(os.path.join(path_seg, 'seg_input.pkl'), 'wb') as f:
            pickle.dump(seg_input, f)

        key = f'img_batch_{imgNo}'
        program, arguments = worker_command('sam', ['image', path_seg, str(imgNo)])
        process_sam = QProcess(self)
        process_sam.setProgram(program)
        process_sam.setArguments(arguments)
        process_sam.readyReadStandardError.connect(lambda:
                            self.handle_error_sam(process_sam, key))
        process_sam.finished.connect(lambda exit_code, exit_status:
                     self.handle_finished_image_sam(
                     process_sam, key, imgNo, exit_code, exit_status))
        process_sam.errorOccurred.connect(lambda error:
                     self.process_failed_image_sam(error, key, imgNo))

        if not hasattr(self, 'running_processes_sam'):
            self.running_processes_sam = {}
        self.running_processes_sam[key] = process_sam
        process_sam.start()

    def process_failed_image_sam(self, error, key, imgNo):
        self.running_processes_sam.pop(key, None)
        self.button_runSeg_img.setEnabled(True)
        self.logger.error("[%s] Single-image segmentation QProcess error occurred: %s",
                           key, error)

    def handle_finished_image_sam(self, process, key, imgNo, exit_code, exit_status):
        """SAM2 single-image subprocess completion handler: load the
        resulting masks (one per batched object - see
        initiate_image_segmentation) into each object's `single_mask` at
        `imgNo`, and refresh the canvas."""
        self.running_processes_sam.pop(key, None)
        self.button_runSeg_img.setEnabled(True)
        self.logger.info(
            "[%s] Single-image segmentation process finished with exit code %s, status %s",
            key, exit_code, exit_status)

        data = process.readAllStandardOutput()
        text = bytes(data).decode("utf-8")
        self.logger.info('text: %s', text)
        try:
            result = json.loads(text.strip())
            if result.get('error') == 'missing_dependency':
                self._show_missing_dependency_dialog(result['message'])
                return
            fn_output = result["path"]
            obj_ids = result["obj_ids"]
            with np.load(fn_output) as f:
                for obj_id in obj_ids:
                    mask = f[f'obj_{obj_id}']
                    # Kept raw (un-eroded) - see apply_edge_mask()/update_canvas().
                    if not isinstance(self.df_obj.at[obj_id, 'single_mask'], np.ndarray):
                        self.df_obj.at[obj_id, 'single_mask'] = np.zeros(self.imgs_8bit.shape, dtype=bool)
                    self.df_obj.loc[obj_id, 'single_mask'][imgNo] = mask
            self.logger.info(
                'SAM2 single-image segmentation completed successfully for '
                '%d object(s), frame %d.', len(obj_ids), imgNo)
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
        # button_fineTuneMask lives in the left object-list panel (see
        # init_ui), not box_3ded, so the sweep above doesn't reach it -
        # toggled explicitly here instead (mirrors ROI Tracker's identical
        # button_fineTuneMask/button_blobSettings handling in its own
        # disable_3ded_widgets).
        self.button_fineTuneMask.setEnabled(state)
    
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
        pattern = '*' if ext == 'All Files' else '*' + glob_ext_for_dtype(ext)
        return sorted(glob(os.path.join(path_4d, pattern)))

    def extract_3ded(self):
        """Kick off background 3DED extraction for every "used" object:
        resolve the 4D signal files (plain folder listing, or the smart-scan
        match), then process one object at a time - each object's own
        frames pooled together via a single batch-driver process (using
        the full configured worker count), the next object's batch only
        starting once the current one fully finishes (see
        _launch_next_object_batch) - per explicit request, rather than
        interleaving every object's frames into one shared queue."""
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
        dtype = resolve_hdf5_dtype(fns_4d[0], self.combo_dtype_4d.currentText())

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

        # One task-list batch per enabled object (df.index), queued here
        # and launched one at a time by _launch_next_object_batch - each
        # object's own frames still parallelize across the full worker
        # pool (self.n_workers below), only the *launch* is sequential
        # across objects. Mask .npy files are written lazily, just before
        # each object's own batch launches (see _launch_next_object_batch),
        # into that batch's own fresh temp dir - not eagerly here for
        # every object at once.
        self._obj_queue = deque()
        self._obj_task_specs = {}
        for idx in df.index:
            self.df_obj.at[idx, 'dp'] = np.zeros((len(self.imgs), shape_d_x,
                                                  shape_d_y), dtype='uint32')
            beg = min(df.loc[idx].frame_idx)
            end = df.loc[idx].end
            specs = []
            for i_fr, fn in enumerate(fns_4d[beg:end]):
                i_fr += beg
                fn_pattern = fns_pattern_4d[i_fr] if fns_pattern_4d is not None else None
                specs.append({
                    'i_index': i_fr, 'fn': fn,
                    'roi': [int(v) for v in df.loc[idx, 'rois'][i_fr]],
                    'dtype': dtype, 'scanSize': list(scanSize),
                    'fn_pattern': fn_pattern, 'det_shape': [shape_d_x, shape_d_y],
                })
            if specs:
                self._obj_queue.append(idx)
                self._obj_task_specs[idx] = specs

        self.n_workers = self.spinbox_threadNum.value()
        self._launch_next_object_batch()

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
        dtype = resolve_hdf5_dtype(fn, self.combo_dtype_4d.currentText())

        scanSize = self.get_scan_size()
        if scanSize is None:  # "Auto": fall back to the loaded nav signal's own shape
            scanSize = tuple(self.imgs.shape[1:])

        mask = self.apply_edge_mask(mask, obj_id, imgNo)

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
        # Force the Clipping Thresholds to reset for this DP (see
        # _apply_dp_clip's own reset=not self._dp_clip_initialized
        # convention) - normally left alone across a frame scrub so a
        # manually-tuned threshold persists through an already-extracted DP
        # stack, but a one-off current-frame check is a fresh, unrelated
        # intensity range (could be a different object/frame entirely) that
        # the OLD thresholds may not even overlap with (e.g. all-clipped-
        # away or no visible clipping at all) - without this, "the DP loads
        # but the thresholds don't update" is exactly what the user sees
        # (same issue reported and fixed for the ROI Tracker tab).
        self._dp_clip_initialized = False
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

    def _launch_next_object_batch(self):
        """Pop the next queued object and launch its 3DED extraction batch -
        one JSON task-list (worker_pool_utils.write_tasks_json), one driver
        process (worker_extract_frame_batch.py) running its own internal
        process pool over every one of that object's frames at once. Only
        called again (by _handle_3ded_driver_finished) once the current
        object's batch has completely finished - if the queue is empty,
        every enabled object has been processed."""
        if not self._obj_queue:
            self._finalize_3ded_extraction()
            return
        idx = self._obj_queue.popleft()
        self._current_obj_idx = idx
        temp_dir = tempfile.mkdtemp(prefix='edyssey_3ded_')
        self._current_obj_temp_dir = temp_dir
        tasks = self._obj_task_specs.pop(idx)
        for task in tasks:
            mask = self.apply_edge_mask(self.df_obj.loc[idx, 'mask'][task['i_index']], idx, task['i_index'])
            mask_path = os.path.join(temp_dir, f"mask_f{task['i_index']}.npy")
            np.save(mask_path, mask)
            task['mask_path'] = mask_path
        tasks_path = os.path.join(temp_dir, 'tasks.json')
        wpu.write_tasks_json(tasks_path, tasks)
        self._launch_3ded_batch_driver(tasks_path, temp_dir)

    def _launch_3ded_batch_driver(self, tasks_path, temp_dir):
        """Start the single worker_extract_frame_batch.py driver QProcess
        for the current object's batch, wiring up the Job Object handshake,
        incremental stdout progress reader, and finished/error handlers -
        same pattern as tab_create_navSignal.py's Navigator batch driver."""
        program, arguments = worker_command(
            'extract_frame_batch', [tasks_path, temp_dir, self.n_workers])
        process = QProcess()
        process.setProgram(program)
        process.setArguments(arguments)
        self._3ded_driver_process = process
        self._3ded_job_handle = None
        self._3ded_worker_pids = []
        self._3ded_stdout_buf = bytearray()
        process.started.connect(lambda: self._on_3ded_driver_started(process))
        process.readyReadStandardOutput.connect(lambda: self._read_3ded_driver_stdout(process))
        process.readyReadStandardError.connect(lambda: self.handle_error_3ded(process))
        process.finished.connect(lambda: self._handle_3ded_driver_finished(process))
        process.errorOccurred.connect(lambda error: self._3ded_driver_failed_to_start(process, error))
        process.start()

    def _on_3ded_driver_started(self, process):
        """Assign the just-started driver process to a Windows Job Object
        before letting it build its own pool - see
        tab_create_navSignal.py's _on_nav_driver_started for the full
        rationale (job membership isn't retroactive)."""
        job_handle = wpu.create_job_object()
        if job_handle is not None and not wpu.assign_process_to_job(job_handle, int(process.processId())):
            job_handle = None
        self._3ded_job_handle = job_handle
        process.write(b'GO\n')

    def _read_3ded_driver_stdout(self, process):
        """Incremental line-buffering reader for the batch driver's stdout -
        see worker_pool_utils.run_pool_batch's DONE/FAIL/WORKERPID
        protocol."""
        self._3ded_stdout_buf += bytes(process.readAllStandardOutput())
        while b'\n' in self._3ded_stdout_buf:
            line, _, rest = self._3ded_stdout_buf.partition(b'\n')
            self._3ded_stdout_buf = bytearray(rest)
            self._handle_3ded_progress_line(line.decode('utf-8', errors='replace').strip())

    def _handle_3ded_progress_line(self, line):
        if not line:
            return
        parts = line.split(' ', 2)
        tag = parts[0]
        if tag == 'DONE' and len(parts) >= 2:
            self._on_3ded_task_done(int(parts[1]))
        elif tag == 'FAIL' and len(parts) >= 2:
            self._on_3ded_task_failed(int(parts[1]), parts[2] if len(parts) > 2 else '')
        elif tag == 'WORKERPID' and len(parts) >= 2:
            self._3ded_worker_pids.append(int(parts[1]))

    def _on_3ded_task_done(self, i_fr):
        idx = self._current_obj_idx
        fn_npy = os.path.join(self._current_obj_temp_dir, f'{i_fr}.npy')
        try:
            self.df_obj.at[idx, 'dp'][i_fr] = np.load(fn_npy)
            os.remove(fn_npy)
        except Exception as e:
            self._3ded_failed = True
            self.logger.error('Failed to load DP for object %s frame %s: %s', idx, i_fr, e)
        self.tomo_counter += 1
        self.update_progress_bar(self.tomo_counter, self.tomo_counter_total)

    def _on_3ded_task_failed(self, i_fr, message):
        self._3ded_failed = True
        self.logger.error('3DED extraction failed for object %s frame %s: %s',
                          self._current_obj_idx, i_fr, message)
        self.tomo_counter += 1
        self.update_progress_bar(self.tomo_counter, self.tomo_counter_total)

    def handle_error_3ded(self, process):
        # worker_extract_frame.py loads tpx3 via eventem, whose progress bar
        # (and any other routine diagnostics) writes straight to stderr on
        # every run, success or failure - this is not itself an error (see
        # ProcessStderrBuffer). A worker that genuinely fails to produce
        # output is instead caught via a FAIL line - see
        # _handle_3ded_progress_line/_on_3ded_task_failed.
        if self._cancelling:
            return
        self._stderr_buffer.log_info(process, self.logger, 'Worker')

    def _handle_3ded_driver_finished(self, process):
        """Drain remaining stdout, release the Job Object, clean up this
        object's own temp dir, and - unless this was a Cancel - mark it
        extracted in the tree and move on to the next queued object (or
        finalize the whole run if none are left)."""
        self._3ded_stdout_buf += bytes(process.readAllStandardOutput())
        while b'\n' in self._3ded_stdout_buf:
            line, _, rest = self._3ded_stdout_buf.partition(b'\n')
            self._3ded_stdout_buf = bytearray(rest)
            self._handle_3ded_progress_line(line.decode('utf-8', errors='replace').strip())
        process.deleteLater()
        wpu.kill_job(self._3ded_job_handle)
        self._3ded_job_handle = None
        idx = self._current_obj_idx
        temp_dir = getattr(self, '_current_obj_temp_dir', None)
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        if self._cancelling:
            return
        self.toggle_tree_icon(self.df_obj.index.get_loc(idx), 'ext', True)
        self._launch_next_object_batch()

    def _3ded_driver_failed_to_start(self, process, error):
        """Handle the current object's batch driver failing to start - with
        one driver process per object instead of one per (object, frame)
        task, this now simply means that one object's extraction never
        ran, rather than needing to individually track and skip past a
        single stuck task slot."""
        if self._cancelling:
            process.deleteLater()
            return
        self._3ded_failed = True
        self.logger.error(
            '3DED extraction batch driver failed to start for object %s (error code %s).',
            self._current_obj_idx, error)
        process.deleteLater()
        temp_dir = getattr(self, '_current_obj_temp_dir', None)
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        qtw.QMessageBox.critical(self, 'Process Error',
            f'The 3DED extraction batch worker failed to start for object {self._current_obj_idx} '
            f'(error code {error}).\nCheck that Python is on PATH and '
            'worker_extract_frame_batch.py exists.')
        self.button_cancel.setDisabled(True)

    def _finalize_3ded_extraction(self):
        """Called once every enabled object's batch has been accounted for
        (successfully or not) - the tail end of what handle_finished_3ded
        used to do once tomo_counter reached tomo_counter_total, now
        triggered by the object queue emptying instead of a per-task
        counter."""
        self.toc = perf_counter()
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
        self.add_scalebar()
        self.button_cancel.setDisabled(True)
        if self.checkbox_autosave.isChecked():
            self.save_results()

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
        # button_fineTuneMask lives in the left object-list panel (see
        # init_ui), not box_3ded, so the sweep above doesn't reach it -
        # toggled explicitly here instead (mirrors ROI Tracker's identical
        # button_fineTuneMask/button_blobSettings handling in its own
        # disable_3ded_widgets).
        self.button_fineTuneMask.setDisabled(state)
    
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

    def _open_pets2_dialog(self, uncheck_on_cancel=True):
        """Open Pets2ParamsDialog pre-filled with voltage/exposure/pixel-size
        read from the current metadata and UI fields. `uncheck_on_cancel`
        (False when opened via button_checkPets2Options, which can be
        clicked regardless of the checkbox's own state) unchecks "Make
        *.pts2" again if the user cancels - only makes sense when this
        dialog is what's actually turning the feature on in the first
        place (on_makePets2_toggled's own trigger)."""
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
        elif uncheck_on_cancel:
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
            # Edge Detection/Dilate-Erode/Mesh are all per-object Segments
            # (see MaskEditDialog's class docstring/get_edge_settings()/
            # get_dilate_erode_settings()/get_mesh_settings()) - each is a
            # `{'segments': [...]}` dict already fully describing exactly
            # what was used for every frame range of this object's
            # extraction, so it's recorded here as-is.
            df['edge_detection'] = self._edge_settings_for(idx) or {'segments': []}
            df['mesh'] = self._mesh_settings_for(idx) or {'segments': []}
            df['dilate_erode'] = self._dilate_erode_settings_for(idx) or {'segments': []}
            df.to_json(os.path.join(path_save_objID, f'roi No {idx}.json'), orient='index', indent=4)
            if not (np.all(pd.isna(self.df_obj.loc[idx, 'rois']))):
                np.save(os.path.join(path_save_objID, 'rois.npy'),
                    self.df_obj.loc[idx, 'rois'])
            if not (np.all(pd.isna(self.df_obj.loc[idx, 'dp']))):
                # Saved as actually used for extraction (edge/mesh view
                # applied) - see apply_edge_mask_stack().
                np.save(os.path.join(path_save_objID, 'output_mask.npy'),
                        self.apply_edge_mask_stack(self.df_obj.loc[idx, 'mask'], idx))

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
                mask_effective = self.apply_edge_mask_stack(self.df_obj.loc[idx, 'mask'], idx)
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
    
    def kill_3ded_driver(self):
        """Forcefully kill the currently-running 3DED batch driver process
        for whichever object is in flight (if any) - along with every pool
        worker it spawned, via the Job Object it was assigned to on start
        plus a direct force-kill of each reported worker PID as a
        redundant backstop (kill_job() alone was empirically found not
        fully reliable at that - see worker_pool_utils.kill_pid's
        docstring). Must tolerate being called (e.g. from cleanup() on
        window close) before any extraction ever ran."""
        process = getattr(self, '_3ded_driver_process', None)
        if process is not None and process.state() != QProcess.NotRunning:
            wpu.kill_job(getattr(self, '_3ded_job_handle', None))
            self._3ded_job_handle = None
            for pid in getattr(self, '_3ded_worker_pids', []):
                wpu.kill_pid(pid)
            process.kill()

    def cancel_running_work(self):
        """Stop SAM2 tracking/segmentation or 3DED extraction and suppress
        the error popups that killing those workers would otherwise
        trigger.

        QThreadPool has no way to forcibly interrupt a runnable that has
        already started (only queued-but-not-started ones can be dropped),
        so an in-flight helper job will still finish in the background and
        is simply ignored when it does. The current object's 3DED batch
        driver is a real OS process tree though (see kill_3ded_driver), so
        that's actually killed outright, and no further queued objects are
        launched."""
        self._cancelling = True
        self.threadpool.clear()
        n_killed = len(getattr(self, 'running_processes_sam', {}))
        process = getattr(self, '_3ded_driver_process', None)
        if process is not None and process.state() != QProcess.NotRunning:
            n_killed += 1
        self.stop_processes()  # kills any running_processes_sam (SAM2 tracking/segmentation)
        self.kill_3ded_driver()
        if hasattr(self, '_obj_queue'):
            self._obj_queue.clear()
        self.button_cancel.setDisabled(True)
        self.button_runSeg_clip.setEnabled(True)
        self.button_runSeg_img.setEnabled(True)
        self.logger.warning('Cancelled by user (%d running worker process(es) killed).', n_killed)
        qtw.QMessageBox.information(self, 'Cancelled',
            'Tracking/segmentation/3DED extraction was cancelled.\n\n'
            'Any helper job already running in the background will still '
            'finish silently - only queued work and the running 3DED extraction '
            'batch were stopped.')

    def get_duplicate_state(self):
        """Snapshot of this tab's in-progress analysis, for "Duplicate
        Current Tab" (see EDyssey_MainWindow.duplicate_current_tab) - a
        synchronous, in-memory equivalent of Save Results/Load Saved
        Analysis (_on_saved_analysis_loaded), just enough to make the
        duplicate tab look and behave like this one immediately. Every
        mutable value (arrays, the object dataframe, dicts) is copied,
        never shared by reference, so the two tabs stay fully independent
        afterward - df_obj specifically is copied column-by-column (not
        deepcopy(DataFrame), which doesn't deep-copy object-dtype cell
        contents - see Tab_Tracking_CV2.get_duplicate_state's identical
        note, and add_item_tree's own duplicate-row handling elsewhere).

        Returns None if no navigation signal has been loaded yet (checked
        via self.imgs, set only once _apply_loaded_nav_signal() has run)."""
        if not isinstance(getattr(self, 'imgs', None), np.ndarray) or len(self.imgs) == 0:
            return None
        df_obj_rows = []
        for idx in self.df_obj.index:
            row = {col: deepcopy(self.df_obj.at[idx, col]) for col in self.cols_df}
            row['idx'] = idx
            df_obj_rows.append(row)
        return {
            # File/scan parameters
            'lineEdit_dir_navSignal': self.lineEdit_dir_navSignal.text(),
            'fn_navSignal': getattr(self, 'fn_navSignal', None),
            'lineEdit_dir_4d': self.lineEdit_dir_4d.text(),
            'lineEdit_dir_save': self.lineEdit_dir_save.text(),
            'combo_dtype_4d': self.combo_dtype_4d.currentText(),
            'spinbox_dwellTime_acquisition': self.spinbox_dwellTime_acquisition.value(),
            'checkbox_smartScan': self.checkbox_smartScan.isChecked(),
            'lineEdit_patternDir': self.lineEdit_patternDir.text(),
            'lineEdit_detectionDir': self.lineEdit_detectionDir.text(),
            'smart_scan_rows': ([dict(row) for row in self._smart_scan_rows]
                                if getattr(self, '_smart_scan_rows', None) else None),
            'smart_scan_summary': self.label_smartScanSummary.text(),
            'nav_4d_files': list(self._nav_4d_files) if self._nav_4d_files else None,
            'nav_4d_directory': self._nav_4d_directory,
            'checkbox_detectorSizeAuto': self.checkbox_detectorSizeAuto.isChecked(),
            'detectorSize': (self.spinbox_detectorSize_x.value(), self.spinbox_detectorSize_y.value()),
            'checkbox_scanSize': self.checkbox_scanSize.isChecked(),
            'scanSize_spin': (self.spinbox_scanSize_x.value(), self.spinbox_scanSize_y.value()),
            'metadata_path_override': self.metadata_path_override,
            'spinbox_metadataCount': self.spinbox_metadataCount.value(),
            'scale_real': self.lineEdit_scale_real.text(),
            'scale_recip': self.lineEdit_scale_recip.text(),
            'dp_center': self.dp_center,
            # Navigation signal + images
            's_navSignal': self.s_navSignal.deepcopy() if hasattr(self, 's_navSignal') else None,
            'imgs': self.imgs.copy(),
            'imgs_8bit': self.imgs_8bit.copy(),
            'imgNo': self.slider_imgNo.value(),
            'spinbox_stackNum': self.spinbox_stackNum.value(),
            # Contrast
            'contrast': self.box_contrast.get_state(),
            'clip_dp': self.clip_dp.get_state(),
            # Extraction settings - Edge Detection/Dilate-Erode/Mesh have no
            # main-tab widgets to copy anymore (all per-object Segments,
            # already riding along inside df_obj_rows below).
            'spinbox_threadNum': self.spinbox_threadNum.value(),
            'spinbox_fps': self.spinbox_fps.value(),
            'checkbox_autosave': self.checkbox_autosave.isChecked(),
            'checkbox_makePets2': self.checkbox_makePets2.isChecked(),
            'pets2_params': deepcopy(self.pets2_params),
            # Tracked/segmented objects
            'df_obj_rows': df_obj_rows,
        }

    def apply_duplicate_state(self, state):
        """Restore a dict from get_duplicate_state() into this (freshly
        constructed, otherwise-empty) tab, and redraw everything it
        touches so the tab looks right immediately - see that method's
        docstring. No-op on None/empty.

        checkbox_makePets2 (opens a modal dialog when checked) and every
        other widget restored here are set with signals blocked - the
        already-computed/copied results are applied directly, so none of
        their change handlers need to (re)run."""
        if not state:
            return
        self.lineEdit_dir_navSignal.setText(state['lineEdit_dir_navSignal'])
        self.fn_navSignal = state['fn_navSignal']
        self.lineEdit_dir_4d.setText(state['lineEdit_dir_4d'])
        self.lineEdit_dir_save.setText(state['lineEdit_dir_save'])
        idx = self.combo_dtype_4d.findText(state['combo_dtype_4d'])
        if idx >= 0:
            self.combo_dtype_4d.setCurrentIndex(idx)
        self.spinbox_dwellTime_acquisition.setValue(state['spinbox_dwellTime_acquisition'])
        self.checkbox_smartScan.setChecked(state['checkbox_smartScan'])
        self.lineEdit_patternDir.setText(state['lineEdit_patternDir'])
        self.lineEdit_detectionDir.setText(state['lineEdit_detectionDir'])
        self._smart_scan_rows = state['smart_scan_rows']
        self._set_smart_scan_summary(state['smart_scan_summary'])
        self._nav_4d_files = state['nav_4d_files']
        self._nav_4d_directory = state['nav_4d_directory']
        self.checkbox_detectorSizeAuto.setChecked(state['checkbox_detectorSizeAuto'])
        self.spinbox_detectorSize_x.setValue(state['detectorSize'][0])
        self.spinbox_detectorSize_y.setValue(state['detectorSize'][1])
        self.checkbox_scanSize.setChecked(state['checkbox_scanSize'])
        self.spinbox_scanSize_x.setValue(state['scanSize_spin'][0])
        self.spinbox_scanSize_y.setValue(state['scanSize_spin'][1])
        self.metadata_path_override = state['metadata_path_override']
        self.spinbox_metadataCount.setValue(state['spinbox_metadataCount'])
        self.lineEdit_scale_real.setText(state['scale_real'])
        self.lineEdit_scale_recip.setText(state['scale_recip'])
        self.dp_center = state['dp_center']

        # Navigation signal + images (mirrors the tail of
        # _apply_loaded_nav_signal(), minus the parts that would recompute
        # imgs_8bit from scratch)
        if state['s_navSignal'] is not None:
            self.s_navSignal = state['s_navSignal']
        self.imgs = state['imgs']
        self.imgs_8bit = state['imgs_8bit']
        self._dp_center_cache_key = None
        self.box_contrast.set_state(state['contrast'])

        self.spinbox_stackNum.blockSignals(True)
        self.spinbox_stackNum.setMaximum(len(self.imgs))
        self.spinbox_stackNum.setValue(state['spinbox_stackNum'])
        self.spinbox_stackNum.blockSignals(False)
        shape_x, shape_y = self.imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['nav'].set_clim(vmin=self.imgs_8bit.min(), vmax=self.imgs_8bit.max())
        self.img_display['seg'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['seg_mask'].set_extent([0, shape_y, shape_x, 0])
        for ax in (self.ax_nav, self.ax_seg):
            ax.set_xlim(0, shape_y)
            ax.set_ylim(shape_x, 0)
        self.toolbar.update()
        self.toolbar.push_current()
        self.slider_imgNo.setRange(0, len(self.imgs) - 1)
        self.frame_flag_bar.set_range(len(self.imgs))
        # Not recomputed here (see _compute_tracking_quality) - a restored
        # object just starts back at "not yet quality-checked" (the same
        # blank-icon state a freshly-tracked one is in before its own
        # check runs), same spirit as this being a derived diagnostic
        # rather than persisted state (see its own init comment).
        self._quality_flags = {}
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.imgs)))
        self.button_runSeg_clip.setEnabled(True)
        self.button_runSeg_img.setEnabled(True)
        self.button_fineTuneMask.setEnabled(True)
        self.button_autoDetector.setEnabled(True)

        # Extraction settings - signals blocked so setting them doesn't
        # trigger a redundant redraw/recompute/dialog (see docstring); the
        # already-copied imgs/df_obj already reflect these settings'
        # effect. Edge Detection/Dilate-Erode/Mesh ride along inside
        # df_obj_rows below (per-object Segments, no main-tab widgets to
        # restore here).
        for wid, value, setter in (
            (self.spinbox_threadNum, state['spinbox_threadNum'], 'setValue'),
            (self.spinbox_fps, state['spinbox_fps'], 'setValue'),
            (self.checkbox_autosave, state['checkbox_autosave'], 'setChecked'),
            (self.checkbox_makePets2, state['checkbox_makePets2'], 'setChecked'),
        ):
            wid.blockSignals(True)
            getattr(wid, setter)(value)
            wid.blockSignals(False)
        self.pets2_params = deepcopy(state['pets2_params'])

        # Tracked/segmented objects - reconstructs the tree the same way
        # _on_saved_analysis_loaded() does from a loaded dataframe.
        any_tracked = False
        for row in state['df_obj_rows']:
            idx = row['idx']
            self.df_obj.loc[idx] = [row[col] for col in self.cols_df]
            self.add_item_tree(idx, row['frame_idx'], row['end'], row['use'])
            row_index = self.df_obj.index.get_loc(idx)
            if row['mask'] is not None:
                self.toggle_tree_icon(row_index, 'trk', True)
                any_tracked = True
            if row['dp'] is not None:
                self.toggle_tree_icon(row_index, 'ext', True)
        self.activate_3ded_widgets(any_tracked)

        self.clip_dp.set_state(state['clip_dp'])
        self._dp_clip_initialized = True
        self.slider_imgNo.blockSignals(True)
        self.slider_imgNo.setValue(state['imgNo'])
        self.slider_imgNo.blockSignals(False)
        self.update_canvas(state['imgNo'])
        self.add_scalebar()

    def cleanup(self):
        """Release resources held by this tab. Called by MainWindow.closeEvent
        so repeated runs of the app in the same console/kernel don't leave
        threadpools, running subprocesses, and matplotlib figures alive."""
        self.threadpool.clear()
        self.stop_processes()  # kills any running_processes_sam
        self.kill_3ded_driver()
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
