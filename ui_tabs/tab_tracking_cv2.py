# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:55:17 2024

@author: SGholam
"""

import ast
import os
import json
import tempfile
from glob import glob
from PyQt5.QtCore import Qt, QTimer
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from matplotlib.colors import SymLogNorm, to_rgb
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
from matplotlib.lines import Line2D
import cv2
import datetime
from copy import deepcopy
from .worker_thread import WorkerThread_General, ProcessStderrBuffer
from .asset_download_dialog import confirm_and_download
from EDyssey.tracking_utils import asset_fetch
from worker_extract_frame import load_dp
import worker_pool_utils as wpu
from .worker_launch import worker_command
from .contrast_scaling import ContrastScalingBox
from .logging_utils import LogConsole
from .base_tab import (TabBase, get_existing_directory, resolve_hdf5_dtype, glob_ext_for_dtype,
                       HDF5_EVENTEM_LABEL)
from .display_settings import DisplaySettings
from .clipping_thresholds import ClippingThresholdsWidget
from .transposed_object_table import TransposedObjectTable
from .pets2_dialog import Pets2ParamsDialog
from .ribbon import RibbonPanel, RibbonTool
from .smart_scan_dialog import SmartScanCheckDialog
from .mask_edit_dialog import MaskEditDialog
from .blob_segmentation_dialog import BlobSegmentationDialog
from .frame_flag_bar import FrameFlagBar
from skimage.filters import threshold_otsu, threshold_li, threshold_mean, threshold_yen
import gc
from time import perf_counter
# from dask.distributed import Client, LocalCluster, as_completed
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
        self._layout_frozen = False
        # Cached "clean" background (everything except the per-frame image/
        # ROI-box/title artists) used to blit slider-driven frame changes
        # instead of a full canvas redraw. None means "needs (re)capture" -
        # see _blit_canvas().
        self._bg = None

        self.init_widget()

        # cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='2GB')
        # client = Client(cluster)

    def init_widget(self):
        button_w = 100
        button_h_lrg = 25
        self.layout = qtw.QVBoxLayout(self)
        self.setLayout(self.layout)

        #%% ribbon (top parameter ribbon, Word-style - see Tab_ROI_on_4D for
        # the original design, and TabBase for the shared helpers). This
        # tab has more groups than Tab_ROI_on_4D, so it gets more ribbon
        # columns instead of stacking everything into fewer of them - see
        # each column's own comment below for the reasoning.
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
        self.dp_center = None  # (x, y) - auto-found or last manually-clicked center
        # id(dp_array) at the time dp_center was last auto-found - lets
        # update_scalebar('reciprocal') skip re-running find_dp_center_blurred
        # (a real HyperSpy call) when the displayed DP hasn't actually
        # changed since, e.g. on every keystroke in the scale-recip field.
        # See _on_auto_center_toggled.
        self._dp_center_cache_key = None

        # Acquisition Dwell T. - moved here from Input Parameters (mirrors
        # Navigator's convention: dwell time lives with the File/Smart Scan
        # controls, not with Detector/Scan Size). 3DED extraction always
        # reads the acquisition (smart-scanned) file when Smart Scanned is
        # checked (see the Smart Scan groupbox below), so a single dwell
        # time covers both the smart-scan and plain cases - unlike
        # Navigator, there's no separate "Detection"-role extraction path
        # here that would need its own dwell spinbox.
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

        #### Smart-scan support
        # Grid layout matches Tab_SAM2's identical "Smart Scan" groupbox:
        # left column = activation controls (checkbox, then Check Files
        # button), right column = directory pickers (pattern dir, then
        # detection dir).
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

        # Summary of the last "Check Files" review (see
        # _set_smart_scan_summary) - was referenced throughout this file
        # (open_smart_scan_check_dialog, get_duplicate_state, etc.) but
        # never actually created here, unlike Tab_SAM2's identical widget -
        # any code path setting it (e.g. accepting the SmartScanCheckDialog)
        # crashed with AttributeError. Matches Tab_SAM2's own
        # label_smartScanSummary exactly (hidden until there's text).
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
        self.box_scanSize, layout_box_scanSize = self._ribbon_group_start(layout_ribbon, stretch=0)

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
        self.lineEdit_scale_recip.textChanged.connect(lambda: self.update_scalebar('reciprocal'))
        self.lineEdit_scale_real.textChanged.connect(lambda: self.update_scalebar('real'))

        layout_box_scanSize.addLayout(layout_exp_groups)

        # Load Signal / Load Saved Analysis - moved here from Files, at the
        # bottom of this column (per user request). Added BEFORE
        # _ribbon_group_end() below (not after, as this used to do) - that
        # call adds this column's "Input Parameters" caption label right
        # where it's called, so calling it right after the Input Parameters
        # grid but before these buttons sandwiched the caption between the
        # grid and the buttons, reading as unwanted blank space/separation
        # between them instead of one clean column with its caption at the
        # very bottom, like every other ribbon column.
        layout_loadSignal = qtw.QHBoxLayout()
        layout_box_scanSize.addLayout(layout_loadSignal)
        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        self.button_loadNavigation.setFixedSize(button_w, button_h_lrg*2)
        # Centered (matches SAM2's identical row) - without this, the pair
        # ends up left-anchored instead, its actual position then drifting
        # against SAM2's own centered pair depending on how much wider this
        # column happens to be than the two buttons combined.
        layout_loadSignal.addWidget(self.button_loadNavigation, alignment=Qt.AlignCenter)
        self.button_loadNavigation.clicked.connect(self.load_navSignal)

        self.button_loadSavedAnalysis = qtw.QPushButton('Load Saved\nAnalysis')
        self.button_loadSavedAnalysis.setFixedSize(button_w, button_h_lrg*2)
        layout_loadSignal.addWidget(self.button_loadSavedAnalysis, alignment=Qt.AlignCenter)
        self.button_loadSavedAnalysis.clicked.connect(self.load_saved_analysis)

        self._ribbon_group_end(layout_ribbon, layout_box_scanSize, 'Input Parameters', stretch=True)


        # Adjust Contrast and Feature Handling have moved out of the
        # ribbon, into one stacked column beside the canvas (same position
        # as the Navigator tab's file list) - see the #%% canvas section
        # below, where self.box_contrast/self.tree_objects etc. are built.

        #%% Threshold / Edge Detection
        # Fixed width (rather than sizing to content) - matches SAM2
        # Tracker's own Threshold/Tracking/Extract column so the two tabs'
        # ribbons don't visibly shift width against each other.
        self.box_3ded, layout_box_3ded = self._ribbon_group_start(layout_ribbon, stretch=0, width=320)
        #### threshold
        layout_thresh_method = qtw.QHBoxLayout()
        label_thresh_method = qtw.QLabel('Threshold Method')
        layout_thresh_method.addWidget(label_thresh_method)
        
        self.combo_thresh_method = qtw.QComboBox()
        layout_thresh_method.addWidget(self.combo_thresh_method)
        self.combo_thresh_method.addItems(['li', 'otsu', 'yen', 'mean'])
        self.combo_thresh_method.currentIndexChanged.connect(self._on_threshold_control_changed) #TODO change to update mask
        
        label_blur = qtw.QLabel('ROI Blur')
        layout_thresh_method.addWidget(label_blur)
        self.spinbox_blur = qtw.QDoubleSpinBox()
        self.spinbox_blur.setFixedWidth(60)
        # Same Gaussian-blur-sigma convention as the "Adjust Contrast" box's
        # own Denoise control (io.DENOISE_PARAM_SPECS['Gaussian Blur']) - 0
        # means no blur.
        self.spinbox_blur.setRange(0.0, 20.0)
        self.spinbox_blur.setSingleStep(0.1)
        self.spinbox_blur.setDecimals(1)
        self.spinbox_blur.setValue(0.0)
        layout_thresh_method.addWidget(self.spinbox_blur)
        self.spinbox_blur.valueChanged.connect(self._on_threshold_control_changed)
        layout_box_3ded.addLayout(layout_thresh_method)

        layout_deviation = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_deviation)
        label_thresh_dev = qtw.QLabel('Deviation')
        layout_deviation.addWidget(label_thresh_dev)
        
        self.slider_thresh = qtw.QSlider(1)
        layout_deviation.addWidget(self.slider_thresh)
        self.slider_thresh.setDisabled(True)
        self.slider_thresh.valueChanged.connect(self._on_threshold_control_changed) # TODO plot only mask ax
        self.slider_thresh.setRange(0, 200)
        
        self.button_thresh = qtw.QPushButton('Reset')
        layout_deviation.addWidget(self.button_thresh)
        self.button_thresh.clicked.connect(self.reset_thresh)
        self._ribbon_group_end(layout_ribbon, layout_box_3ded, 'Threshold', stretch=False)

        # Edge Detection used to live here as a tab-wide control - it's now
        # per-ROI, set only from the Fine-Tune Mask dialog's Segments
        # feature (same as Mesh/Dilate-Erode already were - see
        # apply_edge_mask/_edge_settings_for), so there's no main-tab
        # widget for it anymore.
        # Edge Detection used to live here as a tab-wide control too - it's
        # now per-ROI, set only from the Fine-Tune Mask dialog's Segments
        # feature (same as Mesh/Dilate-Erode already were). Blob Selection
        # (when a ROI's threshold mask actually contains more than one
        # separate object - e.g. two nearby particles - lets the user pick
        # just one of them for extraction) is per-ROI too, but lives as its
        # own "Blob" column in the object list below (see add_item_tree)
        # instead of a tab-wide control here - click directly on the "ROI
        # with Threshold" panel, once that column's checkbox is on for the
        # selected ROI, to seed/reseed which blob to follow (see on_press's
        # ax_mask branch/_resolve_blob_mask - runs before Dilate/Erode/Edge
        # Detection/Mesh, in apply_edge_mask).
        sep_tracking = qtw.QFrame()
        sep_tracking.setFrameShape(qtw.QFrame.HLine)
        sep_tracking.setFrameShadow(qtw.QFrame.Sunken)
        layout_box_3ded.addWidget(sep_tracking)

        #%% Tracking - moved here from the bottom of the left object-list
        # panel (below tree_objects), directly above Extract in the same
        # stacked ribbon column, per user request.
        layout_box_tracking = layout_box_3ded

        # One row: label + tracker-choice combo + the button that acts on
        # the tracked result (Track!) - Fine-Tune Mask.../Blob Settings...
        # used to live here too, but now live in the left object-list panel
        # instead, directly below the Auto Detector/Reset ROIs row (see
        # button_fineTuneMask/button_blobSettings below) since they act on
        # a selected ROI from that panel, not on the tracker choice above it.
        layout_tracking = qtw.QHBoxLayout()
        label_track = qtw.QLabel('Tracker')
        layout_tracking.addWidget(label_track)
        self.combo_trackMethod = qtw.QComboBox()
        # Closed-state width tracks a fixed character count instead of the
        # longest item ('xcorr-template') - the popup itself still shows
        # full item text, only the always-visible closed box is capped.
        self.combo_trackMethod.setSizeAdjustPolicy(qtw.QComboBox.AdjustToMinimumContentsLength)
        self.combo_trackMethod.setMinimumContentsLength(8)
        layout_tracking.addWidget(self.combo_trackMethod)
        self.combo_trackMethod.addItems(['csrt', 'nano', 'mil', 'dasiamrpn', 'xcorr-phase', 'xcorr-template'])

        self.button_track = qtw.QPushButton('Track!')
        self.button_track.setFixedWidth(button_w)
        layout_tracking.addWidget(self.button_track)
        self.button_track.clicked.connect(self.track_rois)
        self.button_track.setDisabled(True)

        layout_tracking.addStretch(1)
        layout_box_tracking.addLayout(layout_tracking)
        self._ribbon_group_end(layout_ribbon, layout_box_tracking, 'Tracking', separator=False, stretch=False)

        sep_extract = qtw.QFrame()
        sep_extract.setFrameShape(qtw.QFrame.HLine)
        sep_extract.setFrameShadow(qtw.QFrame.Sunken)
        layout_box_3ded.addWidget(sep_extract)

        #%% Extract
        layout_box_extract = layout_box_3ded

        layout_threadNo = qtw.QHBoxLayout()
        label_threadNo = qtw.QLabel('CPU Cores')
        layout_threadNo.addWidget(label_threadNo)
        self.spinbox_threadNo = qtw.QSpinBox(self)
        self.spinbox_threadNo.setMaximumWidth(80)
        layout_threadNo.addWidget(self.spinbox_threadNo)
        self.spinbox_threadNo.setRange(1, os.cpu_count() or 1)
        self.spinbox_threadNo.setValue(2)
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
        layout_box_extract.addLayout(layout_threadNo)

        # Own row (rather than sharing layout_threadNo with the CPU/FPS/
        # Autosave row above) so the checkbox label has enough room and
        # doesn't get clipped by the panel's width.
        layout_saveOptions = qtw.QHBoxLayout()
        layout_box_extract.addLayout(layout_saveOptions)

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

        layout_extract = qtw.QHBoxLayout()
        layout_box_extract.addLayout(layout_extract)

        self.button_3ded = qtw.QPushButton('Extract All')
        layout_extract.addWidget(self.button_3ded)
        self.button_3ded.setFixedHeight(button_h_lrg)
        self.button_3ded.clicked.connect(self.extract_3ded)

        self.button_extractCurrentFrame = qtw.QPushButton('Extract Frame')
        layout_extract.addWidget(self.button_extractCurrentFrame)
        self.button_extractCurrentFrame.setFixedHeight(button_h_lrg)
        self.button_extractCurrentFrame.setToolTip(
            'Compute the DP for the current frame only (not saved)')
        self.button_extractCurrentFrame.clicked.connect(self.extract_dp_current_frame)

        for btn in (self.button_3ded, self.button_extractCurrentFrame):
            btn.setFixedHeight(button_h_lrg)

        #### Cancel
        layout_extract_2 = qtw.QVBoxLayout()
        layout_ribbon.addLayout(layout_extract_2)
        
        self.button_save_results = qtw.QPushButton('Save Results')
        layout_extract_2.addWidget(self.button_save_results)
        self.button_save_results.setFixedSize(button_w, button_h_lrg*3)
        self.button_save_results.clicked.connect(self.save_results)

        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.setFixedSize(button_w, button_h_lrg*3)
        self.button_cancel.setStyleSheet("background-color: red; color: white;")
        self.button_cancel.setDisabled(True)
        self.button_cancel.setToolTip('Stop the running tracking/extraction')
        self.button_cancel.clicked.connect(self.cancel_running_work)
        layout_extract_2.addWidget(self.button_cancel)

        self._ribbon_group_end(layout_ribbon, layout_box_extract, 'Extract', separator=False)
        layout_ribbon.addStretch(1)
        #%% Adjust Contrast (top) + Feature Handling
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

        # top
        layout_featureTop = qtw.QHBoxLayout()
        layout_featurePanel.addLayout(layout_featureTop)
        self.button_autoDetection = qtw.QPushButton('Auto Detector')
        layout_featureTop.addWidget(self.button_autoDetection)
        self.button_autoDetection.clicked.connect(self.launch_auto_detector)

        self.button_reset_rois = qtw.QPushButton('Reset ROIs')
        layout_featureTop.addWidget(self.button_reset_rois)
        self.button_reset_rois.clicked.connect(self.reset_rois)

        # Both act on the selected ROI below (tree_objects), not on the
        # tracker choice up in the ribbon - moved down here from the
        # ribbon's Tracking group for that reason, and kept in the same row
        # since both are "do something to the selected ROI's mask" actions.
        # Still governed by disable_3ded_widgets (see its own docstring)
        # even though neither is one of box_3ded's own children anymore.
        row_maskActions = qtw.QHBoxLayout()
        layout_featurePanel.addLayout(row_maskActions)
        self.button_fineTuneMask = qtw.QPushButton('Fine-Tune Mask...')
        self.button_fineTuneMask.setToolTip('Manually edit the ROI\'s mask, frame by frame')
        row_maskActions.addWidget(self.button_fineTuneMask)
        self.button_fineTuneMask.clicked.connect(self.open_fine_tune_mask_dialog)
        self.button_fineTuneMask.setDisabled(True)

        self.button_blobSettings = qtw.QPushButton('Blob Settings...')
        self.button_blobSettings.setToolTip(
            "Configure the selected ROI's Blob Selection - which segmentation "
            'method splits its threshold mask into individual blobs (for '
            "touching/overlapping particles), and which one's currently chosen. "
            'Also opens automatically the first time this ROI\'s "Blob" column '
            'checkbox is checked.')
        row_maskActions.addWidget(self.button_blobSettings)
        self.button_blobSettings.clicked.connect(self.open_blob_settings_dialog)
        self.button_blobSettings.setDisabled(True)

        # Moved here (from right after the Extract ribbon group) - needs
        # button_blobSettings to already exist, since disable_3ded_widgets
        # toggles it explicitly too (see its own docstring).
        self.disable_3ded_widgets(True)

        # tree - stretches to fill the rest of this column's height now that
        # it sits beside the (tall) canvas, rather than being capped to fit
        # inside a short ribbon column. Transposed (see
        # TransposedObjectTable): property names run down the fixed first
        # column instead of across the top, and each tracked ROI is one
        # column instead of one row, so adding a ROI adds a column - still
        # called tree_objects (not literally a QTreeWidget anymore) since
        # renaming the many existing references below wasn't worth it.
        self.cols_tree = ["use", "idx", "init", "end", "ref", "blob", "trk", "ext",
                          "qlty", "dup", "del"]
        # Kept at or under "Start"'s own length (see TransposedObjectTable.
        # _HEADER_WIDTH_REF) - the ones that don't fit unabbreviated get a
        # row_tooltips entry with their full word instead.
        row_labels = ["Use", "Idx", "Start", "End", "Ref", "Blob",
                     "Track", "Extr", "Qlty", "Dup", "Del"]
        row_tooltips = [None, None, None, None, None, None, "Tracked", "Extracted",
                        "Tracking Quality - flagged (see the frame-flag bar under the "
                        "slider) if any frame's mask area looks anomalous after "
                        "tracking, e.g. the tracker may have lost the object",
                        "Duplicate", "Delete"]
        # Selected Object / All Active Objects - governs ax_mask (2) only:
        # the nav overlay and DP panel still follow whichever object is
        # actually selected in the table below, regardless of this choice
        # (see update_canvas) - "All Active Objects" only changes what the
        # mask panel itself shows, from one object's own cropped threshold
        # view to every active ("Use" checked) object's mask composited
        # onto the full frame, each in its own color with its index label
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
        layout_featurePanel.addWidget(self.tree_objects)
        self.tree_objects.setMinimumWidth(200)
        # Tall enough for their content: dup/del hold a 48/30px button, end
        # holds a QSpinBox with up/down arrows, ref/blob hold a
        # QComboBox/QCheckBox.
        row_heights = {'use': 24, 'idx': 24, 'init': 24, 'end': 28, 'ref': 28, 'blob': 24,
                       'trk': 24, 'ext': 24, 'qlty': 24, 'dup': 34, 'del': 34}
        for i, col in enumerate(self.cols_tree):
            self.tree_objects.setRowHeight(i, row_heights[col])
        self.tree_objects.itemSelectionChanged.connect(self.update_canvas)
        self.tree_objects.itemChanged.connect(self.on_item_check_changed)
        # tree_objects is now fixed to its own (small) content height - see
        # TransposedObjectTable.__init__ - so this absorbs the rest of the
        # column's height as plain background instead of the table itself
        # stretching all the way down to it.
        layout_featurePanel.addStretch(1)

        self.patches_axNav = []
        self.patches_axTrack = []
        # {idx: {frame: (x, y)}} - Blob Selection's per-ROI auto-follow
        # cache (see _resolve_blob_mask), cleared whenever a fresh seed is
        # clicked or a ROI's tracking/threshold settings that would change
        # what's in its threshold mask are touched, so a stale chosen-blob
        # trail is never carried into a differently-thresholded/tracked run.
        self._blob_centroid_cache = {}
        # Contour/number-label artists drawn on ax_mask when Blob Selection
        # is enabled (see _draw_blob_overlay) - cleared/rebuilt every
        # update_canvas() call, tracked here so they can be removed again
        # without touching anything else on that axis.
        self._blob_overlay_artists = []
        # Mask/label artists for the "All Active Objects" mask-panel mode
        # (see _draw_all_object_masks) - cleared/rebuilt every call, same
        # convention as _blob_overlay_artists just above.
        self._all_mask_artists = []
        self._ax_mask_full_shape_seen = None
        # {idx: [flagged_frame_idx, ...]} - tracking-quality flags (see
        # _compute_tracking_quality/frame_flag_bar.FrameFlagBar), recomputed
        # fresh after every Track! run - deliberately NOT a df_rois column
        # (or persisted in Save Results/Load Saved Analysis): purely a
        # derived diagnostic, cheap to recompute, not part of this ROI's
        # actual tracked/extracted data.
        self._quality_flags = {}
        self.empty_main_dataframe()

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

        # Canvas scroll area + slider + progress bar share one pane of the
        # vertical splitter above (the log console is the other pane, see
        # below).
        _canvas_pane = qtw.QWidget()
        layout_canvas = qtw.QVBoxLayout(_canvas_pane)
        layout_canvas.setContentsMargins(0, 0, 0, 0)
        layout_canvas_splitter.addWidget(_canvas_pane)
        
        # 3 subplots in one figure, one row (plt.subplots(1, 3)'s
        # arrangement) - Figure()+add_subplot() rather than pyplot's own
        # plt.subplots() to match every other tab's idiom (a bare Figure
        # handed straight to FigureCanvas, not registered with pyplot's
        # global figure manager). Nav. Signal and Tracking Results used to
        # be two separate subplots - now one merged axis (ax_nav): the
        # input ROI (red) and tracked ROI (tab:orange) are distinguishable
        # by color alone, so a second copy of the same frame image added
        # nothing - see draw_rois_in/draw_rois_out.
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax_nav = self.figure.add_subplot(1, 3, 1)
        self.ax_mask = self.figure.add_subplot(1, 3, 2)
        self.ax_dp = self.figure.add_subplot(1, 3, 3)

        # titles for axes
        self.ax_nav.set_title('(1) Navigation / Tracking')
        self.ax_mask.set_title('(2) Roi with Threshold')
        self.ax_dp.set_title('(3) DP')
        self.img_display = {}
        self.img_zero = np.zeros((512,512), dtype='uint16')
        # One-off "Extract DP (Current Frame)" result - only shown while the
        # slider/selection still matches the (idx, i_fr) it was computed
        # for; update_canvas() clears it and falls back to the normal
        # per-series dp display as soon as either changes. See
        # extract_dp_current_frame()/_on_current_frame_dp().
        self._current_frame_dp_preview = None
        axes = ['nav', 'dp']

        for i, ax in enumerate([self.ax_nav, self.ax_dp]):
            self.img_display[axes[i]] = ax.imshow(self.img_zero, cmap='viridis')
            # ax.set_axis_off()
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


        # The Ctrl+Scroll zoom hint applies to every axis on the canvas, so
        # it's one figure-wide supxlabel rather than repeated per-axis text.
        self.figure.supxlabel('Hold "Ctrl" + Scroll wheel to zoom the axis under the cursor',
                              fontsize=10)

        self.ax_mask.set_axis_off()
        self.img_display['img_mask'] = self.ax_mask.imshow(self.img_zero, cmap='gray')
        # 'mask' gets an RGBA array (not scalar+cmap) from update_ax_mask()
        # on every real update - see there, matching Tab_SAM2's show_mask().
        # cmap/alpha here only ever apply to this one placeholder frame.
        self.img_display['mask'] = self.ax_mask.imshow(self.img_zero, cmap='gray')
        self.img_display['dp'].set_norm(SymLogNorm(linthresh=1))
        self.img_display['dp'].set_cmap('inferno')

        # Created once here (not per-frame) - update_canvas() only updates
        # the underlying image data, which keeps these in sync for free.
        # 'mask' is excluded: it's a low-alpha threshold overlay on top of
        # 'img_mask', not an independently meaningful scalar image.
        self.colorbars = {}
        self.colorbars['nav'] = self.figure.colorbar(
            self.img_display['nav'], ax=self.ax_nav, fraction=0.046, pad=0.04)
        self.colorbars['img_mask'] = self.figure.colorbar(
            self.img_display['img_mask'], ax=self.ax_mask, fraction=0.046, pad=0.04)
        self.colorbars['dp'] = self.figure.colorbar(
            self.img_display['dp'], ax=self.ax_dp, fraction=0.046, pad=0.04)

        self.canvas.setMinimumHeight(650)
        # Kept alive (not shown) purely for its view-stack bookkeeping
        # (.update()/.push_current(), used to seed the ribbon's Home button)
        # and as the target of the ribbon's own Pan/Zoom/Home actions below -
        # the toolbar strip itself is no longer shown under the canvas.
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()

        #%% ribbon
        # Docked along the right edge - an additional way to reach the same
        # canvas interactions already available via Ctrl-click/drag on the
        # canvas (see on_press) and matplotlib's own toolbar; deliberately
        # does NOT duplicate the left panel's buttons (Track!, Extract All,
        # Save Results, ...), only actions that act directly on the plot
        # itself. 'select_roi' is the only tool mode on_press actually
        # checks (see RibbonPanel.active_tool there). Pan/Zoom/Home act on
        # the one shared toolbar.
        self.ribbon = RibbonPanel([
            RibbonTool('select_roi', 'select_roi', 'Select ROI - same as Ctrl+drag', 'tool'),
            RibbonTool('sep1', kind='separator'),
            RibbonTool('pan', 'pan', 'Toggle pan mode on the canvas',
                      'action', self.toolbar.pan),
            RibbonTool('zoom', 'zoom', 'Toggle rectangle-zoom mode on the canvas',
                      'action', self.toolbar.zoom),
            RibbonTool('home', 'home', 'Reset the canvas view',
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
        layout_right_outer.addWidget(self.ribbon)

        self._canvas_stack_widget = qtw.QWidget()
        layout_canvas_stack = qtw.QVBoxLayout(self._canvas_stack_widget)

        # Clipping Thresholds beside the canvas (only the DP axis's own
        # clipping, per the decision that the Adjust Contrast box already
        # covers the nav image on this tab - unlike Tab_ROI_on_4D/Navigator,
        # which had no equivalent).
        layout_canvas_row = qtw.QHBoxLayout()
        layout_canvas_row.addWidget(self.wrap_canvas_in_scroll(self.canvas), 1)
        self.clip_dp = ClippingThresholdsWidget(title='DP Clipping\nThresh.')
        layout_canvas_row.addWidget(self.clip_dp)
        layout_canvas_stack.addLayout(layout_canvas_row)
        # DP clip range is only reset the first time real data is shown
        # (see update_ax) - after that, the user's chosen thresholds
        # persist across frame scrubs instead of resetting every tick.
        self._dp_clip_initialized = False
        self.clip_dp.valueChanged.connect(self._update_dp_clip)

        self._canvas_scroll = qtw.QScrollArea()
        self._canvas_scroll.setWidget(self._canvas_stack_widget)
        self._canvas_scroll.setWidgetResizable(True)
        self._canvas_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout_canvas.addWidget(self._canvas_scroll)

        # Connect mouse events - on_press/on_click_dp each already check
        # event.inaxes themselves and no-op outside their own axes, so both
        # can safely share the one canvas's button_press_event.
        self.rect = None            # Currently drawn rectangle
        self.press = None           # Mouse press coordinates

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_click_dp)
        # A resized canvas invalidates the cached blit background (wrong
        # pixel dimensions), so force a recapture on the next frame update.
        self.canvas.mpl_connect('resize_event', lambda evt: setattr(self, '_bg', None))

        self.backgrounds = {}
        self.backgrounds['nav'] = self.canvas.copy_from_bbox(self.ax_nav.bbox)

        #%% slider img num
        layout_slider = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_slider)

        # Frame-flag bar (see frame_flag_bar.FrameFlagBar/
        # _compute_tracking_quality): a thin strip right under the frame
        # navigation row, marking frames flagged as possibly mistracked for
        # whichever ROI is currently selected - passive, click a mark to
        # jump there. Not pixel-aligned to just slider_imgNo's own sub-
        # width (that row also has Prev/Next/Start/Mid flanking it) -
        # spans the full row instead, which reads fine as a status strip
        # without needing a QGridLayout rework of an already-busy row.
        self.frame_flag_bar = FrameFlagBar()
        layout_canvas.addWidget(self.frame_flag_bar)
        # frameClicked wired to slider_imgNo further below, once that
        # widget actually exists (constructed later in this same row).

        self.label_imgCounter = qtw.QLabel('Img No.')
        layout_slider.addWidget(self.label_imgCounter)

        self.lineEdit_imgNo = qtw.QLineEdit()
        layout_slider.addWidget(self.lineEdit_imgNo)
        self.lineEdit_imgNo.setFixedWidth(35)
        self.lineEdit_imgNo.setValidator(QIntValidator(0, 0))
        self.lineEdit_imgNo.returnPressed.connect(self.jump_to_frame_no)

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

        self.slider_imgNo.valueChanged.connect(self._on_slider_imgNo_changed)
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
            fixed-width) is really just "as small as it's allowed to be"
            below - but a QSplitter.setSizes() call made before the window
            has ever actually been shown (i.e. still has no real geometry,
            as here - this runs during __init__, well before
            MainWindow.show()) only stores those sizes proportionally
            against whatever placeholder width Qt reports at that moment,
            not real pixels - so calling it only once, here, left the
            ribbon pane rendered collapsed to nothing until the user
            manually dragged it open. Re-running the exact same calls once
            more via QTimer.singleShot(0, ...) - after the event loop has
            actually processed the window's first show/resize, so every
            widget's real minimum size is now known - fixes that.
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
        self.button_loadNavigation.setToolTip('Load navigation signal (Ctrl+O)')
        self.button_loadSavedAnalysis.setToolTip('Load a saved analysis folder (Ctrl+Shift+O)')
        self.button_track.setToolTip('Track all enabled ROIs across frames (Ctrl+T)')
        self.button_3ded.setToolTip('Extract 3D electron diffraction patterns (Ctrl+E)')
        self.button_save_results.setToolTip('Save tracking and 3DED results to disk (Ctrl+S)')
        self.combo_trackMethod.setToolTip(
            'Tracking algorithm - CSRT is most accurate; xcorr-phase/template '
            'track by cross-correlation shift (best for drift-only motion)')
        self.combo_thresh_method.setToolTip('Thresholding algorithm used to create the binary mask')
        self.spinbox_blur.setToolTip('Gaussian blur sigma applied to the ROI before thresholding - 0 = no blur')
        self.spinbox_threadNo.setToolTip('Number of CPU cores used for parallel 4D extraction')
        self.checkbox_autosave.setToolTip('Automatically save results when extraction finishes')
        self.checkbox_makePets2.setToolTip(
            "Write a PETS2 project file (.pts2) into each ROI's folder on save")

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

        # Picks up any non-default DisplaySettings already set by the Edit
        # tab (e.g. this instance is a duplicate opened after adjusting
        # sizes) - see TabBase.apply_display_settings.
        self.apply_display_settings()
    #%% load data
    def apply_display_settings(self):
        """TabBase's own ribbon/figure-size handling, plus this tab's own
        nav/DP colormap - see display_settings.py's nav_colormap/
        dp_colormap and the Edit menu's Display Size dialog. img_mask/mask
        (the mask-editing crop view) deliberately keep their own 'gray'
        default - a translucent color mask overlay reads better against a
        plain grayscale background than a colored one."""
        super().apply_display_settings()
        settings = DisplaySettings.instance()
        self.img_display['nav'].set_cmap(settings.nav_colormap)
        self.img_display['dp'].set_cmap(settings.dp_colormap)
        self.canvas.draw_idle()

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
        start_dir = self.lineEdit_patternDir.text() or self.lineEdit_dir_4d.text()
        path = get_existing_directory(self, "Select Pattern Files Folder", start_dir)
        if path:
            self.lineEdit_patternDir.setText(path)
            self._smart_scan_rows = None
            self._set_smart_scan_summary('')

    def get_pattern_dir(self):
        return self.lineEdit_patternDir.text() or self.lineEdit_dir_4d.text()

    def browse_detection_dir(self):
        start_dir = self.lineEdit_detectionDir.text() or self.lineEdit_dir_4d.text()
        path = get_existing_directory(self, "Select Detection Files Folder", start_dir)
        if path:
            self.lineEdit_detectionDir.setText(path)
            self._smart_scan_rows = None
            self._set_smart_scan_summary('')

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
            self._set_smart_scan_summary(f'{n_ok} / {len(dlg.rows)} angle(s) included')
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
        # background to be recaptured on the next frame update.
        self._bg = None

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
        """Contrast settings changed: the currently-displayed frame is
        rescaled immediately (cheap, instant feedback), while the full
        stack (used for tracking, and to keep every other frame in sync)
        rescales in the background - a long stack no longer blocks/lags the
        GUI on every settings tweak. Rapid retuning cancels (i.e. discards
        the result of) any still-running previous background rescale - see
        ContrastScalingBox.rescale_async. Also used as
        _apply_denoise_to_all_frames's own worker - both end up wanting
        exactly this same full-stack-refresh-from-current-settings."""
        if not hasattr(self, 's'):
            return
        self._refresh_current_frame_display()
        self.box_contrast.set_denoise_apply_all_busy(True)
        self.progress_bar.setRange(0, len(self.nav_imgs))
        self.progress_bar.setValue(0)
        self.box_contrast.rescale_async(self.s, self.threadpool, self.logger,
                                        on_done=self._on_nav_signal_rescaled,
                                        on_error=self._on_nav_signal_rescale_failed,
                                        on_progress=self.update_progress_bar)

    def _refresh_current_frame_display(self, imgNo=None):
        """Rescale (contrast + denoise, current settings) and redraw just
        the currently-displayed frame - cheap, so safe to call on every
        Denoise parameter tweak (see _on_denoise_preview_changed) or frame
        navigation (see _on_slider_imgNo_changed) without waiting for a full
        background stack rescale.

        update_canvas() already blits (see _blit_canvas) - it paints the
        canvas itself, so no trailing canvas.draw_idle() belongs here: that
        used to schedule a full, unblitted redraw of the whole figure right
        after the cheap blit, silently undoing it and making every single
        Denoise parameter nudge as expensive as a full draw."""
        if not hasattr(self, 's'):
            return
        if imgNo is None:
            imgNo = self.slider_imgNo.value()
        frame_8bit = self.box_contrast.rescale_frame(self.nav_imgs_raw[imgNo])
        self.nav_imgs[imgNo] = frame_8bit
        self.img_display['nav'].set_clim(vmin=frame_8bit.min(), vmax=frame_8bit.max())
        self.update_canvas(imgNo)

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
        if not hasattr(self, 'nav_imgs_raw'):
            qtw.QMessageBox.warning(self, 'No Signal Loaded',
                'Load a signal first to compare denoising methods on it.')
            return
        imgNo = self.slider_imgNo.value()
        self._check_methods_dlg = self.box_contrast.open_check_methods_dialog(
            self.nav_imgs_raw[imgNo], parent=self)

    def _on_nav_signal_rescaled(self, s_8bit):
        """Callback for box_contrast.rescale_async(): apply the freshly
        rescaled full image stack once the background rescale finishes -
        every frame now reflects the current Denoise settings too, so the
        stack is no longer "dirty" (see _on_denoise_preview_changed)."""
        self.box_contrast.set_denoise_apply_all_busy(False)
        self.s_8bit = s_8bit
        self.nav_imgs = deepcopy(s_8bit.data)
        self._denoise_dirty = False
        self.img_display['nav'].set_clim(vmin=self.nav_imgs.min(), vmax=self.nav_imgs.max())
        self.update_canvas()  # already blits and paints - no trailing draw_idle() needed

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
        # Forces update_ax_mask()'s shape-change check to reset ax_mask's
        # view on this signal's first ROI crop, even if it happens to match
        # whatever the previous signal's last-shown crop size was.
        self._ax_mask_shape_seen = None
        self.spinner.stop()
        
        shape_x, shape_y = self.nav_imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['nav'].set_clim(vmin=self.nav_imgs.min(), vmax=self.nav_imgs.max())
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.nav_imgs)))

        # Reset the view to the newly loaded data's full extent (in case the
        # user had already zoomed in on a previous signal, which disables
        # autoscale), then re-seed the toolbar's view stack so its "Home"
        # button resets to *this* view instead of doing nothing (it does
        # nothing until something pushes at least one view onto its stack,
        # which our own scroll-wheel zoom deliberately bypasses).
        # ax_mask is deliberately excluded here: unlike ax_nav, it shows a
        # per-ROI *cropped* image whose size varies with each ROI, and whose
        # view is kept in sync with that in update_ax_mask() instead.
        self.ax_nav.set_xlim(0, shape_y)
        self.ax_nav.set_ylim(shape_x, 0)
        self.toolbar.update()
        self.toolbar.push_current()

        self.update_canvas(0)
        self.canvas.draw()
        # Scale fields may already hold a value from a previous session/load -
        # update_scalebar() is otherwise only triggered by the fields' own
        # textChanged signal, so a fresh load wouldn't show it until touched.
        self.update_scalebar('real')
        self.update_scalebar('reciprocal')
        self.slider_imgNo.setRange(0, len(self.nav_imgs)-1)
        self.frame_flag_bar.set_range(len(self.nav_imgs))
        self._quality_flags = {}
        self.logger.info('No. of Images: %s', len(self.nav_imgs))
        
        self.button_reset_rois.setEnabled(True)
        # self.button_cur_roi.setEnabled(True)
        self.button_track.setEnabled(True)
        self.button_fineTuneMask.setEnabled(True)
        self.button_blobSettings.setEnabled(True)

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
                # Old saved analyses simply lack these keys - default to
                # "no segments" (nothing configured) rather than KeyError.
                # 'edge_detection' (not 'edge') matches _save_results_impl's
                # own JSON field name.
                'mesh': row.get('mesh', {'segments': []}),
                'dilate_erode': row.get('dilate_erode', {'segments': []}),
                'edge': row.get('edge_detection', {'segments': []}),
                'blob': row.get('blob', {'segments': []}),
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
                                      roi['ref'], roi['out_rois'], roi['mask'], roi['dp'],
                                      roi['mesh'], roi['dilate_erode'], roi['edge'], roi['blob']]
            self.add_item_tree(idx, roi['init'], roi['end'], roi['ref'], roi['use'])
            row_index = self.df_rois.index.get_loc(idx)
            if roi['out_rois'] is not None:
                self.toggle_tree_icon(row_index, 'trk', True)
            if roi['dp'] is not None:
                self.toggle_tree_icon(row_index, 'ext', True)

        self.disable_3ded_widgets(False)
        # Select the first restored ROI, if any, so its tracking/mask/DP
        # actually show up right away - update_canvas() only draws the
        # currently-selected ROI's own results, and nothing in the tree is
        # selected by default just from populating it above (add_item_tree
        # doesn't select what it adds), so without this the loaded results
        # sat in df_rois unseen until the user clicked a row themselves.
        if rois:
            self.tree_objects.setCurrentItem(self.tree_objects.topLevelItem(0))
        self.update_canvas(0)
        # Loaded DPs may have a different center than the placeholder - re-run
        # auto-centering now if enabled.
        self.update_scalebar('reciprocal')
        self.logger.info('Loaded saved analysis from %s (%d ROI(s)).', path, len(rois))

    def disable_3ded_widgets(self, state):
        # box_3ded now stacks Threshold/Blob Selection/Tracking/Extract (CPU
        # Cores/Clip FPS/Autosave/Make *.pts2/the 3 extract buttons) in one
        # combined ribbon column. button_cancel lives in here too, but must
        # stay independent of this sweep - it needs to stay clickable
        # regardless of tracking/extraction state, managed by its own
        # enable/disable calls elsewhere. combo_trackMethod is excluded the
        # same way - it's just a preference (which tracker algorithm to use
        # next time "Track!" is clicked), pickable at any time, not
        # something that needs a loaded signal/tracked ROI first the way
        # everything else in this column does.
        for wid in self.box_3ded.findChildren(qtw.QWidget):
            if isinstance(wid, qtw.QLabel) or wid in (self.button_cancel, self.combo_trackMethod):
                continue
            wid.setDisabled(state)
        # button_fineTuneMask/button_blobSettings live in the left
        # object-list panel (see init_widget), not box_3ded, so the sweep
        # above doesn't reach them - toggled explicitly here instead, same
        # as everything else in this column (they act on a selected ROI's
        # tracked mask, so they shouldn't stay clickable mid-tracking/
        # extraction either).
        self.button_fineTuneMask.setDisabled(state)
        self.button_blobSettings.setDisabled(state)
    
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
                        'ref', 'out_rois', 'mask', 'dp', 'mesh', 'dilate_erode', 'edge', 'blob']
        self.df_rois = pd.DataFrame([], columns=self.cols_df)
        self.df_rois = self.df_rois.astype({'use': int, 'init': object, 'in_rois': object, 'end': int,
                                            'out_rois': object, 'dp': object, 'ref':str, 'mask':object,
                                            'mesh': object, 'dilate_erode': object, 'edge': object,
                                            'blob': object})
        
        self.patches_axTrack.clear()
        self.patches_axNav.clear()

    def reset_rois(self):
        self.tree_objects.clear()
        self.empty_main_dataframe()
        self._quality_flags = {}
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

    def _step_frame(self, delta):
        """Previous/Next Frame buttons: move the slider by one frame,
        clamped to its range - mirrors MaskEditDialog's own _step_frame."""
        self.slider_imgNo.setValue(int(np.clip(
            self.slider_imgNo.value() + delta,
            self.slider_imgNo.minimum(), self.slider_imgNo.maximum())))

    def update_canvas(self, imgNo=None):
        """Redraw the nav/mask/dp panels for `imgNo` (or the slider's
        current value) and the selected ROI, via blit. Nav. Signal and
        Tracking Results share one merged axis (ax_nav) - the input ROI
        (red, draw_rois_in) and tracked ROI (tab:orange, draw_rois_out) are
        both drawn directly on top of the one frame image, so there's no
        separate "track" image to update - just the ROI overlays."""
        # Guards the threshold controls' live-preview wiring - they live in
        # box_3ded, normally disabled until a navigation signal is loaded,
        # but a disabled QWidget still
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

        # Nav overlay + DP panel: always keyed on the table's own selection
        # (idx), regardless of the Selected Object/All Active Objects
        # toggle below - that toggle only changes what ax_mask (2) itself
        # shows (see the mask-panel block further down).
        selected_items = self.tree_objects.selectedItems()
        idx = None
        if selected_items:
            item = selected_items[0]
            idx = int(item.text(1))
            self.frame_flag_bar.set_flags(self._quality_flags.get(idx))
            if not np.all(pd.isna(self.df_rois.loc[idx, 'out_rois'])):
                self.draw_rois_out(imgNo)
            else:
                self._clear_tracked_roi_overlay()

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
        else:
            self._clear_tracked_roi_overlay()
            self.frame_flag_bar.set_flags(None)

        # mask panel (ax_mask, (2)): "All Active Objects" ignores idx
        # entirely - every use==1 object at once, regardless of selection.
        # "Selected Object" needs idx to actually point at a tracked ROI,
        # same conditions this used to gate directly on selected_items.
        if self.radio_maskAll.isChecked():
            self._draw_all_object_masks(img, imgNo)
        elif idx is not None and not np.all(pd.isna(self.df_rois.loc[idx, 'out_rois'])) \
                and self.df_rois.loc[idx, 'out_rois'][imgNo].any():
            img_mask, img_roi = self.threshold_img(
                img, self.df_rois.loc[idx, 'out_rois'][imgNo],
                self.combo_thresh_method.currentText(),
                self.slider_thresh.value(), idx=idx, frame_idx=imgNo) #TODO add thresholding mode to the GUI and function here
            self.update_ax_mask(img_roi, img_mask)
            self._draw_blob_overlay(idx, imgNo)
        else:
            self.update_ax_mask(self.img_zero, self.img_zero)
            self._draw_blob_overlay(None, imgNo)

        # A single blit for the whole (single, 3-subplot) canvas here
        # (instead of a full canvas.draw()/draw_idle() per frame) avoids
        # re-rendering every artist in the figure - scale bars, static
        # titles/labels, axis chrome - on every single slider tick; only
        # the image data, ROI boxes, and the per-frame title text actually
        # change between frames, so only those are redrawn.
        # constrained_layout's spacing solve is also one of the most
        # expensive parts of a full draw and doesn't need to repeat once
        # subplot spacing has settled.
        nav_artists = ([self.img_display['nav'], self.ax_nav.title]
                       + self.patches_axNav + self.patches_axTrack)
        extract_artists = ([self.img_display['img_mask'], self.img_display['mask'],
                           self.img_display['dp']] + self._blob_overlay_artists
                          + self._all_mask_artists)
        self._blit_canvas(
            self.canvas, self.figure, '_bg', nav_artists + extract_artists,
            hide_for_background=[self.img_display['nav']] + extract_artists,
            titles_for_background=(self.ax_nav,))
        if not self._layout_frozen:
            self.figure.set_layout_engine('none')
            self._layout_frozen = True

    def _clear_tracked_roi_overlay(self):
        """Remove the tracked-ROI (tab:orange) overlay - and its reference-
        ROI dashed-yellow box, if any - from the merged nav/track axis,
        e.g. when no ROI is selected or the selected one isn't tracked yet.
        draw_rois_out() would otherwise leave a stale overlay on screen
        from whatever was previously selected, since it's simply not called
        in that case."""
        if len(self.patches_axTrack) > 0:
            for p in self.patches_axTrack:
                p.remove()
            self.patches_axTrack.clear()

    def update_ax(self, img, img_disp, ax, title=None,):
        """Update the image data, title, and color limits for one display
        axis (nav/track/mask/dp)."""
        # Rendering is deferred to the single canvas.draw()/draw_idle() call
        # at the end of update_canvas(), rather than a redraw per axis here.
        self.img_display[img_disp].set_data(img)
        ax.set_title(title)
        if img_disp == 'dp':
            # DP changes every frame (a different extracted pattern each
            # time) - re-anchor clip_dp's slider *bounds* to the new
            # frame's range every time, but only reset=True (jump both
            # thresholds back to "no clipping") the very first time, so a
            # manually-tuned threshold survives normal frame scrubbing
            # instead of resetting on every tick.
            self.clip_dp.set_range(img.min(), img.max(), reset=not self._dp_clip_initialized)
            self._dp_clip_initialized = True
            vmin, vmax = self.clip_dp.values()
            self.img_display['dp'].set_clim(vmin=vmin, vmax=vmax)
        else:
            self.img_display[img_disp].set_clim(vmin=img.min(), vmax=img.max())

    def _update_dp_clip(self):
        """clip_dp.valueChanged slot: re-apply its current vmin/vmax to the
        already-displayed DP image (no new data) and redraw."""
        if 'dp' not in self.img_display:
            return
        vmin, vmax = self.clip_dp.values()
        self.img_display['dp'].set_clim(vmin=vmin, vmax=vmax)
        self.canvas.draw_idle()

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
        `imgNo` onto the merged nav/track axis (ax_nav), replacing whatever
        was drawn there before - tab:orange, distinguishing them from the
        (red) input ROIs drawn by draw_rois_in on the same axis.

        A ROI-in-ROI object's own reference/parent ROI is drawn too (a
        distinct dashed yellow box), even if the parent isn't itself
        enabled ('use' unchecked) or otherwise wouldn't independently pass
        the `use == 1` filter below - the child's box only makes sense
        relative to its parent, so the parent should stay visible on the
        tracking canvas for as long as the child is. Skipped when the
        parent is already being drawn anyway (a normal solid orange box,
        from being independently enabled/tracked itself) to avoid drawing
        it twice."""
        if len(self.patches_axTrack) > 0:
            for p in self.patches_axTrack:
                p.remove()
            self.patches_axTrack.clear()
        df = self.df_rois[self.df_rois.use == 1]
        df = df.loc[df.out_rois.dropna().index]
        plotted_ids = set(df.index)
        drawn_refs = set()
        if len(df) > 0:
            for i in df.index:
                try:
                    roi = self.df_rois.loc[i, 'out_rois'][imgNo]
                    x,y,w,h = roi
                    if (w>0) and (h>0):
                        rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='tab:orange',
                                                 facecolor='none')
                        self.ax_nav.add_patch(rect)
                        self.patches_axTrack.append(rect)

                        # id
                        # pos = (x+w+15, y+h+15)
                        font_size = 8
                        pos = (x+w/2, y-15)
                        # font_size = 12
                        t = self.ax_nav.text(pos[0], pos[1], str(i), horizontalalignment='center',
                                               verticalalignment='center', color='tab:orange', fontsize=font_size)
                        self.patches_axTrack.append(t)
                except Exception:
                    self.logger.debug('Skipped drawing ROI %s at frame %d.', i, imgNo, exc_info=True)

                ref = self.df_rois.loc[i, 'ref']
                if pd.isna(ref) or ref in drawn_refs:
                    continue
                drawn_refs.add(ref)
                ref_idx = int(ref)
                if ref_idx in plotted_ids or ref_idx not in self.df_rois.index:
                    continue
                try:
                    ref_roi = self.df_rois.loc[ref_idx, 'out_rois'][imgNo]
                    rx, ry, rw, rh = ref_roi
                    if (rw > 0) and (rh > 0):
                        ref_rect = patches.Rectangle((rx, ry), rw, rh, linewidth=1.5,
                                                     edgecolor='yellow', linestyle='--',
                                                     facecolor='none')
                        self.ax_nav.add_patch(ref_rect)
                        self.patches_axTrack.append(ref_rect)
                        t = self.ax_nav.text(rx+rw/2, ry-15, str(ref_idx), horizontalalignment='center',
                                               verticalalignment='center', color='yellow', fontsize=8)
                        self.patches_axTrack.append(t)
                except Exception:
                    self.logger.debug('Skipped drawing reference ROI %s at frame %d.',
                                      ref_idx, imgNo, exc_info=True)
        # Rendering is deferred to the single canvas.draw()/draw_idle() call
        # at the end of update_canvas(), rather than a blit here.

    def update_ax_mask(self, img_roi, img_mask):
        """Update ax_mask's cropped ROI image and threshold-mask overlay,
        resetting the view to fit the crop only when its size actually
        changed since the last call."""
        # Stale index-label artists from a previous "All Active Objects"
        # mask-panel session (see _draw_all_object_masks) don't mean
        # anything once back in this single-object view - clear them here
        # rather than requiring every update_canvas() branch that lands on
        # this method to remember to.
        if self._all_mask_artists:
            for artist in self._all_mask_artists:
                try:
                    artist.remove()
                except Exception:
                    self.logger.debug('All-objects mask artist already removed.', exc_info=True)
            self._all_mask_artists = []
        shape_x, shape_y = img_mask.shape
        self.img_display['img_mask'].set_data(img_roi)
        try:
            self.img_display['img_mask'].set_clim(vmin=img_roi.min(), vmax=img_roi.max())
        except ValueError:
            pass
        self.img_display['img_mask'].set_extent([0, shape_y, shape_x, 0])

        # A translucent RGBA overlay (like Tab_SAM2's show_mask) instead of
        # a scalar viridis image at a fixed low alpha - that used to tint
        # the *whole* axis (mask=0 regions included, just a dim viridis(0)
        # purple), rather than only coloring where the mask is actually
        # True and leaving everything else fully see-through.
        mask_color = np.array([*to_rgb('tab:orange'), 0.15])
        mask_rgba = img_mask.reshape(shape_x, shape_y, 1) * mask_color.reshape(1, 1, -1)
        self.img_display['mask'].set_data(mask_rgba)
        self.img_display['mask'].set_extent([0, shape_y, shape_x, 0])
        # The ROI crop's size varies between ROIs/selections, so (unlike
        # nav/track) the view here is reset to fit it whenever that size
        # actually changes, rather than only once at load - otherwise it
        # would stay at whatever size an earlier, differently-sized ROI
        # last used. But a same-size update (frame scrub, edge-detection
        # setting tweak, ...) leaves the user's current zoom/pan alone.
        if (shape_x, shape_y) != getattr(self, '_ax_mask_shape_seen', None):
            self.ax_mask.set_xlim(0, shape_y)
            self.ax_mask.set_ylim(shape_x, 0)
            # adjustable='box' (not the 'datalim' some other call could have
            # left it on) - a tracked ROI's crop aspect ratio can change a
            # lot frame to frame, and 'datalim' would stretch/distort the
            # *data* limits to fill this axis's already-allotted box,
            # exactly the pixel-squashing this is meant to avoid. 'box'
            # instead resizes the box itself (within its allotted subplot
            # space) to fit the image at its correct 1:1 pixel aspect.
            self.ax_mask.set_aspect('equal', adjustable='box')
            self._ax_mask_shape_seen = (shape_x, shape_y)
            # That box resize only actually takes effect during a REAL
            # Axes.draw() (Axes.apply_aspect(), which recomputes the axes'
            # own position/bbox within the figure, only runs there) - but
            # update_canvas's own blit (_blit_canvas) never calls that,
            # only canvas.restore_region(self._bg) (repainting a *cached*
            # bitmap captured back when the box had a DIFFERENT shape) plus
            # draw_artist() on the handful of per-frame artists. The result
            # was the reported stretching: img_mask/mask get freshly drawn
            # at the new, correctly-reshaped box, painted right on top of
            # chrome/other-axes pixels still sitting at the OLD box shape
            # underneath. Invalidating the cached background here forces
            # the next update_canvas() to fall back to a real canvas.draw()
            # (see _blit_canvas's own "if self._bg is None" branch), which
            # re-runs apply_aspect() for every axis and recaptures a fresh,
            # correctly-shaped background before any further blitting.
            self._bg = None
        # Rendering is deferred to the single canvas.draw()/draw_idle() call
        # at the end of update_canvas(), rather than a blit here.
        # self.canvas.draw_idle()

    def _on_mask_mode_changed(self):
        self.update_canvas()

    def _draw_all_object_masks(self, img, imgNo):
        """"All Active Objects" mask-panel mode (see the radio_maskAll/
        radio_maskSelected toggle above tree_objects): composite every
        active ("Use" checked) ROI's own threshold mask onto the FULL nav
        frame at once - unlike the normal "Selected Object" view
        (update_ax_mask), which shows just one object's own cropped
        threshold view - each object in its own tab10 color, with its
        index labeled at its own mask centroid. Ignores the table's own
        selection entirely - every active object is shown regardless of
        which one, if any, is currently selected."""
        for artist in self._all_mask_artists:
            try:
                artist.remove()
            except Exception:
                self.logger.debug('All-objects mask artist already removed.', exc_info=True)
        self._all_mask_artists = []
        # A single-object overlay (Blob Selection's numbered contours)
        # doesn't mean anything once several objects' masks are composited
        # together - clear it rather than leaving a stale one from whatever
        # was last selected in "Selected Object" mode.
        for artist in self._blob_overlay_artists:
            try:
                artist.remove()
            except Exception:
                self.logger.debug('Blob overlay artist already removed.', exc_info=True)
        self._blob_overlay_artists = []

        shape = img.shape
        composite = np.zeros((*shape, 4))
        cmap = plt.get_cmap('tab10')
        labels = []
        for idx2 in self.df_rois[self.df_rois['use'] == 1].index:
            out_rois = self.df_rois.loc[idx2, 'out_rois']
            if np.all(pd.isna(out_rois)):
                continue
            roi = out_rois[imgNo]
            if not roi.any():
                continue
            try:
                img_mask, _ = self.threshold_img(
                    img, roi, self.combo_thresh_method.currentText(),
                    self.slider_thresh.value(), idx=idx2, frame_idx=imgNo)
            except Exception:
                continue
            y, x, h, w = roi
            full_mask = np.zeros(shape, dtype=bool)
            full_mask[x:x + w, y:y + h] = img_mask
            if not full_mask.any():
                continue
            color = np.array([*cmap(idx2 % 10)[:3], 0.28])
            composite[full_mask] = color
            cx, cy = io.mask_centroid(full_mask)
            labels.append((idx2, cx, cy))

        self.img_display['img_mask'].set_data(img)
        try:
            self.img_display['img_mask'].set_clim(vmin=img.min(), vmax=img.max())
        except ValueError:
            pass
        self.img_display['img_mask'].set_extent([0, shape[1], shape[0], 0])
        self.img_display['mask'].set_data(composite)
        self.img_display['mask'].set_extent([0, shape[1], shape[0], 0])

        for idx2, cx, cy in labels:
            text = self.ax_mask.text(cx, cy, str(idx2), color='white', fontsize=9,
                                     fontweight='bold', horizontalalignment='center',
                                     verticalalignment='center')
            self._all_mask_artists.append(text)

        # Same reset-view-only-on-shape-change convention as update_ax_mask,
        # tracked separately from its own _ax_mask_shape_seen (this mode
        # shows the full frame, a different shape than any one object's own
        # cropped ROI) - and invalidates the cached blit background for the
        # same reason update_ax_mask does (see its own comment): a box
        # resize only takes effect on a real canvas.draw().
        if shape != self._ax_mask_full_shape_seen:
            self.ax_mask.set_xlim(0, shape[1])
            self.ax_mask.set_ylim(shape[0], 0)
            self.ax_mask.set_aspect('equal', adjustable='box')
            self._ax_mask_full_shape_seen = shape
            self._ax_mask_shape_seen = None  # force update_ax_mask to reset too, next time
            self._bg = None

    def update_scalebar(self, which):
        """Add/update the scale bar (which='real', on nav/mask) or the
        reciprocal-space rings (which='reciprocal', on the dp axis), based
        on the current scale line-edit text."""
        if which == 'real':
            try:
                scale_real = float(self.lineEdit_scale_real.text())

                for ax in [self.ax_nav, self.ax_mask]:
                    io.add_readable_scalebar(ax, scale_real, 'nm')

                # The scale bar itself is static across frames, so it needs
                # to be baked into the cached blit background - invalidate
                # it here so the next frame update recaptures it, while
                # still doing an immediate full draw for instant feedback now.
                self._bg = None
                self.canvas.draw_idle()

            except ValueError:

                for ax in [self.ax_nav, self.ax_mask]:
                    for artist in ax.artists[:]:
                        if isinstance(artist, ScaleBar):
                            artist.remove()

                self._bg = None
                self.canvas.draw_idle()

        elif which == 'reciprocal':
            # A conventional linear scale bar doesn't read naturally on a
            # radially-symmetric diffraction pattern - concentric dashed
            # rings at every 1 1/A (centered on the DP) work better.
            dp_array = self.img_display['dp'].get_array()
            shape = dp_array.shape
            # Centering is purely manual now (see find_and_center_recip and
            # Ctrl+Click in on_click_dp) - self.dp_center just persists
            # across frames until one of those changes it, no more
            # continuous auto-re-finding on every frame update here.
            self._dp_recip_circles = io.draw_reciprocal_scale_circles(
                self.ax_dp, self.lineEdit_scale_recip.text(), shape,
                center=self.dp_center, old_artists=getattr(self, '_dp_recip_circles', None))
            # The circles are static across frames like the scale bars above.
            self._bg = None
            self.canvas.draw_idle()

    def show_help_dialog(self):
        """Ribbon "?" tool: shortcuts/mouse controls for this tab, moved
        here from each subplot's own xlabel (see the canvas-setup history) -
        crowded, and on a narrow window two adjacent subplots' multi-line
        hints could visibly run into each other."""
        self.show_shortcuts_dialog(
            'Nav. Image / Tracking Results (merged axis):\n'
            '  Hold "Ctrl" + Left Click+Drag  ->  New ROI\n'
            '  Hold "Ctrl" + Right Click  ->  Add init to existing ROI\n'
            '  To make a ROI-in-ROI: draw a plain ROI here, then set its "Ref" '
            'in the object list below to another ROI\'s index - it\'s auto-clamped '
            'to fit inside that reference ROI\'s current bounds.\n'
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
                'Load/track a ROI first, so a beam center can be found.')
            return
        try:
            self.dp_center = io.find_dp_center_blurred(dp_array)
        except Exception:
            self.logger.exception('Auto-centering failed.')
            return
        self.update_scalebar('reciprocal')

    def threshold_img(self, img, roi, thresh_method, thresh_offset, mode='full', idx=None, frame_idx=None):
        """Blur `img`, compute a threshold via `thresh_method` (over the
        whole image if mode='full', or just within `roi` if mode='roi'),
        binarize at `thresh_offset` (percent of the computed threshold),
        then crop both the mask and the raw image to `roi` (see
        _raw_threshold_crop) and apply Blob Selection/the current edge-
        detection settings (and, if `idx` is given, that ROI's Mesh
        restriction for `frame_idx` - see apply_edge_mask) to the mask.
        Returns (img_mask, img_cut)."""
        img_mask, img_cut = self._raw_threshold_crop(img, roi, thresh_method, thresh_offset, mode)
        img_mask = self.apply_edge_mask(img_mask, idx, frame_idx)
        return img_mask, img_cut

    def _raw_threshold_crop(self, img, roi, thresh_method, thresh_offset, mode='full'):
        """The blur+threshold+crop portion of threshold_img, without any of
        apply_edge_mask's post-processing (Blob Selection/Dilate-Erode/Edge
        Detection/Mesh) - split out so the "which blob did the user click"
        handler (see _on_blob_mask_clicked) can get at the same raw,
        possibly-multi-blob mask threshold_img itself starts from, instead
        of the already-restricted-to-one-blob result. Returns (img_mask,
        img_cut), same as threshold_img."""
        blur_sigma = self.spinbox_blur.value()
        threshold_methods = {'otsu': threshold_otsu, 'li': threshold_li,
                             'yen': threshold_yen, 'mean': threshold_mean}
        threshold_func = threshold_methods[thresh_method]

        y,x,h,w = roi
        img_blur = io.denoise_image(img, 'Gaussian Blur', blur_sigma) if blur_sigma > 0 else img
        img_cut = img[x:x+w, y:y+h]
        if mode == 'full':
            th = io.threshold_ignore_zero(threshold_func, img_blur)
        elif mode == 'roi':
            th = io.threshold_ignore_zero(threshold_func, img_cut)
        thresh_offset = thresh_offset / 100
        thresh = thresh_offset * th
        img_mask = img_blur >= thresh
        img_mask = img_mask[x:x+w, y:y+h]
        return img_mask, img_cut

    def _mesh_settings_for(self, idx):
        """This ROI's Mesh settings (see MaskEditDialog/get_mesh_settings),
        or None if it has none set / idx is None."""
        if idx is None:
            return None
        mesh = self.df_rois.at[idx, 'mesh']
        return mesh if isinstance(mesh, dict) else None

    def _dilate_erode_settings_for(self, idx):
        """This ROI's Dilate/Erode segments (see MaskEditDialog/
        get_dilate_erode_settings - `{'segments': [...]}`), or None if it
        has none set / idx is None. Per-ROI, set only from the Fine-Tune
        Mask dialog - no main-tab control (same as Mesh, and, since the
        Segments feature, Edge Detection too - see _edge_settings_for)."""
        if idx is None:
            return None
        dilate_erode = self.df_rois.at[idx, 'dilate_erode']
        return dilate_erode if isinstance(dilate_erode, dict) else None

    def _edge_settings_for(self, idx):
        """This ROI's Edge Detection segments (see MaskEditDialog/
        get_edge_settings - `{'segments': [...]}`), or None if it has none
        set / idx is None. Per-ROI (like Mesh/Dilate-Erode) - Edge
        Detection used to be a single tab-wide setting before the Fine-Tune
        Mask dialog's Segments feature; now it's only ever set there, one
        range at a time, same as the other two."""
        if idx is None:
            return None
        edge = self.df_rois.at[idx, 'edge']
        return edge if isinstance(edge, dict) else None

    def _blob_settings_for(self, idx):
        """This ROI's Blob Selection settings - `{'method', 'params',
        'segments': [{'start','end','enabled','seed_centroid'}, ...]}` (see
        EDyssey/io_utils/blob_segmentation.py for 'method'/'params', which a
        ROI that's never opened the Blob Settings dialog may not have set
        at all yet - see _blob_method_params for the DEFAULT_BLOB_METHOD
        fallback) - or None if it has none set / idx is None. Per-ROI,
        main-tab-only (unlike Dilate/Erode/Edge Detection/Mesh, which live
        in the Fine-Tune Mask dialog) - see the "Blob" object-list column/
        _on_blob_mask_clicked/blob_segmentation_dialog.py."""
        if idx is None:
            return None
        blob = self.df_rois.at[idx, 'blob']
        return blob if isinstance(blob, dict) else None

    def _blob_method_params(self, idx):
        """This ROI's chosen Blob Selection segmentation method/params (see
        blob_segmentation.py), defaulting to DEFAULT_BLOB_METHOD/its own
        default params for a ROI that's never opened the Blob Settings
        dialog (or unset one of the two) - keeps every caller from needing
        its own fallback logic."""
        blob = self._blob_settings_for(idx) or {}
        method = blob.get('method') or io.DEFAULT_BLOB_METHOD
        if method not in io.BLOB_SEGMENTATION_METHODS:
            method = io.DEFAULT_BLOB_METHOD
        params = blob.get('params') or io.default_blob_params(method)
        return method, params

    def _current_intensity_crop(self, idx, frame_idx):
        """The raw-intensity ROI crop matching this frame's mask crop (see
        _raw_threshold_crop/_current_raw_blob_mask) - or None if there's
        nothing to compute it from yet. Needed by the intensity-based Blob
        Selection segmentation method (see blob_segmentation.py), which
        must tell touching particles apart using more than just the binary
        mask's own shape."""
        if idx is None:
            return None
        out_rois = self.df_rois.at[idx, 'out_rois']
        if np.all(pd.isna(out_rois)) or frame_idx >= len(out_rois):
            return None
        roi = out_rois[frame_idx]
        if roi is None or not roi.any():
            return None
        y, x, h, w = roi
        return self.nav_imgs[frame_idx][x:x+w, y:y+h]

    def _label_blobs_for(self, mask, idx, frame_idx, method, params):
        """This ROI's raw threshold `mask` split into individual blobs via
        its own chosen segmentation `method`/`params` (see
        blob_segmentation.label_blobs) - fetching the matching intensity
        crop (see _current_intensity_crop) only when that method actually
        needs one. None (blob_segmentation.label_blobs' own signal to fall
        back to plain connected-components) for 'connected' or an empty
        mask."""
        if method == io.DEFAULT_BLOB_METHOD or not mask.any():
            return None
        img_cut = None
        if io.BLOB_SEGMENTATION_METHODS.get(method, {}).get('needs_intensity'):
            img_cut = self._current_intensity_crop(idx, frame_idx)
        return io.label_blobs(mask, img_cut, method, params)

    def _current_blob_labels(self, idx, frame_idx):
        """(raw_mask, labels) for ROI `idx`'s threshold mask at
        `frame_idx`, labels split according to this ROI's own chosen Blob
        Selection segmentation method (see _label_blobs_for) - shared by
        _on_blob_mask_clicked (click -> blob lookup) and _draw_blob_overlay
        (contour/number display), so both agree exactly with what
        _resolve_blob_mask will itself pick during extraction. raw_mask is
        None (labels then meaningless) if there's nothing to compute
        either from yet - see _current_raw_blob_mask."""
        mask = self._current_raw_blob_mask(idx, frame_idx)
        if mask is None:
            return None, None
        method, params = self._blob_method_params(idx)
        return mask, self._label_blobs_for(mask, idx, frame_idx, method, params)

    def _sync_blob_checkbox(self, idx):
        """Make the "Blob" object-list column checkbox for ROI `idx` match
        _blob_enabled_for(idx) - needed after _open_blob_segmentation_dialog
        enables Blob Selection from the "Blob Settings..." button (which,
        unlike checking the column's own checkbox, doesn't already go
        through on_item_check_changed). Signals blocked while setting it so
        this doesn't itself re-trigger on_item_check_changed/reopen the
        dialog."""
        idx_col = self.cols_tree.index('idx')
        blob_col = self.cols_tree.index('blob')
        for col in range(self.tree_objects.topLevelItemCount()):
            item = self.tree_objects.topLevelItem(col)
            if item.text(idx_col) == str(idx):
                self.tree_objects.blockSignals(True)
                item.setCheckState(blob_col,
                                   Qt.Checked if self._blob_enabled_for(idx) else Qt.Unchecked)
                self.tree_objects.blockSignals(False)
                return

    def _open_blob_segmentation_dialog(self, idx):
        """Open BlobSegmentationDialog for ROI `idx`, pre-filled with its
        current segmentation method/params/seed (see _blob_method_params/
        _blob_settings_for) and a live preview built from its current
        (main-tab slider) frame's own raw threshold mask/intensity crop -
        called both right after a fresh "Blob" checkbox check
        (on_item_check_changed) and from the "Blob Settings..." ribbon
        button, to revisit an already-configured ROI's choice.

        On Accept, writes the dialog's (possibly unchanged) method/params
        back, and - if a blob was actually clicked in the dialog - reseeds
        Blob Selection at the previewed frame via _split_blob_segment
        (exactly as clicking that same blob directly on the "ROI with
        Threshold" panel would). On Cancel/close, leaves whatever was
        already there - the default (Connected Components, no seed)
        _set_blob_enabled already put in place for a fresh checkbox check,
        or this ROI's previous settings if reopened via the ribbon button -
        untouched, so cancelling never leaves Blob Selection worse off than
        before the dialog opened."""
        imgNo = self.slider_imgNo.value()
        mask = self._current_raw_blob_mask(idx, imgNo)
        img_cut = self._current_intensity_crop(idx, imgNo)
        if mask is None:
            qtw.QMessageBox.information(self, 'Blob Selection',
                "This ROI has no tracked mask on the current frame yet, so there's "
                "nothing to preview here - Blob Selection is enabled with the default "
                '(Connected Components) method for now. Track this ROI, then reopen '
                'this from the "Blob Settings..." button to fine-tune it.')
            return
        method, params = self._blob_method_params(idx)
        blob = self._blob_settings_for(idx) or {}
        segments = blob.get('segments') or []
        seg = io.segment_for_frame(segments, imgNo) if segments else None
        seed_centroid = tuple(seg['seed_centroid']) if seg and seg.get('seed_centroid') else None

        dlg = BlobSegmentationDialog(mask, img_cut, method, params, seed_centroid, parent=self)
        if dlg.exec_() != qtw.QDialog.Accepted:
            return
        method, params, seed_centroid, seed_changed = dlg.result()
        self.df_rois.at[idx, 'blob'] = {**blob, 'method': method, 'params': params,
                                        'segments': segments or [
                                            {'start': 0, 'end': len(self.nav_imgs) - 1,
                                             'enabled': True, 'seed_centroid': None}]}
        if seed_changed:
            self._split_blob_segment(idx, imgNo, seed_centroid)  # also clears the centroid cache
        else:
            self._blob_centroid_cache.pop(idx, None)
        self._sync_blob_checkbox(idx)
        self.update_canvas()

    def _has_active_postprocessing(self, idx):
        """Whether ROI `idx` has ANY segment (see MaskEditDialog's Segments
        feature) with Dilate/Erode, Edge Detection, Mesh, or Blob Selection
        actually enabled - lets extract_3ded skip apply_edge_mask's
        per-frame work entirely for a ROI with nothing to do there."""
        def _any_enabled(settings, extra=lambda s: True):
            segs = (settings or {}).get('segments') or []
            return any(s.get('enabled') and extra(s) for s in segs)
        return (_any_enabled(self._edge_settings_for(idx))
               or _any_enabled(self._dilate_erode_settings_for(idx), lambda s: (
                   s.get('kernel', 0) != 0 or s.get('open_kernel', 0) != 0 or s.get('close_kernel', 0) != 0))
               or _any_enabled(self._mesh_settings_for(idx), lambda s: s.get('cells'))
               or _any_enabled(self._blob_settings_for(idx)))

    def _resolve_blob_mask(self, idx, frame_idx, mask):
        """If ROI `idx` has Blob Selection enabled for `frame_idx` (see the
        "Blob Selection" ribbon section/_on_blob_mask_clicked), restrict
        `mask` to just one connected component - the one nearest whatever
        it was last seen at - via io.select_blob_by_centroid. A no-op
        (returns `mask` unchanged) only if Blob Selection isn't enabled at
        all for this frame; with no seed yet (enabled but never clicked),
        io.select_blob_by_centroid's own None-seed default (largest blob)
        still applies, so simply checking "Enable" already shows a
        reasonable starting choice before the user clicks a specific one.

        Auto-follow: seeded from the immediately preceding frame's own
        chosen centroid when that's already cached (true frame-to-frame
        following - the common case, since both live-preview frame
        scrubbing and extract_3ded's own per-frame loop naturally proceed
        through frames in order) and still within the same segment;
        otherwise falls back to reseeding from the segment's own fixed
        seed_centroid (wherever the user last clicked, or None for
        "largest blob") rather than walking every intermediate frame just
        to jump far ahead - a bounded-cost approximation of "auto-follow",
        not an exhaustive one."""
        segments = (self._blob_settings_for(idx) or {}).get('segments')
        seg = io.segment_for_frame(segments, frame_idx) if segments else None
        if not seg or not seg.get('enabled'):
            return mask
        method, params = self._blob_method_params(idx)
        labels = self._label_blobs_for(mask, idx, frame_idx, method, params)
        cache = self._blob_centroid_cache.setdefault(idx, {})
        prev = cache.get(frame_idx - 1)
        fixed_seed = seg.get('seed_centroid')
        seed = prev if (prev is not None and seg['start'] <= frame_idx - 1 <= seg['end']) \
            else (tuple(fixed_seed) if fixed_seed is not None else None)
        restricted, chosen_centroid = io.select_blob_by_centroid(mask, seed, labels=labels)
        if chosen_centroid is not None:
            cache[frame_idx] = chosen_centroid
        return restricted

    def _selected_roi_idx(self):
        """The tree_objects row currently selected, as a df_rois index, or
        None if nothing's selected - shared by the Blob Selection controls,
        which (unlike Dilate/Erode/Edge Detection/Mesh) act on whichever
        ROI is selected in the main tab rather than needing a dialog open."""
        selected_items = self.tree_objects.selectedItems()
        if not selected_items:
            return None
        return int(selected_items[0].text(1))

    def _blob_enabled_for(self, idx):
        """Whether Blob Selection's own "Blob" column checkbox is checked
        for ROI `idx` - reads straight from df_rois (the source of truth,
        kept in sync with that checkbox by on_item_check_changed) rather
        than any UI widget, since (unlike the old single ribbon checkbox
        this replaced) there's no single "the" Blob Selection widget
        anymore - every row has its own."""
        if idx is None:
            return False
        segments = (self._blob_settings_for(idx) or {}).get('segments') or []
        return any(s.get('enabled') for s in segments)

    def _set_blob_enabled(self, idx, enabled):
        """"Blob" column checkbox toggled for ROI `idx` (see
        on_item_check_changed): enable/disable it for the ROI's *whole*
        stack (a fresh single segment spanning every frame, no seed yet -
        see _resolve_blob_mask's own None-seed default) rather than per-
        frame-range like Dilate/Erode/Edge Detection/Mesh - clicking a blob
        (see _on_blob_mask_clicked) is what actually introduces segment
        boundaries, only once the user needs different blobs on different
        frame ranges. The segmentation method/params (see
        _blob_method_params/blob_segmentation_dialog.py) carry over from
        whatever this ROI was last left on, defaulting to
        DEFAULT_BLOB_METHOD the first time - re-enabling after a previous
        disable shouldn't forget a method the user already picked."""
        method, params = self._blob_method_params(idx)
        if enabled:
            n = len(self.nav_imgs) if hasattr(self, 'nav_imgs') else 1
            self.df_rois.at[idx, 'blob'] = {
                'method': method, 'params': params,
                'segments': [{'start': 0, 'end': n - 1, 'enabled': True, 'seed_centroid': None}]}
        else:
            self.df_rois.at[idx, 'blob'] = {'method': method, 'params': params, 'segments': []}
        self._blob_centroid_cache.pop(idx, None)

    def _split_blob_segment(self, idx, frame_idx, seed_centroid):
        """Seed (or re-seed) ROI `idx`'s Blob Selection at `frame_idx` -
        updates the segment already starting exactly there in place, or
        splits a new one starting there (capping the previous one's own
        `end` to `frame_idx - 1`) otherwise - mirrors MaskEditDialog's own
        _split_segment_here, just implemented here directly since Blob
        Selection has no Fine-Tune-Mask-style Split/Merge UI of its own;
        every click either updates or creates exactly one boundary."""
        blob = self._blob_settings_for(idx)
        segments = list((blob or {}).get('segments') or [])
        if not segments:
            n = len(self.nav_imgs)
            segments = [{'start': 0, 'end': n - 1, 'enabled': True, 'seed_centroid': None}]
        seg = io.segment_for_frame(segments, frame_idx)
        if seg['start'] == frame_idx:
            seg['seed_centroid'] = seed_centroid
        else:
            seg_pos = segments.index(seg)
            new_seg = {'start': frame_idx, 'end': seg['end'], 'enabled': True,
                      'seed_centroid': seed_centroid}
            seg['end'] = frame_idx - 1
            segments.insert(seg_pos + 1, new_seg)
        self.df_rois.at[idx, 'blob'] = {**(blob or {}), 'segments': segments}
        # A fresh seed invalidates any auto-follow trail computed forward
        # from the old settings - simplest safe choice is to drop the whole
        # per-ROI cache rather than reason about exactly which frames are
        # still valid.
        self._blob_centroid_cache.pop(idx, None)

    def _current_raw_blob_mask(self, idx, frame_idx):
        """The raw (possibly multi-blob) threshold mask for ROI `idx` at
        `frame_idx` - or None if there's nothing to compute it from yet
        (`idx` is None, not tracked, or an empty box on this frame).
        Shared by _on_blob_mask_clicked and _draw_blob_overlay, both of
        which need the SAME pre-restriction mask Blob Selection itself
        picks a component out of (see _resolve_blob_mask)."""
        if idx is None:
            return None
        out_rois = self.df_rois.at[idx, 'out_rois']
        if np.all(pd.isna(out_rois)):
            return None
        roi = out_rois[frame_idx]
        if not roi.any():
            return None
        img = self.nav_imgs[frame_idx]
        thresh_method = self.combo_thresh_method.currentText()
        thresh_offset = self.slider_thresh.value()
        img_mask_raw, _ = self._raw_threshold_crop(img, roi, thresh_method, thresh_offset)
        return img_mask_raw

    def _on_blob_mask_clicked(self, event):
        """A plain left-click on ax_mask ("ROI with Threshold") while Blob
        Selection is enabled: figure out which of the frame's connected
        components the click landed in (nearest centroid to the click
        point, among the RAW - not yet blob-restricted - threshold mask's
        own blobs - see _current_raw_blob_mask), then seed/re-seed Blob
        Selection there (see _split_blob_segment) and redraw."""
        idx = self._selected_roi_idx()
        if not self._blob_enabled_for(idx):
            return
        imgNo = self.slider_imgNo.value()
        img_mask_raw, labels = self._current_blob_labels(idx, imgNo)
        if img_mask_raw is None:
            return
        click = (event.xdata, event.ydata)
        _, chosen_centroid = io.select_blob_by_centroid(img_mask_raw, click, labels=labels)
        if chosen_centroid is None:
            self.logger.info('No blob under the click on ROI %d, frame %d.', idx, imgNo)
            return
        self._split_blob_segment(idx, imgNo, chosen_centroid)
        self.update_canvas(imgNo)

    def _draw_blob_overlay(self, idx, frame_idx):
        """Outline every detected blob in ax_mask's current (raw, possibly
        multi-blob) threshold mask, numbered, so the user can see which is
        which before clicking one (see _on_blob_mask_clicked) - split
        according to this ROI's own chosen segmentation method (see
        _current_blob_labels/blob_segmentation_dialog.py), with whichever
        one _resolve_blob_mask itself just picked for this exact frame
        (already cached in _blob_centroid_cache by the threshold_img call
        update_canvas makes just before this one - see _resolve_blob_mask)
        highlighted in green instead of cyan. Only draws anything when Blob
        Selection is enabled for this ROI; otherwise just clears whatever
        was drawn for a previously-selected ROI. Cleared/rebuilt every call
        rather than diffed, matching draw_rois_in/out's own convention for
        other per-frame overlays."""
        for artist in self._blob_overlay_artists:
            try:
                artist.remove()
            except Exception:
                self.logger.debug('Blob overlay artist already removed.', exc_info=True)
        self._blob_overlay_artists = []
        if not self._blob_enabled_for(idx):
            return
        img_mask_raw, labels = self._current_blob_labels(idx, frame_idx)
        if img_mask_raw is None:
            return
        if labels is None:
            mask_u8 = img_mask_raw.astype('uint8')
            _, labels = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)[:2]

        chosen_centroid = self._blob_centroid_cache.get(idx, {}).get(frame_idx)
        chosen_label = None
        if chosen_centroid is not None:
            row, col = int(round(chosen_centroid[1])), int(round(chosen_centroid[0]))
            if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]:
                chosen_label = labels[row, col]

        for label in (lid for lid in np.unique(labels) if lid != 0):
            blob_u8 = (labels == label).astype('uint8')
            is_chosen = label == chosen_label
            color = 'lime' if is_chosen else 'cyan'
            ys, xs = np.where(blob_u8)
            cx, cy = float(xs.mean()), float(ys.mean())
            contours, _ = cv2.findContours(blob_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                pts = contour.reshape(-1, 2)  # (col, row) = (x, y), matching ax_mask's own extent
                if len(pts) < 2:
                    continue
                line = Line2D(pts[:, 0], pts[:, 1], color=color, linewidth=1.6 if is_chosen else 1.2)
                self.ax_mask.add_line(line)
                self._blob_overlay_artists.append(line)
            text = self.ax_mask.text(cx, cy, str(int(label)), color=color, fontsize=9, fontweight='bold',
                                     horizontalalignment='center', verticalalignment='center')
            self._blob_overlay_artists.append(text)

    def apply_edge_mask(self, mask, idx=None, frame_idx=None):
        """Restrict `mask` to a single blob first when Blob Selection is
        enabled for `frame_idx` (see _resolve_blob_mask - runs first since
        Dilate/Erode/Edge Detection/Mesh below should all act on just the
        one selected object, not the raw possibly-multi-blob threshold
        result). Then grow/shrink it uniformly when `idx`'s Dilate/Erode
        setting for `frame_idx` (see MaskEditDialog's Segments - resolved
        via io.segment_for_frame) is enabled (see io.dilate_erode_mask),
        then reduce it to just its edge/outline when that frame's Edge
        Detection segment is enabled (isotropic, or one-sided along
        "Directional"'s angle when that's also set - see
        io.erode_mask_edge), then - if `idx` is given and that frame's Mesh
        segment has a restriction set - restrict it to the selected mesh
        cell(s), relative to the object's own position on THIS frame (see
        io.mesh_restrict_mask/io.mask_centroid, and MaskEditDialog.
        _effective_mask's identical convention) so a tracked ROI's motion
        across frames doesn't throw off which part of it the selection
        actually covers. A no-op otherwise."""
        mask = self._resolve_blob_mask(idx, frame_idx, mask)

        mesh_segments = (self._mesh_settings_for(idx) or {}).get('segments')
        mesh = io.segment_for_frame(mesh_segments, frame_idx) if mesh_segments else None
        mesh_on = bool(mesh and mesh.get('enabled') and mesh.get('cells'))
        origin = io.mask_centroid(mask) if mesh_on else None

        de_segments = (self._dilate_erode_settings_for(idx) or {}).get('segments')
        de = io.segment_for_frame(de_segments, frame_idx) if de_segments else None
        if de and de.get('enabled'):
            if de.get('kernel', 0) != 0:
                mask = io.dilate_erode_mask(mask, de['kernel'])
            if de.get('open_kernel', 0) != 0:
                mask = io.open_mask(mask, de['open_kernel'])
            if de.get('close_kernel', 0) != 0:
                mask = io.close_mask(mask, de['close_kernel'])

        edge_segments = (self._edge_settings_for(idx) or {}).get('segments')
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

    def _on_ribbon_tool_changed(self, tool_id):
        self.logger.debug('Ribbon tool changed to %s', tool_id)
        self._apply_ribbon_cursor()

    def _apply_ribbon_cursor(self):
        """Set the canvas's cursor to match
        the ribbon's active tool - besides the ribbon button's own
        highlighted (QToolButton:checked) style, this gives the active tool
        a distinct cursor too, since which mode is armed wasn't obvious
        enough from the ribbon alone. Also called (deferred - see the
        'draw_event' connection in init_widget) on every canvas redraw:
        NavigationToolbar2's _wait_cursor_for_draw_cm() wraps every
        canvas.draw() call and restores its own internally-tracked cursor
        afterward, which would otherwise silently undo this on the very
        next on_press/etc. redraw."""
        cursor = {'select_roi': Qt.CrossCursor}.get(self.ribbon.active_tool)
        self.canvas.setCursor(cursor if cursor is not None else Qt.ArrowCursor)

    def on_press(self, event):
        """Mouse-button-press handler: ax_mask ("ROI with Threshold") takes
        a plain left-click as "select this blob" whenever Blob Selection is
        armed (see _on_blob_mask_clicked) - checked first since it's a
        different axis/gesture entirely from everything below. Otherwise
        ax_nav only (no-ops for a click elsewhere on the shared canvas, see
        on_click_dp for ax_dp): with Ctrl held (or the ribbon's "Select
        ROI" tool active), start a new ROI rectangle at the click position.
        ROI-in-ROI is no longer a separate drag gesture - draw a plain ROI
        here, then set its Ref via the object list's own combo (see
        add_item_tree/_on_ref_changed)."""
        if event.inaxes == self.ax_mask:
            if event.xdata is not None and event.ydata is not None and event.button == 1:
                self._on_blob_mask_clicked(event)
            self.press = None
            return
        ribbon_tool = self.ribbon.active_tool
        if event.inaxes != self.ax_nav or (
                ribbon_tool != 'select_roi' and 'ctrl' not in event.modifiers):
            # Plain click/drag is reserved for the navigation toolbar's
            # Pan/Zoom tool (and the scroll-wheel zoom) so images can be
            # zoomed into; hold "ctrl" (or activate the ribbon's "Select
            # ROI" tool) to draw/edit a ROI instead.
            self.press = None
            return
        self.press = (event.xdata, event.ydata)
        if self.rect is not None:
            self.rect.remove()
        self.rect = patches.Rectangle(self.press, 0, 0, linewidth=1,
                                      edgecolor='r', facecolor='none')
        self.patches_axNav.append(self.rect)
        self.ax_nav.add_patch(self.rect)
        self.canvas.draw()
        self.backgrounds['nav'] = self.canvas.copy_from_bbox(self.ax_nav.bbox)

    def on_motion(self, event):
        """Mouse-motion handler: while a Ctrl+drag started by on_press is in
        progress, resize the in-progress ROI rectangle and blit it."""
        if self.press is None or event.inaxes is None:
            return
        if event.inaxes != self.ax_nav:
            return
        x0, y0 = self.press
        width = event.xdata - x0
        height = event.ydata - y0
        try:
            self.rect.set_width(width)
            self.rect.set_height(height)
            self.rect.set_xy((x0, y0))
        except AttributeError:
            self.press = None
        self.canvas.restore_region(self.backgrounds['nav'])
        self.ax_nav.draw_artist(self.rect)
        self.canvas.blit(self.ax_nav.bbox)

    def on_release(self, event):
        """Mouse-button-release handler: finalize the rectangle started by
        on_press into a new ROI row (left click, Ref defaults to "Nav" -
        see add_item_tree's Ref combo to make it a ROI-in-ROI afterward) or
        an additional init frame/box on the selected ROI (right click), and
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
        imgNo = self.slider_imgNo.value()
        ref = None

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

        self.rect = None
        if new_row:
            self.df_rois.loc[idx] = [1, init, [roi], len(self.nav_imgs),
                                                   ref, None, None, None, None, None, None, None]
            self.add_item_tree(idx=idx, init=init, end=None, ref=ref)

        else:
            self.df_rois.at[idx, 'init'] = init
            self.df_rois.at[idx, 'in_rois'].append(roi)
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
        event.canvas.draw_idle()

    def on_click_dp(self, event):
        """Ctrl+Click on the DP plot sets the reciprocal-space rings' center
        manually."""
        if (event.inaxes == self.ax_dp and event.button == 1 and event.xdata is not None
                and 'ctrl' in event.modifiers):
            self.dp_center = (event.xdata, event.ydata)
            self.update_scalebar('reciprocal')
#%%
    def add_item_tree(self, idx, init=[0], end=None, ref=None, use=1):
        """Add one column to tree_objects for ROI `idx`, with its end-frame
        spinbox, Blob checkbox and Duplicate/Delete buttons wired up."""
        cols = {col: i for i,col in enumerate(self.cols_tree)}
        item = self.tree_objects.addTopLevelItem()
        item.setCheckState(cols['use'], Qt.Checked if use else Qt.Unchecked)
        item.setText(cols['idx'], f"{idx}")
        item.setText(cols['init'], f"{init}")
        item.setCheckState(cols['blob'],
                            Qt.Checked if self._blob_enabled_for(idx) else Qt.Unchecked)

        # end frame
        spinbox = qtw.QSpinBox()
        spinbox.setRange(0, len(self.nav_imgs))
        spinbox.setValue(end if end is not None else len(self.nav_imgs))
        self.tree_objects.setItemWidget(item, cols['end'], spinbox)
        spinbox.valueChanged.connect(lambda value: self.on_spinboxEnd_changed(idx, value))
        # spinbox.valueChanged.connect(partial(self.on_spinbox_changed, item, idx))
        
        # ref - a combo (not plain text) so ROI-in-ROI is set directly here
        # instead of a separate mouse gesture (see on_press/on_release):
        # draw any ROI normally, then pick its reference ROI from this
        # combo. _refresh_ref_combos() (called below) populates its actual
        # options (every OTHER currently-listed ROI's index) and selects
        # `ref` - both need the full row list to exist first.
        combo_ref = qtw.QComboBox()
        combo_ref.setToolTip(
            "Reference ROI this one is defined relative to (\"ROI-in-ROI\") - "
            "its box is kept inside the reference's own current box automatically. "
            '"Nav" (default) means it isn\'t a ROI-in-ROI.')
        self.tree_objects.setItemWidget(item, cols['ref'], combo_ref)
        combo_ref.currentIndexChanged.connect(lambda *_: self._on_ref_changed(idx, combo_ref))

        # tracked
        cancel_icon = self.style().standardIcon(self.style().SP_DialogCancelButton)
        item.setIcon(cols['trk'], cancel_icon)
        item.setData(cols['trk'], Qt.UserRole, False)  # Store status boolean (False = not checked)
        
        # extracted
        item.setIcon(cols['ext'], cancel_icon)
        item.setData(cols['ext'], Qt.UserRole, False)  # Store status boolean (False = not checked)

        # quality - left blank (no icon at all) rather than a false-looking
        # "bad" cancel icon, until an actual quality check has run for this
        # ROI (see _refresh_quality_icon/_compute_tracking_quality) -
        # there's nothing to report yet before that.

        # duplicate
        duplicate_button = qtw.QPushButton('Dup')
        duplicate_button.setFixedSize(48, 30)
        duplicate_button.setToolTip('Duplicate this item (deep-copies everything) into a new one')

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
            self._quality_flags.pop(deleted_idx, None)
            self._refresh_ref_combos()
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

        # ref/ref combo depend on the full row list existing first
        # (excluding-self options, and this row's own initial selection).
        if ref is not None:
            self.df_rois.at[idx, 'ref'] = ref
        self._refresh_ref_combos()

    def on_item_check_changed(self, item):
        """tree_objects.itemChanged handler - unlike QTreeWidget's own
        itemChanged(item, column), a real QTableWidgetItem's own signal
        only carries the cell itself; its row/column give which property
        (self.cols_tree[item.row()]) and which ROI-column this cell
        belongs to."""
        row = item.row()
        if row >= len(self.cols_tree):
            return  # some other cell (e.g. the row-0 anchor) changed, not a checkbox row
        key = self.cols_tree[row]
        if key not in ('use', 'blob'):
            return
        idx_item = self.tree_objects.item(self.cols_tree.index('idx'), item.column())
        if idx_item is None or not idx_item.text():
            return
        idx = int(idx_item.text())
        checked = item.checkState() == Qt.Checked
        if key == 'use':
            self.df_rois.at[idx, 'use'] = 1 if checked else 0
        else:  # 'blob'
            was_enabled = self._blob_enabled_for(idx)
            self._set_blob_enabled(idx, checked)
            self.update_canvas()
            if checked and not was_enabled:
                # Fresh enable (not a re-sync from _open_blob_segmentation_
                # dialog's own signal-blocked checkbox update) - offer to
                # configure the segmentation method right away, same as
                # picking one from the "Blob Settings..." button. Leaves
                # _set_blob_enabled's own default (Connected Components, no
                # seed) in place if cancelled - see
                # _open_blob_segmentation_dialog's own docstring.
                self._open_blob_segmentation_dialog(idx)


    def on_spinboxEnd_changed(self, idx, value):
        self.df_rois.at[idx, 'end'] = value

    def _refresh_ref_combos(self):
        """Repopulate every row's Ref combo with every OTHER currently-
        listed ROI's index (plus "Nav") and reselect that row's own
        `df_rois.ref` - called whenever the object list itself changes
        (a ROI added/duplicated/deleted), since a row's own valid Ref
        choices depend on which other rows currently exist. Signals are
        blocked while rebuilding so this never itself triggers
        _on_ref_changed."""
        cols = {col: i for i, col in enumerate(self.cols_tree)}
        for i in range(self.tree_objects.topLevelItemCount()):
            item = self.tree_objects.topLevelItem(i)
            combo = self.tree_objects.itemWidget(item, cols['ref'])
            if combo is None:
                continue
            idx = int(item.text(cols['idx']))
            if idx not in self.df_rois.index:
                continue
            current_ref = self.df_rois.at[idx, 'ref']
            target = None if pd.isna(current_ref) else int(current_ref)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem('Nav', None)
            for other_idx in self.df_rois.index:
                if other_idx != idx:
                    combo.addItem(str(other_idx), other_idx)
            found = combo.findData(target)
            combo.setCurrentIndex(found if found >= 0 else 0)
            combo.blockSignals(False)

    def _roi_current_rect(self, idx, imgNo):
        """This ROI's own representative (y, x, w, h) rectangle "as of"
        frame `imgNo`, for Ref validation/clamping - out_rois[imgNo] if
        already tracked and defined there, else its most recently drawn
        in_rois entry (typically the box the user just finished drawing).
        None if this ROI has no geometry at all yet (shouldn't normally
        happen - every row gets at least one in_rois entry when created)."""
        out_rois = self.df_rois.at[idx, 'out_rois']
        if not np.all(pd.isna(out_rois)) and imgNo < len(out_rois):
            roi = out_rois[imgNo]
            if roi is not None and np.any(roi):
                return tuple(int(v) for v in roi)
        in_rois = self.df_rois.at[idx, 'in_rois']
        if in_rois:
            return tuple(int(v) for v in in_rois[-1])
        return None

    def _apply_roi_rect(self, idx, imgNo, rect):
        """Write a (possibly Ref-clamped) rectangle back into whichever
        source currently defines ROI `idx`'s geometry at `imgNo` - see
        _roi_current_rect, the same resolution order."""
        out_rois = self.df_rois.at[idx, 'out_rois']
        if not np.all(pd.isna(out_rois)) and imgNo < len(out_rois):
            out_rois[imgNo] = np.array(rect)
        else:
            in_rois = self.df_rois.at[idx, 'in_rois']
            if in_rois:
                in_rois[-1] = rect

    @staticmethod
    def _clamp_rect_to_reference(rect, ref_rect):
        """`rect` resized (if larger than `ref_rect`) and repositioned so it
        fits entirely inside `ref_rect` - both (y, x, w, h) tuples. A no-op
        if `rect` already fits."""
        x, y, w, h = rect
        xr, yr, wr, hr = ref_rect
        w = min(w, wr)
        h = min(h, hr)
        x = max(xr, min(x, xr + wr - w))
        y = max(yr, min(y, yr + hr - h))
        return (int(x), int(y), int(w), int(h))

    def _on_ref_changed(self, idx, combo):
        """Ref combo changed for ROI `idx`: clamp its current-frame
        rectangle to fit inside the newly chosen reference ROI's own
        current-frame rectangle (resizing/moving it if it doesn't already),
        then commit the new Ref - see _clamp_rect_to_reference. Rejected
        (combo reverted) if either ROI has no drawn geometry at all yet."""
        new_ref = combo.currentData()
        current_ref = self.df_rois.at[idx, 'ref']
        current_ref = None if pd.isna(current_ref) else int(current_ref)
        if new_ref == current_ref:
            return
        if new_ref is not None:
            imgNo = self.slider_imgNo.value()
            child_rect = self._roi_current_rect(idx, imgNo)
            ref_rect = self._roi_current_rect(new_ref, imgNo)
            if child_rect is None or ref_rect is None:
                qtw.QMessageBox.warning(self, 'No Geometry',
                    "Can't set this ROI's Ref - it or the reference ROI has no "
                    'drawn box yet.')
                self._refresh_ref_combos()  # revert the combo to its actual current value
                return
            clamped = self._clamp_rect_to_reference(child_rect, ref_rect)
            if clamped != child_rect:
                self._apply_roi_rect(idx, imgNo, clamped)
                self.logger.info(
                    'ROI %d resized/moved to fit inside reference ROI %d (%s -> %s).',
                    idx, new_ref, child_rect, clamped)
        self.df_rois.at[idx, 'ref'] = new_ref
        self.update_canvas()
        
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
    
    def _compute_tracking_quality(self, idx):
        """Per-frame mask-area quality check for ROI `idx`, run right after
        tracking (see get_tracking_results) - reuses the exact same
        threshold+apply_edge_mask pipeline "ROI with Threshold" itself
        shows live (Blob Selection, Dilate/Erode, Edge Detection, Mesh -
        whatever's currently configured for this ROI included), so the
        flagged frames reflect what would actually get extracted, not just
        the raw tracked box. See io.flag_anomalous_mask_areas for the
        flagging heuristic itself. Returns [] if this ROI isn't tracked at
        all yet."""
        out_rois = self.df_rois.at[idx, 'out_rois']
        if np.all(pd.isna(out_rois)):
            return []
        thresh_method = self.combo_thresh_method.currentText()
        thresh_offset = self.slider_thresh.value()
        areas = np.full(len(out_rois), np.nan)
        for frame_idx, roi in enumerate(out_rois):
            if roi is None or not np.any(roi):
                continue
            img_mask, _ = self.threshold_img(self.nav_imgs[frame_idx], roi, thresh_method,
                                             thresh_offset, idx=idx, frame_idx=frame_idx)
            areas[frame_idx] = img_mask.sum()
        return io.flag_anomalous_mask_areas(areas)

    def _refresh_quality_icon(self, idx):
        """Set ROI `idx`'s "Qlty" column icon from self._quality_flags -
        a warning icon if any frame is flagged, a plain check if none are
        (still distinguishable from the blank/no-icon-yet state a ROI
        starts in before its first quality check - see add_item_tree)."""
        row_index = self.df_rois.index.get_loc(idx)
        item = self.tree_objects.topLevelItem(row_index)
        if item is None:
            return
        col = self.cols_tree.index('qlty')
        flagged = self._quality_flags.get(idx) or []
        icon = self.style().standardIcon(
            self.style().SP_MessageBoxWarning if flagged else self.style().SP_DialogApplyButton)
        item.setIcon(col, icon)
        tip = (f'{len(flagged)} possibly mistracked frame(s): {", ".join(map(str, flagged[:20]))}'
              + (f' (+{len(flagged) - 20} more)' if len(flagged) > 20 else '')) if flagged \
            else 'No flagged frames'
        item.setToolTip(col, tip)

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
                                           'None', None, None, None, None, None, None, None]
            # self.df_rois.loc[idx] = [1, init, [roi], len(self.nav_imgs),
            #                                        ref, None, None, None]
            self.add_item_tree(idx=i+idx_max, init=[self.imgNo_autoDet], end=None, ref=None)
        # print(self.df_rois)
        self.update_canvas(self.imgNo_autoDet)
        self.canvas.draw()
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

        tracking_method = self.combo_trackMethod.currentText()
        # 'nano'/'dasiamrpn' need external .onnx weight files not bundled
        # with the app (see THIRD_PARTY_NOTICES.md) - checked/downloaded
        # once here for the whole batch, not per-ROI inside track_roi_cv2
        # itself (which runs on a background worker thread and can't show
        # Qt dialogs) - asks once instead of duplicating the prompt per ROI.
        if asset_fetch.tracker_models_available(tracking_method):
            self._start_tracking_workers(df, tracking_method)
        else:
            confirm_and_download(
                self, self.threadpool, 'Download Tracker Model',
                f'The "{tracking_method}" tracker needs model weight files, not bundled '
                'with the app. Download them now? An internet connection is needed.',
                asset_fetch.ensure_tracker_models,
                lambda _result: self._start_tracking_workers(df, tracking_method),
                self._on_tracker_download_failed,
                download_kwargs={'tracking_method': tracking_method})

    def _on_tracker_download_failed(self, error_msg):
        """confirm_and_download's on_failed for the tracker-model download -
        error_msg is empty if the user simply declined/cancelled, not a
        real failure."""
        if error_msg:
            self.logger.error('Tracker model download failed:\n%s', error_msg)
            qtw.QMessageBox.warning(self, 'Download Failed',
                f'Could not download the tracker model:\n{error_msg}')

    def _start_tracking_workers(self, df, tracking_method):
        """Launch tr.track_roi_cv2 for each ROI in `df` - the part of
        track_rois() that actually starts tracking, split out so it can run
        either immediately (models already cached) or after a confirmed
        on-demand download finishes."""
        self.load_spinner()
        self._cancelling = False
        self.button_cancel.setEnabled(True)
        self.tracking_counter = 0
        self.tracking_finished = False
        # tracking_method comes from the caller (not re-read from the combo
        # box here) - it must match whichever method's models were just
        # confirmed/downloaded, even if the user changed the dropdown while
        # that download was still running.

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
            # 'ref' is None for a plain (non-ROI-in-ROI) ROI - nothing to
            # translate against, so this whole block is skipped silently
            # rather than landing in the except below and logging a
            # spurious "translation failed" warning for every ordinary ROI
            # tracked (int(None) raised unconditionally here before this
            # guard existed).
            ref_value = df.loc[ind, 'ref']
            if pd.notna(ref_value):
                try:
                    ref = int(ref_value)
                    # Looked up from self.df_rois (not the use==1-filtered
                    # `df`), same as draw_rois_out's identical fallback - a
                    # reference ROI can be un-checked ("Use") after being
                    # tracked without invalidating its already-computed
                    # out_rois. Sliced to [beg:end] to align frame-for-frame
                    # with `imgs`/`rois_in` above, which are already shifted
                    # to start at this ROI's own `beg` - passing the
                    # reference's full, un-sliced out_rois here (as this
                    # code used to, via the broken `df.idx == ref` lookup
                    # below - `idx` was never a real column, df.index is)
                    # misaligned every frame and effectively always failed.
                    rois_ref = np.array(self.df_rois.loc[ref, 'out_rois'])[beg:end]
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
        # A re-track moves/resizes this ROI's crop every frame - any
        # cached Blob Selection auto-follow centroid (see
        # _resolve_blob_mask) was only ever valid relative to the OLD crop.
        self._blob_centroid_cache.pop(index, None)
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
                    # Both are already plain (N, 4) numpy arrays (see
                    # get_tracking_results' own np.zeros(...) assignment
                    # above) - .at[] gives that directly; the previous
                    # .to_numpy() calls here were left over from an earlier
                    # version where these were pandas Series, and had
                    # started raising (out_rois has no .to_numpy() of its
                    # own) once that stopped being true.
                    rois_ref = self.df_rois.at[ref, 'out_rois']
                    rois_pre = self.df_rois.at[idx, 'out_rois']
                    translated = tr.translate_roiInRoi(rois_pre, rois_ref, fwd=False)
                    # Each frame was tracked independently for this ROI and
                    # its reference, so the translated-back box can drift
                    # outside the reference's own current box on some
                    # frames even though it fit at assignment time (see
                    # _on_ref_changed) - clamp every frame the same way.
                    self.df_rois.at[idx, 'out_rois'] = [
                        self._clamp_rect_to_reference(
                            tuple(int(v) for v in roi), tuple(int(v) for v in rois_ref[i]))
                        for i, roi in enumerate(translated)]
                except Exception:
                    self.logger.warning(
                        'Re-translating ROI-in-ROI coordinates for ROI %s failed; '
                        'its out_rois were left untranslated.', idx, exc_info=True)
                            
            # Tracking-quality check (see _compute_tracking_quality) - every
            # just-tracked ROI ('use'==1, the same set track_rois() itself
            # tracked), run after the ROI-in-ROI re-translation above so it
            # checks each ROI's FINAL out_rois, not the pre-translation ones.
            flagged_summary = {}
            for idx in self.df_rois[self.df_rois['use'] == 1].index:
                flags = self._compute_tracking_quality(idx)
                self._quality_flags[idx] = flags
                self._refresh_quality_icon(idx)
                if flags:
                    flagged_summary[idx] = flags

            # self.slider_imgNo.setValue(0)
            # activating widgets
            self.slider_thresh.setEnabled(True)
            self.slider_thresh.setValue(100)
            self.disable_3ded_widgets(False)
            # self.checkbox_roiInRoi.setEnabled(True)
            item = self.tree_objects.topLevelItem(0)
            item.setSelected(True)
            self.update_canvas(0)
            self.canvas.draw()
            self.spinner.stop()
            self.button_cancel.setDisabled(True)

            duration = perf_counter() - self._track_tic
            self.logger.info(
                'CV2 tracking completed successfully for %d ROI(s) in %s.',
                self.tracking_counter_end, io.format_duration_hms(duration))

            if flagged_summary:
                lines = [f'  ROI {idx}: {len(flags)} frame(s) '
                        f'({", ".join(map(str, flags[:10]))}{", ..." if len(flags) > 10 else ""})'
                        for idx, flags in flagged_summary.items()]
                qtw.QMessageBox.warning(self, 'Tracking Quality Check',
                    f'{len(flagged_summary)} of {self.tracking_counter_end} ROI(s) have '
                    'possibly mistracked frames (an abrupt mask-area change, or the mask '
                    'vanishing entirely):\n\n' + '\n'.join(lines) +
                    '\n\nCheck them via the red marks on the frame-flag bar under the '
                    'slider (click one to jump there), or the "Qlty" column in the object '
                    'list.')

    def open_blob_settings_dialog(self):
        """"Blob Settings..." button: open _open_blob_segmentation_dialog
        for whichever ROI is currently selected - unlike the "Blob" column
        checkbox (which only opens the dialog on a fresh check), this is
        the way to revisit an already-configured ROI's segmentation method
        without unchecking/rechecking it first."""
        idx = self._selected_roi_idx()
        if idx is None:
            qtw.QMessageBox.warning(self, 'No ROI Selected',
                "Select a ROI in the list to configure its Blob Selection first.")
            return
        self._open_blob_segmentation_dialog(idx)

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
        blur_sigma = self.spinbox_blur.value()
        default_mask_stack = tr.create_masks(self.nav_imgs, out_rois, thresh_method,
                                             thresh_offset, blur_sigma)
        mask_stack = self.df_rois.at[idx, 'mask']
        if np.all(pd.isna(mask_stack)):
            mask_stack = default_mask_stack
        if self._blob_enabled_for(idx):
            # Blob Selection, like Dilate/Erode/Edge Detection/Mesh, is
            # never baked into the stored mask - it's applied fresh at
            # display/extraction time everywhere else (update_canvas,
            # extract_dp_current_frame, _draw_all_object_masks, all via
            # apply_edge_mask), and Fine-Tune Mask needs the same
            # treatment so it opens on just the selected blob rather than
            # the raw, possibly-multi-blob threshold result (previously
            # the whole ROI). _resolve_blob_mask is apply_edge_mask's own
            # blob-only first step - not the full apply_edge_mask, since
            # this dialog re-applies its own Dilate/Erode/Edge Detection/
            # Mesh live from the settings seeded below; applying the whole
            # pipeline here would double them. Order matters (auto-follow's
            # own per-frame centroid cache - see _resolve_blob_mask), so
            # both stacks are walked frame-by-frame in order;
            # default_mask_stack (the "Reset to Tracking" target) gets the
            # exact same treatment so resetting doesn't reintroduce the
            # un-restricted mask through a different path.
            mask_stack = mask_stack.copy()
            default_mask_stack = default_mask_stack.copy()
            for f in range(mask_stack.shape[0]):
                mask_stack[f] = self._resolve_blob_mask(idx, f, mask_stack[f])
            for f in range(default_mask_stack.shape[0]):
                default_mask_stack[f] = self._resolve_blob_mask(idx, f, default_mask_stack[f])
        edge_settings = self._edge_settings_for(idx)
        thresh_settings = {
            'method': thresh_method, 'offset_raw': self.slider_thresh.value(), 'blur': blur_sigma}
        mesh_settings = self._mesh_settings_for(idx)
        dilate_erode_settings = self._dilate_erode_settings_for(idx)
        # Contrast-only (no denoise) - MaskEditDialog applies its own
        # Denoise box fresh on top of this, seeded from box_contrast's own
        # current state below, so its preview starts out looking the same
        # as self.nav_imgs (which already has that denoise baked in)
        # without double-applying it - see MaskEditDialog's class docstring.
        bg_stack_contrast_only = io.convert_to_8bit(self.s, **self.box_contrast.get_kwargs()).data
        dialog = MaskEditDialog(self, mask_stack, bg_stack=bg_stack_contrast_only,
                                start_frame=self.slider_imgNo.value(), logger=self.logger,
                                default_mask_stack=default_mask_stack, edge_settings=edge_settings,
                                thresh_settings=thresh_settings, mesh_settings=mesh_settings,
                                dilate_erode_settings=dilate_erode_settings,
                                denoise_state=self.box_contrast.box_denoise.get_state(),
                                recompute_thresh_fn=lambda method, offset, blur:
                                    tr.create_masks(self.nav_imgs, out_rois, method, offset, blur))
        if dialog.exec_() == qtw.QDialog.Accepted:
            self.df_rois.at[idx, 'mask'] = dialog.get_mask_stack()
            # Mesh/Dilate-Erode/Edge Detection are all per-ROI (no main-tab
            # equivalent to sync against anymore - see _apply_dialog_settings_to_ui,
            # which now only has Threshold left to sync) - they round-trip
            # straight into this ROI's own columns instead.
            self.df_rois.at[idx, 'mesh'] = dialog.get_mesh_settings()
            self.df_rois.at[idx, 'dilate_erode'] = dialog.get_dilate_erode_settings()
            self.df_rois.at[idx, 'edge'] = dialog.get_edge_settings()
            self._apply_dialog_settings_to_ui(dialog)
            self.logger.info('Fine-tuned mask saved for ROI %d.', idx)
            self.update_canvas()

    def _apply_dialog_settings_to_ui(self, dialog):
        """Sync MaskEditDialog's Threshold box values back into this tab's
        own main controls on Save && Close, so whatever was left set there
        is what "Extract!"/the live preview use next, instead of silently
        reverting to whatever was set before the dialog was opened. Edge
        Detection/Dilate-Erode/Mesh have no main-tab equivalent anymore (all
        three are per-ROI, set only via the dialog's own Segments feature -
        see open_fine_tune_mask_dialog, which round-trips them directly)."""
        thresh = dialog.get_thresh_settings()
        if thresh is not None:
            self.combo_thresh_method.setCurrentText(thresh['method'])
            self.spinbox_blur.setValue(thresh['blur'])
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
        pattern = '*' if ext == 'All Files' else '*' + glob_ext_for_dtype(ext)
        return sorted(glob(os.path.join(path_4d, pattern)))

    def extract_3ded(self):
        """Handler for the Extract! button: build each enabled ROI's
        threshold mask stack, resolve the (smart-scan-aware) list of 4D
        signal files, then process one tracked object at a time - each
        object's own frames pooled together via a single batch-driver
        process (worker_extract_frame_batch.py, using the full configured
        worker count), the next object's batch only starting once the
        current one fully finishes (see _launch_next_object_batch) - per
        explicit request, rather than interleaving every object's frames
        into one shared queue."""
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

        dtype = resolve_hdf5_dtype(fns_4d[0], self.combo_dtype_4d.currentText())
        blur_sigma = self.spinbox_blur.value()
        thresh_method = self.combo_thresh_method.currentText()
        thresh_offset = self.slider_thresh.value() / 100
        for ind in self.df_rois[self.df_rois.use == 1].index:
            masks = tr.create_masks(
                self.nav_imgs, self.df_rois.loc[ind, 'out_rois'],
                thresh_method, thresh_offset, blur_sigma)
            if self._has_active_postprocessing(ind):
                # Applied per-frame - erode_mask_edge/mesh_restrict_mask are
                # single-2D-mask transforms, and this stack is (N frames, H, W).
                masks = np.stack([self.apply_edge_mask(m, ind, i) for i, m in enumerate(masks)])
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
        lengths = df.end - [min(df.init[idx]) for idx in df.index]
        self.tomo_counter_total = np.sum(lengths)
        self.update_progress_bar(0, self.tomo_counter_total)
        self.logger.info('Starting 3DED extraction for %d ROI(s), %d frame(s) total...',
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
            self.df_rois.at[idx, 'dp'] = np.zeros((len(self.nav_imgs), shape_d_x,
                                                   shape_d_y), dtype='uint32')
            out_rois = self.df_rois.loc[idx, 'out_rois']
            specs = []
            for i_fr, fn in enumerate(fns_4d):
                if out_rois[i_fr].any():
                    fn_pattern = fns_pattern_4d[i_fr] if fns_pattern_4d is not None else None
                    specs.append({
                        'i_index': i_fr, 'fn': fn,
                        'roi': [int(v) for v in out_rois[i_fr]],
                        'dtype': dtype, 'scanSize': list(scanSize),
                        'fn_pattern': fn_pattern, 'det_shape': [shape_d_x, shape_d_y],
                    })
            if specs:
                self._obj_queue.append(idx)
                self._obj_task_specs[idx] = specs

        self.n_workers = self.spinbox_threadNo.value()
        self._launch_next_object_batch()

    def _launch_next_object_batch(self):
        """Pop the next queued object and launch its 3DED extraction batch -
        one JSON task-list (worker_pool_utils.write_tasks_json), one driver
        process (worker_extract_frame_batch.py) running its own internal
        process pool over every one of that object's tracked frames at
        once. Only called again (by _handle_3ded_driver_finished) once the
        current object's batch has completely finished - if the queue is
        empty, every enabled object has been processed."""
        if not self._obj_queue:
            self._finalize_3ded_extraction()
            return
        idx = self._obj_queue.popleft()
        self._current_obj_idx = idx
        temp_dir = tempfile.mkdtemp(prefix='edyssey_3ded_')
        self._current_obj_temp_dir = temp_dir
        tasks = self._obj_task_specs.pop(idx)
        for task in tasks:
            mask_path = os.path.join(temp_dir, f"mask_f{task['i_index']}.npy")
            np.save(mask_path, self.df_rois.loc[idx, 'mask'][task['i_index']])
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
        process.readyReadStandardError.connect(lambda: self.handle_error(process))
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
            self.df_rois.loc[idx, 'dp'][i_fr] = np.load(fn_npy)
            os.remove(fn_npy)
        except Exception as e:
            self._3ded_failed = True
            self.logger.error('Failed to load DP for ROI %s frame %s: %s', idx, i_fr, e)
        self.tomo_counter += 1
        self.update_progress_bar(self.tomo_counter, self.tomo_counter_total)

    def _on_3ded_task_failed(self, i_fr, message):
        self._3ded_failed = True
        self.logger.error('3DED extraction failed for ROI %s frame %s: %s',
                          self._current_obj_idx, i_fr, message)
        self.tomo_counter += 1
        self.update_progress_bar(self.tomo_counter, self.tomo_counter_total)

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
        self.toggle_tree_icon(self.df_rois.index.get_loc(idx), 'ext', True)
        self._launch_next_object_batch()

    def _3ded_driver_failed_to_start(self, process, error):
        """Handle the current object's batch driver failing to start - with
        one driver process per object instead of one per (ROI, frame) task,
        this now simply means that one object's extraction never ran,
        rather than needing to individually track and skip past a single
        stuck task slot."""
        if self._cancelling:
            process.deleteLater()
            return
        self._3ded_failed = True
        self.logger.error('3DED extraction batch driver failed to start for ROI %s (error code %s).',
                          self._current_obj_idx, error)
        process.deleteLater()
        temp_dir = getattr(self, '_current_obj_temp_dir', None)
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        self.spinner.stop()
        qtw.QMessageBox.critical(self, 'Process Error',
            f'The 3DED extraction batch worker failed to start for ROI {self._current_obj_idx} '
            f'(error code {error}).\nCheck that Python is on PATH and '
            'worker_extract_frame_batch.py exists.')
        self.button_cancel.setDisabled(True)

    def _finalize_3ded_extraction(self):
        """Called once every enabled object's batch has been accounted for
        (successfully or not) - the tail end of what handle_finished used
        to do once tomo_counter reached tomo_counter_total, now triggered
        by the object queue emptying instead of a per-task counter."""
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
        self.update_scalebar('reciprocal')
        self.spinner.stop()
        self.button_cancel.setDisabled(True)
        if self.checkbox_autosave.isChecked():
            self.save_results()

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
        dtype = resolve_hdf5_dtype(fn, self.combo_dtype_4d.currentText())

        scanSize = self.get_scan_size()
        if scanSize is None:  # "Auto": fall back to the loaded nav signal's own shape
            scanSize = tuple(self.nav_imgs.shape[1:])

        thresh_method = self.combo_thresh_method.currentText()
        thresh_offset = self.slider_thresh.value() / 100
        blur_sigma = self.spinbox_blur.value()
        mask = tr.create_masks(self.nav_imgs[i_fr:i_fr + 1], out_rois[i_fr:i_fr + 1],
                               thresh_method, thresh_offset, blur_sigma)[0]
        mask = self.apply_edge_mask(mask, idx, i_fr)

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
        # Force the Clipping Thresholds to reset for this DP (see
        # update_ax's own reset=not self._dp_clip_initialized convention) -
        # normally left alone across a frame scrub so a manually-tuned
        # threshold persists through an already-extracted DP stack, but a
        # one-off current-frame check is a fresh, unrelated intensity range
        # (could be a different ROI/frame entirely) that the OLD thresholds
        # may not even overlap with (e.g. all-clipped-away or no visible
        # clipping at all) - without this, "the DP loads but the thresholds
        # don't update" is exactly what the user sees.
        self._dp_clip_initialized = False
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

    def handle_error(self, process):
        """Handler for QProcess.readyReadStandardError."""
        # worker_extract_frame.py loads tpx3 via eventem, whose progress bar
        # (and any other routine diagnostics) writes straight to stderr on
        # every run, success or failure - this is not itself an error (see
        # ProcessStderrBuffer). A worker that genuinely fails to produce
        # output is instead caught via a FAIL line - see
        # _handle_3ded_progress_line/_on_3ded_task_failed.
        if self._cancelling:
            return
        self._stderr_buffer.log_info(process, self.logger, 'Worker')

    def update_progress_bar(self, value, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f'%v / {total}')

    def reset_thresh(self):
        self.slider_thresh.setValue(100)
        self.update_canvas()

    def _on_threshold_control_changed(self, *_args):
        """Threshold Method/ROI Blur/Deviation changed: these all change
        what's actually IN every ROI's raw threshold mask, which any
        cached Blob Selection auto-follow centroid (see
        _resolve_blob_mask) was only ever valid relative to - drop the
        whole cache (every ROI, not just the selected one) rather than
        risk a stale trail silently picking the wrong blob under the new
        settings."""
        self._blob_centroid_cache.clear()
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

    def _open_pets2_dialog(self, uncheck_on_cancel=True):
        """Seed Pets2ParamsDialog from current metadata/settings (voltage,
        exposure, Å^-1/px scale, dp center) and open it. `uncheck_on_cancel`
        (False when opened via button_checkPets2Options, which can be
        clicked regardless of the checkbox's own state) unchecks 'Make
        *.pts2' again if the user cancels - only makes sense when this
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
            df['thresh'] = [('blur sigma', self.spinbox_blur.value()),
                            ('thresh method', self.combo_thresh_method.currentText()),
                            ('thresh offset', self.slider_thresh.value())]
            # Edge Detection/Dilate-Erode/Mesh are all per-ROI Segments (see
            # MaskEditDialog's class docstring/get_edge_settings()/
            # get_dilate_erode_settings()/get_mesh_settings()), and Blob
            # Selection (see _blob_settings_for/_split_blob_segment) is
            # Segments-shaped the same way despite living on the main tab
            # instead - each is a `{'segments': [...]}` dict already fully
            # describing exactly what was used for every frame range of
            # this ROI's extraction, so it's recorded here as-is rather
            # than flattened to one set of values.
            df['edge_detection'] = self._edge_settings_for(idx) or {'segments': []}
            df['mesh'] = self._mesh_settings_for(idx) or {'segments': []}
            df['dilate_erode'] = self._dilate_erode_settings_for(idx) or {'segments': []}
            df['blob'] = self._blob_settings_for(idx) or {'segments': []}
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

            # For a ROI-in-ROI object, also draw its reference/parent ROI's
            # own tracked box in the exported clip (see
            # io.create_clip_tracking's ref_rois docstring) - not just this
            # object's own inner box in isolation.
            ref = self.df_rois.loc[idx, 'ref']
            ref_rois = None
            if pd.notna(ref):
                ref_idx = int(ref)
                if ref_idx in self.df_rois.index:
                    ref_rois = self.df_rois.loc[ref_idx, 'out_rois']

            fn = os.path.join(path_save_roi, 'tracking clip')
            worker_clip_tr_ref = WorkerThread_General(
                io.create_clip_tracking, 0, fn, self.nav_imgs,
                self.df_rois.loc[idx, 'out_rois'], ref_rois=ref_rois, scale=scale_real,
                fps=self.spinbox_fps.value(), logger=self.logger)
            self.threadpool.start(worker_clip_tr_ref)
            
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
        """Stop tracking or 3DED extraction and suppress the error popups
        that killing those workers would otherwise trigger.

        QThreadPool has no way to forcibly interrupt a runnable that has
        already started (only queued-but-not-started ones can be dropped),
        so an in-flight tracking job for a single ROI will still finish in
        the background - its result is applied harmlessly since it's still
        valid data, just not launched further. The current object's 3DED
        batch driver is a real OS process tree though (see
        kill_3ded_driver), so that's actually killed outright, and no
        further queued objects are launched."""
        self._cancelling = True
        self.threadpool.clear()
        process = getattr(self, '_3ded_driver_process', None)
        n_killed = 1 if (process is not None and process.state() != QProcess.NotRunning) else 0
        self.kill_3ded_driver()
        if hasattr(self, '_obj_queue'):
            self._obj_queue.clear()
        if hasattr(self, 'spinner'):
            self.spinner.stop()
        self.button_cancel.setDisabled(True)
        self.logger.warning(
            'Cancelled by user (%d/%d frame(s) already processed, %d running batch process killed).',
            getattr(self, 'tomo_counter', 0), getattr(self, 'tomo_counter_total', 0), n_killed)
        qtw.QMessageBox.information(self, 'Cancelled',
            'Tracking/3DED extraction was cancelled.\n\n'
            'Any tracking job already running for a single ROI will still '
            'finish in the background (its result is kept) - only queued '
            'work and the running 3DED extraction batch were stopped.')

    def get_duplicate_state(self):
        """Snapshot of this tab's in-progress analysis, for "Duplicate
        Current Tab" (see EDyssey_MainWindow.duplicate_current_tab) - a
        synchronous, in-memory equivalent of Save Results/Load Saved
        Analysis (_on_saved_analysis_loaded), just enough to make the
        duplicate tab look and behave like this one immediately. Every
        mutable value (arrays, the ROI dataframe, dicts) is copied, never
        shared by reference, so the two tabs stay fully independent
        afterward - df_rois specifically is copied column-by-column (not
        deepcopy(DataFrame), which doesn't deep-copy object-dtype cell
        contents - the same pandas gotcha the tree's own "Dup" button
        already works around, see add_item_tree's duplicate_row()).

        Returns None if no navigation signal has been loaded yet (checked
        via self.nav_imgs, set only once initiate_processing() has run)."""
        if not isinstance(getattr(self, 'nav_imgs', None), np.ndarray) or len(self.nav_imgs) == 0:
            return None
        df_rois_rows = []
        for idx in self.df_rois.index:
            row = {col: deepcopy(self.df_rois.at[idx, col]) for col in self.cols_df}
            row['idx'] = idx
            df_rois_rows.append(row)
        return {
            # File/scan parameters
            'lineEdit_dir_navSignal': self.lineEdit_dir_navSignal.text(),
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
            'nav_4d_files': list(getattr(self, '_nav_4d_files', []) or []) or None,
            'nav_4d_directory': getattr(self, '_nav_4d_directory', None),
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
            's': self.s.deepcopy() if hasattr(self, 's') else None,
            's_8bit': self.s_8bit.deepcopy() if hasattr(self, 's_8bit') else None,
            'nav_imgs_raw': (self.nav_imgs_raw.copy() if hasattr(self, 'nav_imgs_raw') else None),
            'nav_imgs': self.nav_imgs.copy(),
            'imgNo': self.slider_imgNo.value(),
            # Contrast
            'contrast': self.box_contrast.get_state(),
            'clip_dp': self.clip_dp.get_state(),
            # Threshold / tracking settings - Edge Detection/Dilate-Erode/
            # Mesh have no main-tab widgets to copy anymore (all per-ROI
            # Segments, already riding along inside df_rois_rows below).
            'combo_thresh_method': self.combo_thresh_method.currentText(),
            'spinbox_blur': self.spinbox_blur.value(),
            'slider_thresh': self.slider_thresh.value(),
            'combo_trackMethod': self.combo_trackMethod.currentText(),
            'spinbox_threadNo': self.spinbox_threadNo.value(),
            'spinbox_fps': self.spinbox_fps.value(),
            'checkbox_autosave': self.checkbox_autosave.isChecked(),
            'checkbox_makePets2': self.checkbox_makePets2.isChecked(),
            'pets2_params': deepcopy(self.pets2_params),
            # Tracked/segmented objects
            'df_rois_rows': df_rois_rows,
        }

    def apply_duplicate_state(self, state):
        """Restore a dict from get_duplicate_state() into this (freshly
        constructed, otherwise-empty) tab, and redraw everything it
        touches so the tab looks right immediately - see that method's
        docstring. No-op on None/empty.

        Several restored widgets (checkbox_makePets2,
        combo_thresh_method/spinbox_blur/slider_thresh/edge-detection
        controls) are wired to handlers that open a dialog, recompute
        something from scratch, or just redraw - all skippable here since
        the already-computed/copied results are applied directly, so
        those widgets are set with signals blocked."""
        if not state:
            return
        self.lineEdit_dir_navSignal.setText(state['lineEdit_dir_navSignal'])
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

        # Navigation signal + images (mirrors the tail of initiate_processing(),
        # minus the parts that would recompute s_8bit/nav_imgs from scratch)
        if state['s'] is not None:
            self.s = state['s']
        if state['s_8bit'] is not None:
            self.s_8bit = state['s_8bit']
        if state['nav_imgs_raw'] is not None:
            self.nav_imgs_raw = state['nav_imgs_raw']
        self.nav_imgs = state['nav_imgs']
        self._dp_center_cache_key = None
        self._ax_mask_shape_seen = None
        self.box_contrast.set_state(state['contrast'])

        shape_x, shape_y = self.nav_imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['nav'].set_clim(vmin=self.nav_imgs.min(), vmax=self.nav_imgs.max())
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.nav_imgs)))
        self.ax_nav.set_xlim(0, shape_y)
        self.ax_nav.set_ylim(shape_x, 0)
        self.toolbar.update()
        self.toolbar.push_current()
        self.slider_imgNo.setRange(0, len(self.nav_imgs) - 1)
        self.frame_flag_bar.set_range(len(self.nav_imgs))
        # Not recomputed here (see _compute_tracking_quality) - a restored
        # ROI just starts back at "not yet quality-checked" (the same
        # blank-icon state a freshly-tracked one is in before its own
        # check runs), same spirit as this being a derived diagnostic
        # rather than persisted state (see its own init comment).
        self._quality_flags = {}
        self.button_reset_rois.setEnabled(True)
        self.button_track.setEnabled(True)
        self.button_fineTuneMask.setEnabled(True)
        self.button_blobSettings.setEnabled(True)

        # Threshold / tracking settings - signals blocked so setting them
        # doesn't trigger a redundant re-blur/dialog/redraw (see docstring);
        # the already-copied nav_imgs/df_rois already reflect these
        # settings' effect. Edge Detection/Dilate-Erode/Mesh ride along
        # inside df_rois_rows below (per-ROI Segments, no main-tab widgets
        # to restore here).
        for wid, value, setter in (
            (self.combo_thresh_method, state['combo_thresh_method'], 'setCurrentText'),
            (self.spinbox_blur, state['spinbox_blur'], 'setValue'),
            (self.slider_thresh, state['slider_thresh'], 'setValue'),
            (self.combo_trackMethod, state['combo_trackMethod'], 'setCurrentText'),
            (self.spinbox_threadNo, state['spinbox_threadNo'], 'setValue'),
            (self.spinbox_fps, state['spinbox_fps'], 'setValue'),
            (self.checkbox_autosave, state['checkbox_autosave'], 'setChecked'),
            (self.checkbox_makePets2, state['checkbox_makePets2'], 'setChecked'),
        ):
            wid.blockSignals(True)
            getattr(wid, setter)(value)
            wid.blockSignals(False)
        self.set_threadNo(state['spinbox_threadNo'])
        self.pets2_params = deepcopy(state['pets2_params'])

        # Tracked/segmented objects - reconstructs the tree the same way
        # _on_saved_analysis_loaded() does from a loaded dataframe.
        any_tracked = False
        for row in state['df_rois_rows']:
            idx = row['idx']
            self.df_rois.loc[idx] = [row[col] for col in self.cols_df]
            self.add_item_tree(idx, row['init'], row['end'], row['ref'], row['use'])
            row_index = self.df_rois.index.get_loc(idx)
            if row['out_rois'] is not None:
                self.toggle_tree_icon(row_index, 'trk', True)
                any_tracked = True
            if row['dp'] is not None:
                self.toggle_tree_icon(row_index, 'ext', True)
        self.disable_3ded_widgets(not any_tracked)
        if any_tracked:
            self.slider_thresh.setEnabled(True)

        self.clip_dp.set_state(state['clip_dp'])
        self._dp_clip_initialized = True
        self._bg = None
        self.slider_imgNo.blockSignals(True)
        self.slider_imgNo.setValue(state['imgNo'])
        self.slider_imgNo.blockSignals(False)
        self.update_canvas(state['imgNo'])
        self.canvas.draw()
        self.update_scalebar('real')
        self.update_scalebar('reciprocal')

    def cleanup(self):
        """Release resources held by this tab. Called by MainWindow.closeEvent
        so repeated runs of the app in the same console/kernel don't leave
        threadpools, running subprocesses, and matplotlib figures alive."""
        self.threadpool.clear()
        self.kill_3ded_driver()
        self.log_console.disconnect_log()
        plt.close(self.figure)

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

