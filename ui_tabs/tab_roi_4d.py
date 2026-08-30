# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:34:09 2024

@author: SGholam
"""



import os
import json
import pickle
import tempfile
import shutil
from time import perf_counter
from PyQt5.QtCore import (pyqtSignal, Qt, QRunnable, QObject, QProcess, QTimer)
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QDoubleValidator, QKeySequence
from PyQt5.QtWidgets import QShortcut
from matplotlib.colors import SymLogNorm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import EDyssey.io_utils as io
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as patches
from dask.diagnostics import ProgressBar
from .logging_utils import get_tab_logger, LogConsole
from .base_tab import TabBase
from .threshold_dialog import ThresholdDialog
from .worker_thread import ProcessStderrBuffer, WorkerThread_General
from .worker_launch import worker_command
from .ribbon import RibbonPanel, RibbonTool
from .clipping_thresholds import ClippingThresholdsWidget
from worker_extract_frame import load_dp
# import matplotlib.gridspec as gridspec
# from skimage.filters import threshold_otsu, threshold_li, threshold_mean, threshold_yen
# from skimage import exposure
#%% wdiget
class Tab_ROI_on_4D(TabBase):
    def __init__(self, parent=None):
        super().__init__('Tab_ROI_on_4D', parent)
        self._stderr_buffer = ProcessStderrBuffer()
        self.init_widget()
        
    def init_widget(self):
        self.central_widget = qtw.QWidget(self)
        self.layout = qtw.QVBoxLayout(self)
        self.setLayout(self.layout)
        
        button_w = 100
        button_h = 25
        #%% ribbon (top parameter ribbon, Word-style)
        ribbon_page = qtw.QWidget()
        self.ribbon_page = ribbon_page  # exposed for the Edit tab's ribbon-text-scale control
        layout_ribbon = qtw.QHBoxLayout(ribbon_page)
        layout_ribbon.setContentsMargins(4, 2, 4, 2)
        layout_ribbon.setSpacing(2)
        self.layout.addWidget(ribbon_page)
        #%% Experiment Info (ribbon group)
        # stretch=0 (+ Preferred override below): this column must stay at
        # its natural content width when the main window is resized, not
        # grow to fill the extra space - only the ribbon's own trailing
        # addStretch(1) (see end of init_widget) should absorb that.
        self.box_dir, layout_exp = self._ribbon_group_start(layout_ribbon, stretch=0)
        # self.box_dir.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Preferred)

        layout_file_entry = qtw.QHBoxLayout()
        label_4dSignal = qtw.QLabel('4D Signal')
        label_4dSignal.setFixedWidth(60)
        layout_file_entry.addWidget(label_4dSignal)
        self.lineEdit_dir_signal = qtw.QLineEdit()
        self.lineEdit_dir_signal.setFixedWidth(340)
        layout_file_entry.addWidget(self.lineEdit_dir_signal)
        self.lineEdit_dir_signal.textChanged.connect(lambda:self.enable_dwellTime_spinbox(
            self.lineEdit_dir_signal.text()))
        self.lineEdit_dir_signal.textChanged.connect(self.refresh_file_list)

        self.button_dir_navSignal = qtw.QPushButton('...')
        self.button_dir_navSignal.setFixedWidth(30)
        layout_file_entry.addWidget(self.button_dir_navSignal)
        self.button_dir_navSignal.clicked.connect(self.show_dialog)
        layout_exp.addLayout(layout_file_entry)

        # Smart-scan (pattern-file) acquisition support - lives here in
        # File rather than its own group, since it's really just another
        # attribute of the 4D signal being pointed at above.
        layout_smart = qtw.QHBoxLayout()
        # Dwell Time (row 2 of the same grid)
        label_dwell = qtw.QLabel('Dwell (\u03BCs)')
        label_dwell.setFixedWidth(60)
        layout_smart.addWidget(label_dwell)
        self.spinbox_dwellTime = qtw.QSpinBox()
        self.spinbox_dwellTime.setFixedWidth(60)
        self.spinbox_dwellTime.setRange(1, 99999999)
        self.spinbox_dwellTime.setDisabled(True)
        self.spinbox_dwellTime.setToolTip('Dwell time in microseconds')
        layout_smart.addWidget(self.spinbox_dwellTime)
        self.checkbox_smartScan = qtw.QCheckBox('Smart Scanned')
        self.checkbox_smartScan.setToolTip(
            'Smart-scanned (sparsely acquired) file - needs a pattern file to reshape')
        layout_smart.addWidget(self.checkbox_smartScan)
        self.checkbox_smartScan.stateChanged.connect(self.activate_lineEdit_patternFile)

        self.lineEdit_patternFile = qtw.QLineEdit()
        self.lineEdit_patternFile.setPlaceholderText('Pattern file...')
        self.lineEdit_patternFile.setDisabled(True)
        self.lineEdit_patternFile.setMaximumWidth(160)
        layout_smart.addWidget(self.lineEdit_patternFile)

        self.button_browsePattern = qtw.QPushButton('...')
        self.button_browsePattern.setFixedWidth(30)
        self.button_browsePattern.setDisabled(True)
        layout_smart.addWidget(self.button_browsePattern)
        self.button_browsePattern.clicked.connect(self.browse_pattern_file)
        layout_exp.addLayout(layout_smart)

        # Detector Size / Scan Size / Metadata / Scales each get their own
        # QGroupBox (a real bordered+titled box - unlike the ribbon
        # column's own borderless, bottom-captioned style) so the 4
        # concepts read as visually distinct groups, arranged 2x2: Detector
        # Size beside Scan Size, Metadata beside Scales below them.
        layout_exp_groups = qtw.QGridLayout()
        spin_w_size = 55  # detector/scan size spinboxes only hold 3-4 digits - button_w (100) is overkill

        #### Detector Size
        self.groupbox_detectorSize = qtw.QGroupBox('Detector Size')
        layout_detSize = qtw.QHBoxLayout(self.groupbox_detectorSize)
        detSize_tooltip = 'Detector (diffraction pattern) size in pixels - Auto assumes 512x512.'
        self.checkbox_detectorSizeAuto = qtw.QCheckBox('Auto')
        self.checkbox_detectorSizeAuto.setChecked(True)
        self.checkbox_detectorSizeAuto.setToolTip(detSize_tooltip)
        layout_detSize.addWidget(self.checkbox_detectorSizeAuto)
        self.spinbox_detectorSize_x = qtw.QSpinBox()
        self.spinbox_detectorSize_x.setRange(1, 10000)
        self.spinbox_detectorSize_x.setValue(512)
        self.spinbox_detectorSize_x.setFixedWidth(spin_w_size)
        layout_detSize.addWidget(self.spinbox_detectorSize_x)
        layout_detSize.addWidget(qtw.QLabel('×', alignment=Qt.AlignCenter))
        self.spinbox_detectorSize_y = qtw.QSpinBox()
        self.spinbox_detectorSize_y.setRange(1, 10000)
        self.spinbox_detectorSize_y.setValue(512)
        self.spinbox_detectorSize_y.setFixedWidth(spin_w_size)
        layout_detSize.addWidget(self.spinbox_detectorSize_y)
        layout_exp_groups.addWidget(self.groupbox_detectorSize, 0, 0)
        self.activate_detectorSize_spinboxes()
        self.checkbox_detectorSizeAuto.stateChanged.connect(self.activate_detectorSize_spinboxes)

        #### Scan Size
        self.groupbox_scanSize = qtw.QGroupBox('Scan Size')
        layout_scanSize = qtw.QHBoxLayout(self.groupbox_scanSize)
        self.checkbox_scanSize = qtw.QCheckBox('Auto')
        self.checkbox_scanSize.setChecked(True)
        layout_scanSize.addWidget(self.checkbox_scanSize)
        self.spinbox_scanSize_x = qtw.QSpinBox()
        self.spinbox_scanSize_x.setFixedWidth(spin_w_size)
        self.spinbox_scanSize_x.setRange(1, 10000)
        layout_scanSize.addWidget(self.spinbox_scanSize_x)
        layout_scanSize.addWidget(qtw.QLabel('×', alignment=Qt.AlignCenter))
        self.spinbox_scanSize_y = qtw.QSpinBox()
        self.spinbox_scanSize_y.setFixedWidth(spin_w_size)
        self.spinbox_scanSize_y.setRange(1, 10000)
        layout_scanSize.addWidget(self.spinbox_scanSize_y)
        layout_exp_groups.addWidget(self.groupbox_scanSize, 0, 1)
        self.activate_lineEdit_scanSize()
        self.checkbox_scanSize.stateChanged.connect(self.activate_lineEdit_scanSize)

        self.double_validator = QDoubleValidator(0.0, 1e5, 5)
        self.dp_center = None  # (x, y) - auto-found or last manually-clicked center
        # id(self.dp) at the time dp_center was last auto-found - lets
        # update_canvas() skip re-running find_dp_center_blurred (a real
        # HyperSpy call) when the DP hasn't actually changed since, e.g. on
        # every tick of a contrast-slider drag. See _on_auto_center_toggled.
        self._dp_center_cache_key = None

        #### Metadata (comment.txt) auto-fill - tpx3 acquisitions log scan
        # size/dwell time there, alongside the .tpx3 file itself.
        self.groupbox_metadata = qtw.QGroupBox('Metadata')
        layout_metadata = qtw.QVBoxLayout(self.groupbox_metadata)
        layout_metadata_1 = qtw.QHBoxLayout()
        layout_metadata.addLayout(layout_metadata_1)
        self.button_browseMetadata = qtw.QPushButton('...')
        self.button_browseMetadata.setFixedWidth(30)
        self.button_browseMetadata.setToolTip('Browse for the metadata file')
        layout_metadata_1.addWidget(self.button_browseMetadata)
        self.button_browseMetadata.clicked.connect(self.browse_metadata_file)

        self.button_loadMetadata = qtw.QPushButton('Load')
        # self.button_loadMetadata.setFixedSize(button_w//2, button_h)
        self.button_loadMetadata.setToolTip(
            'Fill Scan Size/Dwell Time from comment.txt next to the signal (tpx3 only) - '
            '"key: value" lines, e.g. "scan size x: 128". Multiple acquisitions? '
            'Pick the block with Block # (right).')
        layout_metadata_1.addWidget(self.button_loadMetadata)
        self.button_loadMetadata.clicked.connect(lambda: self.load_metadata(silent=False))

        self.button_viewMetadata = qtw.QPushButton('View')
        # self.button_viewMetadata.setFixedSize(button_w//2, button_h)
        self.button_viewMetadata.setToolTip('View the full raw comment.txt content')
        layout_metadata_1.addWidget(self.button_viewMetadata)
        self.button_viewMetadata.clicked.connect(self.show_metadata_dialog)

        # Block # sits on its own row (rather than beside Browse/Load/View)
        # so this box stays narrow instead of widening the whole column.
        layout_metadata_2 = qtw.QHBoxLayout()
        layout_metadata.addLayout(layout_metadata_2)
        label_block = qtw.QLabel('Block')
        label_block.setFixedWidth(30)
        layout_metadata_2.addWidget(label_block)
        self.spinbox_metadataCount = qtw.QSpinBox()
        self.spinbox_metadataCount.setFixedWidth(50)
        self.spinbox_metadataCount.setRange(0, 99999)
        self.spinbox_metadataCount.setValue(0)
        self.spinbox_metadataCount.setDisabled(True)  # re-enabled once >1 block is found
        self.spinbox_metadataCount.setToolTip(
            'Which metadata block to read (enabled if comment.txt logs more than one)')
        # Re-reads comment.txt (a cheap text-file parse, not the 4D data
        # file itself) for the newly-selected block as soon as the value
        # changes, instead of requiring an extra "Load Metadata" click every time.
        self.spinbox_metadataCount.valueChanged.connect(lambda: self.load_metadata(silent=True))
        layout_metadata_2.addWidget(self.spinbox_metadataCount, alignment=Qt.AlignLeft)
        layout_exp_groups.addWidget(self.groupbox_metadata, 1, 0)

        self.metadata_path_override = None  # set by browse_metadata_file(); cleared on new 4D signal

        #### Scales - Real (row 0), then Recip. + "Center" (row 1) - two
        # rows rather than one wide row, so this box stays narrow instead
        # of widening the whole column (same reasoning as Metadata above).
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
        # self.button_centerRecip.setFixedWidth(button_w//2)
        self.button_centerRecip.setToolTip(
            'Find the beam center now, or Ctrl+Click the DP to set it manually')
        layout_scales.addWidget(self.button_centerRecip, 1, 2)
        layout_exp_groups.addWidget(self.groupbox_scales, 1, 1)
        self.button_centerRecip.clicked.connect(self.find_and_center_recip)
        self.lineEdit_scale_recip.textChanged.connect(
            lambda: self.update_canvas(ax='dp') if hasattr(self, 'dp') else None)
        self.lineEdit_scale_real.textChanged.connect(
            lambda: self.update_canvas(ax='dp') if hasattr(self, 'dp') else None)

        layout_exp.addLayout(layout_exp_groups)
        self._ribbon_group_end(layout_ribbon, layout_exp, 'Experiment Info')
        #%% Virtual Imaging (ribbon group)
        # Kept early (right after Experiment Info, where "Load Signal" used
        # to sit) since "Compute Virtual Image" below is now this tab's
        # only way to load a signal in the first place - Edge Detection/
        # SAM2 Segmentation/Summed DP Threshold are all downstream,
        # later-stage steps that need something loaded first.
        self.box_virtualImaging, layout_virtualImaging = self._ribbon_group_start(layout_ribbon, stretch=0)
        self.box_virtualImaging.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Preferred)

        self.checkbox_useVirtualMask = qtw.QCheckBox('Use Virtual Mask')
        self.checkbox_useVirtualMask.setToolTip(
            'Use only the annular region(s) below, instead of the whole detector')

        # Center and Radius share one QGridLayout (rather than two
        # independent QHBoxLayouts) so their cells land in exactly the same
        # columns - X directly above In, Y directly above Out - instead of
        # merely hoping two separate row layouts happen to line up.
        grid_vi_geometry = qtw.QGridLayout()
        grid_vi_geometry.setContentsMargins(0, 0, 0, 0)

        # grid_vi_geometry.addWidget(qtw.QLabel('Center'), 0, 0)
        grid_vi_geometry.addWidget(qtw.QLabel('X'), 0, 1)
        self.spinbox_vi_centerX = qtw.QSpinBox()
        self.spinbox_vi_centerX.setMaximumWidth(70)
        self.spinbox_vi_centerX.setRange(0, 8192)
        self.spinbox_vi_centerX.setValue(256)
        grid_vi_geometry.addWidget(self.spinbox_vi_centerX, 0, 2)
        grid_vi_geometry.addWidget(qtw.QLabel('Y'), 0, 3)
        self.spinbox_vi_centerY = qtw.QSpinBox()
        self.spinbox_vi_centerY.setMaximumWidth(70)
        self.spinbox_vi_centerY.setRange(0, 8192)
        self.spinbox_vi_centerY.setValue(256)
        grid_vi_geometry.addWidget(self.spinbox_vi_centerY, 0, 4)
        self.button_vi_autoCenter = qtw.QPushButton('Center')
        self.button_vi_autoCenter.setToolTip(
            'Find the beam center now and copy it into Center X/Y, or Ctrl+drag '
            'the mask\'s "+"/circle edge on the DP to move/resize it manually')
        self.button_vi_autoCenter.clicked.connect(self.vi_auto_center)
        grid_vi_geometry.addWidget(self.button_vi_autoCenter, 0, 0)

        grid_vi_geometry.addWidget(qtw.QLabel('Radius'), 1, 0)
        grid_vi_geometry.addWidget(qtw.QLabel('In'), 1, 1)
        self.spinbox_vi_rIn = qtw.QSpinBox()
        self.spinbox_vi_rIn.setMaximumWidth(70)
        self.spinbox_vi_rIn.setRange(0, 4096)
        self.spinbox_vi_rIn.setSingleStep(10)
        grid_vi_geometry.addWidget(self.spinbox_vi_rIn, 1, 2)
        grid_vi_geometry.addWidget(qtw.QLabel('Out'), 1, 3)
        self.spinbox_vi_rOut = qtw.QSpinBox()
        self.spinbox_vi_rOut.setMaximumWidth(70)
        self.spinbox_vi_rOut.setRange(1, 4096)
        self.spinbox_vi_rOut.setValue(256)
        self.spinbox_vi_rOut.setSingleStep(10)
        grid_vi_geometry.addWidget(self.spinbox_vi_rOut, 1, 4)

        for sb in (self.spinbox_vi_centerX, self.spinbox_vi_centerY,
                   self.spinbox_vi_rIn, self.spinbox_vi_rOut):
            sb.setToolTip('Up/down arrows step by 10; type a value directly for finer control')
            sb.valueChanged.connect(self._sync_active_detector_vi)

        # Every virtual detector - including the very first, spinbox-defined
        # one - always has an entry in the list below (matches the Navigator
        # tab's identical list_detectors behavior), combined together into a
        # single virtual image at calculation time (OR-combined mask for
        # every format except .tpx3, natively summed by eventem there) - see
        # get_active_virtual_detectors/io.calculate_nav_img_masked.
        # self._active_detector_row_vi tracks which entry the center/radii
        # spinboxes currently drive live (see _sync_active_detector_vi/
        # _load_selected_detector_vi) - "Add Detector" freezes the current
        # one and starts a new active entry instead of editing in place.
        self._extra_detectors_vi = [{'center': (self.spinbox_vi_centerX.value(),
                                                 self.spinbox_vi_centerY.value()),
                                     'r_in': self.spinbox_vi_rIn.value(),
                                     'r_out': self.spinbox_vi_rOut.value()}]
        self._active_detector_row_vi = 0
        widget_vi_detectors = qtw.QWidget()
        layout_vi_detectors = qtw.QVBoxLayout(widget_vi_detectors)
        layout_vi_detectors.setContentsMargins(0, 0, 0, 0)
        layout_vi_detectors.addWidget(qtw.QLabel('Detectors'))
        layout_vi_list_buttons = qtw.QHBoxLayout()
        layout_vi_detectors.addLayout(layout_vi_list_buttons)
        self.button_vi_addDetector = qtw.QPushButton('Add Detector')
        self.button_vi_addDetector.setToolTip(
            'Add the center/radii above as a new virtual detector - all listed '
            'detectors are combined into the virtual image')
        layout_vi_list_buttons.addWidget(self.button_vi_addDetector)
        self.button_vi_addDetector.clicked.connect(self.add_extra_detector_vi)
        self.button_vi_removeDetector = qtw.QPushButton('Remove Selected')
        self.button_vi_removeDetector.setToolTip(
            'Remove the selected detector - at least one always remains listed')
        layout_vi_list_buttons.addWidget(self.button_vi_removeDetector)
        self.button_vi_removeDetector.clicked.connect(self.remove_extra_detector_vi)
        layout_vi_list_buttons.addStretch(1)

        self.list_detectors_vi = qtw.QListWidget()
        self.list_detectors_vi.setToolTip(
            'All virtual detectors in play - select one to edit via the spinboxes above')
        self.list_detectors_vi.setMaximumWidth(220)
        # Capped rather than stretched to fill the ribbon row - the row's
        # actual height can be taller than this column needs (set by other,
        # unrelated ribbon columns), and an unstretched list would otherwise
        # balloon to match that instead of the "Use Virtual Mask"
        # checkbox+grid's own natural height on the left.
        self.list_detectors_vi.setMaximumHeight(80)
        layout_vi_detectors.addWidget(self.list_detectors_vi)
        self.list_detectors_vi.addItem(self._format_detector_vi(self._extra_detectors_vi[0]))
        self.list_detectors_vi.setCurrentRow(0)
        self.list_detectors_vi.itemSelectionChanged.connect(self._load_selected_detector_vi)

        # Center/Radius grid and the Detectors list sit side by side (rather
        # than stacked, as they were in the old narrow left-panel column) -
        # two different concepts sharing a row (numeric geometry vs. a list
        # of saved detectors), so a separator marks the boundary, same
        # convention as elsewhere on this ribbon. "Use Virtual Mask" sits
        # above the grid, in the same left-hand QVBoxLayout, so the
        # Detectors list on the right stretches the full height starting
        # from that checkbox's row, not just alongside the grid below it.
        layout_vi_left = qtw.QVBoxLayout()
        layout_vi_left.addWidget(self.checkbox_useVirtualMask)
        layout_vi_left.addLayout(grid_vi_geometry)
        layout_vi_middle = qtw.QHBoxLayout()
        layout_vi_middle.addLayout(layout_vi_left)
        self._ribbon_inline_separator(layout_vi_middle)
        layout_vi_middle.addWidget(widget_vi_detectors, 1)
        layout_virtualImaging.addLayout(layout_vi_middle)

        #### Mode | Compute Virtual Image + Cancel 
        layout_vi_actions = qtw.QHBoxLayout()
        layout_vi_actions_mode = qtw.QVBoxLayout()
        layout_vi_actions.addLayout(layout_vi_actions_mode)
        layout_vi_actions_mode.addWidget(qtw.QLabel('Mode'))
        self.combo_virtualMode = qtw.QComboBox()
        self.combo_virtualMode.setFixedWidth(button_w)
        self.combo_virtualMode.addItems(['Sum', 'Variance'])
        self.combo_virtualMode.setToolTip(
            'Sum: total scattered intensity per scan position (standard vSTEM). '
            'Variance: highlights local structural variation instead')
        layout_vi_actions_mode.addWidget(self.combo_virtualMode)

        self.button_computeVirtualImage = qtw.QPushButton('Compute\nVirtual Image')
        # self.button_computeVirtualImage.setFixedSize(button_w, button_h)
        self.button_computeVirtualImage.clicked.connect(self.compute_virtual_image)
        self.button_computeVirtualImage.setToolTip(
            'Compute a navigation image over the whole scan using the mode/mask above')
        layout_vi_actions.addWidget(self.button_computeVirtualImage)
        
        
        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Expanding)
        # self.button_cancel.setFixedSize(button_w, button_h)
        self.button_cancel.setStyleSheet("background-color: red; color: white;")
        self.button_cancel.setDisabled(True)
        self.button_cancel.setToolTip('Stop the running segmentation/computation')
        self.button_cancel.clicked.connect(self.cancel_running_work)
        
        # button_sumDpWhole lives in the Summed DP Threshold sub-section
        # below (beside button_sumDpFromThreshold) instead of here - both
        # produce a whole-tab-level reference DP without an ROI/mask, so
        # they belong together rather than one of them being off in
        # Virtual Imaging.
        layout_vi_actions.addWidget(self.button_cancel)
        layout_virtualImaging.addLayout(layout_vi_actions)

        # Artists for the virtual-detector circle overlay drawn on the DP
        # axis, redrawn by update_virtual_mask_overlay() - the live
        # (spinbox-driven) detector's 3 artists are also tracked directly
        # (_vi_circle_out/_vi_circle_in/_vi_center_marker) for Ctrl+drag
        # (see _try_start_vi_mask_drag/_on_motion_vi_mask/_on_release_vi_mask).
        self._vi_mask_artists = []
        self._vi_circle_out = None
        self._vi_circle_in = None
        self._vi_center_marker = None
        self._vi_mask_drag_mode = None
        self._vi_mask_bg = None
        # Toggled by the ribbon's "hide virtual detectors" button - hides
        # the overlay circles without touching the underlying detector
        # data, for an unobstructed look at the DP itself.
        self._vi_mask_hidden = False
        self._ribbon_group_end(layout_ribbon, layout_virtualImaging, 'Virtual Imaging')

        #%% SAM2 Segmentation / Edge Detection / Summed DP Threshold
        # These three used to each be their own full-height ribbon column;
        # they're now stacked vertically inside ONE column (box_edgeDetection)
        # instead, each sub-section separated by an HLine and captioned on
        # its own - a deliberate way to fit more, lower-priority/action-
        # oriented groups into fewer ribbon columns. _ribbon_group_end() is
        # called once per sub-section (all still targeting the same
        # layout_edgeDetection), with `stretch=False` only for the very
        # first one (SAM2 Segmentation - its controls should sit right above
        # its caption, not be pushed down by addStretch) and
        # `separator=False` for the second and third (no vertical-line
        # ribbon separator needed between them - the HLine above already
        # marks the boundary; only the group's outer edge needs one).
        self.box_edgeDetection, layout_edgeDetection = self._ribbon_group_start(layout_ribbon, stretch=0)
        self.box_edgeDetection.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Preferred)

        #%% SAM2 segmentation (first sub-section in this combined column)
        layout_segmentation_row = qtw.QHBoxLayout()

        self.button_segment_image = qtw.QPushButton('Segment Image')
        self.button_segment_image.setFixedSize(button_w, button_h)
        layout_segmentation_row.addWidget(self.button_segment_image)
        self.button_segment_image.clicked.connect(self.segment_image)
        self.button_segment_image.setDisabled(True)
        self.button_segment_image.setToolTip('Run SAM2 on the points added below (Shift+Click)')

        self.button_clear_points = qtw.QPushButton('Clear Points')
        self.button_clear_points.setFixedSize(button_w, button_h)
        layout_segmentation_row.addWidget(self.button_clear_points)
        self.button_clear_points.clicked.connect(self.clear_seg_points)
        self.button_clear_points.setDisabled(True)
        self.button_clear_points.setToolTip('Remove all SAM2 points and the segmentation mask')

        self.button_clear_roi = qtw.QPushButton('Clear Box')
        self.button_clear_roi.setFixedSize(button_w, button_h)
        layout_segmentation_row.addWidget(self.button_clear_roi)
        self.button_clear_roi.clicked.connect(self.clear_roi)
        # Deactivated for now, at Saleh's request, pending his own review of
        # the box-prompt path - segment_image() no longer reads self.roi
        # either (see its docstring), so there's nothing left for this
        # button to affect while it's off.
        self.button_clear_roi.setDisabled(True)
        self.button_clear_roi.setToolTip('Temporarily deactivated (pending review)')
        layout_edgeDetection.addLayout(layout_segmentation_row)
        self._ribbon_group_end(layout_ribbon, layout_edgeDetection, 'SAM2 Segmentation', stretch=False)

        sep = qtw.QFrame()
        sep.setFrameShape(qtw.QFrame.HLine)
        sep.setFrameShadow(qtw.QFrame.Sunken)
        layout_edgeDetection.addWidget(sep)

        #%% Edge Detection
        layout_edgeDetection_row_1 = qtw.QHBoxLayout()
        self.checkbox_edgeOnly = qtw.QCheckBox('Activate')
        self.checkbox_edgeOnly.setToolTip('Reduce the mask to just its outline')
        layout_edgeDetection_row_1.addWidget(self.checkbox_edgeOnly)
        self.checkbox_edgeOnly.stateChanged.connect(self._preview_edge_mask)
        self.checkbox_edgeDirectional = qtw.QCheckBox('Directional')
        self.checkbox_edgeDirectional.setToolTip(
            'Keep only the edge facing one direction (angle below)')
        layout_edgeDetection_row_1.addWidget(self.checkbox_edgeDirectional)
        self.checkbox_edgeDirectional.stateChanged.connect(self._on_edge_directional_toggled)
        self.checkbox_revertMask = qtw.QCheckBox('Reverse Mask')
        self.checkbox_revertMask.setToolTip(
            'With Activate: keep the interior, cut the edge band (inverse)')
        layout_edgeDetection_row_1.addWidget(self.checkbox_revertMask)
        self.checkbox_revertMask.stateChanged.connect(self._preview_edge_mask)
        
        layout_edgeDetection.addLayout(layout_edgeDetection_row_1)
        
        # self._ribbon_inline_separator(layout_edgeDetection_row_1)
        layout_edgeDetection_row_2 = qtw.QHBoxLayout()
        layout_edgeDetection_row_2.addWidget(qtw.QLabel('Kernel'))
        self.spinbox_edgeKernel = qtw.QSpinBox()
        self.spinbox_edgeKernel.setRange(1, 99)
        self.spinbox_edgeKernel.setValue(3)
        self.spinbox_edgeKernel.setToolTip('Erosion kernel size (pixels) - larger = wider edge band')
        self.spinbox_edgeKernel.valueChanged.connect(self._preview_edge_mask)
        layout_edgeDetection_row_2.addWidget(self.spinbox_edgeKernel)

        self._ribbon_inline_separator(layout_edgeDetection_row_2)
        layout_edgeDetection_row_2.addWidget(qtw.QLabel('Angle (°)'))
        self.spinbox_edgeDirection = qtw.QDoubleSpinBox()
        self.spinbox_edgeDirection.setRange(-360, 360)
        self.spinbox_edgeDirection.setDecimals(1)
        self.spinbox_edgeDirection.setSingleStep(5)
        self.spinbox_edgeDirection.setValue(0)
        self.spinbox_edgeDirection.setDisabled(True)
        self.spinbox_edgeDirection.setToolTip(
            '0°=right, 90°=down, 180°=left, 270°=up (clockwise); needs "Directional"')
        self.spinbox_edgeDirection.valueChanged.connect(self._preview_edge_mask)
        layout_edgeDetection_row_2.addWidget(self.spinbox_edgeDirection)

        self._ribbon_inline_separator(layout_edgeDetection_row_2)
        self.button_computeEdgeDp = qtw.QPushButton('Re-compute DP')
        self.button_computeEdgeDp.setFixedSize(button_w, button_h)
        self.button_computeEdgeDp.clicked.connect(self._refresh_edge_mask)
        self.button_computeEdgeDp.setToolTip(
            'Recompute the diffraction pattern with the settings above (slower, reads disk)')
        layout_edgeDetection_row_2.addWidget(self.button_computeEdgeDp)
        layout_edgeDetection.addLayout(layout_edgeDetection_row_2)
        self._ribbon_group_end(layout_ribbon, layout_edgeDetection, 'Edge Detection', separator=False, stretch=True)

        sep = qtw.QFrame()
        sep.setFrameShape(qtw.QFrame.HLine)
        sep.setFrameShadow(qtw.QFrame.Sunken)
        layout_edgeDetection.addWidget(sep)

        #%% Sum DP 
        layout_sumDp_row = qtw.QHBoxLayout()
        self.button_sumDpWhole = qtw.QPushButton('Sum DPs')
        self.button_sumDpWhole.setFixedSize(button_w, button_h)
        self.button_sumDpWhole.clicked.connect(self.compute_sum_dp_whole)
        self.button_sumDpWhole.setToolTip(
            'Sum every DP in the whole scan into one reference DP - no ROI/mask needed')
        layout_sumDp_row.addWidget(self.button_sumDpWhole)

        self.button_sumDpFromThreshold = qtw.QPushButton('DP by Threshold')
        self.button_sumDpFromThreshold.setFixedSize(button_w+10, button_h)
        self.button_sumDpFromThreshold.clicked.connect(self.open_threshold_dialog)
        self.button_sumDpFromThreshold.setToolTip(
            'Sum diffraction patterns at scan positions above a real-space threshold')
        layout_sumDp_row.addWidget(self.button_sumDpFromThreshold)

        layout_edgeDetection.addLayout(layout_sumDp_row)
        self._ribbon_group_end(layout_ribbon, layout_edgeDetection, 'Sum DP', separator=False, stretch=True)

        sep = qtw.QFrame()
        sep.setFrameShape(qtw.QFrame.HLine)
        sep.setFrameShadow(qtw.QFrame.Sunken)
        layout_edgeDetection.addWidget(sep)

        layout_ribbon.addStretch(1)
        #%% canvas layout (below the ribbon, using the tab's full width)
        self._right_widget = qtw.QWidget()
        self.layout.addWidget(self._right_widget, 1)
        layout_right_outer = qtw.QHBoxLayout(self._right_widget)
        layout_right_outer.setContentsMargins(0, 0, 0, 0)
        layout_right_outer.setSpacing(0)

        # Clipping Thresholds (vertical vmin/vmax sliders, see
        # ui_tabs/clipping_thresholds.py) sit directly beside the two edge
        # subplots of the 3-subplot canvas below - Nav. Image is the
        # leftmost subplot, Dif. Pattern the rightmost, so a narrow column
        # on each side of the canvas ends up beside its matching axis. The
        # middle "ROI Image" subplot doesn't get its own (main nav image
        # only, matching the other tabs).
        # File list - same convention as the Navigator tab: a data-type
        # filter combo above a list of sibling files (in the loaded 4D
        # signal's own folder), sitting beside clip_nav's sliders so a
        # different file in the same folder can be picked without
        # re-browsing. Double-click fills in the 4D Signal path (same as
        # browsing); it doesn't auto-compute - "Compute Virtual Image"
        # still starts that.
        widget_fileList = qtw.QWidget()
        layout_fileList = qtw.QVBoxLayout(widget_fileList)
        layout_fileList.setContentsMargins(2, 2, 2, 2)
        self.combo_dtype = qtw.QComboBox()
        self.combo_dtype.addItems(['All files', '.tpx3', '.hdf5', '.hspy', '.zspy', '.mib', '.tif'])
        self.combo_dtype.setToolTip(
            'Filter the file list below to one data type - ".tif" matches both '
            '.tif and .tiff files')
        self.combo_dtype.currentIndexChanged.connect(self.refresh_file_list)
        layout_fileList.addWidget(self.combo_dtype)
        self.file_list_widget = qtw.QListWidget()
        self.file_list_widget.setMinimumWidth(150)
        self.file_list_widget.setMaximumWidth(220)
        self.file_list_widget.setToolTip(
            'Other files in the same folder as the loaded 4D Signal - double-click '
            'to fill in its path (does not compute automatically)')
        self.file_list_widget.itemDoubleClicked.connect(self._on_file_list_double_clicked)
        layout_fileList.addWidget(self.file_list_widget, 1)
        layout_right_outer.addWidget(widget_fileList)

        self.clip_nav = ClippingThresholdsWidget()
        layout_right_outer.addWidget(self.clip_nav)

        self._canvas_container = qtw.QWidget()
        layout_right_outer.addWidget(self._canvas_container, 1)
        layout_canvas = qtw.QVBoxLayout(self._canvas_container)
        
        # self.figure = Figure(figsize=(5,5))
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout_canvas.addWidget(self.wrap_canvas_in_scroll(self.canvas))
        
        self.ax_nav = self.figure.add_subplot(131)
        self.ax_nav_roi = self.figure.add_subplot(132)
        self.ax_dp = self.figure.add_subplot(133)

# =============================================================================
#         gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1])  # Top row: equal, bottom: double height
#         self.ax_nav = self.figure.add_subplot(gs[0, 0])
#         self.ax_nav_roi = self.figure.add_subplot(gs[0, 1])
#         self.ax_dp = self.figure.add_subplot(gs[1, :])
# =============================================================================


        # self.ax_nav_roi.set_position([0.75, 0.75, 0.15, 0.15])
        # self.ax_nav.set_position([0.05, 0.1, 0.4, 0.8])
        # self.ax_dp.set_position([0.5, 0.1, 0.4, 0.8])

        # Image artists + colorbars are created once here (not in
        # reset_canvas, which only clears their data) - re-imshow()ing on
        # every "Compute Virtual Image" click would otherwise stack a new
        # AxesImage (and, once colorbars existed, a new colorbar axes) on
        # top of the old ones each time, growing the figure without bound.
        self._setup_canvas()

        self.rect = None            # Currently drawn rectangle
        self.roi = None             # (x, y, w, h) of the last-drawn ROI, or None
        self.press = None           # Mouse press coordinates
        self.backgrounds = {}       # blit-cached canvas snapshots, keyed by axis
        # SAM2 segmentation point prompts for the currently loaded nav image
        self.seg_points = []
        self.seg_labels = []
        self.seg_mask = None        # raw (un-eroded) mask - see _refresh_edge_mask()
        self._mask_source = None    # 'sam2' or 'threshold' - which path produced self.seg_mask
        self._threshold_method = None  # method string, only set when _mask_source == 'threshold'
        self.scatter_plots = []
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        # Kept alive (not shown) purely for its view-stack bookkeeping
        # (.update()/.push_current(), used to seed the ribbon's Home button)
        # and as the target of the ribbon's own Pan/Zoom/Home actions below -
        # the toolbar strip itself is no longer shown under the canvas.
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()

        self.clip_dp = ClippingThresholdsWidget()
        layout_right_outer.addWidget(self.clip_dp)
        self.clip_nav.valueChanged.connect(lambda: self.update_canvas(ax='nav'))
        self.clip_dp.valueChanged.connect(lambda: self.update_canvas(ax='dp'))

        #%% ribbon
        # Docked along the right edge (see layout_right_outer above) - an
        # additional way to reach the same canvas interactions already
        # available via Ctrl/Shift-click (see on_press); deliberately does
        # NOT duplicate the left panel's buttons (Segment Image, Clear
        # Points/ROI, ...), only actions that act directly on the plot
        # itself. 'select_roi'/'add_point' are the only two tool modes
        # on_press actually checks (see RibbonPanel.active_tool there).
        self.ribbon = RibbonPanel([
            RibbonTool('select_roi', 'select_roi', 'Select ROI: click+drag on the Nav. Image '
                      '(same as Ctrl+drag)', 'tool'),
            RibbonTool('add_point', 'add_point', 'Add SAM2 point (left=+/right=-, same as Shift+click)',
                      'tool'),
            RibbonTool('remove_point', 'remove_point', 'Remove last SAM2 point (same as middle-click)',
                      'action', self.delete_last_seg_point),
            RibbonTool('sep1', kind='separator'),
            RibbonTool('center_recip', 'center_recip', 'Find the beam center, re-center the '
                      'reciprocal-space rings (same as "Center" in Scale bars)',
                      'action', self.find_and_center_recip),
            RibbonTool('center_mask', 'center_mask', 'Find the beam center, re-center the virtual detector mask',
                      'action', self.vi_auto_center),
            RibbonTool('hide_mask', 'hide_mask', 'Hide the virtual detector overlay on the DP plot',
                      'toggle', self._on_toggle_vi_mask_hidden),
            RibbonTool('sep2', kind='separator'),
            RibbonTool('pan', 'pan', 'Toggle pan mode',
                      'action', self.toolbar.pan),
            RibbonTool('zoom', 'zoom', 'Toggle rectangle-zoom mode',
                      'action', self.toolbar.zoom),
            RibbonTool('home', 'home', 'Reset the view',
                      'action', self.toolbar.home),
        ], parent=self)
        self.ribbon.toolChanged.connect(self._on_ribbon_tool_changed)
        # NavigationToolbar2's _wait_cursor_for_draw_cm() wraps every
        # canvas.draw() call and restores its own internally-tracked cursor
        # in a `finally` block that runs after 'draw_event' listeners fire -
        # so reapplying the ribbon's cursor directly from 'draw_event' still
        # gets silently overwritten a moment later by mpl itself (verified:
        # this was happening on every on_press-triggered redraw). Deferring
        # via singleShot(0, ...) runs it on the next event-loop tick,
        # after that synchronous mpl-internal reset has already finished.
        self.canvas.mpl_connect(
            'draw_event', lambda evt: QTimer.singleShot(0, self._apply_ribbon_cursor))
        layout_right_outer.addWidget(self.ribbon)

        # The app-wide log console lives here (below this tab's own plot
        # column) rather than under the whole window, so the left parameter
        # panel (a separate splitter pane) can span the full window height.
        self.log_console = LogConsole(self)
        layout_canvas.addWidget(self.log_console)

        # keyboard shortcuts (matching the equivalent actions on the other tabs)
        QShortcut(QKeySequence('Ctrl+O'), self, self.button_computeVirtualImage.click)
        QShortcut(QKeySequence('Ctrl+T'), self, self.button_segment_image.click)

        # Picks up any non-default DisplaySettings already set by the Edit
        # tab (e.g. this instance is a duplicate opened after adjusting
        # sizes) - see TabBase.apply_display_settings.
        self.apply_display_settings()
#%% functions
    def activate_lineEdit_scanSize(self):
        if self.checkbox_scanSize.isChecked():
            self.spinbox_scanSize_x.setDisabled(True)
            self.spinbox_scanSize_y.setDisabled(True)
        else:
            self.spinbox_scanSize_x.setEnabled(True)
            self.spinbox_scanSize_y.setEnabled(True)

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

    def get_scan_size(self):
        """Return the (x, y) scan size from the manual entry fields, or
        None if "Auto" is checked or the fields don't hold valid integers."""
        if not self.checkbox_scanSize.isChecked():
            try:
                x = int(self.spinbox_scanSize_x.text())
                y = int(self.spinbox_scanSize_y.text())
                scanSize = (x,y)
            except ValueError:
                scanSize = None
        else:
            scanSize = None
        return scanSize
    
    def image_handler(self, result):
        """Store a newly computed navigation image (see
        compute_virtual_image/_handle_finished_vi) and reset the nav
        axis view/toolbar history to it."""
        self.navImg = result
        self.dp_center = None  # a new signal may have a different DP shape/center
        self._dp_center_cache_key = None
        self.clip_nav.set_range(self.navImg.min(), self.navImg.max())
        # Navigation images commonly have literal 0-count background/
        # unscanned positions - start the low threshold at the lowest
        # non-zero value instead, so those don't wash out real contrast.
        self.clip_nav.set_low(io.nonzero_display_min(self.navImg))
        self.update_canvas('nav')
        # Reset the view to the newly loaded image's full extent (in case
        # the user had already zoomed in on a previous signal, which
        # disables autoscale), then re-seed the toolbar's view stack so its
        # "Home" button resets to *this* view instead of doing nothing (it
        # does nothing until something pushes at least one view onto its
        # stack, which our own scroll-wheel zoom deliberately bypasses).
        shape_x, shape_y = self.navImg.shape
        self.ax_nav.set_xlim(0, shape_y)
        self.ax_nav.set_ylim(shape_x, 0)
        self.toolbar.update()
        self.toolbar.push_current()
        self.button_clear_points.setEnabled(True)
        self.button_cancel.setDisabled(True)

    def show_dialog(self):
        file_filter = "supported signals (*.zspy *.hspy *.hdf5 *.h5 *.tpx3 *.mib *.pmf);;All Files (*)"
        path = qtw.QFileDialog.getOpenFileName(self, "Select 4D Signals Folder", '', file_filter)
        # if path and os.path.isdir(path[0]):
        if path:
            self.metadata_path_override = None  # new signal - re-derive comment.txt location
            self.lineEdit_dir_signal.setText(path[0])

    def _ext_filter_for_combo(self):
        """Extensions matching the current combo_dtype selection - '.tif'
        matches both '.tif' and '.tiff' (see combo_dtype's tooltip);
        'All files' returns every supported extension. Matches the
        Navigator tab's identical helper."""
        selected = self.combo_dtype.currentText()
        if selected == 'All files':
            return ['.tpx3', '.hdf5', '.zspy', '.hspy', '.mib', '.pmf', '.tif', '.tiff']
        if selected == '.tif':
            return ['.tif', '.tiff']
        return [selected]

    def refresh_file_list(self):
        """(Re)populate the file list from the folder containing the
        currently-entered 4D Signal FILE (unlike the Navigator tab, this
        tab's 4D Signal field points at one file, not a folder - so the
        folder to list is that file's parent), filtered to the data type
        selected in combo_dtype."""
        fn = self.lineEdit_dir_signal.text()
        directory = os.path.dirname(fn) if fn else ''
        self.file_list_widget.clear()
        if not os.path.isdir(directory):
            return
        ext_filter = self._ext_filter_for_combo()
        for f in sorted(os.listdir(directory)):
            if os.path.splitext(f)[1] in ext_filter:
                self.file_list_widget.addItem(f)

    def _on_file_list_double_clicked(self, item):
        """Fill in the 4D Signal path from a file list double-click - same
        end result as browsing to it via show_dialog(), then immediately run
        "Compute Virtual Image" on it, same as clicking that button would."""
        directory = os.path.dirname(self.lineEdit_dir_signal.text())
        self.metadata_path_override = None  # new signal - re-derive comment.txt location
        self.lineEdit_dir_signal.setText(os.path.join(directory, item.text()))
        self.compute_virtual_image()

    def enable_dwellTime_spinbox(self, txt):
        """Enable the scan-size/dwell-time widgets only for .tpx3 files
        (smart-scan controls live in the File box and are handled
        separately - see activate_lineEdit_patternFile()); auto-loads
        metadata when enabling."""
        enable = False
        if os.path.isfile(txt):
            dtype = os.path.splitext(txt)[1]
            if dtype == '.tpx3':
                enable = True
        # Explicit widget list rather than a findChildren() scan over the
        # whole "Experiment Info" group - that group now also holds the 4D
        # Signal/Smart Scan/Metadata/Scale controls (merged into one ribbon
        # column), none of which should be tpx3-gated the way Detector
        # Size/Scan Size/Dwell Time are.
        scan_widgets = (self.checkbox_detectorSizeAuto, self.spinbox_detectorSize_x,
                        self.spinbox_detectorSize_y, self.checkbox_scanSize,
                        self.spinbox_scanSize_x, self.spinbox_scanSize_y,
                        self.spinbox_dwellTime)
        for wid in scan_widgets:
            wid.setEnabled(enable)
        self.checkbox_scanSize.setChecked(not enable)
        self.checkbox_scanSize.setDisabled(enable)
        # Attempted for every format, not just .tpx3 - comment.txt is
        # written for smart-scanned .mib/.hspy/.zspy acquisitions too, and
        # load_metadata(silent=True) already no-ops quietly when comment.txt
        # is missing or unparsable.
        if os.path.isfile(txt):
            self.load_metadata(silent=True)
        self.activate_lineEdit_patternFile()

    def browse_metadata_file(self):
        fn = self.lineEdit_dir_signal.text()
        start_dir = os.path.dirname(fn) if fn else ''
        path, _ = qtw.QFileDialog.getOpenFileName(
            self, "Select Metadata File", start_dir, "Text files (*.txt);;All Files (*)")
        if path:
            self.metadata_path_override = path
            self.load_metadata(silent=False)

    def load_metadata(self, silent=True):
        """Fill scan size / dwell time from a comment.txt next to the 4D
        signal file, if present (tpx3 acquisitions log scan metadata there).
        silent=True swallows a missing/unparsable comment.txt quietly (used
        for the automatic per-file attempt); silent=False (the "Load
        Metadata" button, or a manually-browsed file) surfaces the failure
        to the user."""
        fn = self.lineEdit_dir_signal.text()
        path_main = self.metadata_path_override or os.path.dirname(fn)
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

    def activate_lineEdit_patternFile(self):
        """Enable the pattern-file picker, only when "Smart Scanned" is
        checked - dwell time always comes from the single Dwell T. spinbox
        above (Input Parameters), smart-scanned or not."""
        enable = self.checkbox_smartScan.isChecked()
        self.lineEdit_patternFile.setEnabled(enable)
        self.button_browsePattern.setEnabled(enable)

    def browse_pattern_file(self):
        fn = self.lineEdit_dir_signal.text()
        start_dir = os.path.dirname(fn) if fn else ''
        path, _ = qtw.QFileDialog.getOpenFileName(
            self, "Select Pattern File", start_dir, "Text files (*.txt);;All Files (*)")
        if path:
            self.lineEdit_patternFile.setText(path)

    def get_fn_pattern(self):
        """Return the pattern-file path when smart-scan is enabled and a
        path is set, else None."""
        if self.checkbox_smartScan.isChecked() and self.lineEdit_patternFile.text():
            return self.lineEdit_patternFile.text()
        return None

    def apply_edge_mask(self, mask):
        """Reduce `mask` to just its edge/outline when "Edge Only" is
        checked (see io.erode_mask_edge) - a no-op copy otherwise. Called on
        every mask right as it becomes "final" (SAM2 segmentation result,
        accepted threshold-dialog mask), so both the on-screen overlay and
        the DP extraction that follows see the same, possibly-eroded, mask."""
        if self.checkbox_edgeOnly.isChecked():
            direction = (self.spinbox_edgeDirection.value()
                        if self.checkbox_edgeDirectional.isChecked() else None)
            return io.erode_mask_edge(mask, self.spinbox_edgeKernel.value(), direction=direction,
                                       revert=self.checkbox_revertMask.isChecked())
        return mask

    def _on_edge_directional_toggled(self):
        self.spinbox_edgeDirection.setEnabled(self.checkbox_edgeDirectional.isChecked())
        self._preview_edge_mask()

    # _ribbon_group_start/_ribbon_group_end/_ribbon_inline_separator now
    # live on TabBase (base_tab.py) - shared by all 4 tabs' ribbons.

    def _on_ribbon_tool_changed(self, tool_id):
        self.logger.debug('Ribbon tool changed to %s', tool_id)
        self._apply_ribbon_cursor()

    def _apply_ribbon_cursor(self):
        """Set the canvas cursor to match the ribbon's active tool - besides
        the ribbon button's own highlighted (QToolButton:checked) style,
        this gives the active tool a distinct cursor too, since which mode
        is armed wasn't obvious enough from the ribbon alone. Also called
        on every canvas redraw (see the 'draw_event' connection in
        init_widget) - matplotlib's own toolbar/backend resets the cursor
        to a plain arrow on canvas.draw() (e.g. right after on_press draws
        the in-progress ROI rectangle), which would otherwise silently undo
        this on the very next interaction."""
        cursor = {'select_roi': Qt.CrossCursor, 'add_point': Qt.PointingHandCursor}.get(
            self.ribbon.active_tool)
        self.canvas.setCursor(cursor if cursor is not None else Qt.ArrowCursor)

    def on_press(self, event):
        """Mouse-press dispatch for the nav/DP canvas: manual DP
        re-centering (Ctrl+Click on the DP axis), removing the last SAM2
        point (middle click), adding a SAM2 point (Shift+Click, or a plain
        click while the ribbon's "Add SAM2 point" tool is active), or
        starting a new ROI rectangle (Ctrl+Drag, or a plain drag while the
        ribbon's "Select ROI" tool is active) - depending on click location,
        modifiers, and the ribbon's currently active tool."""
        ribbon_tool = self.ribbon.active_tool
        if (event.inaxes == self.ax_dp and event.button == 1 and event.xdata is not None
                and 'ctrl' in event.modifiers and hasattr(self, 'dp')):
            # Grabbing the virtual mask's own center/edge handles always
            # takes priority over the reciprocal-space center below - the
            # two centers are independent (see find_and_center_recip vs.
            # vi_auto_center).
            if self._try_start_vi_mask_drag(event):
                return
            # Manual re-centering of the reciprocal-space rings.
            self.dp_center = (event.xdata, event.ydata)
            self.update_canvas(ax='dp')
            return
        if event.inaxes != self.ax_nav:
            self.press = None
            return
        if event.button == 2:
            # Middle click removes the last-added SAM2 point.
            self.delete_last_seg_point()
            return
        if (ribbon_tool == 'add_point' or 'shift' in event.modifiers) and event.button in (1, 3):
            # Shift+click (or the ribbon's "Add SAM2 point" tool, plain
            # click) adds a SAM2 point prompt (left = positive, right =
            # negative) - deliberately a different modifier than the
            # Ctrl+drag ROI below so the two annotation modes can't be
            # confused with each other.
            self.add_seg_point(event)
            return
        if (ribbon_tool != 'select_roi' and 'ctrl' not in event.modifiers) or event.button != 1:
            # Plain click/drag is reserved for the navigation toolbar's
            # Pan/Zoom tool (and the scroll-wheel zoom) so images can be
            # zoomed into; hold "ctrl" (or activate the ribbon's "Select
            # ROI" tool) to draw a new ROI instead.
            self.press = None
            return
        self.press = (event.xdata, event.ydata)
        if self.rect is not None:
            self.rect.remove()
        self.rect = patches.Rectangle(self.press, 0, 0, linewidth=1,
                                      edgecolor='r', facecolor='none')
        self.ax_nav.add_patch(self.rect)
        # One full draw() here "bakes in" the current state (nav image, the
        # just-added zero-size rect) into a cached background snapshot, so
        # on_motion below can cheaply blit just the growing rectangle on top
        # of it instead of redrawing the whole figure on every mouse-move.
        self.canvas.draw()
        self.backgrounds['nav'] = self.canvas.copy_from_bbox(self.ax_nav.bbox)

    def on_motion(self, event):
        """Resize the in-progress ROI rectangle as the mouse moves,
        blitting just the rectangle onto the cached background for speed.
        Dispatches to the virtual-mask drag instead when one is active (see
        _try_start_vi_mask_drag)."""
        if self._vi_mask_drag_mode is not None:
            self._on_motion_vi_mask(event)
            return
        if self.press is None or event.inaxes is None:
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
            return
        self.canvas.restore_region(self.backgrounds['nav'])
        self.ax_nav.draw_artist(self.rect)
        self.canvas.blit(self.ax_nav.bbox)

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

    def on_release(self, event):
        """Finalize the ROI rectangle on mouse-up, normalize a reversed
        drag into (x, y, w, h), then kick off a background
        diffraction-pattern computation for it. Dispatches to the
        virtual-mask drag instead when one is active (see
        _try_start_vi_mask_drag)."""
        if self._vi_mask_drag_mode is not None:
            self._on_release_vi_mask()
            return
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
        self.roi = (int(x0), int(y0), int(width), int(height))

        self.press = None
        self.canvas.draw()
        self.logger.info('ROI: %s', self.roi)
        if not hasattr(self, 'dwellTime'):
            try:
                self.dwellTime = self.spinbox_dwellTime.value()
            except Exception:
                self.dwellTime = None
        worker = Worker_CalculateDP(self.fn, self.roi, self.scanSize, self.dwellTime,
                                    self.get_fn_pattern(), self.get_detector_shape(self.fn))
        worker.signals.result.connect(self.get_dp)
        self.threadpool.start(worker)
    
    def get_dp(self, result):
        """Slot for Worker_CalculateDP's result: store the rectangle-ROI
        diffraction pattern/nav crop and refresh the display."""
# =============================================================================
#         s_cut = io.load_signal(self.fn, lazy=False, roi=roi, scanSize=scanSize, dwellTime=dwellTime)
#         self.navImg_cut = s_cut.sum(axis=(2,3)).data
#         self.dp = s_cut.sum(axis=(0,1)).data
#         if hasattr(self.dp, 'compute'): # lazy signals
#             self.dp.compute()
# =============================================================================
        self.dp, self.navImg_cut = result
        self.logger.info('max = %s', self.dp.max())
        self.ax_nav_roi.set_title(f'ROI Image: {self.roi}')
        # A rectangle ROI was just drawn, replacing whatever the "ROI Image"
        # axis was previously showing - clear any leftover SAM2 mask overlay
        # so it doesn't linger on top of this unrelated crop.
        self.img_display['seg_mask'].set_data(np.zeros((512, 512, 4)))
        self._reset_dp_clip_range()
        self.update_canvas(roiUpdate=True)
        self.update_virtual_mask_overlay()

    def _reset_dp_clip_range(self):
        """(Re)anchor clip_dp's Clipping Thresholds to the freshly-loaded
        self.dp's own range, starting at vmin=1 (not the data's true
        minimum, usually 0) - matches this tab's long-standing convention
        for the DP display specifically."""
        self.clip_dp.set_range(self.dp.min(), self.dp.max())
        self.clip_dp.set_low(1)

    def _setup_canvas(self):
        """One-time creation of the image artists and their colorbars.
        reset_canvas() (called on every "Compute Virtual Image") only
        clears their data afterwards - re-imshow()ing here on every load
        would otherwise stack a new image (and, now, a new colorbar axes)
        on the figure each time."""
        self.img_display = {}
        img_temp = np.zeros((512,512), dtype='uint16')
        self.img_display['nav'] = self.ax_nav.imshow(img_temp, cmap='viridis')
        self.img_display['nav_roi'] = self.ax_nav_roi.imshow(img_temp, cmap='viridis')

        self.img_display['dp'] = self.ax_dp.imshow(img_temp, cmap='inferno')
        self.img_display['dp'].set_norm(SymLogNorm(linthresh=1))

        # SAM2 segmentation mask overlay, drawn on top of a crop of the nav
        # image in the "ROI Image" axis (show_seg_mask() re-targets nav_roi's
        # own image + view to that crop, then this sits on top of it).
        # Starts fully transparent (all-zero RGBA); show_seg_mask() fills in
        # color+alpha only where the mask is True.
        self.img_display['seg_mask'] = self.ax_nav_roi.imshow(np.zeros((512, 512, 4)))

        self.ax_nav.set_title('Nav. Image')
        self.ax_nav_roi.set_title('ROI Image')
        self.ax_dp.set_title('Dif. Pattern')

        self.ax_nav_roi.set_axis_off()
        # ax_nav/ax_dp keep their x-axis label visible (for the interaction
        # hints below), so their ticks/spines are hidden individually
        # instead of via set_axis_off(), which would hide the label too.
        for spine in self.ax_nav.spines.values():
            spine.set_visible(False)
        self.ax_nav.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax_nav.set_xlabel(
            'Hold "ctrl" + Drag => New ROI (diffraction-pattern extraction)\n'
            'Hold "shift" + Click => Add SAM2 point (Left=positive, Right=negative)\n'
            'Middle Click => Remove last SAM2 point', fontsize=9)
        self.ax_nav.xaxis.label.set_visible(True)

        for spine in self.ax_dp.spines.values():
            spine.set_visible(False)
        self.ax_dp.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax_dp.xaxis.label.set_visible(True)

        # The Ctrl+Scroll zoom hint applies to every axis on this canvas, so
        # it's a figure-wide supxlabel rather than repeated per-axis text.
        self.figure.supxlabel('Hold "Ctrl" + Scroll wheel to zoom the axis under the cursor',
                              fontsize=10)

        self.colorbars = {}
        self.colorbars['nav'] = self.figure.colorbar(
            self.img_display['nav'], ax=self.ax_nav, fraction=0.046, pad=0.04)
        self.colorbars['nav_roi'] = self.figure.colorbar(
            self.img_display['nav_roi'], ax=self.ax_nav_roi, fraction=0.046, pad=0.04)
        self.colorbars['dp'] = self.figure.colorbar(
            self.img_display['dp'], ax=self.ax_dp, fraction=0.046, pad=0.04)

        # self.figure.tight_layout()

    def reset_canvas(self):
        """Clear displayed data back to blank - used when a brand new 4D
        signal is about to be loaded (not on a same-signal recompute, see
        _reset_nav_image_only). The image artists/colorbars themselves are
        created once in _setup_canvas() and reused."""
        img_temp = np.zeros((512,512), dtype='uint16')
        self.img_display['nav'].set_data(img_temp)
        self.img_display['nav_roi'].set_data(img_temp)
        self.img_display['dp'].set_data(img_temp)
        self.ax_nav_roi.set_title('ROI Image')
        # Forces update_canvas()'s shape-change check to reset the view on
        # the new signal's first draw, even if its shape happens to match
        # whatever the previous signal last showed.
        self._dp_shape_seen = None
        self._navroi_shape_seen = None
        self.clear_seg_points()
        self.clear_roi()
        for artist in self._vi_mask_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._vi_mask_artists.clear()
        self.canvas.draw_idle()

    def _reset_nav_image_only(self):
        """Clear just the Nav. Image display, ahead of a "Compute Virtual
        Image" recompute - unlike reset_canvas(), leaves the already-plotted
        diffraction pattern, its virtual-detector mask circles, the ROI
        crop, and any SAM2 points untouched, since none of those are
        invalidated by re-running the SAME signal through a different
        mode/mask."""
        img_temp = np.zeros((512, 512), dtype='uint16')
        self.img_display['nav'].set_data(img_temp)
        self.canvas.draw_idle()

    def update_canvas(self, ax='dp', roiUpdate=False):
        """Refresh the 'dp' or 'nav' image display from current data -
        roiUpdate=True also refreshes the "ROI Image" crop. Scale bars and
        reciprocal-space circles are redrawn on every call regardless of
        `ax`."""
        if ax == 'dp':
            vmin, vmax = self.clip_dp.values()
            self.img_display['dp'].set_data(self.dp)

            shape_x, shape_y = self.dp.shape
            self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])
            self.img_display['dp'].set_clim(vmin, vmax)
            # ax_dp/ax_nav_roi show a per-ROI crop whose size varies with
            # each ROI selection, so (unlike ax_nav) their view is reset to
            # fit whenever that size actually changes, rather than relying
            # on the toolbar's Home button - otherwise, once the user has
            # zoomed in, the next (differently-sized) ROI selection would
            # stay at the old view. A same-shape update (contrast slider,
            # edge-detection setting tweak, ...) leaves the current zoom/pan
            # alone instead of resetting it on every such call.
            if (shape_x, shape_y) != getattr(self, '_dp_shape_seen', None):
                self.ax_dp.set_xlim(0, shape_y)
                self.ax_dp.set_ylim(shape_x, 0)
                self._dp_shape_seen = (shape_x, shape_y)

            if roiUpdate and hasattr(self, 'navImg_cut'):
                self.img_display['nav_roi'].set_data(self.navImg_cut)
                self.img_display['nav_roi'].set_clim(
                    io.nonzero_display_min(self.navImg_cut), self.navImg_cut.max())
                shape_x, shape_y = self.navImg_cut.shape
                self.img_display['nav_roi'].set_extent([0, shape_y, shape_x, 0])
                if (shape_x, shape_y) != getattr(self, '_navroi_shape_seen', None):
                    self.ax_nav_roi.set_xlim(0, shape_y)
                    self.ax_nav_roi.set_ylim(shape_x, 0)
                    self._navroi_shape_seen = (shape_x, shape_y)

        elif ax == 'nav':
            vmin, vmax = self.clip_nav.values()
            self.img_display['nav'].set_data(self.navImg)
            shape_x, shape_y = self.navImg.shape
            self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
            self.img_display['nav'].set_clim(vmin=vmin, vmax=vmax)
        
        # scale bars
        #TODO adding and removing the artist is not efficient
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
            for scale_ax in [self.ax_nav, self.ax_nav_roi]:
                io.add_readable_scalebar(scale_ax, scale_real, 'nm')
        except Exception:
            pass
        # A conventional linear scale bar doesn't read naturally on a
        # radially-symmetric diffraction pattern - concentric dashed rings
        # at every 1 1/A (centered on the DP) work better.
        dp_shape = self.img_display['dp'].get_array().shape
        # Centering is purely manual now (see find_and_center_recip/
        # vi_auto_center, and Ctrl+Click/drag in on_press) - self.dp_center
        # just persists across redraws until one of those changes it, no
        # more continuous auto-re-finding here.
        center = self.dp_center if self.dp_center is not None else None
        self._dp_recip_circles = io.draw_reciprocal_scale_circles(
            self.ax_dp, self.lineEdit_scale_recip.text(), dp_shape,
            center=center, old_artists=getattr(self, '_dp_recip_circles', None))
        self.ax_dp.set_xlabel(
            'Circle center: click "Center" (Scale bars) to find it, or hold Ctrl '
            'and click the pattern to set it manually', fontsize=9)

        self.canvas.draw()

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

#%% Virtual Imaging
    def find_and_center_recip(self):
        """Find the beam center now and jump the reciprocal-space rings
        there - the "Center" button (Scale bars) and the right-hand
        ribbon's quick-access action both call this; Ctrl+Click on the
        diffraction pattern does the same thing manually instead."""
        if not hasattr(self, 'dp'):
            qtw.QMessageBox.warning(self, 'No Diffraction Pattern',
                'Draw an ROI, segment an image, use "Summed DP from Threshold", '
                'or "Sum DP (Whole Signal)" first, so a beam center can be found.')
            return
        try:
            self.dp_center = io.find_dp_center_blurred(self.dp)
            self._dp_center_cache_key = id(self.dp)
        except Exception:
            self.logger.exception('Auto-centering failed.')
            return
        self.update_canvas(ax='dp')

    def vi_auto_center(self):
        """Copy the diffraction pattern's beam center (self.dp_center -
        found automatically via the large-sigma blur, or set manually via
        Ctrl+Click on the DP) into the virtual-detector Center X/Y
        spinboxes - finds it fresh if it hasn't been found/set yet."""
        if not hasattr(self, 'dp'):
            qtw.QMessageBox.warning(self, 'No Diffraction Pattern',
                'Draw an ROI, segment an image, or use "Summed DP from Threshold" '
                'first, so a beam center can be found.')
            return
        if self.dp_center is None:
            try:
                self.dp_center = io.find_dp_center_blurred(self.dp)
            except Exception:
                self.logger.exception('Auto-centering failed.')
                return
        cx, cy = self.dp_center
        self.spinbox_vi_centerX.setValue(int(round(cx)))
        self.spinbox_vi_centerY.setValue(int(round(cy)))

    def _sync_active_detector_vi(self):
        """Keep the active list entry (self._active_detector_row_vi) live-
        synced with the center/radii spinboxes as the user edits or drags
        them, so its displayed values are never stale - matches the
        Navigator tab's identical _sync_active_detector."""
        detector = {'center': (self.spinbox_vi_centerX.value(), self.spinbox_vi_centerY.value()),
                    'r_in': self.spinbox_vi_rIn.value(), 'r_out': self.spinbox_vi_rOut.value()}
        self._extra_detectors_vi[self._active_detector_row_vi] = detector
        item = self.list_detectors_vi.item(self._active_detector_row_vi)
        if item is not None:
            item.setText(self._format_detector_vi(detector))
        self.update_virtual_mask_overlay()

    def add_extra_detector_vi(self):
        """Snapshot the current center/radii spinbox values as a NEW
        detector entry, on top of whichever one was already active - the
        previous entry stays frozen, and further spinbox edits/drags target
        the new (now-selected) one instead."""
        detector = {'center': (self.spinbox_vi_centerX.value(), self.spinbox_vi_centerY.value()),
                    'r_in': self.spinbox_vi_rIn.value(), 'r_out': self.spinbox_vi_rOut.value()}
        self._extra_detectors_vi.append(detector)
        self.list_detectors_vi.addItem(self._format_detector_vi(detector))
        self.list_detectors_vi.setCurrentRow(len(self._extra_detectors_vi) - 1)
        self.update_virtual_mask_overlay()

    def remove_extra_detector_vi(self):
        """Remove the selected detector from the list - at least one always
        remains, so removing the last one re-seeds a fresh default entry
        instead of leaving the list empty."""
        row = self.list_detectors_vi.currentRow()
        if row < 0:
            return
        del self._extra_detectors_vi[row]
        self.list_detectors_vi.takeItem(row)
        if not self._extra_detectors_vi:
            default = {'center': (self.spinbox_vi_centerX.value(), self.spinbox_vi_centerY.value()),
                      'r_in': self.spinbox_vi_rIn.value(), 'r_out': self.spinbox_vi_rOut.value()}
            self._extra_detectors_vi.append(default)
            self.list_detectors_vi.addItem(self._format_detector_vi(default))
        self.list_detectors_vi.setCurrentRow(min(row, len(self._extra_detectors_vi) - 1))
        self.update_virtual_mask_overlay()

    def _format_detector_vi(self, detector):
        cx, cy = detector['center']
        return f"center=({cx:.0f}, {cy:.0f}), r={detector['r_in']:.0f}-{detector['r_out']:.0f}"

    def _load_selected_detector_vi(self):
        """Load the selected list entry's values into the spinboxes and
        make it the active (live-synced/draggable) detector - see
        _sync_active_detector_vi."""
        row = self.list_detectors_vi.currentRow()
        if row < 0 or row >= len(self._extra_detectors_vi):
            return
        self._active_detector_row_vi = row
        detector = self._extra_detectors_vi[row]
        for sb, val in ((self.spinbox_vi_centerX, detector['center'][0]),
                        (self.spinbox_vi_centerY, detector['center'][1]),
                        (self.spinbox_vi_rIn, detector['r_in']),
                        (self.spinbox_vi_rOut, detector['r_out'])):
            sb.blockSignals(True)
            sb.setValue(val)
            sb.blockSignals(False)
        self.update_virtual_mask_overlay()

    def get_active_virtual_detectors(self):
        """The full set of virtual detectors to use for the next Compute
        Virtual Image - every entry in the (always non-empty) list (matches
        the Navigator tab's identical get_active_detectors())."""
        return list(self._extra_detectors_vi)

    def _on_toggle_vi_mask_hidden(self, hidden):
        """Ribbon "hide virtual detectors" toggle: show/hide the overlay
        circles on ax_dp without discarding any detector data."""
        self._vi_mask_hidden = hidden
        self.update_virtual_mask_overlay()

    def update_virtual_mask_overlay(self):
        """Redraw the virtual-detector circle(s) over the currently
        displayed diffraction pattern (ax_dp) - a preview of what "Compute
        Virtual Image" will use. The live (spinbox-driven) detector is
        always shown - solid lime outer / khaki inner - with its 3 artists
        tracked directly (_vi_circle_out/_vi_circle_in/_vi_center_marker) so
        on_press/on_motion/on_release can Ctrl+drag it (see
        _try_start_vi_mask_drag); every already-Added detector is drawn on
        top, dashed and non-draggable (select it from the list - via
        _load_selected_detector_vi - to edit it instead), matching the
        Navigator tab's identical two-tier overlay."""
        for artist in self._vi_mask_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._vi_mask_artists.clear()
        self._vi_circle_out = None
        self._vi_circle_in = None
        self._vi_center_marker = None
        if not hasattr(self, 'dp'):
            self.canvas.draw_idle()
            return

        cx = self.spinbox_vi_centerX.value()
        cy = self.spinbox_vi_centerY.value()
        r_in = self.spinbox_vi_rIn.value()
        r_out = self.spinbox_vi_rOut.value()
        # khaki inner circle (item 3) reads clearly against both lime/cyan
        # outer rings and the DP axis's inferno colormap.
        self._vi_circle_out = patches.Circle((cx, cy), r_out, fill=False,
                                             edgecolor='lime', linewidth=1.5)
        self.ax_dp.add_patch(self._vi_circle_out)
        self._vi_mask_artists.append(self._vi_circle_out)
        if r_in > 0:
            self._vi_circle_in = patches.Circle((cx, cy), r_in, fill=False,
                                                edgecolor='khaki', linewidth=1.5)
            self.ax_dp.add_patch(self._vi_circle_in)
            self._vi_mask_artists.append(self._vi_circle_in)
        self._vi_center_marker = self.ax_dp.scatter(
            [cx], [cy], color='lime', marker='+', s=80, linewidth=2)
        self._vi_mask_artists.append(self._vi_center_marker)

        # Every OTHER listed detector (not the active one, already drawn
        # above from the spinboxes) - dashed so it stays visually distinct
        # from the one currently being positioned.
        for i, detector in enumerate(self._extra_detectors_vi):
            if i == self._active_detector_row_vi:
                continue
            dcx, dcy = detector['center']
            circle = patches.Circle((dcx, dcy), detector['r_out'], fill=False,
                                    edgecolor='magenta', linewidth=1.2, linestyle='--')
            self.ax_dp.add_patch(circle)
            self._vi_mask_artists.append(circle)
            if detector['r_in'] > 0:
                circle_in2 = patches.Circle((dcx, dcy), detector['r_in'], fill=False,
                                            edgecolor='khaki', linewidth=1.2, linestyle='--')
                self.ax_dp.add_patch(circle_in2)
                self._vi_mask_artists.append(circle_in2)

        if self._vi_mask_hidden:
            for artist in self._vi_mask_artists:
                artist.set_visible(False)
        # A detector's r_out can exceed the DP itself - re-pin the view to
        # the DP's own extent (only when the DP's shape actually changed,
        # sharing update_canvas()'s _dp_shape_seen tracking on this same
        # axis - this method re-runs on every drag/spinbox tweak, which must
        # not fight the user's current zoom/pan) so such circles clip
        # against its edges instead of autoscaling the axes out to fit them.
        # Matches the identical pin in Tab_Create_NavSignal.update_mask_overlay().
        shape_x, shape_y = self.dp.shape
        if (shape_x, shape_y) != getattr(self, '_dp_shape_seen', None):
            self.ax_dp.set_xlim(0, shape_y)
            self.ax_dp.set_ylim(shape_x, 0)
            self._dp_shape_seen = (shape_x, shape_y)
        self.canvas.draw_idle()

    def _try_start_vi_mask_drag(self, event):
        """Hit-test the live virtual-detector mask's center "+" / radius
        edges on ax_dp for a Ctrl+drag start - called from on_press before
        the reciprocal-space center's own Ctrl+Click (see on_press), so
        grabbing the mask's own handles always takes priority over setting
        the (independent) beam center. Returns True if a drag was started
        (the click should not fall through to recip-center logic), False
        otherwise (click missed every handle, or there's no mask to drag)."""
        self._vi_mask_drag_mode = None
        if not self.checkbox_useVirtualMask.isChecked() or self._vi_circle_out is None:
            return False
        cx = self.spinbox_vi_centerX.value()
        cy = self.spinbox_vi_centerY.value()
        r_in = self.spinbox_vi_rIn.value()
        r_out = self.spinbox_vi_rOut.value()

        # Hit-test in pixel space so the threshold doesn't depend on zoom.
        p0 = self.ax_dp.transData.transform((0, 0))
        p1 = self.ax_dp.transData.transform((1, 0))
        px_per_data = np.hypot(p1[0] - p0[0], p1[1] - p0[1]) or 1
        threshold_data = 8 / px_per_data

        dist = np.hypot(event.xdata - cx, event.ydata - cy)
        if dist < threshold_data:
            self._vi_mask_drag_mode = 'center'
        elif abs(dist - r_out) < threshold_data:
            self._vi_mask_drag_mode = 'r_out'
        elif r_in > 0 and abs(dist - r_in) < threshold_data:
            self._vi_mask_drag_mode = 'r_in'
        if self._vi_mask_drag_mode is None:
            return False

        # Blit setup - snapshot everything on ax_dp except the 3 dragged
        # artists, so on_motion can move just those on every mouse-move tick.
        for artist in (self._vi_circle_out, self._vi_circle_in, self._vi_center_marker):
            if artist is not None:
                artist.set_visible(False)
        self.canvas.draw()
        self._vi_mask_bg = self.canvas.copy_from_bbox(self.ax_dp.bbox)
        for artist in (self._vi_circle_out, self._vi_circle_in, self._vi_center_marker):
            if artist is not None:
                artist.set_visible(True)
        return True

    def _on_motion_vi_mask(self, event):
        """Apply the drag started by _try_start_vi_mask_drag - repositions
        the dragged artists directly and blits (mirrors the Navigator tab's
        on_motion_mask), rather than going through the spinboxes'
        valueChanged -> update_virtual_mask_overlay -> full redraw path on
        every tick (still updates the spinboxes' displayed values, with
        their change signals blocked, so that path isn't retriggered)."""
        if (self._vi_mask_drag_mode is None or event.inaxes != self.ax_dp
                or event.xdata is None or event.ydata is None):
            return
        if self._vi_mask_drag_mode == 'center':
            for sb, val in ((self.spinbox_vi_centerX, event.xdata),
                            (self.spinbox_vi_centerY, event.ydata)):
                sb.blockSignals(True)
                sb.setValue(int(round(val)))
                sb.blockSignals(False)
        else:
            cx = self.spinbox_vi_centerX.value()
            cy = self.spinbox_vi_centerY.value()
            r = int(round(np.hypot(event.xdata - cx, event.ydata - cy)))
            if self._vi_mask_drag_mode == 'r_out':
                r = max(r, self.spinbox_vi_rIn.value() + 1)
                self.spinbox_vi_rOut.blockSignals(True)
                self.spinbox_vi_rOut.setValue(r)
                self.spinbox_vi_rOut.blockSignals(False)
            elif self._vi_mask_drag_mode == 'r_in':
                r = min(r, self.spinbox_vi_rOut.value() - 1)
                self.spinbox_vi_rIn.blockSignals(True)
                self.spinbox_vi_rIn.setValue(r)
                self.spinbox_vi_rIn.blockSignals(False)

        cx = self.spinbox_vi_centerX.value()
        cy = self.spinbox_vi_centerY.value()
        self._vi_circle_out.set_center((cx, cy))
        self._vi_circle_out.set_radius(self.spinbox_vi_rOut.value())
        if self._vi_circle_in is not None:
            self._vi_circle_in.set_center((cx, cy))
            self._vi_circle_in.set_radius(self.spinbox_vi_rIn.value())
        self._vi_center_marker.set_offsets([[cx, cy]])

        self.canvas.restore_region(self._vi_mask_bg)
        self.ax_dp.draw_artist(self._vi_circle_out)
        if self._vi_circle_in is not None:
            self.ax_dp.draw_artist(self._vi_circle_in)
        self.ax_dp.draw_artist(self._vi_center_marker)
        self.canvas.blit(self.ax_dp.bbox)

    def _on_release_vi_mask(self):
        """End a virtual-mask drag (if one was active), sync the active list
        entry to the just-finished drag's final values (see
        _sync_active_detector_vi - also does one full, correct redraw:
        update_virtual_mask_overlay() was skipped during the drag itself,
        see _on_motion_vi_mask)."""
        was_dragging = self._vi_mask_drag_mode is not None
        self._vi_mask_drag_mode = None
        self._vi_mask_bg = None
        if was_dragging:
            self._sync_active_detector_vi()

    def compute_virtual_image(self):
        """Load the 4D signal pointed at above and compute its navigation
        image over the WHOLE scan, using the Sum/Variance mode and
        (optionally) the virtual detector(s) configured above - independent
        of whatever ROI/SAM2 mask is currently drawn elsewhere on this tab.
        This is the tab's only entry point for loading a signal (there is
        no separate "Load Signal" button) - the default Sum mode with no
        virtual mask reproduces what that used to do. Only the stale Nav.
        Image itself is cleared before recomputing (see
        _reset_nav_image_only) - an already-plotted diffraction pattern and
        its virtual-detector mask circles are left showing throughout.

        Runs as a real QProcess subprocess (worker_nav_img.py, the same
        script Tab_Create_NavSignal's own batch launches) rather than an
        in-process QThreadPool worker, specifically so Cancel can actually
        kill it outright - a large .tpx3 load through eventem used to just
        keep running in the background with no way to stop it (see
        cancel_running_work's docstring history)."""
        self._reset_nav_image_only()
        self.fn = self.lineEdit_dir_signal.text()
        if not self.fn or not os.path.exists(self.fn):
            qtw.QMessageBox.critical(self, 'No Signal Selected',
                'Select a 4D signal file first (see "4D Signal" above).')
            return
        self.scanSize = self.get_scan_size()
        self.dwellTime = self.spinbox_dwellTime.value()
        dtype = os.path.splitext(self.fn)[-1]
        if self._scan_size_required(dtype) and self.scanSize is None:
            self.logger.warning('Cannot load %s: scan size is required.', self.fn)
            self.message_box_scan_size_required(dtype)
            return

        mode = self.combo_virtualMode.currentText().lower()
        detectors = self.get_active_virtual_detectors() if self.checkbox_useVirtualMask.isChecked() else None
        self.logger.info('Computing virtual image (mode=%s, %s)...', mode,
                          f'{len(detectors)} detector(s)' if detectors else 'whole detector')
        self.button_computeVirtualImage.setDisabled(True)
        self._cancelling = False
        self.button_cancel.setEnabled(True)

        self._vi_temp_dir = tempfile.mkdtemp(prefix='edyssey_vimg_')
        scanSize_str = str(self.scanSize) if self.scanSize is not None else 'None'
        args = [self.fn, dtype, scanSize_str, str(self.dwellTime), '0', self._vi_temp_dir]
        args += [json.dumps(detectors) if detectors else 'None']
        args += [self.get_fn_pattern() or '']
        args += [str(self.get_detector_shape(self.fn))]
        args += [mode]
        program, arguments = worker_command('nav_img', args)
        self._process_vi = QProcess()
        self._process_vi.setProgram(program)
        self._process_vi.setArguments(arguments)
        self._process_vi.readyReadStandardError.connect(self._handle_error_vi)
        self._process_vi.finished.connect(self._handle_finished_vi)
        self._process_vi.errorOccurred.connect(self._process_failed_vi)
        self._process_vi.start()

    def _handle_error_vi(self):
        # eventem's own tpx3-loading progress lands on stderr, not an error -
        # buffered so it can be surfaced if this run turns out to have
        # actually failed (see _handle_finished_vi), same treatment as every
        # other worker subprocess's stderr in this app.
        self._stderr_buffer.append(self._process_vi)

    def _handle_finished_vi(self, exit_code=0, exit_status=0):
        """Collect the finished virtual-image worker process's result (a
        .npy path printed to stdout, like Tab_Create_NavSignal's batch
        workers) and display it - a virtual image IS a navigation image
        (just possibly computed with a non-default mode/mask instead of
        "Load Signal"'s plain whole-detector sum), so this reuses
        image_handler() to replace self.navImg/the "Nav. Image" panel
        exactly like Load Signal does, rather than showing up as a
        separate plot."""
        self.button_computeVirtualImage.setEnabled(True)
        self.button_cancel.setDisabled(True)
        if self._cancelling:
            self._stderr_buffer.discard(self._process_vi)
            shutil.rmtree(self._vi_temp_dir, ignore_errors=True)
            return
        raw_all = bytes(self._process_vi.readAllStandardOutput()).decode('utf-8', errors='replace')
        result_lines = [line.strip() for line in raw_all.splitlines() if line.strip()]
        raw = result_lines[-1] if result_lines else ''
        try:
            img = np.load(raw)
            try:
                os.remove(raw)
            except OSError:
                pass
        except Exception as e:
            stderr_text = self._stderr_buffer.pop_text(self._process_vi)
            self.logger.error('Failed to compute virtual image: %s%s', e,
                              f'\nWorker stderr:\n{stderr_text}' if stderr_text else '')
            qtw.QMessageBox.critical(self, 'Virtual Image Failed',
                (stderr_text or str(e)).strip().splitlines()[-1])
            shutil.rmtree(self._vi_temp_dir, ignore_errors=True)
            return
        self._stderr_buffer.discard(self._process_vi)
        shutil.rmtree(self._vi_temp_dir, ignore_errors=True)
        self.image_handler(img)
        self.logger.info('Virtual image computed successfully.')

    def _process_failed_vi(self, error):
        """Slot for the virtual-image worker's QProcess.errorOccurred (e.g.
        it failed to even start)."""
        self.button_computeVirtualImage.setEnabled(True)
        self.button_cancel.setDisabled(True)
        if self._cancelling:
            return
        self.logger.error('Virtual-image worker process failed to start (error code %s).', error)
        qtw.QMessageBox.critical(self, 'Process Error',
            'The virtual-image worker process failed to start.\n'
            'Check that Python is on PATH and worker_nav_img.py exists.')

    def compute_sum_dp_whole(self):
        """Sum every diffraction pattern of the WHOLE scan into one
        reference DP (see button_sumDpWhole) - independent of any ROI/SAM2
        mask, and without needing to draw/segment/threshold anything first."""
        self.fn = self.lineEdit_dir_signal.text()
        if not self.fn or not os.path.exists(self.fn):
            qtw.QMessageBox.critical(self, 'No Signal Selected',
                'Select a 4D signal file first (see "4D Signal" above).')
            return
        self.scanSize = self.get_scan_size()
        self.dwellTime = self.spinbox_dwellTime.value()
        dtype = os.path.splitext(self.fn)[-1]
        if self._scan_size_required(dtype) and self.scanSize is None:
            self.logger.warning('Cannot compute Sum DP for %s: scan size is required.', self.fn)
            self.message_box_scan_size_required(dtype)
            return
        self.logger.info('Computing Sum DP (whole signal) for %s...', self.fn)
        self.button_sumDpWhole.setDisabled(True)
        worker = WorkerThread_General(
            io.get_dp, 0, self.fn, dtype=dtype, scanSize=self.scanSize,
            dwellTime=self.dwellTime, roi=None, logger=self.logger,
            fn_pattern=self.get_fn_pattern(), det_shape=self.get_detector_shape(self.fn))
        worker.signals.results.connect(self._on_sum_dp_whole_computed)
        worker.signals.error.connect(self._on_sum_dp_whole_failed)
        self.threadpool.start(worker)

    def _on_sum_dp_whole_computed(self, result, index):
        """Display the whole-signal Sum DP (see compute_sum_dp_whole) the
        same way an ROI-drawn DP is displayed (get_dp) - also clears any
        stale ROI-crop-specific overlay (the SAM2 mask preview) so it
        doesn't linger over a DP that no longer corresponds to any
        particular crop."""
        self.button_sumDpWhole.setEnabled(True)
        self.dp = np.asarray(result)
        self.ax_nav_roi.set_title('ROI Image')
        self.img_display['seg_mask'].set_data(np.zeros((512, 512, 4)))
        self._reset_dp_clip_range()
        self.update_canvas(ax='dp')
        self.update_virtual_mask_overlay()
        self.logger.info('Sum DP (whole signal) computed successfully.')

    def _on_sum_dp_whole_failed(self, traceback_text, index):
        self.button_sumDpWhole.setEnabled(True)
        self.logger.error('Failed to compute Sum DP (whole signal):\n%s', traceback_text)
        qtw.QMessageBox.critical(self, 'Sum DP Failed',
            f'Computing the whole-signal Sum DP failed:\n\n{traceback_text.strip().splitlines()[-1]}')

#%% SAM2 segmentation
    def add_seg_point(self, event):
        """Add a SAM2 point prompt at the click location (left = positive,
        right = negative), plot it, and enable the relevant buttons."""
        if not hasattr(self, 'navImg'):
            self.logger.warning('SAM2 point requested but no image is loaded yet.')
            return
        p = [event.xdata, event.ydata]
        label = 1 if event.button == 1 else 0  # left = positive, right = negative
        self.seg_points.append(p)
        self.seg_labels.append(label)
        scatter_p = self.ax_nav.scatter(
            p[0], p[1], color='green' if label else 'red',
            marker='o', s=20, linewidth=1.25)
        self.scatter_plots.append(scatter_p)
        self.canvas.draw_idle()
        self.button_segment_image.setEnabled(True)
        self.button_clear_points.setEnabled(True)
        self.logger.info(
            'Added %s SAM2 point at (%.0f, %.0f) (%d point(s) total).',
            'positive' if label else 'negative', p[0], p[1], len(self.seg_points))

    def delete_last_seg_point(self):
        if not self.seg_points:
            return
        self.seg_points.pop()
        self.seg_labels.pop()
        p = self.scatter_plots.pop()
        try:
            p.remove()
        except Exception:
            self.logger.debug('SAM2 point scatter artist already removed.', exc_info=True)
        self.canvas.draw_idle()
        if not self.seg_points:
            self.button_segment_image.setDisabled(True)
        self.logger.info('Removed last SAM2 point (%d point(s) remaining).', len(self.seg_points))

    def clear_seg_points(self):
        had_points = bool(self.seg_points) or self.seg_mask is not None
        self.seg_points = []
        self.seg_labels = []
        self.seg_mask = None
        self._mask_source = None
        self._threshold_method = None
        for p in self.scatter_plots:
            try:
                p.remove()
            except Exception:
                self.logger.debug('SAM2 point scatter artist already removed.', exc_info=True)
        self.scatter_plots.clear()
        self.img_display['seg_mask'].set_data(np.zeros((512, 512, 4)))
        self.button_segment_image.setDisabled(True)
        self.canvas.draw_idle()
        if had_points:
            self.logger.info('Cleared SAM2 points and segmentation mask.')

    def clear_roi(self):
        """Remove the drawn ROI so it stops being used as the rectangle for
        diffraction-pattern extraction. Not reachable via the (currently
        deactivated) "Clear ROI/Box" button - only called internally, e.g.
        by reset_canvas() - draw a new ROI to replace the old one instead."""
        had_roi = self.roi is not None
        self.roi = None
        if self.rect is not None:
            try:
                self.rect.remove()
            except Exception:
                self.logger.debug('ROI rectangle artist already removed.', exc_info=True)
            self.rect = None
            self.canvas.draw_idle()
        if not self.seg_points:
            self.button_segment_image.setDisabled(True)
        if had_roi:
            self.logger.info('Cleared ROI/box.')

    def show_seg_mask(self, mask, title='SAM2 Segmentation'):
        """Display a mask (SAM2's or a real-space threshold's) overlaid on
        the *full* nav image on the "ROI Image" axis, as a guide to what's
        being summed - shown at full scale rather than cropped to the
        mask's own bounding box, so small/thin masks stay readable in
        context (matches the SAM2 tab's equivalent view; cropping tightly
        to the mask made this look "too zoomed in" for small selections).
        Also stores the mask's padded bounding box (x, y, w, h) in
        self.seg_roi - still used (mask-restricted) by compute_seg_dp()."""
        pad = 10
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        h_img, w_img = mask.shape
        y0 = max(0, int(y_idx[0]) - pad)
        y1 = min(h_img - 1, int(y_idx[-1]) + pad)
        x0 = max(0, int(x_idx[0]) - pad)
        x1 = min(w_img - 1, int(x_idx[-1]) + pad)
        self.seg_roi = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

        shape_y, shape_x = self.navImg.shape
        self.img_display['nav_roi'].set_data(self.navImg)
        self.img_display['nav_roi'].set_clim(
            vmin=io.nonzero_display_min(self.navImg), vmax=self.navImg.max())
        self.img_display['nav_roi'].set_extent([0, shape_x, shape_y, 0])
        # Only reset the view when the full nav image's own shape actually
        # changed (shares update_canvas()'s _navroi_shape_seen tracking,
        # since both display slots use the same axis/artist) - this method
        # re-runs on every Edge Detection tweak (Activate/Kernel/Directional/
        # Angle - see _preview_edge_mask), which must not fight the user's
        # current zoom/pan on every keystroke.
        if (shape_y, shape_x) != getattr(self, '_navroi_shape_seen', None):
            self.ax_nav_roi.set_xlim(0, shape_x)
            self.ax_nav_roi.set_ylim(shape_y, 0)
            self._navroi_shape_seen = (shape_y, shape_x)
        self.ax_nav_roi.set_title(title)

        # tab:orange stands out clearly against the viridis nav-image colormap,
        # unlike tab10's default blue (index 0), which blends into it.
        color = np.array([*mcolors.to_rgb('tab:orange'), 0.5])
        mask_image = mask.reshape(shape_y, shape_x, 1) * color.reshape(1, 1, -1)
        self.img_display['seg_mask'].set_data(mask_image)
        self.img_display['seg_mask'].set_extent([0, shape_x, shape_y, 0])
        self.canvas.draw_idle()

    def _get_seg_temp_dir(self):
        """Return the temp directory used to exchange SAM2 inputs/outputs
        with the segmentation subprocess, creating it if necessary."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(os.path.dirname(script_dir), 'EDyssey', 'io_utils', 'temp', 'roi4d_seg')
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def segment_image(self):
        """Run SAM2's single-image predictor on the currently loaded nav
        image using the point prompts added via Shift+Click. Otherwise this
        mirrors the SAM2 tab's "Seg Image" feature, just without object
        tracking.

        Box-prompt support (combining points with the last-drawn ROI as a
        box, the way the box narrows the region and points refine it
        further) is temporarily deactivated - see button_clear_roi/
        clear_roi's own docstring."""
        if not self.seg_points:
            self.logger.warning('Segment Image requested but no points have been added.')
            qtw.QMessageBox.critical(self, 'No Points Added',
                'Hold Shift and click on the image to add at least one point '
                '(left click = positive, right click = negative) before segmenting.')
            return

        self.logger.info('Starting SAM2 image segmentation (%d point(s))...', len(self.seg_points))
        self.button_segment_image.setDisabled(True)
        self._cancelling = False
        self.button_cancel.setEnabled(True)

        path_seg = self._get_seg_temp_dir()
        img_8bit = io.convert_img_to_8bit(self.navImg)
        # worker_sam.py's 'image' branch takes a {obj_id: {points,labels}}
        # map (batches multiple objects against one image encoding - see
        # Tab_SAM2.initiate_image_segmentation) - this tab only ever
        # segments one object at a time, so obj_id 0 is just a placeholder.
        seg_input = {'image': img_8bit,
                    'objects': {0: {'points': np.array(self.seg_points),
                                     'labels': np.array(self.seg_labels)}}}
        with open(os.path.join(path_seg, 'seg_input.pkl'), 'wb') as f:
            pickle.dump(seg_input, f)

        program, arguments = worker_command('sam', ['image', path_seg, '0'])
        self._process_sam = QProcess(self)
        self._process_sam.setProgram(program)
        self._process_sam.setArguments(arguments)
        self._process_sam.readyReadStandardError.connect(self._handle_error_sam)
        self._process_sam.finished.connect(self._handle_finished_sam)
        self._process_sam.errorOccurred.connect(self._process_failed_sam)
        self._process_sam.start()

    def _process_failed_sam(self, error):
        """Slot for the SAM2 subprocess's QProcess.errorOccurred: report
        that it failed to start (skipped if the user cancelled)."""
        self.button_segment_image.setEnabled(True)
        self.button_cancel.setDisabled(True)
        if self._cancelling:
            return
        self.logger.error('SAM2 segmentation QProcess error occurred: %s', error)
        qtw.QMessageBox.critical(self, 'Process Error',
            'The SAM2 segmentation process failed to start.\n'
            'Check that Python is on PATH and worker_sam.py exists.')

    def _handle_error_sam(self):
        # SAM2/PyTorch write progress bars and warnings to stderr that
        # aren't necessarily errors - real failures surface via the
        # JSON-decode check in _handle_finished_sam below.
        self._stderr_buffer.log_info(self._process_sam, self.logger, 'SAM2')

    def _handle_finished_sam(self, exit_code=0, exit_status=0):
        """Slot for the SAM2 subprocess's QProcess.finished: parse its
        JSON+npz result into self.seg_mask (or report a failure), then
        refresh the mask overlay/DP via _refresh_edge_mask()."""
        self.button_segment_image.setEnabled(True)
        self.button_cancel.setDisabled(True)
        if self._cancelling:
            return
        text = bytes(self._process_sam.readAllStandardOutput()).decode('utf-8')
        try:
            result = json.loads(text.strip())
            with np.load(result['path']) as f:
                mask = f['obj_0']
        except json.JSONDecodeError:
            self.logger.error('Could not decode SAM2 segmentation result: %s', text)
            qtw.QMessageBox.warning(self, 'SAM2 Error',
                f'Could not decode SAM2 output. Check console for details.\n'
                f'Raw output (first 200 chars): {text[:200]}')
            return

        if not mask.any():
            self.logger.warning('SAM2 returned an empty mask (no pixels selected).')
            qtw.QMessageBox.warning(self, 'Empty Mask',
                'SAM2 did not select any pixels. Try adjusting or adding more points.')
            return

        # Kept raw (un-eroded) so toggling "Edge Only"/the kernel size later
        # can re-derive from the original SAM2 result instead of compounding
        # erosion on an already-eroded mask - see _refresh_edge_mask().
        self.seg_mask = mask
        self._mask_source = 'sam2'
        self.button_clear_points.setEnabled(True)
        self.logger.info('SAM2 image segmentation completed successfully.')
        self._refresh_edge_mask()

    def compute_seg_dp(self, mask):
        """Sum diffraction patterns over the SAM2 mask's scan positions and
        display the result on the DP axis, immediately after segmentation -
        the masked-DP equivalent of drawing a rectangle ROI."""
        self.logger.info('Computing diffraction pattern summed over SAM2 mask...')
        dtype = os.path.splitext(self.fn)[-1]
        worker = Worker_CalculateDP_Mask(self.fn, self.seg_roi, mask, dtype,
                                         self.scanSize, self.dwellTime, self.get_fn_pattern(),
                                         self.get_detector_shape(self.fn))
        worker.signals.result.connect(self.get_dp_from_mask)
        self.threadpool.start(worker)

    def get_dp_from_mask(self, dp):
        self.dp = dp
        self.ax_dp.set_title('DP (SAM2 Mask)')
        self._reset_dp_clip_range()
        self.update_canvas(ax='dp')
        self.update_virtual_mask_overlay()

#%% Summed DP from threshold
    def open_threshold_dialog(self):
        """Open the ThresholdDialog popup to check/adjust the real-space
        threshold on the loaded navigation image before committing to the
        actual (file-reading) summed-DP computation - shared with the "Make
        Nav. Sig." tab's identical feature."""
        if not hasattr(self, 'navImg'):
            qtw.QMessageBox.critical(self, 'No Image',
                'Load a 4D signal first - the threshold is applied to its '
                'navigation image.')
            return
        dlg = ThresholdDialog(self, self.navImg, self.fn)
        if dlg.exec_() == qtw.QDialog.Accepted:
            # Kept raw (un-eroded) - see _refresh_edge_mask().
            self.seg_mask = dlg.mask
            self._mask_source = 'threshold'
            self._threshold_method = dlg.combo_threshMethod.currentText()
            self._refresh_edge_mask()

    def _preview_edge_mask(self):
        """Live preview of the edge-detection mask (Activate/Directional/
        Revert Mask/Kernel/Angle) - just the on-screen overlay, no
        diffraction-pattern recomputation. Wired to those controls directly
        so tuning them stays fast; button_computeEdgeDp (_refresh_edge_mask)
        is the separate, deliberate action that re-reads the masked DP from
        disk with whatever mask these controls currently describe."""
        if self.seg_mask is None or self._mask_source is None:
            return
        mask = self.apply_edge_mask(self.seg_mask)
        if not mask.any():
            self.logger.warning(
                'Edge Only removed the entire mask (kernel too large for this mask\'s size).')
            return
        self.show_seg_mask(mask, title='Threshold Mask' if self._mask_source == 'threshold'
                                  else 'SAM2 Segmentation')

    def _refresh_edge_mask(self):
        """Re-derive the mask from the last raw SAM2 or threshold result
        (self.seg_mask) with the current edge-detection settings and
        (re-)run its display+DP computation - the disk-reading half of
        _preview_edge_mask, called right after a new mask is produced
        (segmentation finishes, a threshold is accepted) and by
        button_computeEdgeDp ("Compute") once the user is done tuning
        Activate/Directional/Revert Mask/Kernel/Angle above."""
        if self.seg_mask is None or self._mask_source is None:
            qtw.QMessageBox.critical(self, 'No Mask',
                'Segment an image (SAM2) or set a threshold (Summed DP from Threshold) '
                'first - Edge Detection refines one of those, it doesn\'t create one.')
            return
        mask = self.apply_edge_mask(self.seg_mask)
        if not mask.any():
            self.logger.warning(
                'Edge Only removed the entire mask (kernel too large for this mask\'s size).')
            return
        if self._mask_source == 'sam2':
            self.show_seg_mask(mask)
            self.compute_seg_dp(mask)
        elif self._mask_source == 'threshold':
            self.compute_sum_dp_from_threshold(mask, self._threshold_method)

    def compute_sum_dp_from_threshold(self, mask, method):
        """Sum diffraction patterns only at the scan positions in `mask`
        (confirmed via the ThresholdDialog popup), instead of a rectangular
        ROI - reuses the same masked-DP worker as the SAM2 segmentation path."""
        # Show the mask on the "ROI Image" axis right away, as a guide to
        # what's about to be summed - not gated on the (background) DP
        # computation finishing.
        self.show_seg_mask(mask, title='Threshold Mask')

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        y0, y1 = int(y_idx[0]), int(y_idx[-1])
        x0, x1 = int(x_idx[0]), int(x_idx[-1])
        roi = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        dtype = os.path.splitext(self.fn)[-1]
        self.logger.info(
            'Computing summed DP from %s-thresholded scan positions...', method)
        self.button_sumDpFromThreshold.setDisabled(True)
        self._cancelling = False
        self.button_cancel.setEnabled(True)
        worker = Worker_CalculateDP_Mask(self.fn, roi, mask, dtype, self.scanSize,
                                         self.dwellTime, self.get_fn_pattern(),
                                         self.get_detector_shape(self.fn), patch_mode=True)
        worker.signals.result.connect(self._on_sum_dp_from_threshold_computed)
        worker.signals.error.connect(self._on_sum_dp_from_threshold_failed)
        self.threadpool.start(worker)

    def _on_sum_dp_from_threshold_computed(self, dp):
        self.button_sumDpFromThreshold.setEnabled(True)
        self.button_cancel.setDisabled(True)
        self.dp = dp
        self.ax_dp.set_title('Summed DP (thresholded)')
        self._reset_dp_clip_range()
        self.update_canvas(ax='dp')
        self.update_virtual_mask_overlay()

    def _on_sum_dp_from_threshold_failed(self, traceback_text):
        self.button_sumDpFromThreshold.setEnabled(True)
        self.button_cancel.setDisabled(True)
        if self._cancelling:
            return
        # Without this, a failure in the background computation would leave
        # "Summed DP from Threshold..." disabled forever, with no way to
        # retry and no visible sign anything went wrong.
        self.logger.error('Failed to compute summed DP from threshold:\n%s', traceback_text)
        qtw.QMessageBox.critical(self, 'Summed DP Failed',
            'Computing the summed DP failed - see the log for details.')

    def cancel_running_work(self):
        """Stop the running SAM2 segmentation, virtual-image computation, or
        DP computation, and suppress the error popups that killing those
        workers would otherwise trigger.

        Compute Virtual Image and the SAM2 segmentation each run as a real
        QProcess subprocess, so both are killed outright. Sum DP (Whole
        Signal)/Summed DP from Threshold/SAM2 mask-DP still run as
        QThreadPool QRunnables, which Qt has no way to forcibly interrupt
        once started (only queued-but-not-started ones can be dropped by
        threadpool.clear()) - an in-flight one of those will still finish in
        the background, its result simply ignored. Either way, every
        button a computation could have left disabled is force-re-enabled
        below regardless of whether its own worker ever gets the chance to
        call back and do that itself - otherwise a runnable that
        threadpool.clear() drops while still queued (never starts, so its
        result/error signal never fires at all) would leave that button
        disabled permanently, with no way to start a new computation short
        of restarting the app."""
        self._cancelling = True
        self.threadpool.clear()
        n_killed = 0
        if hasattr(self, '_process_sam') and self._process_sam.state() != QProcess.NotRunning:
            self._process_sam.kill()
            n_killed += 1
        if hasattr(self, '_process_vi') and self._process_vi.state() != QProcess.NotRunning:
            self._process_vi.kill()
            n_killed += 1
        self.button_computeVirtualImage.setEnabled(True)
        self.button_sumDpWhole.setEnabled(True)
        self.button_sumDpFromThreshold.setEnabled(True)
        self.button_cancel.setDisabled(True)
        self.logger.warning('Cancelled by user (%d running process(es) killed).', n_killed)
        qtw.QMessageBox.information(self, 'Cancelled',
            'Running computation was cancelled.\n\n'
            'A Sum DP/DP computation already running in the background will '
            'still finish silently (its result is discarded) - the virtual-image '
            'and SAM2 segmentation processes were stopped outright.')

    def cleanup(self):
        """Release resources held by this tab. Called by MainWindow.closeEvent
        so repeated runs of the app in the same console/kernel don't leave
        threadpools, running subprocesses, and matplotlib figures alive."""
        self.threadpool.clear()
        if hasattr(self, '_process_sam'):
            self._process_sam.kill()
        if hasattr(self, '_process_vi'):
            self._process_vi.kill()
        self.log_console.disconnect_log()
        plt.close(self.figure)

class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(object)  # Formatted traceback string, emitted on failure

class Worker_CalculateDP(QRunnable):
    """Background QRunnable that loads a rectangular ROI's diffraction
    pattern (and its summed nav-image crop), emitting both via
    signals.result."""
    def __init__(self, fn, roi, scanSize, dwellTime, fn_pattern=None, det_shape=(512, 512)):
        super().__init__()
        self.logger = get_tab_logger('Tab_ROI_on_4D')
        self._tic = perf_counter()
        self.logger.info('calculating the dp...')
        self.fn = fn
        self.roi = roi
        self.scanSize = scanSize
        self.dwellTime = dwellTime
        self.fn_pattern = fn_pattern
        self.det_shape = det_shape

        self.signals = WorkerSignals()

    def run(self):
        # load_signal/load_tpx3/load_hs/load_hdf5 no longer return HyperSpy
        # Signal2D objects (nor a (signal, file_handle) tuple for hdf5 - the
        # file is now closed internally): .tpx3 returns the raw eventem.Roi
        # object itself, everything else a plain numpy/dask array already
        # cropped to `roi`.
        dtype = os.path.splitext(self.fn)[-1]
        try:
            if dtype == '.tpx3':
                x, y, w, h = self.roi
                roi_obj = io.load_tpx3(self.fn, roi=self.roi, scanSize=self.scanSize,
                                       dwellTime=self.dwellTime, fn_pattern=self.fn_pattern,
                                       logger=self.logger, get_4d=False,
                                       det_shape=self.det_shape)
                dp = np.array(roi_obj.Roi_diffraction_pattern).reshape(
                    self.det_shape[1], self.det_shape[0])
                navImg_cut = np.array(roi_obj.Roi_scan_image).reshape(h, w)
            else:
                s_cut = io.load_signal(self.fn, roi=self.roi,
                                       scanSize=self.scanSize, dwellTime=self.dwellTime,
                                       logger=self.logger, fn_pattern=self.fn_pattern)
                navImg_cut = s_cut.sum(axis=(2,3))
                dp = s_cut.sum(axis=(0,1))
                if hasattr(dp, 'compute'): # lazy signals
                    with ProgressBar():
                        dp = dp.compute()
                if hasattr(navImg_cut, 'compute'): # lazy signals
                    navImg_cut = navImg_cut.compute()
        except Exception:
            self.logger.exception('Failed to calculate diffraction pattern after %s.',
                                   io.format_duration_hms(perf_counter() - self._tic))
            return
        self.signals.result.emit((dp, navImg_cut))
        self.logger.info('Diffraction pattern calculated successfully in %s.',
                          io.format_duration_hms(perf_counter() - self._tic))
        # self.signals.finished.emit()

class Worker_CalculateDP_Mask(QRunnable):
    """Sum diffraction patterns at the scan positions where an arbitrary
    (e.g. SAM2-segmented) mask is True, restricted to `roi` (the mask's
    bounding box) for efficiency. Reuses the same per-format masked loaders
    already used by the CV2/SAM2 tabs' 3DED extraction.

    `patch_mode` (.tpx3, smart-scanned frames only - ignored for a normal
    dense-raster scan, which always applies the mask directly via eventem
    regardless of this flag): the SAM2 segmentation-DP caller passes a
    naturally small `roi` (the segmentation's own bounding box) and leaves
    this False; the "Summed DP from Threshold" caller has no such small
    ROI (a threshold can select scan positions anywhere), so it passes
    True to split the mask into small per-patch extractions instead of one
    covering its full (potentially huge/scattered) bounding box - see
    load_dp/load_tpx3_patches in worker_extract_frame.py."""
    def __init__(self, fn, roi, mask, dtype, scanSize, dwellTime, fn_pattern=None,
                det_shape=(512, 512), patch_mode=False):
        super().__init__()
        self.logger = get_tab_logger('Tab_ROI_on_4D')
        self._tic = perf_counter()
        self.logger.info('calculating the dp from mask...')
        self.fn = fn
        self.roi = roi
        self.mask = mask
        self.dtype = dtype
        self.scanSize = scanSize
        self.dwellTime = dwellTime
        self.fn_pattern = fn_pattern
        self.det_shape = det_shape
        self.patch_mode = patch_mode
        self.signals = WorkerSignals()

    def run(self):
        try:
            dp = load_dp(self.fn, roi=self.roi, mask=self.mask, dtype=self.dtype,
                        scanSize=self.scanSize, dwellTime=self.dwellTime,
                        fn_pattern=self.fn_pattern, det_shape=self.det_shape,
                        patch_mode=self.patch_mode)
            if hasattr(dp, 'compute'):
                dp = dp.compute()
        except Exception:
            import traceback
            self.logger.exception(
                'Failed to calculate mask-based diffraction pattern after %s.',
                io.format_duration_hms(perf_counter() - self._tic))
            self.signals.error.emit(traceback.format_exc())
            return
        self.signals.result.emit(dp)
        self.logger.info(
            'Mask-based diffraction pattern calculated successfully in %s.',
            io.format_duration_hms(perf_counter() - self._tic))
# =============================================================================
# if __name__ == "__main__":
#     app = qtw.QApplication(sys.argv)
#     
#     # Create and show the main window
#     window = Tab_ROI_on_4D()
#     window.show()
#     
#     # Start the event loop
#     sys.exit(app.exec_())
# =============================================================================
