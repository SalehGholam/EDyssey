# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:55:17 2024

@author: SGholam
"""

import os
import json
from glob import glob
import sys
from PyQt5.QtCore import (Qt, QThreadPool)
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
import numpy as np
import py4DTomo.io_utils as io
import hyperspy.api as hs
from hyperspy.api import load
import py4DTomo.tracking_utils as tr
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as patches
import datetime
from copy import deepcopy
from .worker_thread import WorkerThread_General
from .logging_utils import get_tab_logger, LogConsole
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
class Tab_Tracking_CV2(qtw.QWidget):
# class Tab_Create_NavSignal(qtw.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.logger = get_tab_logger('Tab_Tracking_CV2')

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

        # threadpool to use in the entire tab
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(max(1, os.cpu_count() - 2))
        self._tracking_lock = threading.Lock()
        self._cancelling = False  # set by cancel_running_work(); suppresses error popups it causes

    def init_widget(self):
        button_w = 110
        button_h_sml = 30
        button_h_lrg = 50
        height_userInput = 200
        width_userInput = 320
        self.layout = qtw.QHBoxLayout(self)
        self.setLayout(self.layout)
        self._splitter = qtw.QSplitter(Qt.Horizontal)
        self.layout.addWidget(self._splitter)

        self._left_widget = qtw.QWidget()
        self._left_widget.setFixedWidth(width_userInput)
        self._splitter.addWidget(self._left_widget)

        # layout top
        layout_userInput = qtw.QVBoxLayout(self._left_widget)
        spacer = qtw.QSpacerItem(40, 20, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
        #%% directory
        self.box_dir = qtw.QGroupBox('Directories', self)
        self.box_dir.setFixedHeight(height_userInput)
        
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
        #%% box for scales
        layout_loadSignal = qtw.QHBoxLayout()
        self.box_scale = qtw.QGroupBox('Scale bars')
        layout_box_scale = qtw.QVBoxLayout()
        self.box_scale.setLayout(layout_box_scale)
        # layout_3ded.addWidget(self.box_scale)
        layout_loadSignal.addWidget(self.box_scale)
        
        # real space
        self.double_validator = QDoubleValidator(0.0, 1e5, 5)
        layout_scale_real = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_real)
        label_scale_real = qtw.QLabel('Real (nm)')
        label_scale_real.setFixedWidth(55)
        layout_scale_real.addWidget(label_scale_real)
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        layout_scale_real.addWidget(self.lineEdit_scale_real)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        # reciprocal space
        layout_scale_recip = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_recip)
        label_scale_recip = qtw.QLabel('Recip. (\u00C5<sup>-1</sup>)')
        label_scale_recip.setFixedWidth(55)
        layout_scale_recip.addWidget(label_scale_recip)
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        layout_scale_recip.addWidget(self.lineEdit_scale_recip)
        self.lineEdit_scale_recip.setValidator(self.double_validator)

        self.checkbox_autoCenterDp = qtw.QCheckBox('Auto-center')
        self.checkbox_autoCenterDp.setChecked(True)
        self.checkbox_autoCenterDp.setToolTip(
            'When checked, the reciprocal-space rings are re-centered on the '
            'direct beam automatically (found via a large-sigma blur) after '
            'every redraw. When unchecked, hold Ctrl and click on the DP '
            'plot to set the center manually.')
        layout_box_scale.addWidget(self.checkbox_autoCenterDp)
        self.checkbox_autoCenterDp.stateChanged.connect(lambda: self.update_scalebar('reciprocal'))
        self.dp_center = None  # (x, y) - auto-found or last manually-clicked center

        self.lineEdit_scale_recip.textChanged.connect(lambda: self.update_scalebar('reciprocal'))
        self.lineEdit_scale_real.textChanged.connect(lambda: self.update_scalebar('real'))
        
        # layout_loadSignal.addItem(spacer)
        
        layout_dir.addLayout(layout_loadSignal)
        layout_load_buttons = qtw.QVBoxLayout()
        layout_loadSignal.addLayout(layout_load_buttons)

        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        self.button_loadNavigation.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        layout_load_buttons.addWidget(self.button_loadNavigation)
        self.button_loadNavigation.clicked.connect(self.load_navSignal)

        self.button_loadSavedAnalysis = qtw.QPushButton('Load Saved\nAnalysis')
        self.button_loadSavedAnalysis.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        layout_load_buttons.addWidget(self.button_loadSavedAnalysis)
        self.button_loadSavedAnalysis.clicked.connect(self.load_saved_analysis)
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
        
        # bottom 
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
        
        self.button_track = qtw.QPushButton('Track!')
        layout_featureBottom.addWidget(self.button_track, alignment=Qt.AlignCenter)
        self.button_track.clicked.connect(self.track_rois)
        self.button_track.setDisabled(True)
        #%% box thresh and 3DED
        layout_3ded = qtw.QVBoxLayout()
        layout_userInput.addLayout(layout_3ded)

        #%% box scan size (raw scan dims used to extract DPs for 3DED)
        self.box_scanSize = qtw.QGroupBox('Scan Size')
        layout_box_scanSize = qtw.QVBoxLayout()
        self.box_scanSize.setLayout(layout_box_scanSize)
        layout_3ded.addWidget(self.box_scanSize)

        layout_scanSize_row1 = qtw.QHBoxLayout()
        layout_box_scanSize.addLayout(layout_scanSize_row1)

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

        # Deliberately kept outside box_3ded (whose contents are disabled/
        # enabled together elsewhere) so it stays clickable regardless of
        # tracking/extraction state - it needs to work throughout both.
        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.setFixedSize(button_w, button_h_sml)
        self.button_cancel.setStyleSheet("background-color: red; color: white;")
        self.button_cancel.setDisabled(True)
        self.button_cancel.setToolTip(
            'Stop the running tracking or 3DED extraction. Already-running '
            'background computations finish silently; their results are discarded.')
        layout_3ded.addWidget(self.button_cancel, alignment=Qt.AlignLeft)
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

        layout_threadNo.addItem(spacer)
        
        layout_extract = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_extract)
        
        self.button_3ded = qtw.QPushButton('Extract!')
        layout_extract.addWidget(self.button_3ded)
        self.button_3ded.setFixedSize(button_w, button_h_lrg)
        self.button_3ded.clicked.connect(self.extract_3ded)
        
        self.button_save_results = qtw.QPushButton('Save Results')
        layout_extract.addWidget(self.button_save_results)
        self.button_save_results.setFixedSize(button_w, button_h_lrg)
        self.button_save_results.clicked.connect(self.save_results)
        
        self.disable_3ded_widgets(True)
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
        axes = ['nav', 'track', 'dp']

        for i, ax in enumerate([self.ax_nav, self.ax_track, self.ax_dp]):
            self.img_display[axes[i]] = ax.imshow(self.img_zero, cmap='viridis')
            # ax.set_axis_off()
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        self.ax_track.set_xlabel(
            'Select the reference ROI and Hold "ctrl" + Drag to draw ROIinROI', fontsize=7)
        self.ax_nav.set_xlabel(
            'Hold "ctrl" + Left Click+Drag => New ROI\n'
            'Hold "ctrl" + Right Click => Add init to existing ROI', fontsize=7)
        self.ax_nav.xaxis.label.set_visible(True)
        self.ax_track.xaxis.label.set_visible(True)
        self.ax_dp.xaxis.label.set_visible(True)

        # The Ctrl+Scroll zoom hint applies to every axis on its canvas, so
        # it's a figure-wide supxlabel (one per figure) rather than repeated
        # per-axis text.
        self.figure_nav.supxlabel('Hold "Ctrl" + Scroll wheel to zoom the axis under the cursor',
                                  fontsize=7)
        self.figure_extract.supxlabel('Hold "Ctrl" + Scroll wheel to zoom the axis under the cursor',
                                      fontsize=7)

        self.ax_mask.set_axis_off()
        self.img_display['img_mask'] = self.ax_mask.imshow(self.img_zero, cmap='gray')
        self.img_display['mask'] = self.ax_mask.imshow(self.img_zero, cmap='viridis', alpha=0.1)
        self.img_display['dp'].set_norm(SymLogNorm(linthresh=1))
        self.img_display['dp'].set_cmap('inferno')

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

        # keyboard shortcuts
        QShortcut(QKeySequence('Ctrl+O'), self, self.button_loadNavigation.click)
        QShortcut(QKeySequence('Ctrl+Shift+O'), self, self.button_loadSavedAnalysis.click)
        QShortcut(QKeySequence('Ctrl+T'), self, self.button_track.click)
        QShortcut(QKeySequence('Ctrl+E'), self, self.button_3ded.click)
        QShortcut(QKeySequence('Ctrl+S'), self, self.button_save_results.click)
    #%% load data
    def show_dialog(self, f):
        sender = self.sender()
        if sender == self.button_dir_navSignal:
            file_filter = "supported signals (*.zspy *.hspy);;All Files (*)"
            # path = qtw.QFileDialog.getOpenFileNames(self, "Select 4D Signals Folder", '', file_filter)
            path = qtw.QFileDialog.getOpenFileName(self, "Select 4D Signals Folder", '', file_filter)
            if path and os.path.isfile(path[0]):
                self.lineEdit_dir_navSignal.setText(path[0])
                path_save = os.path.join(os.path.dirname(path[0]), '5DED Analysis')
                self.lineEdit_dir_save.setText(path_save)
                
        elif sender == self.button_dir_4dSignals:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select 4D Folder")
            # if path and os.path.isdir(path[0]):
            if path:
                self.metadata_path_override = None  # new folder - re-derive comment.txt location
                self.lineEdit_dir_4d.setText(path)
                if any(f.endswith('.tpx3') for f in os.listdir(path)):
                    self.load_metadata(silent=True)

        elif sender == self.button_dir_save:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select Destination Folder")
            if path and os.path.isdir(path[0]):
                self.lineEdit_dir_save.setText(path)

    def activate_lineEdit_scanSize(self):
        if self.checkbox_scanSize.isChecked():
            self.lineEdit_scanSize_x.setDisabled(True)
            self.lineEdit_scanSize_y.setDisabled(True)
        else:
            self.lineEdit_scanSize_x.setEnabled(True)
            self.lineEdit_scanSize_y.setEnabled(True)

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

    def load_navSignal(self):
        def get_signal(fn):
            return load(fn)

        fn = self.lineEdit_dir_navSignal.text()
        if not os.path.isfile(fn):
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
        for p in self.ax_nav.patches:
            try:
                p.remove()
            except: pass
        for p in self.ax_track.patches:
            try:
                p.remove()
            except: pass
        for p in self.patches_axNav:
            try:
                p.remove()
            except: pass
        for p in self.patches_axTrack:
            try:
                p.remove()
            except: pass
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
        
    
    def initiate_processing(self, result, index):
        self.s = result
        self.s_8bit = io.convert_to_8bit(self.s)
        self.nav_imgs_raw = self.s.data
        self.nav_imgs = deepcopy(self.s_8bit.data)
        self.dp_center = None  # a new signal may have a different DP shape/center
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
        self.slider_imgNo.setRange(0, len(self.nav_imgs)-1)
        self.logger.info('No. of Images: %s', len(self.nav_imgs))
        
        self.button_reset_rois.setEnabled(True)
        # self.button_cur_roi.setEnabled(True)
        self.button_track.setEnabled(True)

    def load_saved_analysis(self):
        """Restore a previously saved analysis folder (produced by
        save_results): the navigation signal, per-ROI tracking (init points/
        out_rois/mask), and any extracted diffraction patterns."""
        path = qtw.QFileDialog.getExistingDirectory(
            self, "Select Saved Analysis Folder", self.lineEdit_dir_save.text())
        if not path:
            return
        fn_nav = os.path.join(path, 'navigation_signal.hspy')
        if not os.path.isfile(fn_nav):
            qtw.QMessageBox.critical(self, 'Navigation Signal Not Found',
                f'Cannot find navigation_signal.hspy in:\n{path}\n\n'
                'This folder does not look like a saved analysis (or it was '
                'saved before this feature was added).')
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
        # out_rois/mask are saved unconditionally even when still None (as a
        # pickled 0-d object array), so both "file missing" and "file holds
        # a pickled None" need to come back as None here.
        if not os.path.isfile(fn):
            return None
        arr = np.load(fn, allow_pickle=True)
        if arr.ndim == 0 and arr.item() is None:
            return None
        return arr

    def _load_saved_analysis_worker(self, path, fn_nav):
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
        return s, rois, path

    def _on_saved_analysis_loaded(self, result, index):
        s, rois, path = result
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
        self.logger.info('Reset all ROIs.')
#%% canvas functions    
    def jump_to_frame_no(self):
        num = int(self.lineEdit_imgNo.text())
        self.slider_imgNo.setValue(num)
    
    def update_canvas(self, imgNo=None):
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
            if idx in self.df_rois.dp.dropna().index:
                try:
                    self.update_ax(self.df_rois.loc[idx, 'dp'][imgNo], 'dp', self.ax_dp)
                    #TODO set the content size after getting the data
                except:
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
        # Rendering is deferred to the single canvas.draw()/draw_idle() call
        # at the end of update_canvas(), rather than a redraw per axis here.
        self.img_display[img_disp].set_data(img)
        ax.set_title(title)
        self.img_display[img_disp].set_clim(vmin=img.min(), vmax=img.max())

    def draw_rois_in(self, imgNo):
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
                except:
                    pass
        # Rendering is deferred to the single canvas.draw()/draw_idle() call
        # at the end of update_canvas(), rather than a blit here.

    def update_ax_mask(self, img_roi, img_mask):
        shape_x, shape_y = img_mask.shape
        self.img_display['img_mask'].set_data(img_roi)
        try:
            self.img_display['img_mask'].set_clim(vmin=img_roi.min(), vmax=img_roi.max())
        except:
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
            if self.checkbox_autoCenterDp.isChecked():
                try:
                    self.dp_center = io.find_dp_center_blurred(dp_array)
                except Exception:
                    self.logger.exception('Auto-centering the diffraction pattern failed.')
            self._dp_recip_circles = io.draw_reciprocal_scale_circles(
                self.ax_dp, self.lineEdit_scale_recip.text(), shape,
                center=self.dp_center, old_artists=getattr(self, '_dp_recip_circles', None))
            if self.checkbox_autoCenterDp.isChecked():
                self.ax_dp.set_xlabel('Circle center: auto (large-sigma blur)', fontsize=6)
            else:
                self.ax_dp.set_xlabel('Circle center: manual - Ctrl+Click DP plot to set', fontsize=6)
            # The circles are static across frames like the scale bars above.
            self._bg_extract = None
            self.canvas_extract.draw_idle()

    def threshold_img(self, img, roi, thresh_method, thresh_offset, mode='full'):
        # thresh_method = self.combo_thresh_method.currentText()
        blur_kernel = int(self.combo_blur.currentText())
        threshold_methods = {'otsu': threshold_otsu, 'li': threshold_li, 
                             'yen': threshold_yen, 'mean': threshold_mean}
        threshold_func = threshold_methods[thresh_method]

        y,x,h,w = roi
        img_blur = io.gaussian_blur(img, blur_kernel)
        img_cut = img[x:x+w, y:y+h]
        if mode == 'full':
            th = threshold_func(img_blur)
        elif mode == 'roi':
            th = threshold_func(img_cut)
        thresh_offset = thresh_offset / 100
        thresh = thresh_offset * th
        img_mask = img_blur >= thresh
        img_mask = img_mask[x:x+w, y:y+h]
        return img_mask, img_cut

    def on_press(self, event):
        if event.inaxes not in (self.ax_nav, self.ax_track) or 'ctrl' not in event.modifiers:
            # Plain click/drag is reserved for the navigation toolbar's
            # Pan/Zoom tool (and the scroll-wheel zoom) so images can be
            # zoomed into; hold "ctrl" to draw/edit a ROI instead.
            self.press = None
            return
        # Mouse press event: record the starting point
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
        # Mouse motion event: update the rectangle size as the mouse moves
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
            except:
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
        # Mouse release event: finalize the rectangle
        if self.press is None or event.inaxes is None:
            return
        # Finalize the rectangle and store it
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
            init = eval(item.text(2))
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
            "Duplicate this item: copies its use flag, start frame(s)/box(es), "
            "end frame, and reference into a new item (not yet tracked/extracted)")

        def duplicate_row():
            index = self.tree_objects.indexOfTopLevelItem(item)
            orig_idx = self.df_rois.index[index]
            new_idx = 1
            while new_idx in self.df_rois.index:
                new_idx += 1
            orig = self.df_rois.loc[orig_idx]
            self.df_rois.loc[new_idx] = [
                orig['use'], deepcopy(orig['init']), deepcopy(orig['in_rois']),
                orig['end'], orig['ref'], None, None, None]
            self.add_item_tree(idx=new_idx, init=deepcopy(orig['init']), end=orig['end'],
                               ref=orig['ref'], use=orig['use'])
            self.update_canvas()
            self.logger.info('Duplicated ROI %s as ROI %s.', orig_idx, new_idx)

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
        checked = []
        for i in range(self.tree_objects.topLevelItemCount()):
            item = self.tree_objects.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                checked.append(item.text(1))
        self.logger.info("Checked Items: %s", checked)
    
    def load_spinner(self,):
        # Create spinner as a floating overlay on the main widget
        self.spinner = LoadingSpinner(parent=self)
        self.spinner.setAttribute(Qt.WA_TransparentForMouseEvents)  # Optional: let clicks pass through
        self.spinner.setWindowFlags(Qt.SubWindow)  # Optional: prevent it from behaving like a popup
    
        # Center it
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
        try:
            idx_max = self.df_rois.index.to_numpy().max()
        except:
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
                except:
                    raise ValueError(f'The reference roi for roi #{ind} is not available') 
                imgs = tr.cut_imgs_by_roi(imgs, rois_ref)
                rois_in = tr.translate_roiInRoi(rois_in, rois_ref, fwd=True)
            except:
                pass
            worker = WorkerThread_General(tr.track_roi_cv2, ind, imgs, rois_in, 
                                          init, tracking_method)
            worker.signals.results.connect(self.get_tracking_results)  # Connect to result signal
            # worker.signals.finished.connect(self.plot_tracking_result)
            self.threadpool.start(worker)
            
    def get_tracking_results(self, result, index):
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
                except:
                    pass
                            
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
                'CV2 tracking completed successfully for %d ROI(s) in %.1f s.',
                self.tracking_counter_end, duration)

    def extract_3ded(self):
        self.load_spinner()
        
        path_4d = self.lineEdit_dir_4d.text()
        if path_4d == '': # no entry in 4D signals path
            self.spinner.stop()
            qtw.QMessageBox.critical(self, 'No Entry',
                'Enter the path to the folder containing 4D signal files '
                '(.hdf5, .tpx3, .zspy, etc.) before extraction.')
            return

        # check if no of files matches with no of images
        fns_4d = glob(os.path.join(path_4d, '*'))
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
        # make masks
        thresh_method = self.combo_thresh_method.currentText()
        thresh_offset = self.slider_thresh.value() / 100
        for ind in self.df_rois[self.df_rois.use == 1].index:
            self.df_rois.at[ind, 'mask'] = tr.create_masks(
                self.nav_imgs, self.df_rois.loc[ind, 'out_rois'],
                thresh_method, thresh_offset, blur_kernel)

        # set detector size for tpx3
        if dtype == '.tpx3': # TODO not good
            shape_d_x, shape_d_y = 512, 512
        else:
            shape_d_x, shape_d_y = io.get_det_size(fns_4d[0])
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
                    self.tasks.append([fn, df.loc[idx, 'out_rois'][i_fr],
                                       os.path.join(self.temp_dir, f"mask_r{idx}_f{i_fr}.npy"),
                                       dtype, scanSize, (idx, i_fr)])
        
        
        # path_debug = r'C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\py5DED\other_scripts\debug'
        # with open(os.path.join(path_debug, 'args.txt'), 'r') as f:
        #     f.writelines(self.tasks)
        
        
        self.max_processes = self.spinbox_threadNo.value()
        self.running_processes = []
        self.process_task_map = {}
        # self.launch_initial_tasks()
        for _ in range(min(self.max_processes, len(self.tasks))):
            self.launch_next_task()

    def get_temp_dir(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(os.path.dirname(script_dir), 'py4DTomo', 'io_utils', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        self.logger.info('temp directory: %s', temp_dir)
        return temp_dir
    
# =============================================================================
#     def launch_initial_tasks(self):
#         for _ in range(min(self.max_processes, len(self.tasks))):
#             self.launch_next_task()
# =============================================================================

    def launch_next_task(self):
        if not self.tasks or len(self.running_processes) >= self.max_processes:
            return
    
        args = self.tasks.popleft()
        mask_path = args[2]
        idx, i_fr = args[-1]
        np.save(mask_path, self.df_rois.loc[idx, 'mask'][i_fr])
        process = QProcess()
        process.setProgram(sys.executable)
        process.setArguments(["worker_extract_frame.py"] + list(map(str, args)))
        process.readyReadStandardOutput.connect(lambda: self.handle_output(process))
        process.readyReadStandardError.connect(lambda: self.handle_error(process))
        process.finished.connect(lambda: self.handle_finished(process))
        process.errorOccurred.connect(self.process_failed)


        self.running_processes.append(process)
        self.process_task_map[process] = args
        process.start()
        
    def process_failed(self, error):
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
        if self._cancelling:
            return
        error_output = process.readAllStandardError().data().decode().strip()
        if error_output:
            self._3ded_failed = True
            self.logger.error("Worker ERROR: %s", error_output)
            qtw.QMessageBox.warning(self, 'Worker Error',
                f'A worker process reported an error:\n{error_output[:500]}')

    def handle_output(self, process):
        raw_output = process.readAllStandardOutput().data().decode().strip()
        try:
            result_array = pickle.loads(base64.b64decode(raw_output))
        except Exception as e:
            # print(f"Failed to decode output: {e}")
            # print("Raw output was:", raw_output)
            return

        task_info = self.process_task_map.get(process, None)
        if task_info is None:
            self.logger.warning("Unknown process")
            return
    
        # *_ , (r_id, i_fr) = task_info
        img , i_c = result_array
        idx, i_fr = eval(i_c)
        self.df_rois.loc[idx, 'dp'][i_fr] = img
        
    def handle_finished(self, process):
        if process in self.running_processes:
            self.running_processes.remove(process)
        _ = self.process_task_map.pop(process, None)
        process.deleteLater()
        if self._cancelling:
            return

        # progress bar update
        self.tomo_counter += 1
        self.update_progress_bar(self.tomo_counter, self.tomo_counter_total)
        
        if self.tomo_counter >= self.tomo_counter_total:
            self.toc = perf_counter()
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            duration = self.toc - self.tic
            if self._3ded_failed:
                self.logger.error(
                    '3DED extraction finished with errors after %.1f min '
                    '(see log above for details).', duration / 60)
            else:
                self.logger.info(
                    '3DED extraction completed successfully (%d frame(s)) in %.1f min.',
                    self.tomo_counter_total, duration / 60)
            self.update_canvas()
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
    
    def save_results(self):
        tic = perf_counter()
        try:
            self._save_results_impl()
        except Exception:
            self.logger.exception('Failed to save results after %.1f s.', perf_counter() - tic)
            return
        self.logger.info(
            'Results saved successfully in %.1f s (background clip/frame '
            'generation for each ROI continues asynchronously).',
            perf_counter() - tic)

    def _save_results_impl(self):
        path_save = self.lineEdit_dir_save.text()
        if not os.path.isdir(path_save):
            os.mkdir(path_save)

        fld_1 = datetime.date.today()
        fld_2 = datetime.datetime.now().strftime("%H-%M-%S")

        path_save = os.path.join(path_save, f'{fld_1}__{fld_2}')
        os.mkdir(path_save)
        self.logger.info('Saving results to %s...', path_save)

        # Navigation signal: saved once at the top level (shared by every
        # ROI) so "Load Saved Analysis" can restore it later, since it's
        # otherwise never persisted anywhere.
        if hasattr(self, 's'):
            self.s.save(os.path.join(path_save, 'navigation_signal.hspy'), overwrite=True)

        # tracking results, rois, dp
        for idx in self.df_rois.index:
            path_save_roi = os.path.join(path_save, f'roi No {idx}')
            os.mkdir(path_save_roi)
            df = self.df_rois.loc[idx, ['use', 'init', 'in_rois', 'end', 'ref']]
            df['thresh'] = [('blur kernel', self.combo_blur.currentText()),
                            ('thresh method', self.combo_thresh_method.currentText()),
                            ('thresh offset', self.slider_thresh.value())]
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
                fld_frames = os.path.join(path_save_roi, 'frames')
                worker_frames = WorkerThread_General(io.create_frames, 0, fld_frames, dp)
                self.threadpool.start(worker_frames)
                scale_recip = self.lineEdit_scale_recip.text()
                try:
                    scale_recip = float(scale_recip)
                except:
                    scale_recip = None
                fn_dp = os.path.join(path_save_roi, 'tomo clip')
                worker_clip_dp = WorkerThread_General(io.create_clip_dp, 0, fn_dp, dp,
                                                      scale_recip, fps=self.spinbox_fps.value(),
                                                      logger=self.logger)
                self.threadpool.start(worker_clip_dp)

            # clip for tracking
            scale_real = self.lineEdit_scale_real.text()
            try:
                scale_real = float(scale_real)
            except:
                scale_real = None

            fn = os.path.join(path_save_roi, 'tracking clip')
            worker_clip_tr_ref = WorkerThread_General(
                io.create_clip_tracking, 0, fn, self.nav_imgs,
                self.df_rois.loc[idx, 'out_rois'], scale_real,
                fps=self.spinbox_fps.value(), logger=self.logger)
            self.threadpool.start(worker_clip_tr_ref)
            
    def kill_running_process(self):
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

