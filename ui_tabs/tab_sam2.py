# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:43:00 2024

@author: SGholam
"""

import json
import sys
import hyperspy.api as hs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QThreadPool, QProcess
import pickle
import base64
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QKeySequence
from PyQt5.QtWidgets import QShortcut
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import os
import re
from PIL import Image
import gc
from copy import deepcopy
import datetime
from time import perf_counter
import py4DTomo.io_utils as io
from typing import Literal
from .worker_thread import WorkerThread_General
from .logging_utils import get_tab_logger, LogConsole
from glob import glob
from matplotlib.colors import SymLogNorm
# import py4DTomo.tracking_utils as tr
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
class Tab_SAM2(qtw.QWidget):
    def __init__(self):
        super().__init__()

        self.logger = get_tab_logger('Tab_SAM2')

        # Recomputing constrained_layout's spacing solve on every redraw
        # (canvas.draw()'s default behavior) is one of the most expensive
        # parts of a redraw; update_canvas() freezes it after the first
        # real draw, once subplot spacing has settled.
        self._layout_frozen = False

        # threadpool to use in the entire tab
        self.threadpool = QThreadPool()
        # self.threadpool = QThreadPool.globalInstance()
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
        self.layout = qtw.QHBoxLayout(self)
        self._splitter = qtw.QSplitter(Qt.Horizontal)
        self.layout.addWidget(self._splitter)
        self._left_widget = qtw.QWidget()
        self._splitter.addWidget(self._left_widget)

        button_w = 95
        button_h_sml = 30
        button_h_lrg = 50
        # height_layout_top = 200
        width_userInput = 300

        # layout top
        layout_userInput = qtw.QVBoxLayout(self._left_widget)
        #%% directory
        self.box_dir = qtw.QGroupBox('Directories', self)
        self.box_dir.setFixedHeight(200)
        # self.box_dir.setFixedWidth(width_userInput)
        layout_dir = qtw.QVBoxLayout()
        # self.layout.addLayout(layout_dir)
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
        layout_dir.addLayout(layout_loadSignal)
        self.box_scale = qtw.QGroupBox('Scale bars')
        layout_box_scale = qtw.QVBoxLayout()
        layout_loadSignal.addWidget(self.box_scale)
        
        self.box_scale.setFixedWidth(150)
        self.box_scale.setLayout(layout_box_scale)
        layout_loadSignal.addWidget(self.box_scale)
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
        # reciprocal space
        layout_scale_recip = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_recip)
        label_scale_recip = qtw.QLabel('Recip. (\u00C5<sup>-1</sup>)')
        label_scale_recip.setFixedWidth(55)
        layout_scale_recip.addWidget(label_scale_recip)
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        layout_scale_recip.addWidget(self.lineEdit_scale_recip)
        self.lineEdit_scale_recip.setValidator(self.double_validator)
        
        self.lineEdit_scale_recip.textChanged.connect(self.add_scalebar)
        self.lineEdit_scale_real.textChanged.connect(self.add_scalebar)
        
        layout_load_buttons = qtw.QVBoxLayout()
        layout_loadSignal.addLayout(layout_load_buttons)

        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        self.button_loadNavigation.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        layout_load_buttons.addWidget(self.button_loadNavigation)
        self.button_loadNavigation.clicked.connect(self.load_navSignal)

        self.button_loadSavedAnalysis = qtw.QPushButton('Load Saved Analysis')
        self.button_loadSavedAnalysis.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        layout_load_buttons.addWidget(self.button_loadSavedAnalysis)
        self.button_loadSavedAnalysis.clicked.connect(self.load_saved_analysis)
        #%% feature handling
        self.box_table = qtw.QGroupBox('Features Handling')
        layout_userInput.addWidget(self.box_table)
        layout_features = qtw.QVBoxLayout()
        self.box_table.setLayout(layout_features)
        
        # tree
        self.tree_objects = qtw.QTreeWidget()
        layout_features.addWidget(self.tree_objects)
        self.cols_tree = ["use", "idx", "fr_idx", "end", "trk", "ext", "del"]
        self.tree_objects.setColumnCount(len(self.cols_tree))
        self.tree_objects.setHeaderLabels(["Use", "Idx", "Frame", "End", "Tracked", "Extracted", "Delete"])
        for i, _ in enumerate(self.cols_tree):
            self.tree_objects.setColumnWidth(i, 20)
        self.tree_objects.setColumnWidth(2, 50)
        self.tree_objects.setColumnWidth(3, 50)
        self.tree_objects.setMinimumWidth(200)
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
        #%% extract 3DED
        self.box_3ded = qtw.QGroupBox('Extract 3DED')
        layout_box_3ded = qtw.QVBoxLayout()
        self.box_3ded.setLayout(layout_box_3ded)
        layout_userInput.addWidget(self.box_3ded)
        
        layout_threadNum = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_threadNum)
        
        # layout_threadNum.addItem(spacer)
        
        label_threadNo = qtw.QLabel('CPU Cores')
        layout_threadNum.addWidget(label_threadNo)
        self.spinbox_threadNum = qtw.QSpinBox(self)
        layout_threadNum.addWidget(self.spinbox_threadNum)
        self.spinbox_threadNum.setRange(1, os.cpu_count() or 1)
        self.spinbox_threadNum.setValue(max(1, (os.cpu_count() or 2) - 2))
        self.spinbox_threadNum.valueChanged.connect(self.set_threadNo)
        
        label_fps = qtw.QLabel('Clip FPS')
        layout_threadNum.addWidget(label_fps)
        self.spinbox_fps = qtw.QSpinBox(self)
        layout_threadNum.addWidget(self.spinbox_fps)
        self.spinbox_fps.setRange(1, 60)
        self.spinbox_fps.setValue(5)
        self.spinbox_fps.setToolTip('Frames per second for saved video clips')

        layout_threadNum.addSpacerItem(spacer)

        self.checkbox_autosave = qtw.QCheckBox('Autosave')
        layout_threadNum.addWidget(self.checkbox_autosave)
        
        layout_extract_button = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_extract_button)
        self.button_3ded = qtw.QPushButton('Extract!')
        self.button_3ded.setFixedSize(button_w, button_h_lrg)
        layout_extract_button.addWidget(self.button_3ded)
        self.button_3ded.clicked.connect(self.extract_3ded)
        
        self.button_save_results = qtw.QPushButton('Save Results')
        # self.button_save_results.setFixedHeight(35)
        self.button_save_results.setFixedSize(button_w, button_h_lrg)
        layout_extract_button.addWidget(self.button_save_results)
        self.button_save_results.clicked.connect(self.save_results)
        
        self.disable_3ded_widgets(True)
        # layout_userInput.addItem(spacer)
        #%% canvas
        self._right_widget = qtw.QWidget()
        self._splitter.addWidget(self._right_widget)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([300, 900])
        layout_canvas = qtw.QVBoxLayout(self._right_widget)
        
        self.figure = Figure(constrained_layout=True)
        # self.figure = Figure(figsize=(16,8)) # with figsize
        self.canvas = FigureCanvas(self.figure)
        self.ax_nav = self.figure.add_subplot(131)
        self.ax_seg = self.figure.add_subplot(132)
        self.ax_dp = self.figure.add_subplot(133)
        self.img_zero = np.zeros((512,512), dtype='int16')
        self.img_display = {}
        self.img_display['nav'] = self.ax_nav.imshow(self.img_zero, cmap='gray')
        self.ax_nav.set_title('Navigation')
        self.img_display['seg'] = self.ax_seg.imshow(self.img_zero, cmap='gray')
        self.img_display['seg_mask'] = self.ax_seg.imshow(self.img_zero, cmap='gray')
        self.ax_seg.set_title('Segmented')
        self.img_display['dp'] = self.ax_dp.imshow(self.img_zero, cmap='inferno', 
                                                    norm=SymLogNorm(linthresh=1))
        self.ax_dp.set_title('Extracted DP')
        for ax in [self.ax_dp, self.ax_nav, self.ax_seg]:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # self.figure.tight_layout()
        layout_canvas.addWidget(self.canvas)
        self.ax_nav.set_xlabel(
            'Hold "ctrl" + Left Click => Positive Point\n'
            'Hold "ctrl" + Right Click => Negative Point\n'
            'Add "shift" to add Points to an Existing Object\n'
            'Middle Click => Delete Last Point\n'
            'Ctrl+Scroll => Zoom, plain Click+Drag => Pan/Zoom Tool', fontsize=8.5)
        self.ax_nav.xaxis.label.set_visible(True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        # self.masks_plotted = []
        self.create_main_dataframe()
        self.imgs = deepcopy([self.img_zero])
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

        # keyboard shortcuts
        QShortcut(QKeySequence('Ctrl+O'), self, self.button_loadNavigation.click)
        QShortcut(QKeySequence('Ctrl+Shift+O'), self, self.button_loadSavedAnalysis.click)
        QShortcut(QKeySequence('Ctrl+T'), self, self.button_runSeg_clip.click)
        QShortcut(QKeySequence('Ctrl+E'), self, self.button_3ded.click)
        QShortcut(QKeySequence('Ctrl+S'), self, self.button_save_results.click)
#%% load data
    def show_dialog(self, f):
        sender = self.sender()
        if sender == self.button_dir_navSignal:
            file_filter = "supported signals (*.zspy *.hspy);;All Files (*)"
            # path = qtw.QFileDialog.getOpenFileNames(self, "Select 4D Signals Folder", '', file_filter)
            path = qtw.QFileDialog.getOpenFileName(self, "Select 4D Signals Folder", '', file_filter)
            if path:
                self.lineEdit_dir_navSignal.setText(path[0])
                path_save = os.path.join(os.path.dirname(path[0]), '5DED Analysis')
                self.lineEdit_dir_save.setText(path_save)
                
        elif sender == self.button_dir_4dSignals:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select 4D Folder")
            if path:
                self.lineEdit_dir_4d.setText(path)
                
        elif sender == self.button_dir_save:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select Destination Folder")
            if path:
                self.lineEdit_dir_save.setText(path)

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

    def load_navSignal(self):
        fn = self.lineEdit_dir_navSignal.text()
        if not os.path.isfile(fn):
            self.logger.error('Cannot find navigation signal at: %s', fn)
            qtw.QMessageBox.critical(self, 'File Not Found',
                f'Cannot find navigation signal at:\n{fn}')
            return
        self.logger.info('Loading navigation signal from %s...', fn)
        self.reset_data()
        self._load_spinner()

        def _load(fn):
            s = hs.load(fn)
            return s, s.data.copy(), io.convert_to_8bit(s).data

        worker = WorkerThread_General(_load, 0, fn)
        worker.signals.results.connect(self._on_navSignal_loaded)
        self.threadpool.start(worker)

    def _on_navSignal_loaded(self, result, index):
        s, imgs, imgs_8bit = result
        self.spinner.stop()
        self.fn_navSignal = self.lineEdit_dir_navSignal.text()
        self.create_main_dataframe()
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

        self.spinbox_stackNum.setMaximum(len(s))
        shape_x, shape_y = self.imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['nav'].set_clim(vmin=self.imgs.min(), vmax=self.imgs.max())
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
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.imgs)))
        self.spinbox_stackNum.setValue(len(self.imgs))

    def load_saved_analysis(self):
        """Restore a previously saved analysis folder (produced by
        save_results): the navigation signal, per-object tracking (points/
        labels/rois/masks), and any extracted diffraction patterns."""
        path = qtw.QFileDialog.getExistingDirectory(
            self, "Select Saved Analysis Folder", self.lineEdit_dir_save.text())
        if not path:
            return
        fn_nav = os.path.join(path, 'navigation_signal.hspy')
        if not os.path.isfile(fn_nav):
            self.logger.error('Cannot find navigation_signal.hspy in: %s', path)
            qtw.QMessageBox.critical(self, 'Navigation Signal Not Found',
                f'Cannot find navigation_signal.hspy in:\n{path}\n\n'
                'This folder does not look like a saved analysis (or it was '
                'saved before this feature was added).')
            return

        self.logger.info('Loading saved analysis from %s...', path)
        self.reset_data()
        self._load_spinner()

        worker = WorkerThread_General(self._load_saved_analysis_worker, 0, path, fn_nav)
        worker.signals.results.connect(self._on_saved_analysis_loaded)
        self.threadpool.start(worker)

    def _load_saved_analysis_worker(self, path, fn_nav):
        s = hs.load(fn_nav)
        imgs = s.data.copy()
        imgs_8bit = io.convert_to_8bit(s).data

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
        return s, imgs, imgs_8bit, objects, path

    def _on_saved_analysis_loaded(self, result, index):
        s, imgs, imgs_8bit, objects, path = result
        self.spinner.stop()
        self.fn_navSignal = os.path.join(path, 'navigation_signal.hspy')
        self.create_main_dataframe()
        self._apply_loaded_nav_signal(s, imgs, imgs_8bit)

        for obj in objects:
            idx = obj['idx']
            self.df_obj.loc[idx] = [obj['use'], idx, obj['frame_idx'], obj['points'],
                                     obj['labels'], obj['end'], None, obj['mask'],
                                     obj['rois'], obj['dp']]
            self.add_item_tree(idx, obj['frame_idx'], obj['end'], obj['use'])
            row_index = self.df_obj.index.get_loc(idx)
            if obj['mask'] is not None:
                self.toggle_tree_icon(row_index, 'trk', True)
            if obj['dp'] is not None:
                self.toggle_tree_icon(row_index, 'ext', True)

        self.activate_3ded_widgets(True)
        self.update_canvas(0)
        self.logger.info('Loaded saved analysis from %s (%d object(s)).', path, len(objects))

    def create_main_dataframe(self):
        self.cols_df = ['use', 'idx', 'frame_idx', 'points', 'labels', 'end', 
                        'single_mask', 'mask', 'rois', 'dp']
        self.df_obj = pd.DataFrame([], columns=self.cols_df)
        self.df_obj = self.df_obj.astype({'use': int, 'idx': int,'frame_idx': object, 
                                          'points': object, 'labels': object, 
                                          'end': int, 'single_mask': object, 
                                          'dp': object,'mask':object, 'rois':object})
        self.initiate_adding_points()
        
    def reset_data(self):
        for p in self.scatter_plots:
            p.remove()
        self.scatter_plots.clear()
        self.tree_objects.clear()
        self.create_main_dataframe()
        self.label_stack.setText('')
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.imgs)))
        self.update_canvas()
        # self.button_runSeg_clip.setEnabled(False)
    
    def initiate_adding_points(self):
        cols = ['new', 'idx', 'point', 'frame']
        self.df_added_points = pd.DataFrame(data=[], columns=cols)
        self.df_added_points = self.df_added_points.astype({'point': object})
    
    def delete_tree_item(self, col:str):
        for i in reversed(range(self.tree_objects.topLevelItemCount())):
            item = self.tree_objects.topLevelItem(i)
            if item.text(1) == col:
                self.tree_objects.takeTopLevelItem(i)
#%% object tree and funcs
    def add_item_tree(self, idx, fr_idx=[0], end=None, use=1):
        cols = {col: i for i,col in enumerate(self.cols_tree)}
        item = qtw.QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(cols['use'], Qt.Checked if use else Qt.Unchecked)
        item.setText(cols['idx'], f"{idx}")
        item.setText(cols['fr_idx'], f"{fr_idx}")
        # set as selected item
        self.tree_objects.itemChanged.connect(self.on_item_check_changed)
        self.tree_objects.addTopLevelItem(item)

        # end frame
        spinbox = qtw.QSpinBox()
        spinbox.setRange(0, len(self.imgs))
        spinbox.setValue(end if end is not None else len(self.imgs))
        self.tree_objects.setItemWidget(item, cols['end'], spinbox)
        spinbox.valueChanged.connect(lambda value: self.on_spinboxEnd_changed(idx, value))
        # spinbox.valueChanged.connect(partial(self.on_spinbox_changed, item, idx))
        
        # self.tree_objects.addTopLevelItem(item)
        
        # tracked
        cancel_icon = self.style().standardIcon(self.style().SP_DialogCancelButton)
        item.setIcon(cols['trk'], cancel_icon)
        item.setData(cols['trk'], Qt.UserRole, False)  # Store status boolean (False = not checked)
        
        # extracted
        item.setIcon(cols['ext'], cancel_icon)
        item.setData(cols['ext'], Qt.UserRole, False)  # Store status boolean (False = not checked)
        
        # delete
        delete_button = qtw.QPushButton()
        delete_button.setIcon(self.style().standardIcon(qtw.QStyle.SP_TrashIcon))
        delete_button.setFixedSize(30, 30)
        delete_button.setToolTip("Delete this item")
        
        def delete_row():
            index = self.tree_objects.indexOfTopLevelItem(item)
            self.logger.info('Deleted object %s.', self.df_obj.index[index])
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
    
        
        self.tree_objects.setColumnWidth(cols['end'], 60)
        self.tree_objects.setColumnWidth(cols['del'], 40)
        
        # Later: select it
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
    
    def on_item_check_changed(self, item, column):
        use_col = self.cols_tree.index('use')  # or `cols['use']` if accessible
        idx_col = self.cols_tree.index('idx')
        idx = int(item.text(idx_col))
        if item.checkState(use_col) == Qt.Checked:
            self.df_obj.at[idx, 'use'] = 1
        else:
            self.df_obj.at[idx, 'use'] = 0
#%% canvas
    def on_click(self, event):
        if event.button == 2: # middle click:
            self.delete_last_point()
            return
        if (event.inaxes != self.ax_nav) or (event.button not in [1,3]):
            return
        if 'ctrl' not in event.modifiers:
            # Plain click/drag is reserved for the navigation toolbar's
            # Pan/Zoom tool (and the scroll-wheel zoom below) so images can
            # be zoomed into; hold "ctrl" to add a point instead.
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
                                    None, None, None, None]
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
        if imgNo is None:
            imgNo = self.slider_imgNo.value()
        if obj_id is None:
            try:
                item_selected = self.tree_objects.currentItem()
                obj_id = int(item_selected.text(1))
            except:
                obj_id = None
        self.remove_plotted_points()     
        self.img_display['nav'].set_data(self.imgs[imgNo])
        self.ax_nav.set(title=f'Nav. Image No: {imgNo}')
    
        if obj_id is not None:
            self.plot_points(imgNo, obj_id)
            # plot segmentation masks for video
            if (not np.all(pd.isna(self.df_obj.loc[obj_id, 'mask']))):
                self.img_display['seg'].set_data(self.imgs[imgNo])
                self.img_display['seg'].set_clim(vmin=self.imgs[imgNo].min(), vmax=self.imgs[imgNo].max())
                
                try:
                    self.show_mask(self.df_obj.loc[obj_id, 'mask'][imgNo], obj_id)
                except:
                    self.show_mask(self.img_zero)
            
            # plot segmentation masks for single images
            else:
                try:
                    self.img_display['seg'].set_data(self.imgs[imgNo])
                    self.img_display['seg'].set_clim(vmin=self.imgs[imgNo].min(), vmax=self.imgs[imgNo].max())
                    mask = self.df_obj.loc[obj_id, 'single_mask'][imgNo]
                    self.show_mask(mask, 0)
                except:
                    self.img_display['seg'].set_data(self.img_zero)
            
            # diffraction pattern
            if (not np.all(pd.isna(self.df_obj.loc[obj_id, 'dp']))):
                try:
                    self.plot_dp(obj_id=obj_id, imgNo=imgNo)
                except:
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
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
            for ax in [self.ax_nav, self.ax_seg]:
                scalebar_real = ScaleBar(scale_real, 'nm', dimension='si-length', 
                                     location='lower left', box_alpha=0.4)
                for artist in ax.artists:
                    if isinstance(artist, ScaleBar):
                        artist.remove()
                ax.add_artist(scalebar_real)
        except Exception as e:
            # print(e)
            pass
        
        # A conventional linear scale bar doesn't read naturally on a
        # radially-symmetric diffraction pattern - concentric dashed rings
        # at every 1 1/A (centered on the DP) work better.
        shape = self.img_display['dp'].get_array().shape
        self._dp_recip_circles = io.draw_reciprocal_scale_circles(
            self.ax_dp, self.lineEdit_scale_recip.text(), shape,
            old_artists=getattr(self, '_dp_recip_circles', None))

        self.canvas.draw()
    
    def show_mask(self, mask, cmap_idx=0):
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
            except:
                pass
        self.scatter_plots.clear()
        
    def plot_points(self, imgNo, obj_id):
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
        self.img_display['dp'].set_clim(vmin=img.min(), vmax=img.max())
        # self.img_display['dp'].set_clim(vmin=1, vmax=img.max())
        shape_x, shape_y = img.shape
        self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])
#%% SAM2 video segmentation
    def update_stack_guide(self):
        """Rebuild the stack-navigation button strip for the currently selected object.

        Each button is labelled with the stack index and, when clicked, moves the slider
        to the global frame number where that stack begins.  The start of each stack is
        shifted by the object's earliest frame_idx so objects that don't start at frame 0
        are handled correctly.
        """
        # Clear previous buttons
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
        # self.button_runSeg_img.setDisabled(True)
        self.button_runSeg_clip.setDisabled(True)
        self._track_tic = perf_counter()
        self._track_failed = False

        # jpg path
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

        # count the total number of workers for creating jpg
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
         self.running_processes_sam = {}
         self.running_processes_sam_total = len(self.df_toSegment.index)
         idx = self.df_toSegment.index.sort_values()[0]
         path = self.df_toSegment.loc[idx, 'path_jpg']
         self.launch_next_video_seg(path, idx)
                
    def launch_next_video_seg(self, path, idx):
        self.logger.info("Next project: %s %s", idx, path)
        process_sam = QProcess(self)
        process_sam.setProgram(sys.executable)
        process_sam.setArguments(["worker_sam.py"] + ['video', path, str(idx)])

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
        error_output = process.readAllStandardError().data().decode()
        if error_output.strip():
            self.logger.info("[%s] %s", idx, error_output)
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

    def handle_finished_sam(self, process, idx, exit_code, exit_status):
        self.logger.info("[%s] Process finished with exit code %s, status %s",
                          idx, exit_code, exit_status)

        data = process.readAllStandardOutput()
        text = bytes(data).decode("utf-8")
        self.logger.info('text: %s', text)
        # process.kill()
        # sleep(3)
        try:
            result = json.loads(text.strip())
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
                        self.df_obj.at[i_ref, 'mask'][
                            start : start + frame_num] = df.loc[idx, 'mask']
                    # toggling tracking icons
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
                        'after %.1f s (see log above for details).',
                        n_objects, duration)
                else:
                    self.logger.info(
                        'SAM2 tracking completed successfully for %d object(s) in %.1f s.',
                        n_objects, duration)
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

        pathSave = self.lineEdit_dir_save.text()
        if not os.path.isdir(pathSave):
            os.mkdir(pathSave)
        path_seg = os.path.join(pathSave, 'JPG Images', 'single_image_seg')
        if os.path.isdir(path_seg):
            shutil.rmtree(path_seg)
        os.makedirs(path_seg)

        seg_input = pd.Series({'image': self.imgs_8bit[imgNo], 'points': points, 'labels': labels})
        seg_input.to_pickle(os.path.join(path_seg, 'seg_input.pkl'))

        process_sam = QProcess(self)
        process_sam.setProgram(sys.executable)
        process_sam.setArguments(["worker_sam.py"] + ['image', path_seg, str(obj_id)])
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
            fn_output = result["path"]
            with np.load(fn_output) as f:
                mask = f['mask']
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
                except:
                    rois.append((0,0,0,0))
            rois = np.array(rois)
            self.df_obj.at[obj_id, 'rois'] = rois
    
    def extract_3ded(self):
        self.make_rois()
        
        path_4d = self.lineEdit_dir_4d.text()
        # check path
        if path_4d == '': # no entry in 4D signals path
            self.logger.error('3DED extraction cancelled: no 4D signals path entered.')
            qtw.QMessageBox.critical(self, 'No 4D path', 'Please enter a valid path for 4D signals.')
            return
        fns_4d = glob(os.path.join(path_4d, '*'))
        if len(fns_4d) == 0:
            self.logger.error('3DED extraction cancelled: no files found in %s', path_4d)
            qtw.QMessageBox.critical(self, 'Wrong Path', 'No files was found in the path for 4D signals!')
            return
        dtype = os.path.splitext(fns_4d[0])[1]
        
        # check if num of files with num of images
        if len(self.imgs) != len(fns_4d):
            self.logger.warning(
                'Number of 4D signal files (%d) does not match the number of '
                'navigation images (%d).', len(fns_4d), len(self.imgs))
            reply = qtw.QMessageBox.question(self, 'Mismatch',
                   'No of 4D signals mismatches the number of images. Do you want to continue?',)
            if reply == qtw.QMessageBox.No:
                self.logger.info('3DED extraction cancelled by user after mismatch warning.')
                return
        # set detector size for tpx3
        if dtype in ['.tpx3', '.hdf5']: # TODO not good
            shape_d_x, shape_d_y = 512, 512
        else:
            shape_d_x, shape_d_y = io.get_det_size(fns_4d[0])
        scanSize = self.imgs.shape[1:]
        
        df = self.df_obj[self.df_obj.use == 1]
        self.tomo_counter = 0
        
        lengths = df.end - [min(df.frame_idx[idx]) for idx in df.index]
        self.tomo_counter_total = np.sum(lengths)
        self.update_progress_bar(0, self.tomo_counter_total)
        self.tic = perf_counter()
        self._3ded_failed = False
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
                self.tasks.append([fn, df.loc[idx, 'rois'][i_fr], 
                                   os.path.join(self.temp_dir, f"mask_r{idx}_f{i_fr}.npy"),
                                   dtype, scanSize, (idx, i_fr)])
        
        self.max_processes = self.spinbox_threadNum.value()
        self.running_processes = []
        self.process_sam_task_map = {}
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
    
    def get_temp_dir(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(os.path.dirname(script_dir), 'py4DTomo', 'io_utils', 'temp')
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
        if not self.tasks or len(self.running_processes) >= self.max_processes:
            return
    
        args = self.tasks.popleft()
        *_, (idx,i_fr) = args
        _ = self.save_mask_to_temp(self.temp_dir, 
                           self.df_obj.loc[idx, 'mask'][i_fr], idx, i_fr)
        
        process = QProcess()
        process.setProgram(sys.executable)
        process.setArguments(["worker_extract_frame.py"] + list(map(str, args)))
        process.readyReadStandardOutput.connect(lambda: self.handle_output_3ded(process))
        process.readyReadStandardError.connect(lambda: self.handle_error_3ded(process))
        process.finished.connect(lambda: self.handle_finished_3ded(process, idx))
        process.errorOccurred.connect(self.process_failed_3ded)


        self.running_processes.append(process)
        self.process_sam_task_map[process] = args
        process.start()
        
    def process_failed_3ded(self, error):
        self._3ded_failed = True
        self.logger.error("QProcess error occurred: %s", error)
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        qtw.QMessageBox.critical(self, 'Process Error',
            f'A worker process failed to start (error code {error}).\n'
            'Check that Python is on PATH and worker_extract_frame.py exists.')

    def handle_error_3ded(self, process):
        error_output = process.readAllStandardError().data().decode().strip()
        if error_output:
            self._3ded_failed = True
            self.logger.error("Worker ERROR: %s", error_output)
            qtw.QMessageBox.warning(self, 'Worker Error',
                f'A worker process reported an error:\n{error_output[:500]}')

    def handle_output_3ded(self, process):
        raw_output = process.readAllStandardOutput().data().decode().strip()
        try:
            result_array = pickle.loads(base64.b64decode(raw_output))
        except Exception as e:
            # print(f"Failed to decode output: {e}")
            # print("Raw output was:", raw_output)
            return

        task_info = self.process_sam_task_map.get(process, None)
        if task_info is None:
            self.logger.warning("Unknown process")
            return
    
        # *_ , (r_id, i_fr) = task_info
        img , r_id = result_array
        idx, i_fr = eval(r_id)
        self.df_obj.at[idx, 'dp'][i_fr] = img
        
    def handle_finished_3ded(self, process, idx):
        if process in self.running_processes:
            self.running_processes.remove(process)
        _ = self.process_sam_task_map.pop(process, None)
        process.deleteLater()
        
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
            for idx in self.df_obj[self.df_obj.use == 1].index:
                self.toggle_tree_icon(self.df_obj.index.get_loc(idx), 'ext', True)
            self.update_canvas()
            if self.checkbox_autosave.isChecked():
                self.save_results()
        else:
            self.launch_next_task()  # trigger next task if any left

    def set_threadNo(self, value):
        self.threadpool.setMaxThreadCount(value)
        
    def disable_3ded_widgets(self, state):
        for wid in self.box_3ded.findChildren(qtw.QWidget):
            if not isinstance(wid, qtw.QLabel):
                wid.setDisabled(state)
    
    def update_progress_bar(self, value, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f'%v / {total}')
    
#%% Save Data
    def save_results(self):
        tic = perf_counter()
        try:
            self._save_results_impl()
        except Exception:
            self.logger.exception('Failed to save results after %.1f s.', perf_counter() - tic)
            return
        self.logger.info(
            'Results saved successfully in %.1f s (background clip/frame '
            'generation for each object continues asynchronously).',
            perf_counter() - tic)

    def _save_results_impl(self):
        path_save = self.lineEdit_dir_save.text()
        if not os.path.isdir(path_save):
            os.mkdir(path_save)

        date = datetime.date.today()
        tim = datetime.datetime.now().strftime("%H-%M-%S")

        path_save = os.path.join(path_save, f'{date}__{tim}')
        os.mkdir(path_save)
        self.logger.info('Saving results to %s...', path_save)

        # Navigation signal: saved once at the top level (shared by every
        # object) so "Load Saved Analysis" can restore it later, since it's
        # otherwise never persisted anywhere.
        if hasattr(self, 's_navSignal'):
            self.s_navSignal.save(os.path.join(path_save, 'navigation_signal.hspy'),
                                   overwrite=True)

        # tracking results, rois, dp
        for idx in self.df_obj.index:
            path_save_objID = os.path.join(path_save, f'roi No {idx}')
            os.mkdir(path_save_objID)

            df = self.df_obj.loc[idx, ['use', 'idx', 'frame_idx', 'points', 'labels',
                                       'end']]
            df.to_json(os.path.join(path_save_objID, f'roi No {idx}.json'), orient='index', indent=4)
            if not (np.all(pd.isna(self.df_obj.loc[idx, 'rois']))):
                np.save(os.path.join(path_save_objID, 'rois.npy'),
                    self.df_obj.loc[idx, 'rois'])
            if not (np.all(pd.isna(self.df_obj.loc[idx, 'dp']))):
                np.save(os.path.join(path_save_objID, 'output_mask.npy'),
                        self.df_obj.loc[idx, 'mask'])

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
                except:
                    scale_recip = None
                fn_clip_dp = os.path.join(path_save_objID, 'tomo clip')
                worker_clip_dp = WorkerThread_General(io.create_clip_dp, 0, fn_clip_dp,
                                self.df_obj.loc[idx, 'dp'], scale_recip,
                                fps=self.spinbox_fps.value())
                self.threadpool.start(worker_clip_dp)

            # clip tracking
            if not (np.all(pd.isna(self.df_obj.loc[idx, 'mask']))):
                np.save(os.path.join(path_save_objID, f'segmentation masks_ obj ID {idx}.npy'),
                        self.df_obj.loc[idx, 'mask'])
                scale_real = self.lineEdit_scale_real.text()
                try:
                    scale_real = float(scale_real)
                except:
                    scale_real = None
                fn_clip_tracking = os.path.join(path_save_objID, 'tracking clip')
                worker_tracking = WorkerThread_General(
                    io.create_clip_tracking_with_mask, 0,
                    fn_clip_tracking, self.imgs,
                    self.df_obj.loc[idx, 'mask'], idx, scale_real,
                    fps=self.spinbox_fps.value(), cmap='Grays_r')
                self.threadpool.start(worker_tracking)
    
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
