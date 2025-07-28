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
from PyQt5.QtGui import QDoubleValidator, QIntValidator
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
from glob import glob
from matplotlib.colors import SymLogNorm
# import py4DTomo.tracking_utils as tr
import shutil
from hyperspy.api import signals as hsSignals
import pandas as pd
from time import sleep
path_ffmpeg = r'C:\Users\sgholam\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'
plt.rcParams['animation.ffmpeg_path'] = path_ffmpeg  # Windows example
#%% tab class
class Tab_SAM2(qtw.QWidget):
    def __init__(self):
        super().__init__()
        
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
        
        button_w = 95
        button_h_sml = 30
        button_h_lrg = 50
        # height_layout_top = 200
        width_userInput = 300
        
        # layout top
        layout_userInput = qtw.QVBoxLayout()
        self.layout.addLayout(layout_userInput)
        #%% directory
        self.box_dir = qtw.QGroupBox('Directories', self)
        self.box_dir.setFixedSize(width_userInput, 200)
        # self.box_dir.setFixedWidth(width_userInput)
        layout_dir = qtw.QVBoxLayout(self)
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
        layout_dir_4dSignals = qtw.QHBoxLayout(self)
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
        
        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        # self.button_loadNavigation.setMaximumHeight(button_h_lrg)
        self.button_loadNavigation.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        layout_loadSignal.addWidget(self.button_loadNavigation)
        # layout_loadSignal.addWidget(self.button_loadNavigation, alignment=Qt.AlignRight)
        # self.button_loadNavigation.setFixedSize(110, 50)
        self.button_loadNavigation.clicked.connect(self.load_navSignal)
        #%% feature handling
        self.box_table = qtw.QGroupBox('Features Handling')
        # self.box_table.setFixedSize(400, height_layout_top)
        self.box_table.setFixedWidth(width_userInput)
        layout_userInput.addWidget(self.box_table)
        layout_features = qtw.QVBoxLayout()
        self.box_table.setLayout(layout_features)
        
        # tree
        self.tree_objects = qtw.QTreeWidget()
        layout_features.addWidget(self.tree_objects)
        self.tree_objects.setMaximumWidth(400)
        self.cols_tree = ["use", "idx", "fr_idx", "end", "trk", "ext", "del"]
        self.tree_objects.setColumnCount(len(self.cols_tree))
        self.tree_objects.setHeaderLabels(self.cols_tree)
        for i, _ in enumerate(self.cols_tree):
            self.tree_objects.setColumnWidth(i, 20)
        self.tree_objects.setColumnWidth(2, 50)
        self.tree_objects.setColumnWidth(3, 50)
        # self.box_table.setFixedSize(self.tree_objects.width(), height_layout_top)
        self.box_table.setFixedWidth(width_userInput)
        self.tree_objects.setSelectionMode(qtw.QTreeWidget.SingleSelection)
        self.tree_objects.itemSelectionChanged.connect(self.update_canvas)
        
        header = self.tree_objects.header()
        # Prevent the last column from auto-stretching
        header.setStretchLastSection(False)
        # Let all columns resize to contents
        header.setSectionResizeMode(qtw.QHeaderView.ResizeToContents)
        #%% run sam2
        layout_sam_buttons_1 = qtw.QHBoxLayout(self)
        layout_features.addLayout(layout_sam_buttons_1)
        layout_sam_buttons_2 = qtw.QHBoxLayout(self)
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
        layout_stack_top.addWidget(label_stackNum)
        self.spinbox_stackNum = qtw.QSpinBox()
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
        # self.box_3ded.setFixedWidth(350)
        # self.box_3ded.setFixedSize(250, height_layout_top)
        self.box_3ded.setFixedWidth(width_userInput)
        self.box_3ded.setLayout(layout_box_3ded)
        layout_userInput.addWidget(self.box_3ded)
        
        layout_threadNum = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_threadNum)
        
        # layout_threadNum.addItem(spacer)
        
        label_threadNo = qtw.QLabel('Thread No')
        layout_threadNum.addWidget(label_threadNo)
        self.spinbox_threadNum = qtw.QSpinBox(self)
        layout_threadNum.addWidget(self.spinbox_threadNum)
        self.spinbox_threadNum.setRange(1,os.cpu_count()-1)
        self.spinbox_threadNum.setValue(3)
        self.spinbox_threadNum.valueChanged.connect(self.set_threadNo)
        
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
        # layout_
        layout_canvas = qtw.QVBoxLayout()
        self.layout.addLayout(layout_canvas)
        
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
        self.ax_nav.set_xlabel('Left Click => Positive Point\nRight Click => Negative Point\n' + 
                               'Hold "shift" for Adding Points to an Existing Object\nMiddle Click => Delete Last Point', fontsize=8.5)
        self.ax_nav.xaxis.label.set_visible(True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        # self.masks_plotted = []
        self.create_main_dataframe()
        self.imgs = deepcopy([self.img_zero])
        self.scatter_plots = []
        #%% slider
        layout_slider = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_slider)
        
        layout_slider.addWidget(NavigationToolbar(self.canvas, self))
        
        vline = qtw.QFrame()
        vline.setFrameShape(qtw.QFrame.VLine)
        vline.setFrameShadow(qtw.QFrame.Sunken)
        layout_slider.addWidget(vline)
        
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
        #%% progress bar
        layout_progress_bar = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_progress_bar)
        
        self.progress_bar = qtw.QProgressBar()
        layout_progress_bar.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)
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
    
    def load_navSignal(self):
        self.reset_data()
        
        self.fn_navSignal = self.lineEdit_dir_navSignal.text()
        s = hs.load(self.fn_navSignal)
        self.imgs = s.data
        self.imgs_8bit = io.convert_to_8bit(s).data
        self.create_main_dataframe()

        self.spinbox_stackNum.setMaximum(len(s))
        # set size and limit of the navigation data
        shape_x, shape_y = self.imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['nav'].set_clim(vmin=self.imgs.min(), vmax=self.imgs.max())
        self.img_display['seg'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['seg_mask'].set_extent([0, shape_y, shape_x, 0])
        self.update_canvas(0)
        self.slider_imgNo.setRange(0, len(self.imgs)-1)
        self.button_runSeg_clip.setEnabled(True)
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.imgs)))
        self.spinbox_stackNum.setValue(len(self.imgs))
    
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
    def add_item_tree(self, idx, fr_idx=[0], end=None):
        cols = {col: i for i,col in enumerate(self.cols_tree)}
        item = qtw.QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        # item.setCheckState(cols['use'], Qt.Unchecked)
        item.setCheckState(cols['use'], Qt.Checked)
        item.setText(cols['idx'], f"{idx}")
        item.setText(cols['fr_idx'], f"{fr_idx}")
        # set as selected item
        self.tree_objects.itemChanged.connect(self.on_item_check_changed)
        self.tree_objects.addTopLevelItem(item)
        
        # end frame
        spinbox = qtw.QSpinBox()
        spinbox.setRange(0, len(self.imgs))
        spinbox.setValue(len(self.imgs))
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
            # print(index)
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
            self.df_rois.at[idx, 'use'] = 1
        else:
            self.df_rois.at[idx, 'use'] = 0
#%% canvas
    def on_click(self, event):
        if event.button == 2: # middle click:
            self.delete_last_point()
            return
        if (event.inaxes != self.ax_nav) or (event.button not in [1,3]):
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
                
                self.show_mask(self.df_obj.loc[obj_id, 'mask'][imgNo], obj_id)
            
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
                self.plot_dp(obj_id=obj_id, imgNo=imgNo)
        shape_x, shape_y = self.imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        self.canvas.draw() 
       
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
        
        scale_recip = self.lineEdit_scale_recip.text()
        try:
            scale_recip = float(scale_recip)
            if scale_recip != 0:
                scalebar_recip = ScaleBar(scale_recip*10, '1/nm', dimension='si-length-reciprocal', 
                    location='lower left', box_alpha=0.4,
                    scale_formatter=lambda value, unit:  f'{value / 10}'r' $\AA^{-1}$', fixed_value=5)
                for artist in self.ax_dp.artists:
                        if isinstance(artist, ScaleBar):
                            artist.remove()
                self.ax_dp.add_artist(scalebar_recip)
        except Exception as e:
            # print(e)
            pass
        
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
        if hasattr(self, 'tomo_ds'):
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
        try:
            stack = self.spinbox_stackNum.value()
            item = self.tree_objects.selectedItems()[0]
            idx = int(item.text(1))
            beg = min(self.df_obj.loc[idx, 'frame_idx'])
            arr = np.arange(beg, len(self.imgs_8bit), stack)[1:]
            self.label_stack.setText(f'Img num guide: {arr.tolist()}')
        except:
            self.label_stack.setText('')

    def initiate_video_segmentation(self):
        # self.button_runSeg_img.setDisabled(True)
        self.button_runSeg_clip.setDisabled(True)
        
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
            self.df_obj.at[idx, 'mask'] = np.zeros(imgs.shape, dtype=bool)
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
        print("Next project:")
        print(idx, path)
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
        print(f"[{idx}] QProcess error occurred:", error) 
        
    def handle_error_sam(self, process, idx):
        error_output = process.readAllStandardError().data().decode()
        print("Worker ERROR:", error_output)
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
        print(f"[{idx}] Process finished with exit code {exit_code}, status {exit_status}")

        data = process.readAllStandardOutput()
        text = bytes(data).decode("utf-8")
        print('text:', text)
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
                        self.df_obj.at[i_ref, 'mask'][
                            (i_c)*self.stack_num : (i_c+1)*self.stack_num] = df.loc[idx, 'mask']
                    # toggling tracking icons
                    row_index = self.df_obj.index.get_loc(i_ref)
                    self.toggle_tree_icon(row_index, 'trk', True)

                _ = gc.collect()
                del self.df_toSegment
                self.activate_3ded_widgets(True)
                self.update_canvas()
        except json.JSONDecodeError:
            print("Could not decode result:", text)
        self.button_runSeg_clip.setEnabled(True)
    
    def stop_processes(self):
        if hasattr(self, 'running_processes_sam'):
            while len(self.running_processes_sam) > 0:
                idx, pr = self.running_processes_sam.popitem()
                pr.kill()
#%% image segmentation    
    def initiate_image_segmentation(self):
        pass

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
            qtw.QMessageBox.critical(self, 'No 4D path', 'Please enter a valid path for 4D signals.')
            return
        fns_4d = glob(os.path.join(path_4d, '*')) 
        if len(fns_4d) == 0:
            qtw.QMessageBox.critical(self, 'Wrong Path', 'No files was found in the path for 4D signals!')
            return
        dtype = os.path.splitext(fns_4d[0])[1]
        
        # check if num of files with num of images
        if len(self.imgs) != len(fns_4d):
            reply = qtw.QMessageBox.question(self, 'Mismatch',
                   'No of 4D signals mismatches the number of images. Do you want to continue?',)
            if reply == qtw.QMessageBox.No:
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
        
        self.tasks = []
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
        print('temp dir', temp_dir)
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
    
        args = self.tasks.pop(0)
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
        print("QProcess error occurred:", error)    
        
    def handle_error_3ded(self, process):
        error_output = process.readAllStandardError().data().decode()
        print("Worker ERROR:", error_output)    
    
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
            print("Unknown process")
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
            time = self.toc - self.tic
            print(f'Data Extraction Time: {time/60:.1f} min')
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
        value = value / total * 100
        value = int(value)
        self.progress_bar.setValue(value)
    
#%% Save Data
    def save_results(self):
        path_save = self.lineEdit_dir_save.text()
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        
        date = datetime.date.today()
        tim = datetime.datetime.now().strftime("%H-%M-%S")
        
        path_save = os.path.join(path_save, f'{date}__{tim}')
        os.mkdir(path_save)
        
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
                np.save(os.path.join(path_save_objID, '3DED.npy'), 
                        self.df_obj.loc[idx, 'dp'])
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
                                self.df_obj.loc[idx, 'dp'], scale_recip)
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
                fn_clip_tracking = os.path.join(os.path.join(path_save_objID, 'tracking clip'))
                worker_tracking = WorkerThread_General(
                    io.create_clip_tracking_with_mask, 0, 
                    fn_clip_tracking, self.imgs, 
                    self.df_obj.loc[idx, 'mask'], idx, scale_real, 
                    300, None,  'Grays_r')
                self.threadpool.start(worker_tracking)
    
    def closeEvent(self,event):
        # empty_cache()
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
