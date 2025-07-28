# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:55:17 2024

@author: SGholam
"""

import os
from glob import glob
import sys
from PyQt5.QtCore import (Qt, QThreadPool)
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from matplotlib.colors import SymLogNorm
import numpy as np
import py4DTomo.io_utils as io
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
from skimage.filters import threshold_otsu, threshold_li, threshold_mean, threshold_yen
import gc
from time import perf_counter
# from dask.distributed import Client, LocalCluster, as_completed
import base64
import pickle
from PyQt5.QtCore import QProcess
import shutil
from .loading_label import LoadingSpinner
from .object_detection_widget import Object_Detector_Widget
import pandas as pd
#%% wdiget
class Tab_Tracking_CV2(qtw.QWidget):
# class Tab_Create_NavSignal(qtw.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.init_widget()
        
        
        # cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='2GB')
        # client = Client(cluster)
        
        # threadpool to use in the entire tab
        self.threadpool = QThreadPool()
        # self.threadpool = QThreadPool.globalInstance()
        logical_processors = os.cpu_count()
        
# =============================================================================
#         if logical_processors > 2:
#             self.threadpool.setMaxThreadCount(logical_processors - 2)
# =============================================================================
        self.threadpool.setMaxThreadCount(3)

    def init_widget(self):
        button_w = 110
        button_h_sml = 30
        button_h_lrg = 50
        height_layout_top = 200
        # self.central_widget = qtw.QWidget(self)
        # self.setCentralWidget(self.central_widget)
        self.layout = qtw.QVBoxLayout(self)
        self.setLayout(self.layout)
        
        # layout top
        layout_top = qtw.QHBoxLayout()
        self.layout.addLayout(layout_top)
        spacer = qtw.QSpacerItem(40, 20, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
        #%% directory
        # layout_top.addItem(spacer)
        self.box_dir = qtw.QGroupBox('Directories', self)
        # self.box_dir.setFixedSize(500, 200)
        self.box_dir.setFixedWidth(350)
        self.box_dir.setFixedHeight(height_layout_top)
        # self.box_dir.setMaximumHeight(250)
        
        # self.box_dir.setFixedWidth()
        layout_dir = qtw.QVBoxLayout(self)
        # self.layout.addLayout(layout_dir)
        layout_top.addWidget(self.box_dir)
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
        self.box_scale = qtw.QGroupBox('Scale bars')
        layout_box_scale = qtw.QVBoxLayout()
        self.box_scale.setFixedWidth(150)
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
        
        self.lineEdit_scale_recip.textChanged.connect(lambda: self.update_scalebar('reciprocal'))
        self.lineEdit_scale_real.textChanged.connect(lambda: self.update_scalebar('real'))
        
        layout_loadSignal.addItem(spacer)
        
        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        layout_dir.addLayout(layout_loadSignal)
        layout_loadSignal.addWidget(self.button_loadNavigation, alignment=Qt.AlignCenter)
        self.button_loadNavigation.setFixedSize(button_w, button_h_lrg)
        self.button_loadNavigation.clicked.connect(self.load_navSignal)
        #%% feature handling
        layout_top_2 = qtw.QVBoxLayout()
        self.box_buttons = qtw.QGroupBox('Feature Handling')
        layout_top.addWidget(self.box_buttons)
        self.box_buttons.setLayout(layout_top_2)
        
        # top
        layout_featureTop = qtw.QHBoxLayout()
        layout_top_2.addLayout(layout_featureTop)
        self.button_autoDetection = qtw.QPushButton('Auto Detector')
        layout_featureTop.addWidget(self.button_autoDetection)
        self.button_autoDetection.clicked.connect(self.launch_auto_detector)
        
        self.button_reset_rois = qtw.QPushButton('Reset ROIs')
        layout_featureTop.addWidget(self.button_reset_rois)
        self.button_reset_rois.clicked.connect(self.reset_rois)
        
        # tree
        self.tree_objects = qtw.QTreeWidget()
        layout_top_2.addWidget(self.tree_objects)
        self.tree_objects.setMaximumWidth(400)
        self.tree_objects.setColumnCount(6)
        self.cols_tree = ["use", "idx", "init", "end", "ref", "trk", "ext", "del"]
        self.tree_objects.setHeaderLabels(self.cols_tree)
        for i, _ in enumerate(self.cols_tree):
            self.tree_objects.setColumnWidth(i, 20)
        self.tree_objects.setColumnWidth(2, 50)
        self.tree_objects.setColumnWidth(3, 50)
        self.box_buttons.setFixedSize(self.tree_objects.width()-10, height_layout_top)
        # self.box_buttons.setFixedSize(280, height_layout_top)
        self.tree_objects.setSelectionMode(qtw.QTreeWidget.SingleSelection)
        self.tree_objects.itemSelectionChanged.connect(self.update_canvas)
        
        self.patches_tracked = []
        self.patches_axNav = []
        self.patches_axTrack = []
        self.empty_main_dataframe()
        
        # bottom 
        layout_featureBottom = qtw.QHBoxLayout()
        layout_top_2.addLayout(layout_featureBottom)
        
        # self.checkbox_roiInRoi = qtw.QCheckBox('Select ROIinROI')
        # layout_featureBottom.addWidget(self.checkbox_roiInRoi)
        # self.checkbox_roiInRoi.setDisabled(True)

        label_blur_track = qtw.QLabel('Blur')
        layout_featureBottom.addWidget(label_blur_track)
        self.combo_blur_track = qtw.QComboBox()
        layout_featureBottom.addWidget(self.combo_blur_track)
        self.combo_blur_track.addItems([str(i) for i in range(1,23,2)])
        self.combo_blur_track.currentIndexChanged.connect(self.blur_navImages)

        spacer = qtw.QSpacerItem(40, 20, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
        layout_featureBottom.addItem(spacer)

        label_track = qtw.QLabel('Tracker')
        layout_featureBottom.addWidget(label_track)
        self.combo_trackMethod = qtw.QComboBox()
        layout_featureBottom.addWidget(self.combo_trackMethod)
        self.combo_trackMethod.addItems(['csrt', 'nano', 'mil', 'dasiamrpn'])
        
        self.button_track = qtw.QPushButton('Track!')
        layout_featureBottom.addWidget(self.button_track, alignment=Qt.AlignCenter)
        self.button_track.clicked.connect(self.track_rois)
        self.button_track.setDisabled(True)
        #%% box thresh and 3DED
        layout_3ded = qtw.QVBoxLayout()
        layout_top.addLayout(layout_3ded)
        layout_top.addItem(spacer)
        
        self.box_3ded = qtw.QGroupBox('Extract 3DED')
        layout_box_3ded = qtw.QVBoxLayout()
        self.box_3ded.setLayout(layout_box_3ded)
        layout_3ded.addWidget(self.box_3ded)
        # self.box_3ded.setFixedWidth(350)
        self.box_3ded.setFixedSize(350, height_layout_top)
        # self.box_3ded.setMaximumSize(300, 400)
        
        layout_thresh_method = qtw.QHBoxLayout()
        layout_thresh_method.addItem(spacer)
        layout_box_3ded.addLayout(layout_thresh_method)
        label_thresh_method = qtw.QLabel('Threshold Method')
        layout_thresh_method.addWidget(label_thresh_method)
        
        
        self.combo_thresh_method = qtw.QComboBox()
        layout_thresh_method.addWidget(self.combo_thresh_method)
        # self.combo_thresh_method.setFixedWidth(100)
        self.combo_thresh_method.addItems(['li', 'otsu', 'yen', 'mean'])
        self.combo_thresh_method.currentIndexChanged.connect(lambda: self.update_canvas()) #TODO change to update mask
        
        label_blur = qtw.QLabel('Blurring Kernel')
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
        # self.slider_thresh.setFixedWidth(100)
        self.slider_thresh.valueChanged.connect(lambda: self.update_canvas()) # TODO plot only mask ax
        self.slider_thresh.setRange(0, 200)
        
        self.button_thresh = qtw.QPushButton('Reset')
        layout_deviation.addWidget(self.button_thresh)
        self.button_thresh.clicked.connect(self.reset_thresh)
            #%% 3DED
        layout_threadNo = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_threadNo)
        layout_threadNo.addItem(spacer)
        
        label_threadNo = qtw.QLabel('Thread No')
        layout_threadNo.addWidget(label_threadNo)
        self.spinbox_threadNo = qtw.QSpinBox(self)
        layout_threadNo.addWidget(self.spinbox_threadNo)
        self.spinbox_threadNo.setRange(1,os.cpu_count()-1)
        self.spinbox_threadNo.setValue(3)
        self.spinbox_threadNo.valueChanged.connect(self.set_threadNo)
        
        
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
        layout_canvas = qtw.QHBoxLayout()
        self.layout.addLayout(layout_canvas)
        
        # self.figure = Figure(figsize=(8,4))
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax_nav = self.figure.add_subplot(141)
        self.ax_track = self.figure.add_subplot(142)
        self.ax_mask = self.figure.add_subplot(143)
        self.ax_dp = self.figure.add_subplot(144)
# =============================================================================
#         self.ax_nav = self.figure.add_subplot(221)
#         self.ax_track = self.figure.add_subplot(222)
#         self.ax_mask = self.figure.add_subplot(223)
#         self.ax_dp = self.figure.add_subplot(224)
# =============================================================================
        
        # titles for axes
        self.ax_nav.set_title('Nav. Signal')
        self.ax_track.set_title('Tracking Results')
        self.ax_mask.set_title('Roi with Threshold')
        self.ax_dp.set_title('DP')
        self.img_display = {}
        self.img_zero = np.zeros((512,512), dtype='uint16')
# =============================================================================
#         axes = ['nav', 'track', 'mask', 'dp']
#         for i, ax in enumerate([self.ax_nav, self.ax_track, self.ax_mask, self.ax_dp]):
#             self.img_display[axes[i]] = ax.imshow(img_temp, cmap='viridis') 
# =============================================================================
        axes = ['nav', 'track', 'dp', 'mask']
        
        for i, ax in enumerate([self.ax_nav, self.ax_track, self.ax_dp]):
            self.img_display[axes[i]] = ax.imshow(self.img_zero, cmap='viridis') 
            # ax.set_axis_off()
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        self.ax_track.set_xlabel('Draw ROIinROI on this plot', fontsize=7)
        self.ax_nav.set_xlabel('Use Left Click to add new ROI and\nRight Click to add init to an existing ROI', fontsize=7)
        self.ax_nav.xaxis.label.set_visible(True)
        self.ax_track.xaxis.label.set_visible(True)
        
        self.ax_mask.set_axis_off()
        self.img_display['img_mask'] = self.ax_mask.imshow(self.img_zero, cmap='gray')
        self.img_display['mask'] = self.ax_mask.imshow(self.img_zero, cmap='viridis', alpha=0.1)
        self.img_display['dp'].set_norm(SymLogNorm(linthresh=1))
        self.img_display['dp'].set_cmap('inferno')
        # self.figure.tight_layout()
        # self.layout.addWidget(self.canvas)
        layout_canvas.addWidget(self.canvas)
        
        
        # Connect mouse events
        self.rect = None            # Currently drawn rectangle
        self.rect_roiInRoi = None
        self.press = None           # Mouse press coordinates

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.axes = [self.ax_nav, self.ax_track, self.ax_dp, self.ax_mask]
        self.backgrounds = {}
        for i, ax in enumerate(self.axes):
            self.backgrounds[axes[i]] = self.canvas.copy_from_bbox(ax.bbox)
        
        #%% slider img num
        layout_slider = qtw.QHBoxLayout(self)
        self.layout.addLayout(layout_slider)

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

        
        self.slider_imgNo = qtw.QSlider(self)
        self.slider_imgNo.setOrientation(1)  # Horizontal slider
        self.slider_imgNo.setRange(0,0)
        layout_slider.addWidget(self.slider_imgNo)
        
        # self.update_canvas(0)
        self.slider_imgNo.valueChanged.connect(self.update_canvas)
        #%% progress bar
        layout_progress_bar = qtw.QHBoxLayout()
        self.layout.addLayout(layout_progress_bar)
        
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
            if path and os.path.isfile(path[0]):
                self.lineEdit_dir_navSignal.setText(path[0])
                path_save = os.path.join(os.path.dirname(path[0]), '5DED Analysis')
                self.lineEdit_dir_save.setText(path_save)
                
        elif sender == self.button_dir_4dSignals:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select 4D Folder")
            # if path and os.path.isdir(path[0]):
            if path:
                self.lineEdit_dir_4d.setText(path)
                
        elif sender == self.button_dir_save:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select Destination Folder")
            if path and os.path.isdir(path[0]):
                self.lineEdit_dir_save.setText(path)
    
    def load_navSignal(self):
        def get_signal(fn):
            return load(fn)
        
        self.load_spinner()
        gc.collect()
        #TODO delete the previous batch
        if hasattr(self, 'rois_tracked'):
            self.reset_rois()
            self.disable_3ded_widgets(True)
                    
        fn = self.lineEdit_dir_navSignal.text()
        # self.s = load(fn)
        worker = WorkerThread_General(get_signal, 0, fn)
        worker.signals.results.connect(self.initiate_processing)
        self.threadpool.start(worker)

    def initiate_processing(self, result, index):
        self.empty_main_dataframe()
        self.s = result
        self.s_8bit = io.convert_to_8bit(self.s)
        self.nav_imgs_raw = self.s.data
        self.nav_imgs = deepcopy(self.s_8bit.data)
        self.spinner.stop()
        
        shape_x, shape_y = self.nav_imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['track'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['nav'].set_clim(vmin=self.nav_imgs.min(), vmax=self.nav_imgs.max())
        self.img_display['track'].set_clim(vmin=self.nav_imgs.min(), vmax=self.nav_imgs.max())
        self.lineEdit_imgNo.setValidator(QIntValidator(0, len(self.nav_imgs)))
        
        self.update_canvas(0)
        self.canvas.draw()
        self.slider_imgNo.setRange(0, len(self.nav_imgs)-1)
        print('No. of Images:', len(self.nav_imgs))
        
        self.button_reset_rois.setEnabled(True)
        # self.button_cur_roi.setEnabled(True)
        self.button_track.setEnabled(True)
    
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
        self.patches_tracked.clear()

    def blur_navImages(self):
        kernelSize = int(self.combo_blur_track.currentText())
        new_images = np.zeros_like(self.nav_imgs)
        for i, img in enumerate(self.s_8bit.data):
            new_images[i] = io.gaussian_blur(img, kernelSize)
        self.nav_imgs = new_images
        self.update_canvas()
        
    def reset_rois(self):
        self.tree_objects.clear()
        self.empty_main_dataframe()
        self.update_canvas()
#%% canvas functions    
    def jump_to_frame_no(self):
        num = int(self.lineEdit_imgNo.text())
        self.slider_imgNo.setValue(num)
    
    def update_canvas(self, imgNo=None):
        if imgNo is None:
            imgNo = self.slider_imgNo.value()
            
        img = self.nav_imgs[imgNo]
        
        self.update_ax(img, 'nav', self.ax_nav, f'Nav Image No. {imgNo+1:d}')
        self.draw_rois_in(imgNo)
        
        selected_items = self.tree_objects.selectedItems()
        if selected_items:
            item = selected_items[0]
            idx = int(item.text(1))
            # track
            if not np.all(pd.isna(self.df_rois.loc[idx, 'out_rois'])):
                self.update_ax(img, 'track', self.ax_track, f'Nav Image No. {imgNo+1:d}')
                self.draw_rois_out(imgNo)

                # mask
                img_mask, img_roi = self.threshold_img(
                    img, self.df_rois.loc[idx, 'out_rois'][imgNo], 
                    self.combo_thresh_method.currentText(),
                    self.slider_thresh.value())
                self.update_ax_mask(img_roi, img_mask)
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
                
    def update_ax(self, img, img_disp, ax, title=None,):
        # self.canvas.restore_region(self.canvas.copy_from_bbox(ax.bbox))
        self.img_display[img_disp].set_data(img)
        ax.set_title(title)
        self.img_display[img_disp].set_clim(vmin=img.min(), vmax=img.max())
        # self.canvas.blit(ax.bbox)
        self.canvas.draw_idle()
    
    def draw_rois_in(self, imgNo):
        self.canvas.restore_region(self.canvas.copy_from_bbox(self.ax_nav.bbox))
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
        self.canvas.blit(self.ax_nav.bbox)
        # self.canvas.draw_idle()
                
    def draw_rois_out(self, imgNo):
        self.canvas.restore_region(self.canvas.copy_from_bbox(self.ax_track.bbox))
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
        self.canvas.blit(self.ax_track.bbox)
            
    def update_ax_mask(self, img_roi, img_mask):
        self.canvas.restore_region(self.canvas.copy_from_bbox(self.ax_mask.bbox))
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
        self.canvas.blit(self.ax_mask.bbox)
        # self.canvas.draw_idle()
        
    def update_scalebar(self, which):
        if which == 'real':
            scale_real = self.lineEdit_scale_real.text()
            try:
                scale_real = float(scale_real)
                for ax in [self.ax_nav, self.ax_track, self.ax_mask]:
                    self.canvas.restore_region(self.canvas.copy_from_bbox(ax.bbox))
                    scalebar_patch = ScaleBar(scale_real, 'nm', dimension='si-length', 
                                              location='lower left', box_alpha=0, color='w')
                    for artist in ax.artists:
                        if isinstance(artist, ScaleBar):
                            artist.remove()
                            ax.add_artist(scalebar_patch)
                    self.canvas.blit(ax.bbox)
            except:
                for ax in [self.ax_nav, self.ax_track, self.ax_mask]:
                    self.canvas.restore_region(self.canvas.copy_from_bbox(ax.bbox))
                    for artist in ax.artists:
                        if isinstance(artist, ScaleBar):
                            artist.remove()
                    self.canvas.blit(ax.bbox)

        elif which == 'reciprocal':
            scale_recip = self.lineEdit_scale_recip.text()
            try:
                scale_recip = float(scale_recip)
                self.canvas.restore_region(self.canvas.copy_from_bbox(self.ax_dp.bbox))
                scalebar_recip = ScaleBar(scale_recip*10, '1/nm', dimension='si-length-reciprocal', location='lower left',
                                    box_alpha=0, color='w',
                                    scale_formatter=lambda value, unit:  f'{value / 10}'r' $\AA^{-1}$', fixed_value=5)
                for artist in self.ax_dp.artists:
                        if isinstance(artist, ScaleBar):
                            artist.remove()
                self.ax_dp.add_artist(scalebar_recip)
                self.canvas.blit(self.ax_dp.bbox)
            except:
                self.canvas.restore_region(self.canvas.copy_from_bbox(self.ax_dp.bbox))
                for artist in self.ax_dp.artists:
                        if isinstance(artist, ScaleBar):
                            artist.remove()
                self.canvas.blit(self.ax_dp.bbox)

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
        img_mask = img_blur > thresh
        img_mask = img_mask[x:x+w, y:y+h]
        return img_mask, img_cut

    def on_press(self, event):
        # Mouse press event: record the starting point
        self.press = (event.xdata, event.ydata)
        if event.inaxes == self.ax_nav:
            if self.rect is not None:
                self.rect.remove()
            self.rect = patches.Rectangle(self.press, 0, 0, linewidth=1, 
                                          edgecolor='r', facecolor='none')
            self.patches_axNav.append(self.rect)
            self.ax_nav.add_patch(self.rect)
            self.canvas.draw()
            self.backgrounds['nav'] = self.canvas.copy_from_bbox(self.ax_nav.bbox)
            
        elif (event.inaxes == self.ax_track):
            self.canvas.restore_region(self.backgrounds['track'])
            if self.rect_roiInRoi is not None:
                self.rect_roiInRoi.remove()
            
            self.rect_roiInRoi = patches.Rectangle(self.press, 0, 0, linewidth=1, 
                                                   edgecolor='r', facecolor='none')
            self.patches_axTrack.append(self.rect_roiInRoi)
            self.ax_track.add_patch(self.rect_roiInRoi)
            self.canvas.draw()
            self.backgrounds['track'] = self.canvas.copy_from_bbox(self.ax_track.bbox)
            
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
            self.canvas.restore_region(self.backgrounds['nav'])
            self.ax_nav.draw_artist(self.rect)
            self.canvas.blit(self.ax_nav.bbox)
            
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
                print('First select a reference ROI')
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
        
            self.canvas.restore_region(self.backgrounds['track'])
            self.ax_track.draw_artist(self.rect_roiInRoi)
            self.canvas.blit(self.ax_track.bbox)


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
            self.backgrounds['track'] = self.canvas.copy_from_bbox(self.ax_track.bbox)
            self.canvas.restore_region(self.backgrounds['track'])
            self.ax_track.draw_artist(t)
            self.canvas.blit(self.ax_track.bbox)
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
#%%
    def add_item_tree(self, idx, init=[0], end=None, ref=None):
        cols = {col: i for i,col in enumerate(self.cols_tree)}
        item = qtw.QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        # item.setCheckState(cols['use'], Qt.Unchecked)
        item.setCheckState(cols['use'], Qt.Checked)
        item.setText(cols['idx'], f"{idx}")
        item.setText(cols['init'], f"{init}")
        self.tree_objects.itemChanged.connect(self.on_item_check_changed)
        
        self.tree_objects.addTopLevelItem(item)

        # end frame
        spinbox = qtw.QSpinBox()
        spinbox.setRange(0, len(self.nav_imgs))
        spinbox.setValue(len(self.nav_imgs))
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
        
        # delete
        delete_button = qtw.QPushButton()
        delete_button.setIcon(self.style().standardIcon(qtw.QStyle.SP_TrashIcon))
        delete_button.setFixedSize(30, 30)
        delete_button.setToolTip("Delete this item")
        
        def delete_row():
            index = self.tree_objects.indexOfTopLevelItem(item)
            # print(index)
            self.tree_objects.takeTopLevelItem(index)
            self.df_rois = self.df_rois.drop(self.df_rois.index[index])
            # print(self.df_rois)
            self.update_canvas()

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
        print("Checked Items:", checked)
    
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
        self.canvas.draw()
            
    def track_rois(self):
        self.load_spinner()
        if len(self.df_rois) == 0:
            return
        df = self.df_rois[self.df_rois.use == 1]
        if len(df) == 0:
            return
        
        self.tracking_counter = 0
        self.tracking_finished = False
        tracking_method = self.combo_trackMethod.currentText()

        self.tracking_counter_end = len(df.index)
        for ind in df.index:
            init = df.loc[ind, 'init']
            beg = min(init)
            end = df.loc[ind, 'end']
            imgs = self.nav_imgs[beg:end]
            
            rois_in = df.loc[ind, 'in_rois']
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
        self.tracking_counter += 1
        self.df_rois.at[index, 'out_rois'] = np.zeros((len(self.nav_imgs), 4), dtype=np.int16)
        st = min(self.df_rois.loc[index, 'init'])
        end = self.df_rois.loc[index, 'end']
        print(st, end, len(result))
        self.df_rois.at[index, 'out_rois'][st:end] = result
        self.toggle_tree_icon(self.df_rois.index.get_loc(index), 'trk', True)
        self.update_progress_bar(self.tracking_counter, self.tracking_counter_end)
        if self.tracking_counter == self.tracking_counter_end:
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
            self.canvas.draw()
            self.spinner.stop()
     
    def extract_3ded(self):
        self.load_spinner()
        
        path_4d = self.lineEdit_dir_4d.text()
        if path_4d == '': # no entry in 4D signals path
            self.spinner.stop()   
            qtw.QMessageBox.critical(self, 'No Entry', 'Enter the path for 4D signals before the extraction!')
            return
        
        # check if no of files matches with no of images
        fns_4d = glob(os.path.join(path_4d, '*'))
        if len(fns_4d) == 0:
            self.spinner.stop()   
            qtw.QMessageBox.critical(self, 'Wrong Path', 'No files was found in the path for 4D signals!')
            return
        if len(self.nav_imgs) != len(fns_4d):
            reply = qtw.QMessageBox.question(self, 'Mismatch',
                   'No. of 4D signals mismatches the number of images. Do you want to continue?',)
            if reply == qtw.QMessageBox.No:
                self.spinner.stop()   
                return

        dtype = os.path.splitext(fns_4d[0])[-1]
        blur_kernel = int(self.combo_blur.currentText())
        # make masks
        thresh_method = self.combo_thresh_method.currentText()
        thresh_offset = self.slider_thresh.value() / 100
        for ind in self.df_rois[self.df_rois.use == 1].index:
            beg = min(self.df_rois.loc[ind, 'init'])
            end = self.df_rois.loc[ind, 'end']
            self.df_rois.at[ind, 'mask'] = tr.create_masks(
                self.nav_imgs[beg:end], self.df_rois.loc[ind, 'out_rois'],
                thresh_method, thresh_offset, blur_kernel)
            

        # set detector size for tpx3
        if dtype == '.tpx3': # TODO not good
            shape_d_x, shape_d_y = 512, 512
        else:
            shape_d_x, shape_d_y = io.get_det_size(fns_4d[0])
        scanSize = self.nav_imgs.shape[1:]
        
        df = self.df_rois[self.df_rois['use'] == 1]
                
        self.tic = perf_counter()
        
        self.tomo_counter = 0
        self.tasks = []
        self.temp_dir = self.get_temp_dir()
        lengths = df.end - [min(df.init[idx]) for idx in df.index]
        self.tomo_counter_total = np.sum(lengths)
        self.update_progress_bar(0, self.tomo_counter_total)
        for idx in df.index:
            self.df_rois.at[idx, 'dp'] = np.zeros((lengths.loc[idx], shape_d_x, 
                                                   shape_d_y), dtype='uint32')
            beg = min(df.loc[idx].init)
            end = df.loc[idx].end
            for i_fr, fn in enumerate(fns_4d[beg:end]):
                i_fr += beg
                self.tasks.append([fn, df.loc[idx, 'out_rois'][i_fr],
                                   os.path.join(self.temp_dir, f"mask_r{idx}_f{i_fr}.npy"),
                                   dtype, scanSize, (idx, i_fr)])

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
        print('temp dir', temp_dir)
        return temp_dir
    
# =============================================================================
#     def launch_initial_tasks(self):
#         for _ in range(min(self.max_processes, len(self.tasks))):
#             self.launch_next_task()
# =============================================================================

    def launch_next_task(self):
        if not self.tasks or len(self.running_processes) >= self.max_processes:
            return
    
        args = self.tasks.pop(0)
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
        print("QProcess error occurred:", error)    
        
    def handle_error(self, process):
        error_output = process.readAllStandardError().data().decode()
        print("Worker ERROR:", error_output)
        self.spinner.stop()
    
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
            print("Unknown process")
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
        
        # progress bar update
        self.tomo_counter += 1
        self.update_progress_bar(self.tomo_counter, self.tomo_counter_total)
        
        if self.tomo_counter >= self.tomo_counter_total:
            self.toc = perf_counter()
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            time = self.toc - self.tic
            print(f'Data Extraction Time: {time/60:.1f} min')
            self.spinner.stop()
            for idx in self.df_rois[self.df_rois.use == 1].index:
                self.toggle_tree_icon(self.df_rois.index.get_loc(idx), 'ext', True)
            
            if self.checkbox_autosave.isChecked():
                self.save_results()
        else:
            self.launch_next_task()  # trigger next task if any left
        
    def update_progress_bar(self, value, total):
        value = value / total * 100
        value = int(value)
        self.progress_bar.setValue(value)
    
    def reset_thresh(self):
        self.slider_thresh.setValue(100)
        self.update_canvas()
    
    def save_results(self):
        path_save = self.lineEdit_dir_save.text()
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        
        fld_1 = datetime.date.today()
        fld_2 = datetime.datetime.now().strftime("%H-%M-%S")
        
        path_save = os.path.join(path_save, f'{fld_1}__{fld_2}')
        os.mkdir(path_save)
        
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
            
            # write frames
            np.save(os.path.join(path_save_roi, '3DED.npy'), self.df_rois.loc[idx, 'dp'])
            fld_frames = os.path.join(path_save_roi, 'frames')
            worker_frames = WorkerThread_General(io.create_frames, 0, fld_frames, self.df_rois.loc[idx, 'dp'])
            self.threadpool.start(worker_frames)
            
            # clip for dp
            scale_recip = self.lineEdit_scale_recip.text()
            try:
                scale_recip = float(scale_recip)
            except:
                scale_recip = None
            fn_dp = os.path.join(path_save_roi, 'tomo clip')
            worker_clip_dp = WorkerThread_General(io.create_clip_dp, 0, fn_dp, 
                                                  self.df_rois.loc[idx, 'dp'], scale_recip)
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
                self.df_rois.loc[idx, 'out_rois'], scale_real)
            self.threadpool.start(worker_clip_tr_ref)
            
    def kill_running_process(self):
        for process in self.running_processes:
            process.kill()  # Forcefully terminates the subprocess
            process.deleteLater()
        self.running_processes.clear()

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

