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
from PyQt5.QtGui import QDoubleValidator
from matplotlib.colors import SymLogNorm
import numpy as np
import py4DTomo.io_utils as io
from hyperspy.api import load, signals
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
#%% wdiget
class Tab_Tracking_CV2(qtw.QWidget):
# class Tab_Create_NavSignal(qtw.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.init_widget()
        
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
        
        self.lineEdit_scale_recip.textChanged.connect(lambda: self.update_canvas(
            self.slider_imgNo.value()))
        self.lineEdit_scale_real.textChanged.connect(lambda: self.update_canvas(
            self.slider_imgNo.value()))
        
        layout_loadSignal.addItem(spacer)
        
        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        layout_dir.addLayout(layout_loadSignal)
        layout_loadSignal.addWidget(self.button_loadNavigation, alignment=Qt.AlignCenter)
        self.button_loadNavigation.setFixedSize(button_w, button_h_lrg)
        self.button_loadNavigation.clicked.connect(self.load_navSignal)
        #%% feature handling
        layout_topMiddle = qtw.QVBoxLayout()
        layout_top.addLayout(layout_topMiddle)
        self.box_buttons = qtw.QGroupBox('Features Handling')
        self.box_buttons.setFixedSize(350, height_layout_top)
        # self.box_buttons.setFixedWidth(350)
        self.box_buttons.setFixedHeight(height_layout_top//3)
        layout_topMiddle.addWidget(self.box_buttons)
        
        # layout_topMiddle.setAlignment(Qt.AlignLeft)
        
        layout_features = qtw.QHBoxLayout(self)
        self.box_buttons.setLayout(layout_features)
        label_roiNo = qtw.QLabel('Roi No')
        layout_features.addWidget(label_roiNo)
        self.combo_roiNo = qtw.QComboBox()
        layout_features.addWidget(self.combo_roiNo)
        self.combo_roiNo.currentIndexChanged.connect(lambda: self.plot_image_mask(
            self.slider_thresh.value()))
        self.combo_roiNo.currentIndexChanged.connect(self.plot_dp)

# =============================================================================
#         # spacer
#         spacer = qtw.QSpacerItem(40, 20, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
#         layout_features.addItem(spacer)
# =============================================================================
        
        self.button_cur_roi = qtw.QPushButton('Delete Current ROI')
        layout_features.addWidget(self.button_cur_roi)
        self.button_cur_roi.setFixedSize(button_w, button_h_sml)
        self.button_cur_roi.clicked.connect(self.del_current_roi)
        self.button_cur_roi.setDisabled(True)
        
        self.button_reset_rois = qtw.QPushButton('Reset ROIs')
        layout_features.addWidget(self.button_reset_rois)
        self.button_reset_rois.setFixedSize(button_w, button_h_sml)
        self.button_reset_rois.clicked.connect(self.reset_rois)
        self.button_reset_rois.setDisabled(True)
        
        #%% tracker
        self.box_tracker = qtw.QGroupBox('OpenCV Tracker')
        # self.box_tracker.setFixedSize(300, 120)
        self.box_tracker.setFixedWidth(350)
        self.box_tracker.setFixedHeight(height_layout_top//3 *2)
        layout_topMiddle.addWidget(self.box_tracker)
        layout_tracker = qtw.QHBoxLayout(self)
        self.box_tracker.setLayout(layout_tracker)
        
        layout_tracker_1 = qtw.QVBoxLayout()
        layout_tracker.addLayout(layout_tracker_1)
        layout_tracker_1_2 = qtw.QHBoxLayout()
        layout_tracker_1.addLayout(layout_tracker_1_2)
        
        layout_tracker_1_2.addItem(spacer)
        label_track = qtw.QLabel('Method')
        layout_tracker_1_2.addWidget(label_track)
        self.combo_trackMethod = qtw.QComboBox()
        layout_tracker_1_2.addWidget(self.combo_trackMethod)
        self.combo_trackMethod.addItems(['csrt', 'nano', 'mil', 'dasiamrpn'])
        layout_tracker_1_2.addItem(spacer)
        
        # layout_tracker.addItem(spacer)
        
        self.button_track = qtw.QPushButton('Track!')
        layout_tracker_1.addWidget(self.button_track, alignment=Qt.AlignCenter)
        self.button_track.setFixedSize(button_w, button_h_lrg)
        self.button_track.clicked.connect(
            lambda: self.track_rois(self.nav_imgs, self.rois))
        self.button_track.setDisabled(True)
        
# =============================================================================
#         # layout_features.addStretch(1)
#         spacer = qtw.QSpacerItem(40, 20, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
#         layout_top.addItem(spacer)
# =============================================================================
        #%% roi in roi box
        self.box_roiInRoi = qtw.QGroupBox('ROI in ROI')
        layout_box_roiInRoi = qtw.QVBoxLayout()
        self.box_roiInRoi.setLayout(layout_box_roiInRoi)
        layout_tracker.addWidget(self.box_roiInRoi)
        # self.box_roiInRoi.setFixedWidth(150)
        # self.box_roiInRoi.setFixedHeight(100)
        
        self.checkbox_roiInRoi = qtw.QCheckBox('Activate')
        layout_box_roiInRoi.addWidget(self.checkbox_roiInRoi)
        self.checkbox_roiInRoi.setDisabled(True)
        
        # TODO fixed position for roi in roi
# =============================================================================
#         self.checkbox_roiInRoi_constPos = qtw.QCheckBox('Constant Position')
#         layout_box_roiInRoi.addWidget(self.checkbox_roiInRoi_constPos)
#         self.checkbox_roiInRoi_constPos.setDisabled(True)
# =============================================================================
        
        self.button_trackAgain = qtw.QPushButton("Track Again")
        layout_box_roiInRoi.addWidget(self.button_trackAgain, alignment=Qt.AlignCenter)
        # self.button_trackAgain.setFixedSize(75, 50)
        self.button_trackAgain.setFixedSize(button_w, button_h_sml)
        self.button_trackAgain.clicked.connect(self.track_roiInRoi)
        self.button_trackAgain.setDisabled(True)
        
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
        self.combo_thresh_method.addItems(['otsu', 'li', 'yen', 'mean'])
        self.combo_thresh_method.currentIndexChanged.connect(lambda: self.plot_image_mask(
            self.slider_thresh.value()))
        layout_thresh_method.addItem(spacer)
        
        layout_deviation = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_deviation)
        label_thresh_dev = qtw.QLabel('Deviation')
        layout_deviation.addWidget(label_thresh_dev)
        
        self.slider_thresh = qtw.QSlider(1)
        layout_deviation.addWidget(self.slider_thresh)
        self.slider_thresh.setDisabled(True)
        # self.slider_thresh.setFixedWidth(100)
        self.slider_thresh.valueChanged.connect(self.plot_image_mask)
        self.slider_thresh.setRange(0, 200)
        
        self.button_thresh = qtw.QPushButton('Reset')
        layout_deviation.addWidget(self.button_thresh)
        self.button_thresh.clicked.connect(self.reset_thresh)
            #%% 3DED
        layout_roi_frame = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_roi_frame)
        label_roi_3ded = qtw.QLabel('Roi No')
        layout_roi_frame.addWidget(label_roi_3ded)
        self.combo_3ded = qtw.QComboBox()
        self.combo_3ded.setFixedWidth(50)
        layout_roi_frame.addWidget(self.combo_3ded)
        # layout_roi_frame.addItem(spacer)
        
        label_threadNo = qtw.QLabel('Thread No')
        layout_roi_frame.addWidget(label_threadNo)
        self.spinbox_threadNo = qtw.QSpinBox(self)
        layout_roi_frame.addWidget(self.spinbox_threadNo)
        self.spinbox_threadNo.setRange(1,os.cpu_count()-1)
        self.spinbox_threadNo.setValue(3)
        self.spinbox_threadNo.valueChanged.connect(self.set_threadNo)
        
        
        # layout_finalFrame = qtw.QHBoxLayout()
        # layout_box_3ded.addLayout(layout_finalFrame)
        label_finalFrame = qtw.QLabel('Final Frame')
        self.spinbox_finalFrame = qtw.QSpinBox()
        self.spinbox_finalFrame.setMinimum(1)
        self.spinbox_finalFrame.setFixedWidth(50)
        for wid in [label_finalFrame, self.spinbox_finalFrame]:
            layout_roi_frame.addWidget(wid)
        
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
        
        layout_plotCombo = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_plotCombo)
        label_dpPlotCombo = qtw.QLabel('DP Plot Ref ROI')
        self.combo_dp_plot = qtw.QComboBox()
        for wid in [label_dpPlotCombo, self.combo_dp_plot]:
            layout_plotCombo.addWidget(wid)

        self.update_combo_3ded()
        self.combo_dp_plot.currentIndexChanged.connect(lambda: self.update_canvas(
            self.slider_imgNo.value()))
        #%% canvas
        layout_canvas = qtw.QHBoxLayout()
        self.layout.addLayout(layout_canvas)
        
        # self.figure = Figure(figsize=(8,4))
        self.figure = Figure()
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
        axes = ['nav', 'track', 'dp']
        for i, ax in enumerate([self.ax_nav, self.ax_track, self.ax_dp]):
            self.img_display[axes[i]] = ax.imshow(self.img_zero, cmap='viridis') 
            ax.set_axis_off()
        self.ax_mask.set_axis_off()
        self.img_display['img_mask'] = self.ax_mask.imshow(self.img_zero, cmap='gray')
        self.img_display['mask'] = self.ax_mask.imshow(self.img_zero, cmap='viridis', alpha=0.25)
        self.img_display['dp'].set_norm(SymLogNorm(linthresh=1))
        self.img_display['dp'].set_cmap('inferno')
        self.figure.tight_layout()
        # self.layout.addWidget(self.canvas)
        layout_canvas.addWidget(self.canvas)
        
        # Connect mouse events
        self.rect = None            # Currently drawn rectangle
        self.rect_roiInRoi = None
        self.press = None           # Mouse press coordinates
        self.rois = {}              # List to store all rois
        self.rois_roiInRoi = {}
        self.roi_counter = 0            # Counter for numbering rois
        self.roi_counter_roiInRoi = 0
        self.patches_toTrack = []
        self.patches_toTrack_roiInRoi = []

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        #%% slider layout
        layout_slider = qtw.QHBoxLayout(self)
        self.layout.addLayout(layout_slider)

        layout_slider.addWidget(NavigationToolbar(self.canvas, self))
        
        vline = qtw.QFrame()
        vline.setFrameShape(qtw.QFrame.VLine)
        vline.setFrameShadow(qtw.QFrame.Sunken)
        layout_slider.addWidget(vline)
        
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
        self.layout.addLayout(layout_progress_bar)
        
        self.progress_bar = qtw.QProgressBar()
        layout_progress_bar.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)
    #%% functions
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
    
    def update_canvas(self, imgNo=None):
        if not imgNo:
            imgNo = self.slider_imgNo.value()
        if hasattr(self, 'nav_imgs'):
            img = self.nav_imgs[imgNo]
            
            # nav axis
            self.img_display['nav'].set_data(img)
            self.img_display['nav'].set_clim(vmin=img.min(), vmax=img.max())
            shape_x, shape_y = img.shape
            self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
            self.ax_nav.set_title(f'Nav Image No. {imgNo+1:d}')
            self.draw_rois_nav(self.patches_toTrack, self.ax_nav, self.rois)
        # draw rois from roi in roi
        if hasattr(self, 'tracking_finished'):
            if (self.tracking_finished) and (hasattr(self, 'rois_tracked_roiInRoi_trans')):
                self.draw_rois_nav(self.patches_toTrack_roiInRoi, self.ax_nav, self.rois_roiInRoi)
            else:
                self.draw_rois_nav(self.patches_toTrack_roiInRoi, self.ax_track, self.rois_roiInRoi)
        
        # draw image mask
        if hasattr(self, 'rois_tracked'):
            self.plot_tracking_results(imgNo)
            th = self.slider_thresh.value()
            self.plot_image_mask(th)
        
        # draw dp
        if hasattr(self, 'tomo_ds'):
            # roiNo = self.combo_roiNo.currentText()
            roiNo = self.combo_dp_plot.currentText()
            roiNo = eval(roiNo)
            self.plot_dp(roiNo, imgNo)
        
        # scale bars 
        #TODO adding and removing the artist is not efficient
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
            for ax in [self.ax_nav, self.ax_track, self.ax_mask]:
                scalebar_real = ScaleBar(scale_real, 'nm', dimension='si-length', location='lower left')
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
                scalebar_recip = ScaleBar(scale_recip*10, '1/nm', dimension='si-length-reciprocal', location='lower left',
                                    scale_formatter=lambda value, unit:  f'{value / 10}'r' $\AA^{-1}$', fixed_value=5)
                for artist in self.ax_dp.artists:
                        if isinstance(artist, ScaleBar):
                            artist.remove()
                self.ax_dp.add_artist(scalebar_recip)
        except Exception as e:
            # print(e)
            pass
        self.canvas.draw()
            
    def load_navSignal(self):
        gc.collect()
        #TODO delete the previous batch
        if hasattr(self, 'rois_tracked'):
            self.reset_rois()
# =============================================================================
#             del self.rois_tracked
#             img_temp = np.zeros((512,512), dtype='int8')
#             axes = ['nav', 'track', 'mask', 'dp']
#             for i, ax in enumerate([self.ax_nav, self.ax_track, self.ax_mask, self.ax_dp]):
#                 self.img_display[axes[i]] = ax.imshow(img_temp, cmap='viridis')
#             if hasattr(self, 'patches_tracked'):
#                 for p in self.patches_tracked:
#                     p.remove()
#             if hasattr(self, 'patches_toTrack'):
#                 for p in self.patches_toTrack:
#                     p.remove()
# =============================================================================
            self.disable_3ded_widgets(True)
                    
            
        fn = self.lineEdit_dir_navSignal.text()
        self.s = load(fn)
        self.s_8bit = io.convert_to_8bit(self.s)
        self.nav_imgs_raw = self.s.data
        self.nav_imgs = self.s_8bit.data
        self.update_canvas(0)
        self.slider_imgNo.setRange(0, len(self.nav_imgs)-1)
        print('No. of Images:', len(self.nav_imgs))
        
        self.button_reset_rois.setEnabled(True)
        self.button_cur_roi.setEnabled(True)
        self.button_track.setEnabled(True)
    
    def on_press(self, event):
        # Mouse press event: record the starting point
        self.press = (event.xdata, event.ydata)
        if event.inaxes == self.ax_nav:
            if self.rect is not None:
                self.rect.remove()
            self.rect = patches.Rectangle(self.press, 0, 0, linewidth=1, 
                                          edgecolor='r', facecolor='none')
            self.patches_toTrack.append(self.rect)
            self.ax_nav.add_patch(self.rect)
            self.canvas.draw()
            
        elif (event.inaxes == self.ax_track) and (self.checkbox_roiInRoi.isChecked()):
            if self.rect_roiInRoi is not None:
                self.rect_roiInRoi.remove()
            
            self.rect_roiInRoi = patches.Rectangle(self.press, 0, 0, linewidth=1, 
                                                   edgecolor='r', facecolor='none')
            self.patches_toTrack_roiInRoi.append(self.rect_roiInRoi)
            self.ax_track.add_patch(self.rect_roiInRoi)
            self.canvas.draw()
        else:
            self.press = None
        

    def on_motion(self, event):
        # Mouse motion event: update the rectangle size as the mouse moves
        if self.press is None or event.inaxes is None:
            return
        if (event.inaxes == self.ax_track) and (not self.checkbox_roiInRoi.isChecked()):
            return
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
            
        elif (event.inaxes == self.ax_track) and (self.checkbox_roiInRoi.isChecked()):
            x0, y0 = self.press
            width = event.xdata - x0
            height = event.ydata - y0
            
            # confine the roi in roi to the ref roi            
            roiRef = int(self.combo_roiNo.currentText())
            imgNo = self.slider_imgNo.value()
            xr,yr,wr,hr = self.rois_tracked[roiRef][imgNo]
            if x0 > xr+wr:
                x0 = xr+wr
            if x0+width > xr+wr:
                width = (xr+wr) - x0
            elif width<0 and abs(x0+width) < xr:
                width = xr - x0
            
            if y0 > yr+hr:
                y0 = yr+hr
            if y0+height > yr+hr:
                height = (yr+hr) - y0
            elif height<0 and abs(y0+height) < yr:
                height = yr - y0
            
            
            try:
                self.rect_roiInRoi.set_width(width)
                self.rect_roiInRoi.set_height(height)
                self.rect_roiInRoi.set_xy((x0, y0))
            except AttributeError: # Although the checkbox was unchecked, the zoom function was causing this error after checkbox was activated again
                self.press = None
        self.canvas.draw()

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
                
        imgNo = self.slider_imgNo.value()
        if (event.inaxes == self.ax_nav):
            # Store the rectangle's properties
            if imgNo == 0:
                self.roi_counter += 1
                self.rois[self.roi_counter] = {0: roi}
                text_id = str(self.roi_counter)
                self.combo_roiNo.addItem(str(self.roi_counter))
            else:
                roiNo = int(self.combo_roiNo.currentText())
                self.rois[roiNo][imgNo] = roi
                text_id = roiNo
    
            # Draw a label (counter) at the center of the rectangle
            t = self.ax_nav.text(x0 + width/2, y0-15, text_id,
                         horizontalalignment='center', verticalalignment='center', 
                         color='red', fontsize=12)
            self.patches_toTrack.append(t)
            # Set the press attribute to None for future drawings
            self.rect = None
            print('The added ROI coordintations: ', roi)
        
        elif (event.inaxes == self.ax_track) and (self.checkbox_roiInRoi.isChecked()):
            # Store the rectangle's properties
            roiNo_ref = int(self.combo_roiNo.currentText())
            
            # (roiInRoi counter, reference ROI) => imgNo => roi
            if imgNo == 0:
                self.roi_counter_roiInRoi += 1
                self.rois_roiInRoi[(self.roi_counter_roiInRoi, roiNo_ref)] = {0: roi}
            else:
                self.rois_roiInRoi[(self.roi_counter_roiInRoi, roiNo_ref)][imgNo] = roi
            text_id = f'{self.roi_counter_roiInRoi}_{roiNo_ref}'
            
# =============================================================================
#             # reference ROI => roiinroi counter => (imgNo, roi)
#             if imgNo == 0:
#                 self.rois_roiInRoi[roiNo_ref] = {self.roi_counter_roiInRoi: [(0, roi)]}
#                 self.roi_counter_roiInRoi += 1
#             else:
#                 self.rois_roiInRoi[roiNo_ref][self.roi_counter_roiInRoi].append(
#                     (imgNo, roi))
#             text_id = f'{roiNo_ref}_{self.roi_counter_roiInRoi}'
# =============================================================================
    
            # Draw a label (counter) at the center of the rectangle
            t = self.ax_track.text(x0, y0-15, text_id,
                         horizontalalignment='center', verticalalignment='center', 
                         color='red', fontsize=6)
            self.patches_toTrack_roiInRoi.append(t)
            self.rect_roiInRoi = None
            print('The addedd ROI coordintations: ', roi)
        self.canvas.draw()
        self.press = None
    
    def draw_rois_nav(self, patch, ax, rois):
        for p in patch:
            try:
                p.remove()
            # except ValueError:
            except:
                pass
        patch.clear()
        imgNo = self.slider_imgNo.value()
        # roiNo = int(self.combo_roiNo.currentText())
        for rn in rois.keys():
            try:
                roi = rois[rn][imgNo]
                x,y,w,h = roi
                rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='r', 
                                         facecolor='none')
                ax.add_patch(rect)
                patch.append(rect)
                if type(rn) == tuple:
                    pos = (x+w+15, y+h+15)
                    font_size = 8
                else:
                    pos = (x+w/2, y-15)
                    font_size = 12
                t = ax.text(pos[0], pos[1], str(rn), horizontalalignment='center', 
                             verticalalignment='center', color='red', fontsize=font_size)
                patch.append(t)
            except KeyError:
                pass
        self.canvas.draw()
    
    def reset_rois(self): #TODO add a button to reset the last ROI
        # self.ax_nav.clear()  # Clear the axes
        imgNo = self.slider_imgNo.value()
        self.rois.clear()  # Clear the list of rois
        self.roi_counter = 0  # Reset the counter
        self.combo_roiNo.clear()
        self.combo_3ded.clear()
        self.checkbox_roiInRoi.setChecked(False)
        self.disable_3ded_widgets(True)
        self.disable_roiInRoi_widgets(True)
        
        if hasattr(self, 'nav_imgs'): # if images are already loaded
            self.img_display['nav'].set_data(self.nav_imgs[imgNo])
            shape_x, shape_y = self.nav_imgs[imgNo].shape
            self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])

        # delete patches/rois drawn
        for p in self.patches_toTrack:
            p.remove()
        for p in self.patches_toTrack_roiInRoi:
            p.remove()

        # remove rois
        if hasattr(self, 'rois_tracked'):
            del self.rois_tracked
            # re-draw axes with empty images (except for navigation)
            for ax in self.img_display.keys():
                if ax != 'nav':
                    self.img_display[ax].set_data(self.img_zero)

        # delete patches tracked
        if hasattr(self, 'patches_tracked'):
            for p in self.patches_tracked:
                p.remove()
        # delete extracted dps
        if hasattr(self, 'tomo_ds'):
            del self.tomo_ds
        
        if hasattr(self, 'rois_tracked_roiInRoi_trans'):
            del self.rois_tracked_roiInRoi_trans
        
        if hasattr(self, 'rois_roiInRoi'):
            self.rois_roiInRoi = {}
            
        self.canvas.draw()
        self.update_progress_bar(0, 100)
        
    def del_current_roi(self):
        roiNo = self.combo_roiNo.currentText()
        try:
            roiNo = int(roiNo)
            imgNo = self.slider_imgNo.value()
                
            try:
                if len(self.rois[roiNo]) == 1:
                    del self.rois[roiNo]
                    index = self.combo_roiNo.findText(str(roiNo))
                    self.combo_roiNo.removeItem(index)
                else:
                    del self.rois[roiNo][imgNo]
            except Exception as e:
                print(f'Could not delete the ROI: {e}')
            self.update_canvas()
        except ValueError: # there might not be any roi
            pass
        gc.collect()
            
    def track_rois(self, imgs, rois, roiInRoi_index=False):
        if len(rois) == 0:
            return
        self.tracking_counter = 0
        self.tracking_finished = False
        tracking_method = self.combo_trackMethod.currentText()
        # threadpool = QThreadPool()
        if not roiInRoi_index:
            self.rois_tracked = {}
        else:
            self.rois_tracked_roiInRoi = {}
            self.tracking_counter_2 += 1
            rois = {0: rois}
        # print(rois)
        for r_id in rois.keys():
            rs = rois[r_id]
            if roiInRoi_index:
                index = roiInRoi_index
            else:
                index = (r_id, False)
            worker = WorkerThread_General(tr.track_roi_cv2, index, imgs, 
                                          rs, tracking_method)
            worker.signals.results.connect(self.get_tracking_results)  # Connect to result signal
            # worker.signals.finished.connect(self.plot_tracking_result)
            self.threadpool.start(worker)
    
    def track_roiInRoi(self):
        self.tracking_counter_2 = 0
        self.tracking_finished = False
        # imgs_cut = deepcopy(self.nav_imgs)
        self.roiInRoi_ids = list(self.rois_roiInRoi.keys())
        for key in self.rois_roiInRoi.keys():
            roiNo, roi_ref = key
            imgs_cut = []
            # crop rois from images
            for i, img in enumerate(self.nav_imgs):
                y,x,h,w = self.rois_tracked[roi_ref][i]
                imgs_cut.append(img[x:x+w, y:y+h])
            imgs_toTrack = io.create_array_from_dissimilar_imgs(imgs_cut, mode='edge', 
                                                                signal=False) #, counter=self.tracking_counter_2
            # self.tracking_counter_2 += 1
            # s.plot()
            
            # create new rois
            rois_toTrack = {}
            for imgNo, roi in self.rois_roiInRoi[key].items():
                # translate roi coords to the cut img
                x,y,w,h = roi
                x0,y0,w0,h0 = self.rois_tracked[roi_ref][imgNo]
                # new width and height might be larger than the first ROI
                if x+w > x0+w0:
                    w = x-(x0+w0)
                if y+h > y0+h0:
                    h = y-(y0+h0)
                rois_toTrack[imgNo] = (x-x0, y-y0, w, h)
            self.track_rois(imgs_toTrack, rois_toTrack, key)
            
            
    def get_tracking_results(self, result, index):
        roiInRoi = False
        if index[1]:
            roiInRoi = True
        
        if not roiInRoi:
            self.tracking_counter += 1
            index = index[0]
            self.rois_tracked[index] = result
            self.update_progress_bar(self.tracking_counter, len(self.rois))
            if self.tracking_counter == len(self.rois):
                self.tracking_finished = True
        
        else:
            self.rois_tracked_roiInRoi[index] = result
            self.update_progress_bar(self.tracking_counter_2, len(self.rois_roiInRoi))
            if self.tracking_counter_2 == len(self.rois_roiInRoi):
                self.tracking_finished = True
            
        if self.tracking_finished:
            # in roiInRoi translate tracked ROIs to the full images
            if roiInRoi:
                self.rois_tracked_roiInRoi_trans = {}
                for key in self.rois_tracked_roiInRoi.keys():
                    roiID, roi_ref = key
                    rois = self.rois_tracked_roiInRoi[key]
                    rois_trans = []
                    for i, roi in enumerate(rois):
                        x,y,w,h = roi
                        x0,y0,w0,h0 = self.rois_tracked[roi_ref][i]
                        roi_translated = [x+x0, y+y0, w, h]
                        rois_trans.append(roi_translated)
                    self.rois_tracked_roiInRoi_trans[key] = rois_trans
            self.slider_imgNo.setValue(0)
            # activating widgets
            self.slider_thresh.setEnabled(True)
            self.slider_thresh.setValue(100)
            self.update_combo_3ded()
            self.disable_3ded_widgets(False)
            self.disable_roiInRoi_widgets(False)
            self.update_canvas(0)
            self.spinbox_finalFrame.setMaximum(len(self.nav_imgs))
            self.spinbox_finalFrame.setValue(len(self.nav_imgs))
     
    def extract_3ded(self):
        self.combo_dp_plot.clear()
        rois_selected = self.combo_3ded.currentText()
        rois_all = [eval(self.combo_3ded.itemText(i)) for i in range(self.combo_3ded.count()) if self.combo_3ded.itemText(i) != 'all']
        if rois_selected == 'all':
            rois_selected = rois_all
        else:
            pass
        self.combo_dp_plot.addItems([str(item) for item in rois_selected])
        path_4d = self.lineEdit_dir_4d.text()
        if path_4d == '': # no entry in 4D signals path
            qtw.QMessageBox.critical(self, 'No Entry', 'Enter the path for 4D signals before the extraction!')
            return

        fns_4d = glob(os.path.join(path_4d, '*'))
        if len(fns_4d) == 0:
            qtw.QMessageBox.critical(self, 'Wrong Path', 'No files was found in the path for 4D signals!')
            return
        dtype = os.path.splitext(fns_4d[0])[-1]
        
        # check if no of files with no of images
        if len(self.nav_imgs) != len(fns_4d):
            reply = qtw.QMessageBox.question(self, 'Mismatch',
                   'No. of 4D signals mismatches the number of images. Do you want to continue?',)
            if reply == qtw.QMessageBox.No:
                return
            
        
        # make masks
        thresh_method = self.combo_thresh_method.currentText()
        
        rois_to_extract = {}
        for item in rois_selected:
            print(item)
            if type(item) == int:
                rois_to_extract[item] = self.rois_tracked[item]
            else:
                rois_to_extract[item] = self.rois_tracked_roiInRoi_trans[item]
        
        self.masks = tr.create_masks(self.nav_imgs, rois_to_extract, 
                                thresh_method, self.thresh_offset)
        
        # set detector size for tpx3
        if dtype == '.tpx3': # TODO not good
            shape_d_x, shape_d_y = 512, 512
        else:
            shape_d_x, shape_d_y = io.get_det_size(fns_4d[0])
        scanSize = self.nav_imgs.shape[1:]
        
        self.tomo_counter = 0
        self.tomo_ds = {}
        
        # extract to a certain frame no
        final_frame = self.spinbox_finalFrame.value()
        fns_4d = fns_4d[:final_frame]
        self.tomo_counter_total = len(rois_to_extract) * len(fns_4d)
        self.update_progress_bar(0, self.tomo_counter_total)
        
        # threading images of each roi
        self.tic = perf_counter()
        for r_id in rois_to_extract.keys():
            self.tomo_ds[r_id] = np.zeros((len(fns_4d), shape_d_x, shape_d_y), dtype='uint32')
            for i_fr, fn in enumerate(fns_4d):
                worker = WorkerThread_General(tr.extract_3ded_mask_single_frame, 
                                              (r_id, i_fr), fn, self.masks[r_id][i_fr], 
                                              dtype, scanSize, rois_to_extract[r_id][i_fr])
            
                worker.signals.results.connect(self.get_tomo_results)  # Connect to result signal
                # worker.signals.finished.connect(self.plot_tracking_result)
                self.threadpool.start(worker)
            
# =============================================================================
#         # threading rois
#         for r_id in self.rois_tracked.keys():
#             worker = WorkerThread_General(tr.extract_3ded_mask, r_id, fns_4d, masks[r_id])
#             
#             worker.signals.results.connect(self.get_tomo_results)  # Connect to result signal
#             # worker.signals.finished.connect(self.plot_tracking_result)
#             self.threadpool.start(worker)
# =============================================================================

    def get_tomo_results(self, result, index):
        r_id, i_fr = index
        # print(f'ROI No: {r_id}, frame No: {i_fr}') # check results
        self.tomo_ds[r_id][i_fr] = result
        
        # progress bar update
        self.tomo_counter += 1
        self.update_progress_bar(self.tomo_counter, self.tomo_counter_total)
        # plot results at the end
        if self.tomo_counter == self.tomo_counter_total:
            roiNo = self.combo_dp_plot.currentText()
            roiNo = eval(roiNo)
            try: # roi in roi is tuple
                roiRef = roiNo[1]
            except:
                roiRef = roiNo
            self.combo_roiNo.setCurrentText(str(roiRef))
            imgNo = self.slider_imgNo.value()
            self.plot_dp(roiNo, imgNo)
            self.update_canvas(imgNo)
            toc = perf_counter()
            print(f'Extraction Duration: {(toc-self.tic)//60:.0f} min')
            # self.combo_roiNo.currentIndexChanged.connect(lambda: self.plot_dp(roiNo, imgNo))
    
    def plot_dp(self, roiNo=None, imgNo=None):
        if hasattr(self, 'tomo_ds'):
            if not imgNo:
                imgNo = self.slider_imgNo.value()
            if not roiNo:
                roiNo = int(self.combo_dp_plot.currentText())
            try:
                img = self.tomo_ds[roiNo][imgNo]
            except IndexError:
                img = self.img_zero
            self.img_display['dp'].set_data(img)
            self.img_display['dp'].set_clim(vmin=img.min(), vmax=img.max())
            shape_x, shape_y = img.shape
            self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])
    
    def update_progress_bar(self, value, total):
        value = value / total * 100
        value = int(value)
        self.progress_bar.setValue(value)
    
    def plot_tracking_results(self, imgNo):
        if hasattr(self, 'patches_tracked'):
            try:
                for p in self.patches_tracked:
                    p.remove()
            except ValueError:
                pass
        self.patches_tracked = []
        
        img = self.nav_imgs[imgNo]
        self.img_display['track'].set_data(img)
        self.img_display['track'].set_clim(vmin=img.min(), vmax=img.max())
        shape_x, shape_y = img.shape
        self.img_display['track'].set_extent([0, shape_y, shape_x, 0])

        for i_r in self.rois_tracked.keys():
            roi = self.rois_tracked[i_r][imgNo]
            x,y,w,h = roi
            rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='tab:orange', 
                                     facecolor='none')
            self.ax_track.add_patch(rect)
            t = self.ax_track.text(x + w/2, y - 15, str(i_r), horizontalalignment='center', 
                                   verticalalignment='center', color='tab:orange', fontsize=12)
            self.patches_tracked.append(rect)
            self.patches_tracked.append(t)
            self.ax_track.set_title(f'Tracking Result: Image No. {imgNo+1:d}')
        
        # plot roi in roi
        if hasattr(self, 'rois_tracked_roiInRoi_trans'):
            for index in self.rois_tracked_roiInRoi_trans.keys():
                roi = self.rois_tracked_roiInRoi_trans[index][imgNo]
                x,y,w,h = roi
                rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='tab:orange', 
                                         facecolor='none')
                self.ax_track.add_patch(rect)
                t = self.ax_track.text(x+w, y+h, f'{index[0]}_{index[1]}', horizontalalignment='center', 
                                       verticalalignment='center', color='tab:orange', fontsize=8)
                self.patches_tracked.append(rect)
                self.patches_tracked.append(t)
        
    def plot_image_mask(self, thresh_offset):
        if hasattr(self, 'rois_tracked') and (self.combo_roiNo.currentText() != ''):
            self.thresh_offset = thresh_offset / 100
            imgNo = self.slider_imgNo.value()
            roiNo = int(self.combo_roiNo.currentText())
            roi = self.rois_tracked[roiNo][imgNo]
            y,x,h,w = roi
            img = deepcopy(self.nav_imgs[imgNo][x:x+w, y:y+h])
            
            thresh_method = self.combo_thresh_method.currentText()
            threshold_methods = {
            'otsu': threshold_otsu,
            'li': threshold_li,
            'yen': threshold_yen,
            'mean': threshold_mean}
            threshold_func = threshold_methods[thresh_method]
            th = threshold_func(img)
            
            self.thresh = self.thresh_offset * th
            img_mask = img > self.thresh
            
            self.img_display['img_mask'].set_data(img)
            self.img_display['img_mask'].set_clim(vmin=img.min(), vmax=img.max())
            self.img_display['mask'].set_data(img_mask)
            
            shape_x, shape_y = img_mask.shape
            self.img_display['mask'].set_clim(vmin=0, vmax=1)
            self.img_display['img_mask'].set_extent([0, shape_y, shape_x, 0])
            self.img_display['mask'].set_extent([0, shape_y, shape_x, 0])
            self.canvas.draw()
    
    def reset_thresh(self):
        self.slider_thresh.setValue(100)
        self.update_canvas()
    
    def update_combo_3ded(self):
        if self.combo_3ded.count() > 0:
            self.combo_3ded.clear()
        rois = [self.combo_roiNo.itemText(i) for i in range(self.combo_roiNo.count())]
        # check whether there is roi in roi for each one
        if hasattr(self, 'rois_tracked_roiInRoi_trans'):
            for key in self.rois_tracked_roiInRoi_trans.keys():
                roi_id, roi_ref = key
                try: #TODO check
                    rois.remove(str(roi_ref))
                except:
                    pass
                rois.append(str(key))
        items = ['all'] + rois
        self.combo_3ded.addItems(items)
        # self.combo_dp_plot.addItems(rois)
    
# =============================================================================
#     def run_func_in_workerThread(self, threadpool, func, *args, **kwargs):
#         worker = WorkerThread_General(func, args, kwargs)
#         worker.signals.results.connect(self.display_result)  # Connect to result signal
#         worker.signals.finished.connect(self.task_complete)  # Connect to finished signal
#         threadpool.start(worker)
# =============================================================================

    def save_results(self):
        path_save = self.lineEdit_dir_save.text()
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        
        fld_1 = datetime.date.today()
        fld_2 = datetime.datetime.now().strftime("%H-%M-%S")
        
        path_save = os.path.join(path_save, f'{fld_1}__{fld_2}')
        os.mkdir(path_save)
        
        # tracking results, rois, dp
        for roi_id, ds in self.tomo_ds.items():
            path_save_roi = os.path.join(path_save, f'roi No {roi_id}')
            os.mkdir(path_save_roi)
            if type(roi_id) == int:
                with open(os.path.join(path_save_roi, f'roi coords_id {roi_id}.npy'), 'wb') as f:
                    np.save(f, self.rois_tracked[roi_id])
            else: # roi in roi
                with open(os.path.join(path_save_roi, f'roi coords_ref_id {roi_id[1]}.npy'), 'wb') as f:
                    np.save(f, self.rois_tracked[roi_id[1]])
                with open(os.path.join(path_save_roi, f'roi coords_id {roi_id}.npy'), 'wb') as f:
                    np.save(f, self.rois_tracked_roiInRoi_trans[roi_id])
            
            s = signals.Signal2D(self.tomo_ds[roi_id])
            s.save(os.path.join(path_save_roi, f'3DED_id {roi_id}.hspy'))

            # write frames
            fld_frames = os.path.join(path_save_roi, 'frames')
            worker_frames = WorkerThread_General(io.create_frames, 0, fld_frames, s)
            self.threadpool.start(worker_frames)
            
            # clip for dp
            scale_recip = self.lineEdit_scale_recip.text()
            try:
                scale_recip = float(scale_recip)
            except:
                scale_recip = None
            fn_dp = os.path.join(path_save_roi, 'tomo clip')
            worker_clip_dp = WorkerThread_General(io.create_clip_dp, 0, fn_dp, s, scale_recip)
            self.threadpool.start(worker_clip_dp)
            
            
            # clip for tracking
            scale_real = self.lineEdit_scale_real.text()
            try:
                scale_real = float(scale_real)
            except:
                scale_real = None
            
            if type(roi_id) == int:
                rois_ref = self.rois_tracked[roi_id]
            else:
                rois_ref = self.rois_tracked[roi_id[1]]
                rois = self.rois_tracked_roiInRoi_trans[roi_id]
                fn_roiInRoi = os.path.join(path_save_roi, 'tracking clip_ROIINROI')
                worker_clip_tr = WorkerThread_General(io.create_clip_tracking, 0, 
                                                      fn_roiInRoi, self.nav_imgs, rois, scale_real)
                self.threadpool.start(worker_clip_tr)
            fn = os.path.join(path_save_roi, 'tracking clip')
            worker_clip_tr_ref = WorkerThread_General(io.create_clip_tracking, 0,
                                                      fn, self.nav_imgs, rois_ref, scale_real)
            self.threadpool.start(worker_clip_tr_ref)
        
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
