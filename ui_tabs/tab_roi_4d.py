# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:34:09 2024

@author: SGholam
"""



import os
import sys
import json
from time import perf_counter
from PyQt5.QtCore import (pyqtSignal, Qt, QRunnable, QObject, QThreadPool, QProcess)
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py4DTomo.io_utils as io
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
from dask.diagnostics import ProgressBar
from .logging_utils import get_tab_logger
# import matplotlib.gridspec as gridspec
# from skimage.filters import threshold_otsu, threshold_li, threshold_mean, threshold_yen
# from skimage import exposure
#%% wdiget
class Tab_ROI_on_4D(qtw.QWidget):
# class Tab_Create_NavSignal(qtw.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_tab_logger('Tab_ROI_on_4D')
        self.threadpool = QThreadPool()
        self.init_widget()
        
    def init_widget(self):
        self.central_widget = qtw.QWidget(self)
        self.layout = qtw.QHBoxLayout(self)
        self.setLayout(self.layout)
        self._splitter = qtw.QSplitter(Qt.Horizontal)
        self.layout.addWidget(self._splitter)
        self._left_widget = qtw.QWidget()
        self._splitter.addWidget(self._left_widget)

        width_userInput = 300

        layout_userInput = qtw.QVBoxLayout(self._left_widget)
        #%% directory
        self.box_dir = qtw.QGroupBox('Directory')
        layout_userInput.addWidget(self.box_dir)
        layout_dir = qtw.QVBoxLayout()
        self.box_dir.setLayout(layout_dir)
        
        # nav signal dir
        layout_dir_entry = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_entry)
        label_dir = qtw.QLabel('4D Signal')
        layout_dir_entry.addWidget(label_dir)
        
        self.lineEdit_dir_signal = qtw.QLineEdit()
        layout_dir_entry.addWidget(self.lineEdit_dir_signal)
        self.lineEdit_dir_signal.textChanged.connect(lambda:self.enable_dwellTime_spinbox(
            self.lineEdit_dir_signal.text()))
        
        self.button_dir_navSignal = qtw.QPushButton('...')
        layout_dir_entry.addWidget(self.button_dir_navSignal)
        self.button_dir_navSignal.clicked.connect(self.show_dialog)
        #%% input layout, box scan size
        self.box_scanSize = qtw.QGroupBox('Scan Size')
        layout_box_scanSize = qtw.QHBoxLayout()
        self.box_scanSize.setLayout(layout_box_scanSize)
        layout_userInput.addWidget(self.box_scanSize)
        
        # label_scanSize = qtw.QLabel('Scan Size')
        # layout_box_scanSize.addWidget(label_scanSize)
        
        self.checkbox_scanSize = qtw.QCheckBox('Auto')
        layout_box_scanSize.addWidget(self.checkbox_scanSize)
        self.checkbox_scanSize.setChecked(True)

        self.lineEdit_scanSize_x = qtw.QLineEdit()
        self.lineEdit_scanSize_x.setAlignment(Qt.AlignLeft)
        layout_box_scanSize.addWidget(self.lineEdit_scanSize_x)
        self.lineEdit_scanSize_x.setFixedWidth(50)
        self.lineEdit_scanSize_x.setValidator(QIntValidator(0,99999))
        
        label_cross = qtw.QLabel('X')
        layout_box_scanSize.addWidget(label_cross)
        
        self.lineEdit_scanSize_y = qtw.QLineEdit()
        layout_box_scanSize.addWidget(self.lineEdit_scanSize_y)
        self.lineEdit_scanSize_y.setFixedWidth(50)
        self.lineEdit_scanSize_y.setValidator(QIntValidator(0,99999))

        self.activate_lineEdit_scanSize()
        self.checkbox_scanSize.stateChanged.connect(self.activate_lineEdit_scanSize)
        
        label_dwellTime = qtw.QLabel('Dwell Time (\u03BCsec)')
        self.spinbox_dwellTime = qtw.QSpinBox()
        self.spinbox_dwellTime.setFixedWidth(75)
        self.spinbox_dwellTime.setRange(1, 99999999)
        self.spinbox_dwellTime.setDisabled(True)
        for wid in [label_dwellTime, self.spinbox_dwellTime]:
            layout_box_scanSize.addWidget(wid)

        layout_box_scanSize.addStretch(1)
        #%% box for scales
        self.box_scale = qtw.QGroupBox('Scale bars')
        layout_box_scale = qtw.QHBoxLayout()
        # self.box_scale.setFixedWidth(150)
        self.box_scale.setLayout(layout_box_scale)
        layout_userInput.addWidget(self.box_scale)
        
        # real space
        self.double_validator = QDoubleValidator(0.0, 1e5, 5)
        layout_scale_real = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_real)
        label_scale_real = qtw.QLabel('Real (nm)')
        layout_scale_real.addWidget(label_scale_real)
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        layout_scale_real.addWidget(self.lineEdit_scale_real)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        # reciprocal space
        layout_scale_recip = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_recip)
        label_scale_recip = qtw.QLabel('Recip. (\u00C5<sup>-1</sup>)')
        layout_scale_recip.addWidget(label_scale_recip)
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        layout_scale_recip.addWidget(self.lineEdit_scale_recip)
        self.lineEdit_scale_recip.setValidator(self.double_validator)
        
        self.lineEdit_scale_recip.textChanged.connect(self.update_canvas)
        self.lineEdit_scale_real.textChanged.connect(self.update_canvas)
        
        #%% load button
        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        self.button_loadNavigation.setFixedSize(110, 50)
        layout_userInput.addWidget(self.button_loadNavigation, alignment=Qt.AlignCenter)
        self.button_loadNavigation.clicked.connect(self.get_nav_image)
        #%% SAM2 segmentation
        self.box_segmentation = qtw.QGroupBox('SAM2 Segmentation')
        layout_userInput.addWidget(self.box_segmentation)
        layout_segmentation = qtw.QHBoxLayout()
        self.box_segmentation.setLayout(layout_segmentation)

        self.button_segment_image = qtw.QPushButton('Segment Image')
        layout_segmentation.addWidget(self.button_segment_image)
        self.button_segment_image.clicked.connect(self.segment_image)
        self.button_segment_image.setDisabled(True)
        self.button_segment_image.setToolTip(
            'Run SAM2 on the points added below to segment the navigation image')

        self.button_clear_points = qtw.QPushButton('Clear Points')
        layout_segmentation.addWidget(self.button_clear_points)
        self.button_clear_points.clicked.connect(self.clear_seg_points)
        self.button_clear_points.setDisabled(True)
        self.button_clear_points.setToolTip('Remove all SAM2 points and the segmentation mask')

        layout_userInput.addStretch(1)
        #%% canvas layout
        self._right_widget = qtw.QWidget()
        self._splitter.addWidget(self._right_widget)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([300, 900])
        layout_canvas = qtw.QVBoxLayout(self._right_widget)
        
        # self.figure = Figure(figsize=(5,5))
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout_canvas.addWidget(self.canvas)
        
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
        # every "Load Signal" click would otherwise stack a new AxesImage
        # (and, once colorbars existed, a new colorbar axes) on top of the
        # old ones each time, growing the figure without bound.
        self._setup_canvas()

        self.rect = None            # Currently drawn rectangle
        self.press = None           # Mouse press coordinates
        # SAM2 segmentation point prompts for the currently loaded nav image
        self.seg_points = []
        self.seg_labels = []
        self.seg_mask = None
        self.scatter_plots = []
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        #%% slider layout
        layout_slider = qtw.QHBoxLayout()
        layout_canvas.addLayout(layout_slider)
        
        self.label_vmin = qtw.QLabel('vmin')
        layout_slider.addWidget(self.label_vmin)
        
        self.slider_vmin = qtw.QSlider(self)
        self.slider_vmin.setOrientation(1)  # Horizontal slider
        self.slider_vmin.setRange(0,0)
        layout_slider.addWidget(self.slider_vmin)
        
        self.label_vmax = qtw.QLabel('vmax')
        layout_slider.addWidget(self.label_vmax)
        
        self.slider_vmax = qtw.QSlider(self)
        self.slider_vmax.setOrientation(1)  # Horizontal slider
        self.slider_vmax.setRange(0,0)
        layout_slider.addWidget(self.slider_vmax)

        self.slider_vmin.valueChanged.connect(lambda: self.update_canvas(ax='dp'))
        self.slider_vmax.valueChanged.connect(lambda: self.update_canvas(ax='dp'))
        
        self.button_reset_slider = qtw.QPushButton('Reset Sliders')
        layout_slider.addWidget(self.button_reset_slider)
        self.button_reset_slider.clicked.connect(self.reset_sliders)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout_canvas.addWidget(self.toolbar)
#%% functions
    def activate_lineEdit_scanSize(self):
        if self.checkbox_scanSize.isChecked():
            self.lineEdit_scanSize_x.setDisabled(True)
            self.lineEdit_scanSize_y.setDisabled(True)
        else:
            self.lineEdit_scanSize_x.setEnabled(True)
            self.lineEdit_scanSize_y.setEnabled(True)

    def get_scan_size(self):
        if not self.checkbox_scanSize.isChecked(): # get scan size
            try:
                x = int(self.lineEdit_scanSize_x.text())
                y = int(self.lineEdit_scanSize_y.text())
                scanSize = (x,y)
            except:
                scanSize = None
        else:
            scanSize = None
        return scanSize
    
    def get_nav_image(self):
        self.reset_canvas()
        self.fn = self.lineEdit_dir_signal.text()
        self.scanSize = self.get_scan_size()
        self.dwellTime = self.spinbox_dwellTime.value()
        dtype = os.path.splitext(self.fn)[-1]
        if dtype == '.tpx3' and self.scanSize == None:
            self.logger.warning('Cannot load %s: scan size is required for .tpx3 files.', self.fn)
            self.message_box_tpx3()
            return

        self.logger.info('Loading 4D signal from %s...', self.fn)
        worker = Worker_NavImg(self.fn, self.scanSize, self.dwellTime)
        
        worker.signals.result.connect(self.image_handler)  # Connect to result signal
        self.threadpool.start(worker)

# =============================================================================
#         self.navImg = io.calculate_nav_signal(self.fn, scanSize=scanSize)
#         self.update_canvas('nav')
# =============================================================================

    def image_handler(self, result):
        self.navImg = result
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

    def show_dialog(self):
        file_filter = "supported signals (*.zspy *.hspy *.hdf5 *.h5 *.tpx3 *.pmf);;All Files (*)"
        path = qtw.QFileDialog.getOpenFileName(self, "Select 4D Signals Folder", '', file_filter)
        # if path and os.path.isdir(path[0]):
        if path:
            self.lineEdit_dir_signal.setText(path[0])
    
    def enable_dwellTime_spinbox(self, txt):
        enable = False
        if os.path.isfile(txt):
            dtype = os.path.splitext(txt)[1]
            if dtype == '.tpx3':
                enable = True
        for wid in self.box_scanSize.findChildren(qtw.QWidget):
            wid.setEnabled(enable)
            # print(e)
        self.checkbox_scanSize.setChecked(not enable)
        self.checkbox_scanSize.setDisabled(enable)
            
    
    def on_press(self, event):
        if event.inaxes != self.ax_nav:
            self.press = None
            return
        if event.button == 2:
            # Middle click removes the last-added SAM2 point.
            self.delete_last_seg_point()
            return
        if 'shift' in event.modifiers and event.button in (1, 3):
            # Shift+click adds a SAM2 point prompt (left = positive, right =
            # negative) - deliberately a different modifier than the
            # Ctrl+drag ROI below so the two annotation modes can't be
            # confused with each other.
            self.add_seg_point(event)
            return
        if 'ctrl' not in event.modifiers or event.button != 1:
            # Plain click/drag is reserved for the navigation toolbar's
            # Pan/Zoom tool (and the scroll-wheel zoom) so images can be
            # zoomed into; hold "ctrl" to draw a new ROI instead.
            self.press = None
            return
        # Mouse press event: record the starting point
        self.press = (event.xdata, event.ydata)
        if self.rect is not None:
            self.rect.remove()
        self.rect = patches.Rectangle(self.press, 0, 0, linewidth=1,
                                      edgecolor='r', facecolor='none')
        self.ax_nav.add_patch(self.rect)
        self.canvas.draw()

    def on_motion(self, event):
        # Mouse motion event: update the rectangle size as the mouse moves
        if self.press is None or event.inaxes is None:
            return
        x0, y0 = self.press
        width = event.xdata - x0
        height = event.ydata - y0
        self.rect.set_width(width)
        self.rect.set_height(height)
        self.rect.set_xy((x0, y0))
        self.canvas.draw()

    def on_scroll(self, event):
        """Zoom the axes under the cursor in/out on the scroll wheel,
        centered on the cursor position."""
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
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
        self.roi = (int(x0), int(y0), int(width), int(height))
        
        
        # Set the press attribute to None for future drawings
        self.press = None
        self.canvas.draw()
        self.logger.info('ROI: %s', self.roi)
        if not hasattr(self, 'dwellTime'):
            try:
                self.dwellTime = self.spinbox_dwellTime.value()
            except:
                self.dwellTime = None
        worker = Worker_CalculateDP(self.fn, self.roi, self.scanSize, self.dwellTime)
        worker.signals.result.connect(self.get_dp)
        self.threadpool.start(worker)
    
    def get_dp(self, result):
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
        self.update_slider_range()
        self.slider_vmax.setValue(self.dp.max())
        self.slider_vmin.setValue(1)
        self.update_canvas(roiUpdate=True)
    
    def reset_sliders(self):
        self.slider_vmin.setValue(0)
        self.slider_vmax.setValue(self.dp.max())
        self.update_canvas(roiUpdate=True)
    
    def _setup_canvas(self):
        """One-time creation of the image artists and their colorbars.
        reset_canvas() (called on every "Load Signal") only clears their
        data afterwards - re-imshow()ing here on every load would otherwise
        stack a new image (and, now, a new colorbar axes) on the figure
        each time."""
        self.img_display = {}
        img_temp = np.zeros((512,512), dtype='uint16')
        self.img_display['nav'] = self.ax_nav.imshow(img_temp, cmap='viridis')
        self.img_display['nav_roi'] = self.ax_nav_roi.imshow(img_temp, cmap='viridis')

        self.img_display['dp'] = self.ax_dp.imshow(img_temp, cmap='inferno')
        self.img_display['dp'].set_norm(SymLogNorm(linthresh=1))

        # SAM2 segmentation mask overlay, drawn on top of the nav image.
        # Starts fully transparent (all-zero RGBA); show_seg_mask() fills in
        # color+alpha only where the mask is True.
        self.img_display['seg_mask'] = self.ax_nav.imshow(np.zeros((512, 512, 4)))

        self.ax_nav.set_title('Nav. Image')
        self.ax_nav_roi.set_title('ROI Image')
        self.ax_dp.set_title('Dif. Pattern')

        self.ax_nav_roi.set_axis_off()
        self.ax_dp.set_axis_off()
        # ax_nav keeps its x-axis label visible (for the interaction hints
        # below), so its ticks/spines are hidden individually instead of via
        # set_axis_off(), which would hide the label too.
        for spine in self.ax_nav.spines.values():
            spine.set_visible(False)
        self.ax_nav.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax_nav.set_xlabel(
            'Hold "ctrl" + Drag => New ROI\n'
            'Hold "shift" + Click => Add SAM2 point (Left=positive, Right=negative)\n'
            'Middle Click => Remove last SAM2 point, Scroll Wheel => Zoom\n'
            'Plain Click+Drag => Pan/Zoom Tool', fontsize=6)
        self.ax_nav.xaxis.label.set_visible(True)

        self.colorbars = {}
        self.colorbars['nav'] = self.figure.colorbar(
            self.img_display['nav'], ax=self.ax_nav, fraction=0.046, pad=0.04)
        self.colorbars['nav_roi'] = self.figure.colorbar(
            self.img_display['nav_roi'], ax=self.ax_nav_roi, fraction=0.046, pad=0.04)
        self.colorbars['dp'] = self.figure.colorbar(
            self.img_display['dp'], ax=self.ax_dp, fraction=0.046, pad=0.04)

        self.figure.tight_layout()

    def reset_canvas(self):
        """Clear displayed data back to blank at the start of every "Load
        Signal" - the image artists/colorbars themselves are created once
        in _setup_canvas() and reused."""
        img_temp = np.zeros((512,512), dtype='uint16')
        self.img_display['nav'].set_data(img_temp)
        self.img_display['nav_roi'].set_data(img_temp)
        self.img_display['dp'].set_data(img_temp)
        self.ax_nav_roi.set_title('ROI Image')
        self.clear_seg_points()
        self.canvas.draw_idle()

    def update_slider_range(self):
        self.slider_vmin.setRange(0, int(self.dp.max()/2))
        self.slider_vmax.setRange(1, self.dp.max())
        self.slider_vmin.setSingleStep(1)
        self.slider_vmax.setSingleStep(1)
    
    def update_canvas(self, ax='dp', roiUpdate=False):
        if ax == 'dp':
            vmax = self.slider_vmax.value()
            self.label_vmax.setText(f'vmax: {vmax:.0f}')
            vmin = self.slider_vmin.value()
            if vmin >= vmax:
                vmin = vmax - 1
                self.slider_vmin.setValue(vmin)
            self.label_vmin.setText(f'vmin: {vmin:.0f}')
            
            self.img_display['dp'].set_data(self.dp)

            # self.img_display['dp'].set_clim(vmin, vmax)
            # self.img_display['dp'].set_norm(SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax))
            shape_x, shape_y = self.dp.shape
            self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])
            # self.img_display['dp'].set_clim(self.dp.min(), self.dp.max())
            self.img_display['dp'].set_clim(vmin, vmax)
            # ax_dp/ax_nav_roi show a per-ROI crop whose size varies with
            # each ROI selection, so (unlike ax_nav) their view is reset to
            # fit every time here, rather than relying on the toolbar's
            # Home button - otherwise, once the user has zoomed in, the next
            # (differently-sized) ROI selection would stay at the old view.
            self.ax_dp.set_xlim(0, shape_y)
            self.ax_dp.set_ylim(shape_x, 0)

            if roiUpdate:
                self.img_display['nav_roi'].set_data(self.navImg_cut)
                self.img_display['nav_roi'].set_clim(self.navImg_cut.min(), self.navImg_cut.max())
                shape_x, shape_y = self.navImg_cut.shape
                self.img_display['nav_roi'].set_extent([0, shape_y, shape_x, 0])
                self.ax_nav_roi.set_xlim(0, shape_y)
                self.ax_nav_roi.set_ylim(shape_x, 0)

        elif ax == 'nav':
            self.img_display['nav'].set_data(self.navImg)
            shape_x, shape_y = self.navImg.shape
            self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
            self.img_display['nav'].set_clim(vmin=self.navImg.min(), vmax=self.navImg.max())
        
        # scale bars 
        #TODO adding and removing the artist is not efficient
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
            for ax in [self.ax_nav, self.ax_nav_roi]:
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
    
    def message_box_tpx3(self):
       msg = qtw.QMessageBox()
       msg.setWindowTitle("Scan Size Error!")
       msg.setText("(Currently,) tpx3 conversion requires scan size input!")
       msg.setInformativeText("Enter scan size and try again.")
       msg.setStandardButtons(qtw.QMessageBox.Ok)
       msg.setIcon(qtw.QMessageBox.Critical)
       retval = msg.exec_()

#%% SAM2 segmentation
    def add_seg_point(self, event):
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
            pass
        self.canvas.draw_idle()
        if not self.seg_points:
            self.button_segment_image.setDisabled(True)
        self.logger.info('Removed last SAM2 point (%d point(s) remaining).', len(self.seg_points))

    def clear_seg_points(self):
        had_points = bool(self.seg_points) or self.seg_mask is not None
        self.seg_points = []
        self.seg_labels = []
        self.seg_mask = None
        for p in self.scatter_plots:
            try:
                p.remove()
            except Exception:
                pass
        self.scatter_plots.clear()
        self.img_display['seg_mask'].set_data(np.zeros((512, 512, 4)))
        self.button_segment_image.setDisabled(True)
        self.canvas.draw_idle()
        if had_points:
            self.logger.info('Cleared SAM2 points and segmentation mask.')

    def show_seg_mask(self, mask):
        cmap = plt.get_cmap('tab10')
        color = np.array([*cmap(0)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        self.img_display['seg_mask'].set_data(mask_image)
        self.img_display['seg_mask'].set_extent(self.img_display['nav'].get_extent())
        self.canvas.draw_idle()

    def _get_seg_temp_dir(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(os.path.dirname(script_dir), 'py4DTomo', 'io_utils', 'temp', 'roi4d_seg')
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def segment_image(self):
        """Run SAM2's single-image predictor on the currently loaded nav
        image using the point prompts added via Shift+Click, exactly like
        the SAM2 tab's "Seg Image" feature but without object tracking."""
        if not self.seg_points:
            self.logger.warning('Segment Image requested but no points have been added.')
            qtw.QMessageBox.critical(self, 'No Points Added',
                'Hold Shift and click on the image to add at least one point '
                '(left click = positive, right click = negative) before segmenting.')
            return

        self.logger.info('Starting SAM2 image segmentation (%d point(s))...', len(self.seg_points))
        self.button_segment_image.setDisabled(True)

        path_seg = self._get_seg_temp_dir()
        img_8bit = io.convert_img_to_8bit(self.navImg)
        seg_input = pd.Series({'image': img_8bit,
                               'points': np.array(self.seg_points),
                               'labels': np.array(self.seg_labels)})
        seg_input.to_pickle(os.path.join(path_seg, 'seg_input.pkl'))

        self._process_sam = QProcess(self)
        self._process_sam.setProgram(sys.executable)
        self._process_sam.setArguments(["worker_sam.py", 'image', path_seg, '0'])
        self._process_sam.readyReadStandardError.connect(self._handle_error_sam)
        self._process_sam.finished.connect(self._handle_finished_sam)
        self._process_sam.errorOccurred.connect(self._process_failed_sam)
        self._process_sam.start()

    def _process_failed_sam(self, error):
        self.button_segment_image.setEnabled(True)
        self.logger.error('SAM2 segmentation QProcess error occurred: %s', error)
        qtw.QMessageBox.critical(self, 'Process Error',
            'The SAM2 segmentation process failed to start.\n'
            'Check that Python is on PATH and worker_sam.py exists.')

    def _handle_error_sam(self):
        # SAM2/PyTorch write progress bars and warnings to stderr that
        # aren't necessarily errors - real failures surface via the
        # JSON-decode check in _handle_finished_sam below.
        text = bytes(self._process_sam.readAllStandardError()).decode('utf-8').strip()
        if text:
            self.logger.info('SAM2: %s', text)

    def _handle_finished_sam(self, exit_code=0, exit_status=0):
        self.button_segment_image.setEnabled(True)
        text = bytes(self._process_sam.readAllStandardOutput()).decode('utf-8')
        try:
            result = json.loads(text.strip())
            with np.load(result['path']) as f:
                mask = f['mask']
            self.seg_mask = mask
            self.show_seg_mask(mask)
            self.button_clear_points.setEnabled(True)
            self.logger.info('SAM2 image segmentation completed successfully.')
        except json.JSONDecodeError:
            self.logger.error('Could not decode SAM2 segmentation result: %s', text)
            qtw.QMessageBox.warning(self, 'SAM2 Error',
                f'Could not decode SAM2 output. Check console for details.\n'
                f'Raw output (first 200 chars): {text[:200]}')

    def cleanup(self):
        """Release resources held by this tab. Called by MainWindow.closeEvent
        so repeated runs of the app in the same console/kernel don't leave
        threadpools, running subprocesses, and matplotlib figures alive."""
        self.threadpool.clear()
        if hasattr(self, '_process_sam'):
            self._process_sam.kill()
        plt.close(self.figure)

class WorkerSignals(QObject):
    finished = pyqtSignal()  # Signal to indicate task completion
    result = pyqtSignal(object)  # Signal to emit the result of the task

# Step 2: Create a WorkerThread class that runs the task in the background
class Worker_NavImg(QRunnable):
    def __init__(self, fn, scanSize=None, dwellTime=None):
        super().__init__()
        self.logger = get_tab_logger('Tab_ROI_on_4D')
        self._tic = perf_counter()
        self.logger.info('Calculating navigation image...')
        self.fn = fn
        self.scanSize = scanSize
        self.dwellTime = dwellTime
        self.signals = WorkerSignals()  # Create an instance of WorkerSignals

    def run(self):
        try:
            navImg = io.calculate_nav_img(self.fn, scanSize=self.scanSize,
                                             dwellTime=self.dwellTime)
        except Exception:
            self.logger.exception('Failed to calculate navigation image after %.1f s.',
                                   perf_counter() - self._tic)
            self.signals.finished.emit()
            return

        # Emit the result when the task is done
        self.signals.result.emit(navImg)
        self.logger.info('Navigation image calculated successfully in %.1f s.',
                          perf_counter() - self._tic)
        self.signals.finished.emit()  # Emit the finished signal when done

class Worker_CalculateDP(QRunnable):
    def __init__(self, fn, roi, scanSize, dwellTime):
        super().__init__()
        self.logger = get_tab_logger('Tab_ROI_on_4D')
        self._tic = perf_counter()
        self.logger.info('calculating the dp...')
        self.fn = fn
        self.roi = roi
        self.scanSize = scanSize
        self.dwellTime = dwellTime

        self.signals = WorkerSignals()

    def run(self):
        try:
            s_cut = io.load_signal(self.fn, roi=self.roi,
                                   scanSize=self.scanSize, dwellTime=self.dwellTime)
            if type(s_cut) == tuple: # for hdf5
                s_cut, self.f = s_cut

            # The h5py file handle (self.f, for the hdf5-lazy case) must stay
            # open through .compute() since the dask array reads from it
            # lazily, but has to be closed here regardless of whether that
            # computation succeeds or raises — otherwise a failed compute
            # leaks an open file handle for the rest of the process/console
            # session.
            try:
                navImg_cut = s_cut.sum(axis=(2,3)).data
                dp = s_cut.sum(axis=(0,1)).data
                if hasattr(dp, 'compute'): # lazy signals
                    with ProgressBar():
                        dp.compute()
                if hasattr(navImg_cut, 'compute'): # lazy signals
                    navImg_cut = navImg_cut.compute()
            finally:
                if hasattr(self, 'f'):
                    self.f.close()
        except Exception:
            self.logger.exception('Failed to calculate diffraction pattern after %.1f s.',
                                   perf_counter() - self._tic)
            return
        self.signals.result.emit((dp, navImg_cut))
        self.logger.info('Diffraction pattern calculated successfully in %.1f s.',
                          perf_counter() - self._tic)
        # self.signals.finished.emit()
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
