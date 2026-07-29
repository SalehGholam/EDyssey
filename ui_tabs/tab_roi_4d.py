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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py4DTomo.io_utils as io
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as patches
from dask.diagnostics import ProgressBar
from .logging_utils import get_tab_logger, LogConsole
from .threshold_dialog import ThresholdDialog
from worker_extract_frame import load_dp
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
        self._cancelling = False  # set by cancel_running_work(); suppresses error popups it causes
        self.init_widget()
        
    def init_widget(self):
        self.central_widget = qtw.QWidget(self)
        self.layout = qtw.QHBoxLayout(self)
        self.setLayout(self.layout)
        self._splitter = qtw.QSplitter(Qt.Horizontal)
        self.layout.addWidget(self._splitter)
        
        width_userInput = 320
        self._left_widget = qtw.QWidget()
        self._left_widget.setFixedWidth(width_userInput)
        self._splitter.addWidget(self._left_widget)

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
        layout_box_scanSize = qtw.QVBoxLayout()
        self.box_scanSize.setLayout(layout_box_scanSize)
        layout_userInput.addWidget(self.box_scanSize)

        layout_scanSize_row1 = qtw.QHBoxLayout()
        layout_box_scanSize.addLayout(layout_scanSize_row1)

        self.checkbox_scanSize = qtw.QCheckBox('Auto')
        layout_scanSize_row1.addWidget(self.checkbox_scanSize)
        self.checkbox_scanSize.setChecked(True)

        self.lineEdit_scanSize_x = qtw.QLineEdit()
        self.lineEdit_scanSize_x.setAlignment(Qt.AlignLeft)
        layout_scanSize_row1.addWidget(self.lineEdit_scanSize_x)
        self.lineEdit_scanSize_x.setFixedWidth(40)
        self.lineEdit_scanSize_x.setValidator(QIntValidator(0,99999))

        label_cross = qtw.QLabel('X')
        layout_scanSize_row1.addWidget(label_cross)

        self.lineEdit_scanSize_y = qtw.QLineEdit()
        layout_scanSize_row1.addWidget(self.lineEdit_scanSize_y)
        self.lineEdit_scanSize_y.setFixedWidth(40)
        self.lineEdit_scanSize_y.setValidator(QIntValidator(0,99999))

        self.activate_lineEdit_scanSize()
        self.checkbox_scanSize.stateChanged.connect(self.activate_lineEdit_scanSize)

        label_dwellTime = qtw.QLabel('Dwell T. (\u03BCs)')
        label_dwellTime.setToolTip('Dwell time in microseconds')
        self.spinbox_dwellTime = qtw.QSpinBox()
        self.spinbox_dwellTime.setFixedWidth(60)
        self.spinbox_dwellTime.setRange(1, 99999999)
        self.spinbox_dwellTime.setDisabled(True)
        for wid in [label_dwellTime, self.spinbox_dwellTime]:
            layout_scanSize_row1.addWidget(wid)

        layout_scanSize_row1.addStretch(1)

        # metadata (comment.txt) auto-fill - tpx3 acquisitions log scan
        # size/dwell time there, alongside the .tpx3 file itself.
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
            'Fill scan size / dwell time from comment.txt next to the 4D '
            'signal file (tpx3 acquisitions only)')
        layout_scanSize_row2.addWidget(self.button_loadMetadata)
        self.button_loadMetadata.clicked.connect(lambda: self.load_metadata(silent=False))

        self.button_browseMetadata = qtw.QPushButton('...')
        self.button_browseMetadata.setFixedWidth(30)
        self.button_browseMetadata.setToolTip(
            'Browse for the metadata file (defaults to comment.txt next to the 4D signal)')
        layout_scanSize_row2.addWidget(self.button_browseMetadata)
        self.button_browseMetadata.clicked.connect(self.browse_metadata_file)
        layout_scanSize_row2.addStretch(1)

        self.metadata_path_override = None  # set by browse_metadata_file(); cleared on new 4D signal
        #%% box for scales
        self.box_scale = qtw.QGroupBox('Scale bars')
        # Real/Recip/Auto-center stack as their own rows (not one wide row)
        # so this box doesn't force the whole left panel wider than intended.
        layout_box_scale = qtw.QVBoxLayout()
        self.box_scale.setLayout(layout_box_scale)
        layout_userInput.addWidget(self.box_scale)
        
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
            'every redraw. When unchecked, hold Ctrl and click on the '
            'diffraction pattern to set the center manually.')
        layout_box_scale.addWidget(self.checkbox_autoCenterDp)
        self.checkbox_autoCenterDp.stateChanged.connect(
            lambda: self.update_canvas(ax='dp') if hasattr(self, 'dp') else None)

        self.dp_center = None  # (x, y) - auto-found or last manually-clicked center

        self.lineEdit_scale_recip.textChanged.connect(self.update_canvas)
        self.lineEdit_scale_real.textChanged.connect(self.update_canvas)

        #%% load button
        layout_load = qtw.QHBoxLayout()
        layout_userInput.addLayout(layout_load)
        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        self.button_loadNavigation.setFixedSize(110, 50)
        layout_load.addWidget(self.button_loadNavigation, alignment=Qt.AlignCenter)
        self.button_loadNavigation.clicked.connect(self.get_nav_image)

        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.setFixedSize(70, 50)
        self.button_cancel.setStyleSheet("background-color: red; color: white;")
        self.button_cancel.setDisabled(True)
        self.button_cancel.setToolTip(
            'Stop the running SAM2 segmentation or DP computation. Already-running '
            'background computations finish silently; their results are discarded.')
        layout_load.addWidget(self.button_cancel, alignment=Qt.AlignCenter)
        self.button_cancel.clicked.connect(self.cancel_running_work)
        #%% SAM2 segmentation
        self.box_segmentation = qtw.QGroupBox('SAM2 Segmentation')
        layout_userInput.addWidget(self.box_segmentation)
        layout_segmentation = qtw.QHBoxLayout()
        self.box_segmentation.setLayout(layout_segmentation)

        self.button_segment_image = qtw.QPushButton('Segment\nImage')
        layout_segmentation.addWidget(self.button_segment_image)
        self.button_segment_image.clicked.connect(self.segment_image)
        self.button_segment_image.setDisabled(True)
        self.button_segment_image.setToolTip(
            'Run SAM2 on the points added below (Shift+Click), optionally combined '
            'with the last-drawn ROI (Ctrl+Drag) as a box prompt')

        self.button_clear_points = qtw.QPushButton('Clear\nPoints')
        layout_segmentation.addWidget(self.button_clear_points)
        self.button_clear_points.clicked.connect(self.clear_seg_points)
        self.button_clear_points.setDisabled(True)
        self.button_clear_points.setToolTip('Remove all SAM2 points and the segmentation mask')

        self.button_clear_roi = qtw.QPushButton('Clear\nROI/Box')
        layout_segmentation.addWidget(self.button_clear_roi)
        self.button_clear_roi.clicked.connect(self.clear_roi)
        self.button_clear_roi.setToolTip(
            'Remove the drawn ROI so it stops being used as a SAM2 box prompt '
            '(and as the rectangle for diffraction-pattern extraction)')

        #%% Summed DP from threshold
        self.box_sumDpThreshold = qtw.QGroupBox('Summed DP from Threshold')
        layout_userInput.addWidget(self.box_sumDpThreshold)
        layout_sumDpThreshold = qtw.QHBoxLayout()
        self.box_sumDpThreshold.setLayout(layout_sumDpThreshold)

        self.button_sumDpFromThreshold = qtw.QPushButton('Summed DP from\nThreshold...')
        layout_sumDpThreshold.addWidget(self.button_sumDpFromThreshold)
        self.button_sumDpFromThreshold.clicked.connect(self.open_threshold_dialog)
        self.button_sumDpFromThreshold.setToolTip(
            'Open a window to check/adjust a real-space threshold on the loaded '
            'navigation image, then sum diffraction patterns only at the scan '
            'positions above it, instead of a rectangular ROI')

        layout_userInput.addStretch(1)
        #%% canvas layout
        self._right_widget = qtw.QWidget()
        self._splitter.addWidget(self._right_widget)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([width_userInput, 900])
        layout_canvas = qtw.QVBoxLayout(self._right_widget)
        
        # self.figure = Figure(figsize=(5,5))
        self.figure = Figure(constrained_layout=True)
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
        self.roi = None             # (x, y, w, h) of the last-drawn ROI, or None
        self.press = None           # Mouse press coordinates
        self.backgrounds = {}       # blit-cached canvas snapshots, keyed by axis
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

        # The app-wide log console lives here (below this tab's own plot
        # column) rather than under the whole window, so the left parameter
        # panel (a separate splitter pane) can span the full window height.
        self.log_console = LogConsole(self)
        layout_canvas.addWidget(self.log_console)
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
        self._cancelling = False
        self.button_cancel.setEnabled(True)
        worker = Worker_NavImg(self.fn, self.scanSize, self.dwellTime)

        worker.signals.result.connect(self.image_handler)  # Connect to result signal
        self.threadpool.start(worker)

# =============================================================================
#         self.navImg = io.calculate_nav_signal(self.fn, scanSize=scanSize)
#         self.update_canvas('nav')
# =============================================================================

    def image_handler(self, result):
        self.navImg = result
        self.dp_center = None  # a new signal may have a different DP shape/center
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
        if enable:
            self.load_metadata(silent=True)

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
                self.lineEdit_scanSize_x.setText(str(int(metadata['scan size x'])))
                self.lineEdit_scanSize_y.setText(str(int(metadata['scan size y'])))
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

    def on_press(self, event):
        if (event.inaxes == self.ax_dp and event.button == 1 and event.xdata is not None
                and 'ctrl' in event.modifiers and not self.checkbox_autoCenterDp.isChecked()
                and hasattr(self, 'dp')):
            # Manual re-centering of the reciprocal-space rings (only takes
            # effect when auto-centering is off, so it isn't immediately
            # overwritten by the next auto-centered redraw).
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
        # One full draw() here "bakes in" the current state (nav image, the
        # just-added zero-size rect) into a cached background snapshot, so
        # on_motion below can cheaply blit just the growing rectangle on top
        # of it instead of redrawing the whole figure on every mouse-move.
        self.canvas.draw()
        self.backgrounds['nav'] = self.canvas.copy_from_bbox(self.ax_nav.bbox)

    def on_motion(self, event):
        # Mouse motion event: update the rectangle size as the mouse moves
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
        # A box alone is a valid SAM2 prompt (see segment_image()), so
        # drawing one enables Segment Image even without any points yet.
        self.button_segment_image.setEnabled(True)
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
        # A rectangle ROI was just drawn, replacing whatever the "ROI Image"
        # axis was previously showing - clear any leftover SAM2 mask overlay
        # so it doesn't linger on top of this unrelated crop.
        self.img_display['seg_mask'].set_data(np.zeros((512, 512, 4)))
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
            'Hold "ctrl" + Drag => New ROI (also usable as a SAM2 box prompt)\n'
            'Hold "shift" + Click => Add SAM2 point (Left=positive, Right=negative)\n'
            'Middle Click => Remove last SAM2 point', fontsize=6)
        self.ax_nav.xaxis.label.set_visible(True)

        for spine in self.ax_dp.spines.values():
            spine.set_visible(False)
        self.ax_dp.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax_dp.xaxis.label.set_visible(True)

        # The Ctrl+Scroll zoom hint applies to every axis on this canvas, so
        # it's a figure-wide supxlabel rather than repeated per-axis text.
        self.figure.supxlabel('Hold "Ctrl" + Scroll wheel to zoom the axis under the cursor',
                              fontsize=7)

        self.colorbars = {}
        self.colorbars['nav'] = self.figure.colorbar(
            self.img_display['nav'], ax=self.ax_nav, fraction=0.046, pad=0.04)
        self.colorbars['nav_roi'] = self.figure.colorbar(
            self.img_display['nav_roi'], ax=self.ax_nav_roi, fraction=0.046, pad=0.04)
        self.colorbars['dp'] = self.figure.colorbar(
            self.img_display['dp'], ax=self.ax_dp, fraction=0.046, pad=0.04)

        # self.figure.tight_layout()

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
        self.clear_roi()
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
            for scale_ax in [self.ax_nav, self.ax_nav_roi]:
                io.add_readable_scalebar(scale_ax, scale_real, 'nm')
        except Exception as e:
            # print(e)
            pass
        # A conventional linear scale bar doesn't read naturally on a
        # radially-symmetric diffraction pattern - concentric dashed rings
        # at every 1 1/A (centered on the DP) work better.
        dp_shape = self.img_display['dp'].get_array().shape
        if self.checkbox_autoCenterDp.isChecked():
            try:
                self.dp_center = io.find_dp_center_blurred(self.dp)
            except Exception:
                self.logger.exception('Auto-centering the diffraction pattern failed.')
        center = self.dp_center if self.dp_center is not None else None
        self._dp_recip_circles = io.draw_reciprocal_scale_circles(
            self.ax_dp, self.lineEdit_scale_recip.text(), dp_shape,
            center=center, old_artists=getattr(self, '_dp_recip_circles', None))
        if self.checkbox_autoCenterDp.isChecked():
            self.ax_dp.set_xlabel('Circle center: auto (large-sigma blur)', fontsize=6)
        else:
            self.ax_dp.set_xlabel('Circle center: manual - Ctrl+Click pattern to set', fontsize=6)

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

    def clear_roi(self):
        """Remove the drawn ROI so it stops being used as a SAM2 box prompt
        or as the rectangle for diffraction-pattern extraction - without
        this, a stale box from an earlier draw would keep silently feeding
        into segment_image() even after the user no longer wants a box."""
        had_roi = self.roi is not None
        self.roi = None
        if self.rect is not None:
            try:
                self.rect.remove()
            except Exception:
                pass
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
        self.img_display['nav_roi'].set_clim(vmin=self.navImg.min(), vmax=self.navImg.max())
        self.img_display['nav_roi'].set_extent([0, shape_x, shape_y, 0])
        self.ax_nav_roi.set_xlim(0, shape_x)
        self.ax_nav_roi.set_ylim(shape_y, 0)
        self.ax_nav_roi.set_title(title)

        # tab:orange stands out clearly against the viridis nav-image colormap,
        # unlike tab10's default blue (index 0), which blends into it.
        color = np.array([*mcolors.to_rgb('tab:orange'), 0.5])
        mask_image = mask.reshape(shape_y, shape_x, 1) * color.reshape(1, 1, -1)
        self.img_display['seg_mask'].set_data(mask_image)
        self.img_display['seg_mask'].set_extent([0, shape_x, shape_y, 0])
        self.canvas.draw_idle()

    def _get_seg_temp_dir(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(os.path.dirname(script_dir), 'py4DTomo', 'io_utils', 'temp', 'roi4d_seg')
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def segment_image(self):
        """Run SAM2's single-image predictor on the currently loaded nav
        image using the point prompts added via Shift+Click, optionally
        combined with the last-drawn ROI (Ctrl+drag) as a box prompt - SAM2
        accepts points and a box together, the box narrowing the region and
        the points refining it further. Otherwise this mirrors the SAM2
        tab's "Seg Image" feature, just without object tracking."""
        has_roi = getattr(self, 'roi', None) is not None
        if not self.seg_points and not has_roi:
            self.logger.warning('Segment Image requested but no points or ROI have been added.')
            qtw.QMessageBox.critical(self, 'No Points or ROI Added',
                'Hold Shift and click on the image to add at least one point '
                '(left click = positive, right click = negative), and/or hold '
                'Ctrl and drag to draw a box, before segmenting.')
            return

        self.logger.info(
            'Starting SAM2 image segmentation (%d point(s), %s)...',
            len(self.seg_points), f'box={self.roi}' if has_roi else 'no box')
        self.button_segment_image.setDisabled(True)
        self._cancelling = False
        self.button_cancel.setEnabled(True)

        path_seg = self._get_seg_temp_dir()
        img_8bit = io.convert_img_to_8bit(self.navImg)
        seg_input_dict = {'image': img_8bit,
                          'points': np.array(self.seg_points),
                          'labels': np.array(self.seg_labels)}
        if has_roi:
            x, y, w, h = self.roi
            seg_input_dict['box'] = np.array([x, y, x + w, y + h])
        seg_input = pd.Series(seg_input_dict)
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
        text = bytes(self._process_sam.readAllStandardError()).decode('utf-8').strip()
        if text:
            self.logger.info('SAM2: %s', text)

    def _handle_finished_sam(self, exit_code=0, exit_status=0):
        self.button_segment_image.setEnabled(True)
        self.button_cancel.setDisabled(True)
        if self._cancelling:
            return
        text = bytes(self._process_sam.readAllStandardOutput()).decode('utf-8')
        try:
            result = json.loads(text.strip())
            with np.load(result['path']) as f:
                mask = f['mask']
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

        self.seg_mask = mask
        self.button_clear_points.setEnabled(True)
        self.logger.info('SAM2 image segmentation completed successfully.')
        self.show_seg_mask(mask)
        self.compute_seg_dp(mask)

    def compute_seg_dp(self, mask):
        """Sum diffraction patterns over the SAM2 mask's scan positions and
        display the result on the DP axis, immediately after segmentation -
        the masked-DP equivalent of drawing a rectangle ROI."""
        self.logger.info('Computing diffraction pattern summed over SAM2 mask...')
        dtype = os.path.splitext(self.fn)[-1]
        worker = Worker_CalculateDP_Mask(self.fn, self.seg_roi, mask, dtype,
                                         self.scanSize, self.dwellTime)
        worker.signals.result.connect(self.get_dp_from_mask)
        self.threadpool.start(worker)

    def get_dp_from_mask(self, dp):
        self.dp = dp
        self.ax_dp.set_title('DP (SAM2 Mask)')
        self.update_slider_range()
        self.slider_vmax.setValue(self.dp.max())
        self.slider_vmin.setValue(1)
        self.update_canvas(ax='dp')

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
            self.compute_sum_dp_from_threshold(dlg.mask, dlg.combo_threshMethod.currentText())

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
        worker = Worker_CalculateDP_Mask(self.fn, roi, mask, dtype, self.scanSize, self.dwellTime)
        worker.signals.result.connect(self._on_sum_dp_from_threshold_computed)
        worker.signals.error.connect(self._on_sum_dp_from_threshold_failed)
        self.threadpool.start(worker)

    def _on_sum_dp_from_threshold_computed(self, dp):
        self.button_sumDpFromThreshold.setEnabled(True)
        self.button_cancel.setDisabled(True)
        self.dp = dp
        self.ax_dp.set_title('Summed DP (thresholded)')
        self.update_slider_range()
        self.slider_vmax.setValue(self.dp.max())
        self.slider_vmin.setValue(1)
        self.update_canvas(ax='dp')

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
        """Stop the running SAM2 segmentation or DP computation and
        suppress the error popups that killing those workers would
        otherwise trigger.

        QThreadPool has no way to forcibly interrupt a runnable that has
        already started (only queued-but-not-started ones can be dropped),
        so an in-flight nav-image/DP computation will still finish in the
        background - its result is simply ignored. The SAM2 segmentation
        subprocess is a real OS process though, so that's actually killed
        outright."""
        self._cancelling = True
        self.threadpool.clear()
        n_killed = 0
        if hasattr(self, '_process_sam') and self._process_sam.state() != QProcess.NotRunning:
            self._process_sam.kill()
            n_killed = 1
        self.button_cancel.setDisabled(True)
        self.logger.warning('Cancelled by user (%d running process(es) killed).', n_killed)
        qtw.QMessageBox.information(self, 'Cancelled',
            'Running computation was cancelled.\n\n'
            'A nav-image/DP computation already running in the background will '
            'still finish silently - only queued work and the SAM2 segmentation '
            'process were stopped.')

    def cleanup(self):
        """Release resources held by this tab. Called by MainWindow.closeEvent
        so repeated runs of the app in the same console/kernel don't leave
        threadpools, running subprocesses, and matplotlib figures alive."""
        self.threadpool.clear()
        if hasattr(self, '_process_sam'):
            self._process_sam.kill()
        self.log_console.disconnect_log()
        plt.close(self.figure)

class WorkerSignals(QObject):
    finished = pyqtSignal()  # Signal to indicate task completion
    result = pyqtSignal(object)  # Signal to emit the result of the task
    error = pyqtSignal(object)  # Formatted traceback string, emitted on failure

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
                                             dwellTime=self.dwellTime, logger=self.logger)
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
                                   scanSize=self.scanSize, dwellTime=self.dwellTime,
                                   logger=self.logger)
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

class Worker_CalculateDP_Mask(QRunnable):
    """Sum diffraction patterns at the scan positions where an arbitrary
    (e.g. SAM2-segmented) mask is True, restricted to `roi` (the mask's
    bounding box) for efficiency. Reuses the same per-format masked loaders
    already used by the CV2/SAM2 tabs' 3DED extraction."""
    def __init__(self, fn, roi, mask, dtype, scanSize, dwellTime):
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
        self.signals = WorkerSignals()

    def run(self):
        try:
            dp = load_dp(self.fn, roi=self.roi, mask=self.mask, dtype=self.dtype,
                        scanSize=self.scanSize, dwellTime=self.dwellTime)
            if hasattr(dp, 'compute'):
                dp = dp.compute()
        except Exception:
            import traceback
            self.logger.exception(
                'Failed to calculate mask-based diffraction pattern after %.1f s.',
                perf_counter() - self._tic)
            self.signals.error.emit(traceback.format_exc())
            return
        self.signals.result.emit(dp)
        self.logger.info(
            'Mask-based diffraction pattern calculated successfully in %.1f s.',
            perf_counter() - self._tic)
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
