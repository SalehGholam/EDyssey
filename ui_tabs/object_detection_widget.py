# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:40:48 2025

@author: sgholam
"""

import sys
import os
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from skimage.filters import threshold_otsu, threshold_li, threshold_mean, threshold_yen
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import cv2
from copy import deepcopy
import EDyssey.io_utils as io
from .denoise_widget import DenoiseBox
#%%
class Object_Detector_Widget(qtw.QWidget):
    final_objects = pyqtSignal(list)
    
    def __init__(self, img, parent=None):
        """Store `img` and set up a debounced mask-update timer before building the widget."""
        super().__init__(parent)
        self.img = img
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self._do_update_mask)
        self.init_widget()
        
    def init_widget(self):
        self.central_widget = qtw.QWidget(self)
        self.setWindowTitle('Auto Object Detector')
        # self.setCentralWidget(self.central_widget)
        self.layout_main = qtw.QVBoxLayout(self)
        self.setLayout(self.layout_main)
        
        layout_top = qtw.QHBoxLayout()
        self.layout_main.addLayout(layout_top)
        fixed_w = 250
        fixed_h = 100
        #%% input
# =============================================================================
#         self.group_input = qtw.QGroupBox('Input Image')
#         # layout_input = qtw.QGridLayout()
#         layout_input = qtw.QVBoxLayout()
#         self.group_input.setLayout(layout_input)
#         layout_top.addWidget(self.group_input)
#         self.group_input.setFixedSize(fixed_w, fixed_h)
#         
#         layout_imgInput = qtw.QHBoxLayout()
#         layout_input.addLayout(layout_imgInput)
#         label_acqFast = qtw.QLabel('Dwell Time (\u00B5s)')
#         layout_imgInput.addWidget(label_acqFast)
#         self.spinbox_acqFast = qtw.QDoubleSpinBox()
#         layout_imgInput.addWidget(self.spinbox_acqFast)
#         self.spinbox_acqFast.setValue(1)
#         self.spinbox_acqFast.setMinimum(0.01)
#         
#         self.combo_inputImg = qtw.QComboBox()
#         layout_imgInput.addWidget(self.combo_inputImg)
#         self.combo_inputImg.addItems(['MAIN UI', 'HAADF', 'VDF', 'VBF', 'DOSE'])
#         self.combo_inputImg.setFixedWidth(75)
#         self.combo_inputImg.currentIndexChanged.connect(self.activate_layout_dir)
#         
#         self.layout_input_dir = qtw.QHBoxLayout()
#         layout_input.addLayout(self.layout_input_dir)
#         label_dir = qtw.QLabel('Dir.')
#         self.layout_input_dir.addWidget(label_dir)
#         label_dir.setFixedWidth(20)
#         
#         self.lineEdit_dir = qtw.QLineEdit()
#         self.layout_input_dir.addWidget(self.lineEdit_dir)
#         
#         self.button_dir = qtw.QPushButton('...')
#         self.layout_input_dir.addWidget(self.button_dir)
#         self.button_dir.setFixedWidth(30)
#         self.button_dir.clicked.connect(self.file_dialog)
#         
#         self.activate_layout_dir()
#         
#         layout_acq = qtw.QHBoxLayout()
#         layout_input.addLayout(layout_acq)
#         label_acqSlow = qtw.QLabel('Acq. Dwell Time (\u00B5s)')
#         layout_acq.addWidget(label_acqSlow)
#         self.spinbox_acqSlow = qtw.QSpinBox()
#         layout_acq.addWidget(self.spinbox_acqSlow)
#         self.spinbox_acqSlow.setValue(50)
#         self.spinbox_acqSlow.setSingleStep(10)
#         self.spinbox_acqSlow.setMinimum(1)
#         self.spinbox_acqSlow.setMaximum(10_000_000)
# =============================================================================
        #%% thresh
        self.checkbox_extendCanvas = qtw.QCheckBox('Extended Canvas')
        self.checkbox_extendCanvas.stateChanged.connect(self.extend_canvas)
        self.checkbox_extendCanvas.setChecked(False)
        self.layout_main.addWidget(self.checkbox_extendCanvas)
        
        
        self.group_thresh = qtw.QGroupBox('Threshold Method')
        layout_threshMethod = qtw.QVBoxLayout()
        layout_top.addWidget(self.group_thresh)
        self.group_thresh.setLayout(layout_threshMethod)    
        self.group_thresh.setFixedSize(fixed_w, fixed_h)

        self.checkbox_convert = qtw.QCheckBox("Revert")
        layout_threshMethod.addWidget(self.checkbox_convert)
        self.checkbox_convert.stateChanged.connect(self.update_mask)
        
        layout_threshMethod_1 = qtw.QHBoxLayout()
        layout_threshMethod.addLayout(layout_threshMethod_1)
        self.combo_threshMethod = qtw.QComboBox()
        layout_threshMethod_1.addWidget(self.combo_threshMethod)
        self.combo_threshMethod.addItems(['yen', 'otsu', 'li', 'mean'])
        self.combo_threshMethod.currentIndexChanged.connect(self.update_mask)

        self.button_reset_slider = qtw.QPushButton('Reset')
        layout_threshMethod_1.addWidget(self.button_reset_slider)
        self.button_reset_slider.clicked.connect(lambda: self.slider_deviation.setValue(100))
        
        layout_threshMethod_2 = qtw.QHBoxLayout()
        layout_threshMethod.addLayout(layout_threshMethod_2)
        label_deviation = qtw.QLabel('Deviation')
        layout_threshMethod_2.addWidget(label_deviation)
        
        self.slider_deviation = qtw.QSlider(1)
        layout_threshMethod_2.addWidget(self.slider_deviation)
        self.slider_deviation.setRange(0, 200)
        self.slider_deviation.setValue(100)
        self.slider_deviation.setSingleStep(5)
        self.slider_deviation.valueChanged.connect(self.update_mask)
        #%% kernel size layout
        self.group_kernel = qtw.QGroupBox('Kernel Sizes')
        layout_kernel = qtw.QGridLayout()
        layout_top.addWidget(self.group_kernel)
        self.group_kernel.setLayout(layout_kernel)
        self.group_kernel.setFixedSize(fixed_w, fixed_h)
        
        label_morph_openKernel = qtw.QLabel('Opening')
        self.spinbox_morph_openKernel = qtw.QSpinBox()
        self.spinbox_morph_openKernel.setValue(1)
        self.spinbox_morph_openKernel.valueChanged.connect(self.update_mask)
        
        label_morph_closeKernel = qtw.QLabel('Closing')
        self.spinbox_morph_closeKernel = qtw.QSpinBox()
        self.spinbox_morph_closeKernel.setValue(3)
        self.spinbox_morph_closeKernel.valueChanged.connect(self.update_mask)
        
        label_contourMinArea = qtw.QLabel('Min Area')
        self.spinbox_contourMinArea = qtw.QSpinBox()
        self.spinbox_contourMinArea.setValue(5)
        self.spinbox_contourMinArea.setSingleStep(5)
        self.spinbox_contourMinArea.setMaximum(1000)
        self.spinbox_contourMinArea.valueChanged.connect(self.update_mask)
        
        self.checkbox_contourMaxArea = qtw.QCheckBox('Max Area')
        self.checkbox_contourMaxArea.setChecked(False)
        self.checkbox_contourMaxArea.stateChanged.connect(self.update_mask)
        self.spinbox_contourMaxArea = qtw.QSpinBox()
        self.spinbox_contourMaxArea.setSingleStep(50)
        self.spinbox_contourMaxArea.setMaximum(999999)
        self.spinbox_contourMaxArea.setValue(1000)
        self.spinbox_contourMaxArea.valueChanged.connect(self.update_mask)
        
        label_dilate = qtw.QLabel('Dilation (px)')
        self.spinbox_dilateKernel = qtw.QSpinBox()
        self.spinbox_dilateKernel.setValue(3)
        self.spinbox_dilateKernel.setToolTip(
            'Pixels to grow each detected box by on every side, after detection - '
            'purely enlarges the box/ROI around an already-found object, does not '
            'affect detection itself (0 = no growth).')
        self.spinbox_dilateKernel.valueChanged.connect(self.update_mask)

        for i, wid in enumerate([label_morph_openKernel, self.spinbox_morph_openKernel,
                    label_morph_closeKernel, self.spinbox_morph_closeKernel,
                    label_contourMinArea, self.spinbox_contourMinArea,
                    self.checkbox_contourMaxArea, self.spinbox_contourMaxArea,
                    label_dilate, self.spinbox_dilateKernel]):
            layout_kernel.addWidget(wid, i//4, i%4)
        #%% denoise
        # The only smoothing stage now (see _do_update_mask) - Kernel Sizes'
        # own separate "Blurring" control was removed since this box's
        # 'Gaussian Blur' method covers that need identically; pick any
        # other method for a stronger, content-aware alternative instead.
        # 'None' (the default) is an exact no-op.
        self.denoise_box = DenoiseBox(title='Denoise', show_apply_all=False)
        layout_top.addWidget(self.denoise_box)
        self.denoise_box.settingsChanged.connect(self.update_mask)
        #%% blob identification
        # Splits a blob that plain contour-finding would report as ONE
        # merged object (two particles touching/overlapping in the
        # threshold mask) back into its individual particles - see
        # EDyssey.io_utils.blob_segmentation, the same module ROI Tracker's
        # own per-ROI "Blob" column already uses (BlobSegmentationDialog).
        # Off by default ('connected' + unchecked): detect_objects() then
        # takes its original, pixel-identical box-per-contour path.
        self.group_blob = qtw.QGroupBox('Blob Identification')
        layout_blob = qtw.QVBoxLayout(self.group_blob)
        layout_top.addWidget(self.group_blob)

        self.checkbox_blobEnabled = qtw.QCheckBox('Split touching objects')
        self.checkbox_blobEnabled.setToolTip(
            'When checked, each detected object below is further split into '
            'individual blobs by the method below (applied separately to each '
            "one, the same way ROI Tracker's own per-ROI Blob Selection does) "
            "before boxes are drawn - use this when two touching/overlapping "
            'particles are otherwise reported as one merged object.')
        self.checkbox_blobEnabled.stateChanged.connect(self._on_blob_enabled_toggled)
        layout_blob.addWidget(self.checkbox_blobEnabled)

        self.combo_blobMethod = qtw.QComboBox()
        for method_id, spec in io.BLOB_SEGMENTATION_METHODS.items():
            self.combo_blobMethod.addItem(spec['label'], method_id)
        found = self.combo_blobMethod.findData(io.DEFAULT_BLOB_METHOD)
        self.combo_blobMethod.setCurrentIndex(found if found >= 0 else 0)
        self.combo_blobMethod.currentIndexChanged.connect(self._on_blob_method_changed)
        layout_blob.addWidget(self.combo_blobMethod)

        self.form_blobParams = qtw.QFormLayout()
        layout_blob.addLayout(self.form_blobParams)
        self._blob_param_widgets = {}
        self._blob_params = io.default_blob_params(self.combo_blobMethod.currentData())
        self._rebuild_blob_param_form()
        self._on_blob_enabled_toggled(False)  # starts unchecked -> greyed out
        #%% finish button
        layout_end = qtw.QVBoxLayout()
        layout_top.addLayout(layout_end)
        self.button_finish = qtw.QPushButton('Finish')
        layout_end.addWidget(self.button_finish)
        self.button_finish.clicked.connect(lambda: self.finish_widget(True))
        self.button_finish.setFixedWidth(100)
        
        self.button_cancel = qtw.QPushButton('Cancel')
        layout_end.addWidget(self.button_cancel)
        self.button_cancel.setFixedWidth(100)
        self.button_cancel.clicked.connect(lambda: self.finish_widget(False))
        
        
        layout_top.addStretch(1)
        #%% canvas + object list
        layout_canvasRow = qtw.QHBoxLayout()
        self.layout_main.addLayout(layout_canvasRow)

        layout_canvas = qtw.QVBoxLayout()
        layout_canvasRow.addLayout(layout_canvas, 1)

        # self.figure = Figure(figsize=(5,5))
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout_canvas.addWidget(self.canvas)

        #%% object list
        # Lets a false positive be dropped before Finish sends self.boxes to
        # the main UI's ROI list - rows always match self.boxes 1:1 (both
        # rebuilt together in _populate_object_list), and removing one here
        # only edits self.boxes and redraws the two object panels directly
        # (_remove_objects) - no re-run of detection, so it's instant and
        # independent of every slider/checkbox above.
        group_objects = qtw.QGroupBox('Detected Objects')
        layout_objects = qtw.QVBoxLayout(group_objects)
        group_objects.setFixedWidth(220)
        layout_canvasRow.addWidget(group_objects)

        self.list_objects = qtw.QListWidget()
        self.list_objects.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        self.list_objects.setToolTip(
            'Every object currently detected, one row per entry in the '
            '"objects" panel. Select one or more and click "Remove Selected" '
            '(or double-click one row) to drop it before Finish - it will '
            "not be sent to the main UI's object list.")
        self.list_objects.itemDoubleClicked.connect(self._on_object_item_double_clicked)
        layout_objects.addWidget(self.list_objects, 1)

        self.button_removeSelected = qtw.QPushButton('Remove Selected')
        self.button_removeSelected.setToolTip('Drop the selected object(s) - they will not be sent on Finish.')
        self.button_removeSelected.clicked.connect(self._remove_selected_from_list)
        layout_objects.addWidget(self.button_removeSelected)

        # self.figure.tight_layout()
        # self.update_canvas()
        self.extend_canvas(False)
        self.update_mask()

        layout_canvas.addWidget(NavigationToolbar(self.canvas, self))
#%% functions
# =============================================================================
#     def file_dialog(self):
#         # sender = self.sender()
#         # if sender == self.button_dir_navSignal:
#         file_filter = "supported signals (*.tif *.tpx3 *.hdf5);;All Files (*)"
#         # path = qtw.QFileDialog.getOpenFileNames(self, "Select 4D Signals Folder", '', file_filter)
#         path = qtw.QFileDialog.getOpenFileName(self, "Select Nav. Image", '', file_filter)
#         if path and os.path.isfile(path[0]):
#             self.lineEdit_dir_navSignal.setText(path[0])
#             self.lineEdit_dir.setText(path[0])
# =============================================================================
    
# =============================================================================
#     def activate_layout_dir(self):
#         if self.combo_inputImg.currentText() == 'MAIN UI':
#             act = False
#         else:
#             act = True
#         for i in range(self.layout_input_dir.count()):
#             item = self.layout_input_dir.itemAt(i)
#             wid = item.widget()
#             wid.setEnabled(act)
# =============================================================================
    
    def _on_blob_enabled_toggled(self, checked):
        """Grey out the method/params while blob identification is off -
        same convention as the Declustering/Max Area sub-controls
        elsewhere in this app."""
        self.combo_blobMethod.setEnabled(bool(checked))
        for w in self._blob_param_widgets.values():
            w.setEnabled(bool(checked))
        self.update_mask()

    def _on_blob_method_changed(self):
        self._blob_params = io.default_blob_params(self.combo_blobMethod.currentData())
        self._rebuild_blob_param_form()
        self.update_mask()

    def _on_blob_param_changed(self):
        for key, widget in self._blob_param_widgets.items():
            self._blob_params[key] = widget.value()
        self.update_mask()

    def _rebuild_blob_param_form(self):
        """(Re)build the parameter spinboxes for the currently-selected
        blob method - see BlobSegmentationDialog._rebuild_param_form,
        whose pattern this mirrors (same BLOB_SEGMENTATION_METHODS spec
        shape)."""
        while self.form_blobParams.rowCount():
            self.form_blobParams.removeRow(0)
        self._blob_param_widgets = {}
        method = self.combo_blobMethod.currentData()
        spec = io.BLOB_SEGMENTATION_METHODS[method]
        for key, label, low, high, step, decimals, tooltip in spec['param_specs']:
            if decimals > 0:
                widget = qtw.QDoubleSpinBox()
                widget.setDecimals(decimals)
            else:
                widget = qtw.QSpinBox()
            widget.setRange(low, high)
            widget.setSingleStep(step)
            widget.setValue(self._blob_params.get(key, spec['default_params'][key]))
            widget.setToolTip(tooltip)
            widget.setEnabled(self.checkbox_blobEnabled.isChecked())
            widget.valueChanged.connect(self._on_blob_param_changed)
            self.form_blobParams.addRow(label + ':', widget)
            self._blob_param_widgets[key] = widget

    def _split_blobs_per_component(self):
        """Split each connected component of self.mask_contours (post
        area-filter - screens out noise specks before they'd multiply into
        spurious splits below - but pre-dilation, since dilation would only
        ever re-merge a split, never help it) into individual blobs via the
        selected BLOB_SEGMENTATION_METHODS method, one component at a time -
        NOT one call over the whole frame's mask.

        This matters, not just mirrors BlobSegmentationDialog for style:
        K-Means/GMM cluster every foreground pixel handed to them into
        exactly n_clusters groups (see blob_segmentation.py's own
        _cluster_pixel_coords) - correct when what's handed in is one
        already-isolated merged blob (BlobSegmentationDialog's per-ROI
        use), meaningless if handed this whole multi-particle frame at
        once (it would carve the entire image into n_clusters giant zones,
        not split each touching pair). Looping per pre-existing component
        keeps every method - including Watershed, which is already safe
        whole-frame on its own - correct at this whole-frame scale.

        Intensity-based methods (Watershed - Intensity) get self.img_denoised
        as their crop - the mask itself only says WHERE the thresholded
        region is, not which pixels within it are brighter/dimmer, and using
        the actual denoised contrast there (not a further-blurred or binary
        version) is what lets that method find a real intensity dip between
        two touching particles.

        Returns a global int-labeled array, same shape as self.img (0 =
        background, 1..N one id per final split blob, unique across every
        component, not just within one)."""
        method = self.combo_blobMethod.currentData()
        params = dict(self._blob_params)
        mask_u8 = self.mask_contours.astype('uint8')
        n_components, components = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)[:2]
        global_labels = np.zeros(mask_u8.shape, dtype=int)
        next_id = 1
        for comp_id in range(1, n_components):
            ys, xs = np.where(components == comp_id)
            if ys.size == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop_mask = components[y0:y1, x0:x1] == comp_id
            crop_img = self.img_denoised[y0:y1, x0:x1]
            local = io.label_blobs(crop_mask, crop_img, method, params)
            if local is None:  # 'connected' (or unrecognized) - one blob, unsplit
                local = crop_mask.astype(int)
            for lid in np.unique(local):
                if lid == 0:
                    continue
                global_labels[y0:y1, x0:x1][local == lid] = next_id
                next_id += 1
        return global_labels

    def extend_canvas(self, state):
        """Rebuild the figure's axes: an 8-panel pipeline-stage view if `state`
        is truthy, otherwise the plain 2-panel raw/final-objects view."""
        size_x, size_y = self.img.shape
        img_zero = np.zeros((size_x, size_y), dtype='int8')
        self.figure.clf()
        if state:
            # 2 rows x 4 cols: raw/final mirror the plain 2-panel view's own
            # pairing (positions 0/1, for an at-a-glance result even here),
            # the rest read left-to-right, top-to-bottom in pipeline order.
            # No separate "dilated" panel - dilation is now pure box padding
            # applied after detection (see detect_objects/pad_box), not a
            # mask stage with its own image to show.
            self.axes = [self.figure.add_subplot(2, 4, i + 1) for i in range(8)]
            (self.ax_raw, self.ax_maskFinal, self.ax_denoised, self.ax_binarized,
             self.ax_morph, self.ax_contours, self.ax_blobs, self.ax_objects) = self.axes
            titles = ['raw', 'final objects', 'denoised', 'binarized',
                      'morph', 'contours', 'blobs', 'objects']

            self.img_disp = {}
            self.img_disp[0] = self.ax_raw.imshow(self.img)
            self.img_disp[1] = self.ax_maskFinal.imshow(img_zero)
            for i in range(2, 8):
                self.img_disp[i] = self.axes[i].imshow(img_zero)
        else:
            self.img_disp = {}
            self.ax_raw = self.figure.add_subplot(121)
            self.ax_maskFinal = self.figure.add_subplot(122)
            self.axes = [self.ax_raw, self.ax_maskFinal]
            titles = ['raw', 'final objects']
            self.img_disp = {}
            self.img_disp[0] = self.ax_raw.imshow(self.img)
            self.img_disp[1] = self.ax_maskFinal.imshow(img_zero)

        # figure.clf() above wipes any previous colorbars along with the old
        # axes, so these are (re)created fresh every call, one per panel.
        for i, ax in enumerate(self.axes):
            ax.set_axis_off()
            ax.set_title(titles[i])
            self.figure.colorbar(self.img_disp[i], ax=ax, fraction=0.046, pad=0.04)
        # self.canvas.draw()
        
        self.img_disp[0].set_clim(vmin=self.img.min(), vmax=self.img.max())
        
        self.update_mask()
            
   
    def normalize_image(self, img):
        """Min-max scale `img` to uint8 [0, 255]; no-op if already uint8."""
        if img.dtype != 'uint8': # TODO not sure if necessary
            img_8bit = deepcopy(img)
            # img_8bit[img_8bit < 0] = 0
            img_8bit = ((img - img.min()) / (img.max() - img.min())) * 255.0
            img_8bit = img_8bit.astype(np.uint8)
            return img_8bit
        else:
            return img
# =============================================================================
#             img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
#             img_8bit = img_norm.astype(np.uint8)
# =============================================================================
    
    def binarize_image(self, img):
        """Threshold `img` using the selected method, offset by the Deviation slider."""
        thresh_method = self.combo_threshMethod.currentText()
        threshold_methods = {
        'otsu': threshold_otsu,
        'li': threshold_li,
        'yen': threshold_yen,
        'mean': threshold_mean}
        threshold_func = threshold_methods[thresh_method]
        th = threshold_func(img)
        thresh_offset = (200 - self.slider_deviation.value()) / 100
        
        thresh = thresh_offset * th
        img_mask = img > thresh
        img_mask = img_mask.astype('uint8')
        return img_mask
    
    def morphological_transformation(self, mask, kernel_size=3):
        """Apply a closing then an opening to `mask`; the `kernel_size` argument
        is ignored, kernel sizes are always read from the Opening/Closing spinboxes."""
        kernel_size = self.spinbox_morph_closeKernel.value()
        kernel_close = np.ones((kernel_size,kernel_size), np.uint8)

        kernel_size = self.spinbox_morph_openKernel.value()
        kernel_open = np.ones((kernel_size,kernel_size), np.uint8)

        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
        return mask_opened
    
    def find_contours(self, mask, min_area=5):
        """Find external contours in `mask` and rasterize the ones passing the
        area filter into a new mask. `min_area` is ignored - always read from
        the Min Area spinbox - and a Max Area filter is applied too if checked."""
        min_area = self.spinbox_contourMinArea.value()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_selected = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        if self.checkbox_contourMaxArea.isChecked():
            max_area = self.spinbox_contourMaxArea.value()
            contours_selected = [cnt for cnt in contours_selected if cv2.contourArea(cnt) <= max_area]
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, contours_selected, -1, 255, thickness=-1)
        contour_mask = (contour_mask > 1).astype('uint8')
        return contour_mask, contours_selected
    
    def pad_box(self, box):
        """Grow one (x, y, w, h) box by the Dilation spinbox's value on
        every side, clamped to the image bounds - applied identically to
        every detected box regardless of which path produced it (plain
        contours, or Blob Identification's split), right at the end of
        detect_objects. Purely enlarges the ROI drawn around an already-
        detected object; it never touches the mask/detection itself, so it
        can't merge two boxes together or affect what gets found."""
        pad = self.spinbox_dilateKernel.value()
        if pad <= 0:
            return box
        x, y, w, h = box
        size_y, size_x = self.img.shape
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(size_x, x + w + pad), min(size_y, y + h + pad)
        return (x0, y0, x1 - x0, y1 - y0)
    
    def update_mask(self):
        """(Re)start the debounce timer; the actual pipeline runs in _do_update_mask."""
        self._debounce_timer.start()

    def _do_update_mask(self):
        """Run the full detection pipeline (normalize -> denoise -> binarize
        -> morph -> contours -> detect [-> blob split] -> pad) and redraw
        the canvas.

        Denoise (see DenoiseBox) is now the only smoothing stage - pick
        'Gaussian Blur' there for the old small-kernel blur, or a stronger
        content-aware method instead; 'None' (the default) is an exact
        no-op. Blob identification (optional, see
        _split_blobs_per_component) runs on mask_contours, using
        img_denoised as its intensity reference - after the area filter, but
        before dilation, which is now pure box padding applied at the very
        end (see pad_box) rather than a mask step, so it can only ever
        enlarge an already-detected box, never merge or reshape one."""
        self.img_8bit = self.normalize_image(self.img)
        if self.checkbox_convert.isChecked():
            self.img_8bit = 255 - self.img_8bit
        self.img_denoised = self.denoise_box.apply(self.img_8bit)
        self.mask_binarized = self.binarize_image(self.img_denoised)
        self.mask_morph = self.morphological_transformation(self.mask_binarized)
        self.mask_contours, self.contours = self.find_contours(self.mask_morph)
        self.img_objects, self.boxes, self.img_blobs = self.detect_objects()

        self.masks = [self.img, self.img_objects, self.img_denoised,
                      self.mask_binarized, self.mask_morph, self.mask_contours,
                      self.img_blobs, self.img_objects]
        self.update_canvas()
        self._populate_object_list()
    
    def update_canvas(self):
        """Push self.masks into the displayed image panels and refresh the
        final-objects-count title."""
        for k in self.img_disp.keys():
            img = self.masks[k]
            self.img_disp[k].set_data(img)
            self.img_disp[k].set_clim(vmin=img.min(), vmax=img.max())
        self.ax_maskFinal.set_title(f'Final Objects: {len(self.boxes)}')
        self.canvas.draw()
        
    def _draw_boxes(self, boxes):
        """Numbered green rectangles for `boxes` (already final - e.g.
        already padded), on a fresh copy of self.img_8bit. Shared by
        detect_objects (first draw, after a full pipeline run) and
        _remove_objects (redraw after dropping some from the list, with
        no pipeline re-run - boxes are drawn exactly as given, not
        recomputed/re-padded)."""
        img = deepcopy(self.img_8bit)
        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{idx}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

    def detect_objects(self):
        """Return (annotated 8-bit image, list of (x, y, w, h) boxes,
        blob-panel preview image).

        Two box sources, chosen by the Blob Identification checkbox - both
        from self.mask_contours (post area-filter), never a dilated mask:
        - unchecked (default): every external contour of mask_contours -
          one box per merged blob (touching particles included together).
        - checked: one box per split blob from _split_blobs_per_component.

        Either way, every box is then grown by pad_box (the Dilation
        spinbox) before it's drawn/returned - dilation's only remaining
        role, applied uniformly regardless of which path produced the box."""
        if self.checkbox_blobEnabled.isChecked():
            labels = self._split_blobs_per_component()
            ids = [int(lid) for lid in np.unique(labels) if lid != 0]
            raw_boxes = []
            for lid in ids:
                ys, xs = np.where(labels == lid)
                x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
                raw_boxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
            blob_preview = labels
        else:
            contours, _ = cv2.findContours(self.mask_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_boxes = [cv2.boundingRect(cnt) for cnt in contours]
            # The blob panel still previews "what each currently-detected
            # object would be split into" even with the checkbox off, so
            # turning it on is never a surprise.
            blob_preview = self.mask_contours

        boxes = [self.pad_box(b) for b in raw_boxes]
        img = self._draw_boxes(boxes)
        return img, boxes, blob_preview

    def _populate_object_list(self):
        """Rebuild self.list_objects from self.boxes, one row per entry, in
        the same order/numbering the "objects" panel itself draws - so row
        N in the list is always object N in that panel."""
        self.list_objects.blockSignals(True)
        self.list_objects.clear()
        for i, (x, y, w, h) in enumerate(self.boxes, start=1):
            self.list_objects.addItem(qtw.QListWidgetItem(f'{i}:  x={x}, y={y}, w={w}, h={h}'))
        self.list_objects.blockSignals(False)

    def _remove_objects(self, indices):
        """Drop self.boxes[i] for every i in `indices`, then redraw just
        the "final objects"/"objects" panels and the list from what's left -
        no re-run of the detection pipeline (normalize/denoise/binarize/
        .../pad), so this is instant regardless of image size and immune to
        every detection parameter above changing meanwhile."""
        if not indices:
            return
        self.boxes = [b for i, b in enumerate(self.boxes) if i not in indices]
        self.img_objects = self._draw_boxes(self.boxes)
        # self.masks holds self.img_objects at index 1 (the always-visible
        # raw/final-objects pair) and again at its own last index (the full
        # pipeline-order "objects" panel, extended view only) - see
        # _do_update_mask's own masks list.
        for k in {1, len(self.masks) - 1}:
            if k in self.img_disp:
                self.img_disp[k].set_data(self.img_objects)
                self.img_disp[k].set_clim(vmin=self.img_objects.min(), vmax=self.img_objects.max())
        self.ax_maskFinal.set_title(f'Final Objects: {len(self.boxes)}')
        self.canvas.draw()
        self._populate_object_list()

    def _remove_selected_from_list(self):
        rows = {self.list_objects.row(item) for item in self.list_objects.selectedItems()}
        self._remove_objects(rows)

    def _on_object_item_double_clicked(self, item):
        self._remove_objects({self.list_objects.row(item)})

    def finish_widget(self, ret=True):
        """Emit the detected boxes (or None if cancelled) via final_objects, then close."""
        if ret:
            self.final_objects.emit(self.boxes)
        else:
            self.final_objects.emit(None)
        self.close()
    
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    
    window = Object_Detector_Widget(np.random.randint(0, 1000, size=(512,512)))
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())
