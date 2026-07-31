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
#%%
class Object_Detector_Widget(qtw.QWidget):
    final_objects = pyqtSignal(list)
    
    def __init__(self, img, parent=None):
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
        
        label_blur = qtw.QLabel('Blurring')
        self.combo_blurKernel = qtw.QComboBox()
        self.combo_blurKernel.addItems([str(i) for i in range(1,23,2)])
        self.combo_blurKernel.setCurrentText('3')
        self.combo_blurKernel.currentTextChanged.connect(self.update_mask)
        
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
        
        label_dilate = qtw.QLabel('Dilation')
        self.spinbox_dilateKernel = qtw.QSpinBox()
        self.spinbox_dilateKernel.setValue(3)
        self.spinbox_dilateKernel.valueChanged.connect(self.update_mask)
        
        for i, wid in enumerate([label_blur, self.combo_blurKernel, 
                    label_morph_openKernel, self.spinbox_morph_openKernel, 
                    label_morph_closeKernel, self.spinbox_morph_closeKernel, 
                    label_contourMinArea, self.spinbox_contourMinArea, 
                    self.checkbox_contourMaxArea, self.spinbox_contourMaxArea,
                    label_dilate, self.spinbox_dilateKernel]):
            layout_kernel.addWidget(wid, i//4, i%4)
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
        #%% canvas
        layout_canvas = qtw.QVBoxLayout()
        self.layout_main.addLayout(layout_canvas)
        
        # self.figure = Figure(figsize=(5,5))
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout_canvas.addWidget(self.canvas)
        
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
    
    def extend_canvas(self, state):
        size_x, size_y = self.img.shape
        img_zero = np.zeros((size_x, size_y), dtype='int8')
        self.figure.clf()
        if state:
            self.ax_raw = self.figure.add_subplot(241)
            self.ax_maskFinal = self.figure.add_subplot(242)
            self.ax_blur = self.figure.add_subplot(243)
            self.ax_binarized = self.figure.add_subplot(244)
            self.ax_morph = self.figure.add_subplot(245)
            self.ax_contours = self.figure.add_subplot(246)
            self.ax_dilated = self.figure.add_subplot(247)
            self.ax_objects = self.figure.add_subplot(248)
            
            self.axes = [self.ax_raw, self.ax_maskFinal, self.ax_blur, self.ax_binarized,
                         self.ax_morph, self.ax_contours, self.ax_dilated, self.ax_objects]
            titles = ['raw', 'final objects', 'blurred', 'binarized', 'morph', 
                      'contours', 'dilated', 'objects']
            
            self.img_disp = {}
            self.img_disp[0] = self.ax_raw.imshow(self.img)
            self.img_disp[1] = self.ax_maskFinal.imshow(img_zero)

            self.img_disp[2] = self.ax_blur.imshow(img_zero)
            self.img_disp[3] = self.ax_binarized.imshow(img_zero)
            self.img_disp[4] = self.ax_morph.imshow(img_zero)
            self.img_disp[5] = self.ax_contours.imshow(img_zero)
            self.img_disp[6] = self.ax_dilated.imshow(img_zero)
            self.img_disp[7] = self.ax_objects.imshow(img_zero)
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
    
    def gaussian_blur(self, img, kernel_size=3):
        kernel_size = int(self.combo_blurKernel.currentText())
        return cv2.GaussianBlur(img, (kernel_size,kernel_size), 0)
    
    def binarize_image(self, img):
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
        kernel_size = self.spinbox_morph_closeKernel.value()
        kernel_close = np.ones((kernel_size,kernel_size), np.uint8)

        kernel_size = self.spinbox_morph_openKernel.value()
        kernel_open = np.ones((kernel_size,kernel_size), np.uint8)

        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
        return mask_opened
    
    def find_contours(self, mask, min_area=5):
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
    
    def dilate_mask(self, mask, kernel_size=3):
        kernel_size = self.spinbox_dilateKernel.value()
        if kernel_size == 0:
            return mask
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)
    
    def update_mask(self):
        self._debounce_timer.start()

    def _do_update_mask(self):
        self.img_8bit = self.normalize_image(self.img)
        if self.checkbox_convert.isChecked():
            self.img_8bit = 255 - self.img_8bit
        self.img_blurred = self.gaussian_blur(self.img_8bit)
        self.mask_binarized = self.binarize_image(self.img_blurred)
        self.mask_morph = self.morphological_transformation(self.mask_binarized)
        self.mask_contours, self.contours = self.find_contours(self.mask_morph)
        self.mask_dilated = self.dilate_mask(self.mask_contours)
        self.img_objects, self.boxes = self.detect_objects()

        self.masks = [self.img, self.img_objects, self.img_blurred, self.mask_binarized,
                      self.mask_morph, self.mask_contours, self.mask_dilated, self.img_objects]
        self.update_canvas()
    
    def update_canvas(self):
        for k in self.img_disp.keys():
            img = self.masks[k]
            self.img_disp[k].set_data(img)
            self.img_disp[k].set_clim(vmin=img.min(), vmax=img.max())
        no_points = len(np.where(self.mask_dilated > 0)[0])
        self.ax_maskFinal.set_title(f'Final Objects: {len(self.boxes)}')
        self.canvas.draw()
        
    def detect_objects(self):
        boxes = []
        idx = 1
        img = deepcopy(self.img_8bit)
        contours, _ = cv2.findContours(self.mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for _, cnt in enumerate(self.contours):
        for _, cnt in enumerate(contours):
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x,y,w,h))
            # Draw the bounding box on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{idx}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Optionally, extract the ROI for further processing (e.g., classification)
            roi = img[y:y+h, x:x+w]
            # ... perform further operations on roi if needed
            idx += 1
        return img, boxes
    
    def finish_widget(self, ret=True):
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
