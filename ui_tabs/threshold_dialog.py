# -*- coding: utf-8 -*-
"""
Shared "PACBED from Threshold" popup, used by both Tab_Create_NavSignal and
Tab_ROI_on_4D so the two tabs don't duplicate the widget.
"""

import os
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
import py4DTomo.io_utils as io


class ThresholdDialog(qtw.QDialog):
    """Popup for checking/adjusting the real-space threshold mask used by
    "PACBED from Threshold" before committing to it - kept off the main
    navigation-image plot (which stays a plain image, no overlay); this is
    its own small window instead."""

    def __init__(self, parent, img, fn):
        super().__init__(parent)
        self.setWindowTitle('PACBED from Threshold')
        self.resize(600, 650)
        self.img = img
        self.fn = fn
        self.mask = None

        layout = qtw.QVBoxLayout(self)

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot()
        self.img_display = self.ax.imshow(img, cmap='viridis')
        self.img_display.set_clim(img.min(), img.max())
        # Starts fully transparent (all-zero RGBA); update_preview() fills in
        # color+alpha only where the mask is currently True.
        self.img_display_overlay = self.ax.imshow(np.zeros((*img.shape, 4)))
        self.ax.set_axis_off()
        self.ax.set_title(os.path.basename(fn), fontsize=9)
        self.colorbar = self.figure.colorbar(
            self.img_display, ax=self.ax, fraction=0.046, pad=0.04)
        layout.addWidget(self.canvas)
        layout.addWidget(NavigationToolbar(self.canvas, self))

        row = qtw.QHBoxLayout()
        layout.addLayout(row)
        row.addWidget(qtw.QLabel('Threshold'))
        self.combo_threshMethod = qtw.QComboBox()
        self.combo_threshMethod.addItems(['otsu', 'li', 'yen'])
        row.addWidget(self.combo_threshMethod)
        self.combo_threshMethod.currentIndexChanged.connect(self.update_preview)

        row.addWidget(qtw.QLabel('Blur'))
        self.combo_blur = qtw.QComboBox()
        self.combo_blur.addItems([str(i) for i in range(1, 23, 2)])
        row.addWidget(self.combo_blur)
        self.combo_blur.setToolTip(
            'Gaussian blur kernel applied before thresholding (reduces noise in the '
            'mask) - the displayed image itself stays sharp, only the mask is affected')
        self.combo_blur.currentIndexChanged.connect(self.update_preview)

        self.label_threshDev = qtw.QLabel('Deviation: 100%')
        row.addWidget(self.label_threshDev)
        self.slider_threshDev = qtw.QSlider(Qt.Horizontal)
        self.slider_threshDev.setRange(0, 200)
        self.slider_threshDev.setValue(100)
        row.addWidget(self.slider_threshDev)
        self.slider_threshDev.setToolTip(
            "Scales the auto-threshold up/down (100% = the method's own value)")
        self.slider_threshDev.valueChanged.connect(self.update_preview)

        buttons = qtw.QHBoxLayout()
        layout.addLayout(buttons)
        self.button_compute = qtw.QPushButton('Compute PACBED from Threshold')
        self.button_compute.clicked.connect(self.conditional_accept)
        buttons.addWidget(self.button_compute)
        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.button_cancel)

        self.update_preview()

    def update_preview(self):
        method = self.combo_threshMethod.currentText()
        threshold_funcs = {'otsu': threshold_otsu, 'li': threshold_li, 'yen': threshold_yen}
        dev = self.slider_threshDev.value()
        self.label_threshDev.setText(f'Deviation: {dev}%')
        blur_kernel = int(self.combo_blur.currentText())
        # cv2.GaussianBlur doesn't support arbitrary integer dtypes (e.g. the
        # int64 a summed navigation image often comes back as) - casting to
        # float32 keeps this working regardless of the source format.
        img_blur = io.gaussian_blur(self.img.astype(np.float32), blur_kernel)
        thresh = threshold_funcs[method](img_blur) * (dev / 100)
        self.mask = img_blur >= thresh

        color = np.array([*mcolors.to_rgb('tab:orange'), 0.45])
        mask_image = self.mask.reshape(*self.mask.shape, 1) * color.reshape(1, 1, -1)
        self.img_display_overlay.set_data(mask_image)
        self.canvas.draw_idle()

    def conditional_accept(self):
        if self.mask is None or not self.mask.any():
            qtw.QMessageBox.warning(self, 'Empty Mask',
                'The current threshold selects no scan positions - adjust the '
                'method/deviation before computing.')
            return
        self.accept()
