# -*- coding: utf-8 -*-
"""
Shared "Summed DP from Threshold" popup, used by both Tab_Create_NavSignal and
Tab_ROI_on_4D so the two tabs don't duplicate the widget.
"""

import os
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
import EDyssey.io_utils as io
from .ribbon import RibbonPanel, RibbonTool
from .denoise_widget import DenoiseBox


class ThresholdDialog(qtw.QDialog):
    """Popup for checking/adjusting the real-space threshold mask used by
    "Summed DP from Threshold" before committing to it - kept off the main
    navigation-image plot (which stays a plain image, no overlay); this is
    its own small window instead.

    Below the canvas, a small ribbon offers two rectangle-draw tools -
    Include ROI / Exclude ROI - to manually force a region into or out of
    the auto-thresholded mask (e.g. to drop a bright artifact the threshold
    alone can't distinguish, or recover a dim region it missed). Manual
    edits (self._manual_edits) are replayed on top of a freshly
    auto-thresholded mask every time the Threshold/Deviation/Denoise
    controls change, rather than being discarded by them - see
    update_preview()."""

    def __init__(self, parent, img, fn, denoise_state=None):
        """
        Args:
            img: Navigation-image array to display and threshold.
            fn: Source file path, used only for the window title (basename).
            denoise_state: Optional DenoiseBox.get_state() dict to seed this
                dialog's own Denoise box from - typically the calling tab's
                own current Denoise setting (ROI on 4D's box_denoise), so
                this starts out looking the same as what's already on
                screen there. None (e.g. the "Make Nav. Sig." tab, which has
                no Denoise box of its own) starts at 'None'/untouched.
        """
        super().__init__(parent)
        self.setWindowTitle('Summed DP from Threshold')
        self.resize(600, 750)
        self.img = img
        self.fn = fn
        self.mask = None
        # [(row0, row1, col0, col1, value), ...] - manual Include/Exclude
        # ROI edits (see the ribbon below), each applied in order on top of
        # the freshly-recomputed auto-threshold mask every update_preview()
        # call, so tweaking Threshold/Deviation/Denoise afterward doesn't
        # silently discard them.
        self._manual_edits = []
        self._roi_drag = None       # (x0, y0, value) while drawing a rectangle
        self._roi_rect_artist = None
        self._roi_bg = None

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

        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

        # Include ROI / Exclude ROI - reuses the exact same 'rect_in'/
        # 'rect_out' drawn-icon kinds Fine-Tune Mask's own ribbon uses (see
        # mask_edit_dialog.py), so a drag-to-paint rectangle tool looks and
        # behaves consistently across the app. No Pan/Zoom tools here -
        # the NavigationToolbar above already provides those, independent
        # of whichever of these two (if either) is armed.
        self.ribbon = RibbonPanel([
            RibbonTool('include_roi', 'rect_in', 'Draw a rectangular ROI to INCLUDE '
                      '(force into the mask)', 'tool'),
            RibbonTool('exclude_roi', 'rect_out', 'Draw a rectangular ROI to EXCLUDE '
                      '(force out of the mask)', 'tool'),
            RibbonTool('sep1', kind='separator'),
            RibbonTool('clear_edits', 'undo', 'Clear every manual Include/Exclude ROI edit',
                      'action', self._clear_manual_edits),
        ], parent=self, orientation='horizontal')
        layout.addWidget(self.ribbon)

        row = qtw.QHBoxLayout()
        layout.addLayout(row)
        row.addWidget(qtw.QLabel('Threshold'))
        self.combo_threshMethod = qtw.QComboBox()
        self.combo_threshMethod.addItems(['otsu', 'li', 'yen'])
        row.addWidget(self.combo_threshMethod)
        self.combo_threshMethod.currentIndexChanged.connect(self.update_preview)

        self.label_threshDev = qtw.QLabel('Deviation: 100%')
        row.addWidget(self.label_threshDev)
        self.slider_threshDev = qtw.QSlider(Qt.Horizontal)
        self.slider_threshDev.setRange(0, 200)
        self.slider_threshDev.setValue(100)
        row.addWidget(self.slider_threshDev)
        self.slider_threshDev.setToolTip(
            "Scales the auto-threshold up/down (100% = the method's own value)")
        self.slider_threshDev.valueChanged.connect(self.update_preview)

        # Replaces the old plain "Blur" kernel combo - the same Denoise box
        # (method + tunable parameter + "Test Methods...") the main UI
        # offers elsewhere, applied here before thresholding to reduce mask
        # noise, instead of being limited to a single Gaussian-blur kernel
        # size. Seeded from the calling tab's own current Denoise state
        # (denoise_state), so this starts out looking the same as what's
        # already on screen there - see class docstring.
        self.box_denoise = DenoiseBox(title='Denoise', show_apply_all=False)
        self.box_denoise.set_state(denoise_state)
        self.box_denoise.settingsChanged.connect(self.update_preview)
        self.box_denoise.checkMethodsRequested.connect(self._show_denoise_check_methods)
        layout.addWidget(self.box_denoise)

        buttons = qtw.QHBoxLayout()
        layout.addLayout(buttons)
        self.button_compute = qtw.QPushButton('Compute Summed DP from Threshold')
        self.button_compute.clicked.connect(self.conditional_accept)
        buttons.addWidget(self.button_compute)
        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.button_cancel)

        self.update_preview()

    def _bg_8bit(self):
        """This dialog's own 8-bit baseline for thresholding/denoising -
        the single place both update_preview() and
        _show_denoise_check_methods() derive it from, so they can never
        drift out of sync with each other."""
        return io.convert_img_to_8bit(self.img)

    def _show_denoise_check_methods(self):
        self._check_methods_dlg = self.box_denoise.open_check_methods_dialog(
            self._bg_8bit(), parent=self)

    def update_preview(self):
        """Recompute the mask from the current threshold/deviation/denoise
        settings, replay every manual Include/Exclude ROI edit on top (see
        class docstring), and refresh the overlay on the displayed image."""
        method = self.combo_threshMethod.currentText()
        threshold_funcs = {'otsu': threshold_otsu, 'li': threshold_li, 'yen': threshold_yen}
        dev = self.slider_threshDev.value()
        self.label_threshDev.setText(f'Deviation: {dev}%')
        img_denoised = self.box_denoise.apply(self._bg_8bit())
        thresh = io.threshold_ignore_zero(threshold_funcs[method], img_denoised) * (dev / 100)
        self.mask = img_denoised >= thresh
        for row0, row1, col0, col1, value in self._manual_edits:
            self.mask[row0:row1, col0:col1] = value

        color = np.array([*mcolors.to_rgb('tab:orange'), 0.45])
        mask_image = self.mask.reshape(*self.mask.shape, 1) * color.reshape(1, 1, -1)
        self.img_display_overlay.set_data(mask_image)
        self.canvas.draw_idle()

    def _clear_manual_edits(self):
        self._manual_edits = []
        self.update_preview()

    #%% Include ROI / Exclude ROI - drag-to-paint rectangles (mirrors
    # MaskEditDialog's own Shift+drag rect-paint gesture, just always-on
    # via the ribbon tool instead of a modifier key, since there's no other
    # click/drag gesture on this canvas to arbitrate against).
    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        tool = self.ribbon.active_tool
        if tool not in ('include_roi', 'exclude_roi'):
            return
        value = tool == 'include_roi'
        self._roi_drag = (event.xdata, event.ydata, value)
        color = 'lime' if value else 'red'
        self._roi_rect_artist = patches.Rectangle(
            (event.xdata, event.ydata), 0, 0, linewidth=1.5, edgecolor=color, facecolor='none')
        self.ax.add_patch(self._roi_rect_artist)
        self.canvas.draw()
        self._roi_bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def _on_motion(self, event):
        if self._roi_drag is None or event.inaxes != self.ax \
                or event.xdata is None or event.ydata is None:
            return
        x0, y0, _ = self._roi_drag
        try:
            self._roi_rect_artist.set_width(event.xdata - x0)
            self._roi_rect_artist.set_height(event.ydata - y0)
            self._roi_rect_artist.set_xy((x0, y0))
        except AttributeError:
            return
        self.canvas.restore_region(self._roi_bg)
        self.ax.draw_artist(self._roi_rect_artist)
        self.canvas.blit(self.ax.bbox)

    def _on_release(self, event):
        if self._roi_drag is None:
            return
        x0, y0, value = self._roi_drag
        self._roi_drag = None
        if self._roi_rect_artist is not None:
            try:
                self._roi_rect_artist.remove()
            except Exception:
                pass
            self._roi_rect_artist = None
        if event.xdata is not None and event.ydata is not None:
            col0, col1 = sorted((int(round(x0)), int(round(event.xdata))))
            row0, row1 = sorted((int(round(y0)), int(round(event.ydata))))
            # A real drag only - press and release on the same pixel paints
            # nothing, same convention as MaskEditDialog's identical check.
            if row1 > row0 and col1 > col0:
                h, w = self.img.shape[:2]
                row0, row1 = max(row0, 0), min(row1 + 1, h)
                col0, col1 = max(col0, 0), min(col1 + 1, w)
                if row1 > row0 and col1 > col0:
                    self._manual_edits.append((row0, row1, col0, col1, value))
        self.update_preview()

    def conditional_accept(self):
        """Accept the dialog unless the current mask is empty, in which case
        warn instead of closing."""
        if self.mask is None or not self.mask.any():
            qtw.QMessageBox.warning(self, 'Empty Mask',
                'The current threshold selects no scan positions - adjust the '
                'method/deviation before computing.')
            return
        self.accept()
