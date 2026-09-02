# -*- coding: utf-8 -*-
"""Blob Selection's segmentation-method picker - opens from Tab_Tracking_CV2
when a ROI's "Blob" object-list checkbox is freshly checked (or later, via
the "Blob Settings..." ribbon button, to revisit it), so touching/partially-
overlapping particles that plain connected-components analysis would merge
into one blob can still be split apart and picked individually before Blob
Selection ever reaches the main "ROI with Threshold" canvas (see
EDyssey/io_utils/blob_segmentation.py for the actual segmentation methods).

Self-contained: takes the ROI's current raw threshold mask (+ matching
intensity crop, for methods that need it) and an initial method/params/
seed in, returns the (possibly unchanged) method/params/seed back out via
result() on Accept - Tab_Tracking_CV2 itself owns turning that into a
df_rois['blob'] update (see _open_blob_segmentation_dialog), so this dialog
doesn't need to know anything about ROIs, frames, or tracking.
"""
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D
import PyQt5.QtWidgets as qtw

import EDyssey.io_utils as io


class BlobSegmentationDialog(qtw.QDialog):
    """See module docstring. Construct with the ROI's raw threshold mask
    (`mask`) and matching raw-intensity crop (`img_cut`, may be None if
    unavailable - only the intensity-based method needs it), the method/
    params/seed_centroid it should start pre-filled with, then call
    exec_(); on qtw.QDialog.Accepted, result() returns the
    (method, params, seed_centroid) the user left it on."""

    def __init__(self, mask, img_cut, method=None, params=None, seed_centroid=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Blob Selection - Segmentation')
        self.resize(560, 600)
        self._mask = mask
        self._img_cut = img_cut
        self._method = method if method in io.BLOB_SEGMENTATION_METHODS else io.DEFAULT_BLOB_METHOD
        self._params = dict(params) if params else io.default_blob_params(self._method)
        self._seed_centroid = tuple(seed_centroid) if seed_centroid is not None else None
        # Whether the user actually clicked a blob in THIS dialog session -
        # distinct from _seed_centroid itself already being non-None from
        # a previous session's own click, passed in unchanged - so the
        # caller (Tab_Tracking_CV2._open_blob_segmentation_dialog) only
        # re-seeds Blob Selection (splitting a segment boundary at the
        # previewed frame) when the user actually picked something new
        # here, not merely because they reopened the dialog to tweak the
        # method on a ROI that already had a seed from before.
        self._seed_changed = False
        self._labels = None  # recomputed by _redraw()
        self._param_widgets = {}  # key -> widget
        self._overlay_artists = []
        self._build_ui()
        self._redraw()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = qtw.QVBoxLayout(self)

        intro = qtw.QLabel(
            "Pick how this ROI's threshold mask is split into individual "
            'blobs before one is selected for tracking/extraction. Click a '
            'blob below to choose it (same as clicking directly on the "ROI '
            'with Threshold" panel) - the chosen one is highlighted in green; '
            "with none chosen, the largest blob is used, same as before this "
            'dialog existed.')
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = qtw.QFormLayout()
        self.combo_method = qtw.QComboBox()
        for method_id, spec in io.BLOB_SEGMENTATION_METHODS.items():
            if spec['needs_intensity'] and self._img_cut is None:
                continue  # nothing to key intensity-based splitting off of here
            self.combo_method.addItem(spec['label'], method_id)
        found = self.combo_method.findData(self._method)
        self.combo_method.setCurrentIndex(found if found >= 0 else 0)
        self._method = self.combo_method.currentData()
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        form.addRow('Method:', self.combo_method)
        layout.addLayout(form)

        self.label_description = qtw.QLabel()
        self.label_description.setWordWrap(True)
        self.label_description.setStyleSheet('color: #888;')
        layout.addWidget(self.label_description)

        self.form_params = qtw.QFormLayout()
        layout.addLayout(self.form_params)

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 340)
        layout.addWidget(self.canvas, 1)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        bg = self._img_cut if self._img_cut is not None else self._mask.astype(float)
        self.ax.imshow(bg, cmap='gray')
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

        self.label_status = qtw.QLabel()
        self.label_status.setWordWrap(True)
        layout.addWidget(self.label_status)

        buttons = qtw.QDialogButtonBox(qtw.QDialogButtonBox.Ok | qtw.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._rebuild_param_form()

    def _rebuild_param_form(self):
        """(Re)build the parameter spinboxes for the currently-selected
        method - cleared and rebuilt each time the method changes, since
        each has its own parameter set (see BLOB_SEGMENTATION_METHODS)."""
        while self.form_params.rowCount():
            self.form_params.removeRow(0)
        self._param_widgets = {}
        spec = io.BLOB_SEGMENTATION_METHODS[self._method]
        self.label_description.setText(spec['description'])
        for key, label, low, high, step, decimals, tooltip in spec['param_specs']:
            if decimals > 0:
                widget = qtw.QDoubleSpinBox()
                widget.setDecimals(decimals)
            else:
                widget = qtw.QSpinBox()
            widget.setRange(low, high)
            widget.setSingleStep(step)
            widget.setValue(self._params.get(key, spec['default_params'][key]))
            widget.setToolTip(tooltip)
            widget.valueChanged.connect(self._on_param_changed)
            self.form_params.addRow(label + ':', widget)
            self._param_widgets[key] = widget

    # -------------------------------------------------------------- events
    def _on_method_changed(self):
        self._method = self.combo_method.currentData()
        self._params = io.default_blob_params(self._method)
        self._rebuild_param_form()
        self._redraw()

    def _on_param_changed(self):
        for key, widget in self._param_widgets.items():
            self._params[key] = widget.value()
        self._redraw()

    def _on_canvas_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or self._labels is None:
            return
        click = (event.xdata, event.ydata)
        _, chosen_centroid = io.select_blob_by_centroid(self._mask, click, labels=self._labels)
        if chosen_centroid is None:
            return
        self._seed_centroid = chosen_centroid
        self._seed_changed = True
        self._redraw()

    # -------------------------------------------------------------- redraw
    def _redraw(self):
        """Recompute this method/params' own blob labels and redraw the
        contour/number overlay - the currently-chosen blob (see
        _seed_centroid, resolved the same way select_blob_by_centroid
        itself resolves it) is highlighted in green, everything else in
        cyan - same convention as Tab_Tracking_CV2._draw_blob_overlay's own
        main-canvas preview."""
        for artist in self._overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._overlay_artists = []

        labels = io.label_blobs(self._mask, self._img_cut, self._method, self._params)
        if labels is None:
            mask_u8 = self._mask.astype('uint8')
            _, labels = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)[:2]
        self._labels = labels

        restricted, _ = io.select_blob_by_centroid(self._mask, self._seed_centroid, labels=labels)

        ids = [int(lid) for lid in np.unique(labels) if lid != 0]
        for lid in ids:
            blob_mask = labels == lid
            ys, xs = np.where(blob_mask)
            cx, cy = float(xs.mean()), float(ys.mean())
            is_chosen = bool(restricted is not None and restricted[ys[0], xs[0]])
            color = 'lime' if is_chosen else 'cyan'
            blob_u8 = blob_mask.astype('uint8')
            contours, _ = cv2.findContours(blob_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                pts = contour.reshape(-1, 2)  # (col, row) = (x, y)
                if len(pts) < 2:
                    continue
                line = Line2D(pts[:, 0], pts[:, 1], color=color, linewidth=1.6 if is_chosen else 1.0)
                self.ax.add_line(line)
                self._overlay_artists.append(line)
            text = self.ax.text(cx, cy, str(lid), color=color, fontsize=9, fontweight='bold',
                                horizontalalignment='center', verticalalignment='center')
            self._overlay_artists.append(text)

        n = len(ids)
        if n == 0:
            self.label_status.setText("No blob detected in this ROI's current threshold mask.")
        elif n == 1:
            self.label_status.setText('1 blob detected - nothing to split apart here.')
        else:
            self.label_status.setText(
                f'{n} blobs detected - click one above to choose it (green); '
                'with none chosen, the largest is used.')
        self.canvas.draw_idle()

    # ----------------------------------------------------------------- API
    def result(self):
        """(method, params, seed_centroid, seed_changed) as currently left
        in the dialog - read after exec_() returns qtw.QDialog.Accepted.
        seed_changed is True only if the user actually clicked a blob in
        THIS dialog session (see _seed_changed) - tells the caller whether
        seed_centroid is worth re-seeding Blob Selection over, versus just
        being unchanged from whatever was passed in originally."""
        return self._method, dict(self._params), self._seed_centroid, self._seed_changed
