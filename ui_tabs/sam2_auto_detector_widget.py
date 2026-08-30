# -*- coding: utf-8 -*-
"""SAM2-backed "Auto Detector" popup for the SAM2 tab (ui_tabs/tab_sam2.py):
runs SAM2's automatic mask generator on one frame to find candidate objects,
lets the user tune its parameters and pick which candidates to keep, then
emits them (as a center-of-mass point, optionally plus a couple of
background points) back to the tab, ready to seed tracking exactly like a
manually Ctrl+clicked object.

Mirrors ROI Tracker's classical-CV Object_Detector_Widget
(ui_tabs/object_detection_widget.py) as a popup-widget shell, but is SAM2-
backed instead of threshold/contour-backed, and runs via a QProcess
subprocess (worker_sam.py's 'auto' mode) like every other SAM2 op in this
app, rather than in-process - keeps the torch/sam2 import isolated the same
way worker_sam.py already does (not bundled in a frozen build, see
INSTALL.md), and automatic mask generation is materially more expensive than
Object_Detector_Widget's near-instant CV pipeline, so parameter changes
don't auto-rerun - "Run Detection" launches one explicit subprocess call.
"""
import os
import json
import pickle
import datetime
import numpy as np
import cv2
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QProcess, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .worker_launch import worker_command
from .loading_label import LoadingSpinner

# (key, label, is_int, default, minimum, maximum, step, tooltip) - a
# practical subset of SAM2AutomaticMaskGenerator's ~15 constructor kwargs,
# not the full surface (points_per_batch, stability_score_offset,
# crop_nms_thresh, crop_overlap_ratio, mask_threshold, use_m2m,
# multimask_output stay at the library's own defaults).
_PARAM_SPECS = [
    ('points_per_side', 'Points per Side', True, 32, 4, 128, 1,
     'Side length of the square grid of points sampled over the image - '
     'higher finds smaller/more objects but is much slower (roughly N^2 point prompts).'),
    ('pred_iou_thresh', 'Predicted IoU Thresh', False, 0.8, 0.0, 1.0, 0.01,
     "Discard a candidate mask if SAM2's own predicted mask-quality score is below this."),
    ('stability_score_thresh', 'Stability Score Thresh', False, 0.95, 0.0, 1.0, 0.01,
     'Discard a candidate mask that is unstable under small changes to the '
     'logit-to-mask cutoff - a rough proxy for how confidently-defined its edges are.'),
    ('min_mask_region_area', 'Min Mask Area (px)', True, 0, 0, 1_000_000, 10,
     'Remove disconnected regions/holes smaller than this from each candidate mask (0 = off).'),
    ('box_nms_thresh', 'Box NMS Thresh', False, 0.7, 0.0, 1.0, 0.01,
     'Overlapping-box IoU cutoff used to drop duplicate candidates covering the same object.'),
]


class SAM2AutoDetectorWidget(qtw.QWidget):
    """Run SAM2's automatic mask generator on one frame, let the user pick
    which of the found objects to keep, and emit them via `final_objects`.

    final_objects: list of dicts, one per accepted object -
    {'points': [[x, y], ...], 'labels': [1, 0, ...], 'frame_idx': int} - or
    None if the widget was cancelled. `frame_idx` is the frame the widget
    was launched on (self.imgNo), same for every accepted object."""
    final_objects = pyqtSignal(object)

    def __init__(self, img, imgNo, path_save, ensure_sam2_ready, logger, parent=None):
        super().__init__(parent)
        self.img = img
        self.imgNo = imgNo
        self.path_save = path_save
        self._ensure_sam2_ready = ensure_sam2_ready
        self.logger = logger
        self.candidates = []          # list of {'segmentation','area','bbox','predicted_iou','stability_score'}
        self._candidate_artists = []  # one AxesImage overlay per candidate, index-aligned with self.candidates
        self._selection_label_artist = None  # the "#N" label for whichever row is currently selected
        self._selection_edge_artists = []    # red outline of the currently-selected candidate's mask
        self._process = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle('SAM2 Auto Detector')
        self.resize(900, 650)
        layout_main = qtw.QVBoxLayout(self)

        layout_top = qtw.QHBoxLayout()
        layout_main.addLayout(layout_top)

        #%% parameters
        group_params = qtw.QGroupBox('Detection Parameters')
        layout_params = qtw.QGridLayout(group_params)
        layout_top.addWidget(group_params)
        self._param_widgets = {}
        for i, (key, label, is_int, default, minimum, maximum, step, tooltip) in enumerate(_PARAM_SPECS):
            lab = qtw.QLabel(label)
            lab.setToolTip(tooltip)
            layout_params.addWidget(lab, i, 0)
            if is_int:
                wid = qtw.QSpinBox()
                wid.setRange(int(minimum), int(maximum))
                wid.setSingleStep(int(step))
                wid.setValue(int(default))
            else:
                wid = qtw.QDoubleSpinBox()
                wid.setDecimals(2)
                wid.setRange(minimum, maximum)
                wid.setSingleStep(step)
                wid.setValue(default)
            wid.setToolTip(tooltip)
            layout_params.addWidget(wid, i, 1)
            self._param_widgets[key] = wid

        self.checkbox_cropLayers = qtw.QCheckBox('Detect Small Objects (slower)')
        self.checkbox_cropLayers.setToolTip(
            'Also run detection on overlapping image crops, to catch smaller '
            'objects the full-image grid misses - roughly doubles run time.')
        layout_params.addWidget(self.checkbox_cropLayers, len(_PARAM_SPECS), 0, 1, 2)

        #%% run/accept controls
        layout_end = qtw.QVBoxLayout()
        layout_top.addLayout(layout_end)
        self.button_run = qtw.QPushButton('Run Detection')
        self.button_run.setToolTip('Run SAM2\'s automatic mask generator on this frame '
                                   'with the parameters above')
        self.button_run.clicked.connect(self.run_detection)
        layout_end.addWidget(self.button_run)

        self.checkbox_addBackground = qtw.QCheckBox('Add Background Points')
        self.checkbox_addBackground.setChecked(True)
        self.checkbox_addBackground.setToolTip(
            'For each kept object, also add a couple of negative points just '
            'outside its mask - helps SAM2 resist bleeding into the '
            'background/neighboring objects during video tracking.')
        layout_end.addWidget(self.checkbox_addBackground)

        layout_selectButtons = qtw.QHBoxLayout()
        layout_end.addLayout(layout_selectButtons)
        self.button_selectAll = qtw.QPushButton('Select All')
        self.button_selectAll.clicked.connect(lambda: self._set_all_checked(True))
        layout_selectButtons.addWidget(self.button_selectAll)
        self.button_selectNone = qtw.QPushButton('Select None')
        self.button_selectNone.clicked.connect(lambda: self._set_all_checked(False))
        layout_selectButtons.addWidget(self.button_selectNone)

        self.button_delete = qtw.QPushButton('Delete Selected')
        self.button_delete.setToolTip(
            "Permanently remove the candidate currently selected in the list "
            "below (not just uncheck it) - for false detections you don't "
            "want cluttering the list at all")
        self.button_delete.clicked.connect(self.delete_selected)
        layout_end.addWidget(self.button_delete)

        layout_saveLoad = qtw.QHBoxLayout()
        layout_end.addLayout(layout_saveLoad)
        self.button_save = qtw.QPushButton('Save Detection...')
        self.button_save.setToolTip('Save the candidate list (masks + metadata + a preview '
                                    'image) to review/pick from again later')
        self.button_save.clicked.connect(self.save_detection)
        layout_saveLoad.addWidget(self.button_save)
        self.button_load = qtw.QPushButton('Load Detection...')
        self.button_load.setToolTip('Reload a previously saved candidate list, without '
                                    're-running SAM2')
        self.button_load.clicked.connect(self.load_detection)
        layout_saveLoad.addWidget(self.button_load)

        layout_end.addStretch(1)
        layout_finish = qtw.QHBoxLayout()
        layout_end.addLayout(layout_finish)
        self.button_accept = qtw.QPushButton('Accept')
        self.button_accept.setToolTip('Add every checked candidate to the object list')
        self.button_accept.clicked.connect(lambda: self._finish(True))
        layout_finish.addWidget(self.button_accept)
        self.button_cancel = qtw.QPushButton('Cancel')
        self.button_cancel.clicked.connect(lambda: self._finish(False))
        layout_finish.addWidget(self.button_cancel)

        #%% candidate list
        self.list_candidates = qtw.QListWidget()
        self.list_candidates.setMinimumWidth(220)
        self.list_candidates.setMaximumWidth(260)
        self.list_candidates.itemChanged.connect(self._on_item_checked)
        self.list_candidates.currentItemChanged.connect(self._on_selection_changed)
        layout_top.addWidget(self.list_candidates)

        #%% canvas
        # stretch=1 (vs. layout_top's implicit 0 above) so enlarging the
        # widget grows the canvas - the parameter/button/candidate-list row
        # stays at its own natural size instead of stretching too.
        layout_canvas = qtw.QVBoxLayout()
        layout_main.addLayout(layout_canvas, 1)
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout_canvas.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self.img_display = self.ax.imshow(self.img, cmap='gray')
        self.ax.set_title(f'Frame {self.imgNo} - no candidates yet, click "Run Detection"')

        self.spinner = LoadingSpinner(parent=self)
        self.spinner.setAttribute(Qt.WA_TransparentForMouseEvents)

    #%% run detection
    def run_detection(self):
        self.button_run.setDisabled(True)

        def _on_ready():
            self._launch_detection()

        def _on_failed(error_msg):
            self.button_run.setEnabled(True)
            qtw.QMessageBox.warning(self, 'SAM2 Checkpoint Download Failed',
                'Could not download the SAM2 model checkpoint - check your '
                'internet connection and see the log console for details.')

        self._ensure_sam2_ready(_on_ready, _on_failed)

    def _launch_detection(self):
        """Build seg_input.pkl and launch worker_sam.py's 'auto' mode -
        split out from run_detection() so it only runs after
        _ensure_sam2_ready() confirms the checkpoint is present."""
        x = (self.width() - self.spinner.width()) // 2
        y = (self.height() - self.spinner.height()) // 2
        self.spinner.move(x, y)
        self.spinner.raise_()
        self.spinner.start()

        path_seg = os.path.join(self.path_save, 'JPG Images', 'auto_detect')
        os.makedirs(path_seg, exist_ok=True)
        params = {key: wid.value() for key, wid in self._param_widgets.items()}
        params['crop_n_layers'] = 1 if self.checkbox_cropLayers.isChecked() else 0
        seg_input = {'image': self.img, 'params': params}
        with open(os.path.join(path_seg, 'seg_input.pkl'), 'wb') as f:
            pickle.dump(seg_input, f)

        self.logger.info('Starting SAM2 Auto Detector on frame %d...', self.imgNo)
        program, arguments = worker_command('sam', ['auto', path_seg, str(self.imgNo)])
        self._process = QProcess(self)
        self._process.setProgram(program)
        self._process.setArguments(arguments)
        self._process.finished.connect(self._on_detection_finished)
        self._process.errorOccurred.connect(self._on_detection_failed)
        self._process.start()

    def _on_detection_failed(self, error):
        self.spinner.stop()
        self.button_run.setEnabled(True)
        self.logger.error('SAM2 Auto Detector QProcess error occurred: %s', error)
        qtw.QMessageBox.critical(self, 'Process Error',
            'The SAM2 auto-detection process failed to start.\n'
            'Check that Python is on PATH and worker_sam.py exists.')

    def _on_detection_finished(self, exit_code, exit_status):
        self.spinner.stop()
        self.button_run.setEnabled(True)
        text = bytes(self._process.readAllStandardOutput()).decode('utf-8')
        try:
            result = json.loads(text.strip())
            if result.get('error') == 'missing_dependency':
                self.logger.error('SAM2 worker reported a missing dependency: %s', result['message'])
                qtw.QMessageBox.warning(self, 'SAM2 Dependencies Not Installed', result['message'])
                return
            with np.load(result['path']) as f:
                masks = f['masks']
            with open(result['meta_path']) as f:
                meta = json.load(f)
            candidates = [{'segmentation': masks[i], **meta[i]} for i in range(len(meta))]
            self._set_candidates(candidates)
        except json.JSONDecodeError:
            self.logger.error('Could not decode SAM2 Auto Detector result: %s', text)
            qtw.QMessageBox.warning(self, 'SAM2 Error',
                f'Could not decode SAM2 output. Check console for details.\n'
                f'Raw output (first 200 chars): {text[:200]}')

    #%% candidate list / overlay
    def _set_candidates(self, candidates):
        for artist in self._candidate_artists:
            artist.remove()
        self._candidate_artists.clear()
        if self._selection_label_artist is not None:
            self._selection_label_artist.remove()
            self._selection_label_artist = None
        for artist in self._selection_edge_artists:
            artist.remove()
        self._selection_edge_artists = []
        self.list_candidates.clear()
        self.candidates = candidates

        cmap = plt.get_cmap('tab20')
        for i, cand in enumerate(candidates):
            mask = cand['segmentation']
            color = cmap(i % 20)[:3]
            rgba = np.zeros((*mask.shape, 4))
            rgba[mask] = (*color, 0.5)
            artist = self.ax.imshow(rgba)
            self._candidate_artists.append(artist)

            item = qtw.QListWidgetItem(
                f"#{i+1}  area={cand['area']}  iou={cand['predicted_iou']:.2f}  "
                f"stab={cand['stability_score']:.2f}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_candidates.addItem(item)

        self.ax.set_title(f'Frame {self.imgNo} - {len(candidates)} candidate(s) found')
        self.canvas.draw_idle()
        self.logger.info('SAM2 Auto Detector found %d candidate object(s) on frame %d.',
                          len(candidates), self.imgNo)

    def _on_item_checked(self, item):
        i = self.list_candidates.row(item)
        self._candidate_artists[i].set_visible(item.checkState() == Qt.Checked)
        self.canvas.draw_idle()

    def _set_all_checked(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.list_candidates.count()):
            self.list_candidates.item(i).setCheckState(state)

    def _on_selection_changed(self, current, previous):
        """Highlight whichever candidate is currently selected in the list -
        its number labeled at its centroid, and its mask outlined in red -
        so the user can tell which list row is which object on the canvas
        (and vice versa)."""
        if self._selection_label_artist is not None:
            self._selection_label_artist.remove()
            self._selection_label_artist = None
        for artist in self._selection_edge_artists:
            artist.remove()
        self._selection_edge_artists = []
        if current is not None:
            i = self.list_candidates.row(current)
            if 0 <= i < len(self.candidates):
                mask = self.candidates[i]['segmentation']
                ys, xs = np.where(mask)
                if len(ys):
                    self._selection_label_artist = self.ax.annotate(
                        f'#{i+1}', xy=(xs.mean(), ys.mean()), color='white',
                        fontsize=13, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                                 alpha=0.75, edgecolor='white'))
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for cnt in contours:
                        pts = cnt.reshape(-1, 2)
                        line, = self.ax.plot(pts[:, 0], pts[:, 1], color='red', linewidth=2)
                        self._selection_edge_artists.append(line)
        self.canvas.draw_idle()

    def delete_selected(self):
        """Permanently drop the currently-selected candidate (not just
        uncheck it) - for false detections that shouldn't even be an option
        at Accept time."""
        item = self.list_candidates.currentItem()
        if item is None:
            return
        i = self.list_candidates.row(item)
        if self._selection_label_artist is not None:
            self._selection_label_artist.remove()
            self._selection_label_artist = None
        for artist in self._selection_edge_artists:
            artist.remove()
        self._selection_edge_artists = []
        self._candidate_artists[i].remove()
        del self._candidate_artists[i]
        del self.candidates[i]
        self.list_candidates.takeItem(i)
        self._renumber_candidates()
        self.canvas.draw_idle()

    def _renumber_candidates(self):
        """Refresh every remaining list row's "#N ..." text to match its
        current (post-delete) position - the number shown always matches
        what _on_selection_changed's on-canvas label would show for it."""
        for i, cand in enumerate(self.candidates):
            self.list_candidates.item(i).setText(
                f"#{i+1}  area={cand['area']}  iou={cand['predicted_iou']:.2f}  "
                f"stab={cand['stability_score']:.2f}")

    #%% accept / cancel
    def _object_from_mask(self, mask):
        """One accepted object's seed point(s): the mask's center of mass
        as a positive point, plus (if "Add Background Points" is checked) a
        couple of negative points sampled on a dilated ring just outside the
        mask - explicit "not this" context that helps SAM2 resist bleeding
        into neighboring/background regions during video propagation."""
        ys, xs = np.where(mask)
        points = [[float(xs.mean()), float(ys.mean())]]
        labels = [1]
        if self.checkbox_addBackground.isChecked():
            kernel = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            ring_y, ring_x = np.where(dilated & ~mask)
            if len(ring_y):
                n_bg = min(2, len(ring_y))
                choice = np.random.choice(len(ring_y), size=n_bg, replace=False)
                for k in choice:
                    points.append([float(ring_x[k]), float(ring_y[k])])
                    labels.append(0)
        return {'points': points, 'labels': labels, 'frame_idx': self.imgNo}

    def _finish(self, accepted):
        if accepted:
            objects = []
            for i, cand in enumerate(self.candidates):
                if self.list_candidates.item(i).checkState() != Qt.Checked:
                    continue
                objects.append(self._object_from_mask(cand['segmentation']))
            if not objects:
                qtw.QMessageBox.warning(self, 'Nothing Selected',
                    'Check at least one candidate (or Cancel) before accepting.')
                return
            self.final_objects.emit(objects)
        else:
            self.final_objects.emit(None)
        self.close()

    #%% save / load
    def save_detection(self):
        try:
            self._save_detection_impl()
        except Exception as exc:
            self.logger.exception('Failed to save Auto Detector results.')
            qtw.QMessageBox.critical(self, 'Save Failed',
                f'Failed to save detection results:\n{exc}')
            return
        qtw.QMessageBox.information(self, 'Saved', 'Detection results saved successfully.')

    def _save_detection_impl(self):
        if not self.candidates:
            qtw.QMessageBox.warning(self, 'Nothing to Save', 'Run detection first.')
            return
        date = datetime.date.today()
        tim = datetime.datetime.now().strftime('%H-%M-%S')
        path = os.path.join(self.path_save, 'Auto Detection', f'{date}__{tim}')
        os.makedirs(path, exist_ok=True)

        masks = np.stack([c['segmentation'] for c in self.candidates])
        np.savez_compressed(os.path.join(path, 'masks.npz'), masks=masks)

        meta = []
        for i, cand in enumerate(self.candidates):
            meta.append({
                'area': int(cand['area']), 'bbox': [float(v) for v in cand['bbox']],
                'predicted_iou': float(cand['predicted_iou']),
                'stability_score': float(cand['stability_score']),
                'kept': self.list_candidates.item(i).checkState() == Qt.Checked})
        with open(os.path.join(path, 'candidates.json'), 'w') as f:
            json.dump({'imgNo': self.imgNo, 'candidates': meta}, f, indent=4)

        self.figure.savefig(os.path.join(path, 'preview.png'), dpi=150)
        self.logger.info('Auto Detector results saved to %s.', path)

    def load_detection(self):
        path = qtw.QFileDialog.getExistingDirectory(
            self, 'Load Auto Detection', os.path.join(self.path_save, 'Auto Detection'))
        if not path:
            return
        try:
            with np.load(os.path.join(path, 'masks.npz')) as f:
                masks = f['masks']
            with open(os.path.join(path, 'candidates.json')) as f:
                saved = json.load(f)
            # The saved detection may be for a different frame than the one
            # this widget was opened on - only the candidate masks/metadata
            # are restored, the background image stays whatever frame the
            # widget was launched with.
            self.imgNo = saved['imgNo']
            candidates = [
                {'segmentation': masks[i], 'area': m['area'], 'bbox': m['bbox'],
                 'predicted_iou': m['predicted_iou'], 'stability_score': m['stability_score']}
                for i, m in enumerate(saved['candidates'])]
            self._set_candidates(candidates)
            for i, m in enumerate(saved['candidates']):
                self.list_candidates.item(i).setCheckState(Qt.Checked if m['kept'] else Qt.Unchecked)
            self.logger.info('Loaded a saved Auto Detector session from %s.', path)
        except Exception as exc:
            self.logger.exception('Failed to load Auto Detector results.')
            qtw.QMessageBox.critical(self, 'Load Failed',
                f'Failed to load detection results:\n{exc}')
