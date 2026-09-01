# -*- coding: utf-8 -*-
"""Frame-by-frame manual fine-tuning of a tracked object's per-frame mask
stack: grow/shrink the mask directionally (D-pad buttons), paint single
pixels or rectangular regions in/out with the mouse, and preview the same
Edge Detection post-processing the main tab uses - live, without baking it
into the stored mask. Shared by Tab_SAM2 and Tab_Tracking_CV2 (see their own
open_fine_tune_mask_dialog())."""
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import EDyssey.io_utils as io

_MASK_COLOR = np.array([1.0, 0.55, 0.0, 0.45])  # translucent orange overlay
_DIRECTIONS = [('Top', 270), ('Bottom', 90), ('Left', 180), ('Right', 0)]
# Full reference for the "?" help button below the canvas (see
# _show_help_dialog) - promoted from the old always-visible label_tip
# (easy to miss/scroll out of view) into a proper, discoverable help
# affordance, mirroring the main tabs' own ribbon "?" -> show_help_dialog
# pattern (see ui_tabs/base_tab.py's show_shortcuts_dialog). Kept accurate
# to exactly what this dialog supports - update this alongside any future
# interaction change.
_HELP_TEXT = (
    'Canvas:\n'
    '  Hold "Ctrl" + Scroll wheel  ->  Zoom in/out, centered on the cursor\n'
    '  Hold "Ctrl" + Left-Click (or drag)  ->  Paint pixels IN (add to mask)\n'
    '  Hold "Ctrl" + Right-Click (or drag)  ->  Paint pixels OUT (remove from mask)\n'
    '  Hold "Shift" + Left-drag  ->  Paint a rectangular region IN\n'
    '  Hold "Shift" + Right-drag  ->  Paint a rectangular region OUT\n'
    '  Left-Click a mesh cell (while Mesh is enabled, no modifier)  ->  Toggle that cell\n'
    '\n'
    'Grow / Shrink Mask:\n'
    '  Each D-pad arrow adds/removes one row or column of pixels on that '
    'side of the mask (arrow pointing away from center = grow, toward '
    'center = shrink).\n'
    '\n'
    'Threshold (only shown when this mask is threshold-derived):\n'
    '  Method/ROI Blur/Deviation rebuild the mask from the ROI live, for '
    'just the frame on screen - "Apply to All Frames" propagates that to '
    'every frame at once.\n'
    '\n'
    'Edge Detection:\n'
    '  Live preview only - never changes the returned mask. Reduces the '
    'mask to its outline (optionally one-sided via "Directional" + Angle). '
    '"This Frame Only"/"All Frames" controls whether it previews on just '
    'the frame named in the spinbox, or every frame.\n'
    '\n'
    'Mesh:\n'
    '  Divides the mask into a rotated grid; left-click cell(s) to '
    'restrict the mask to just those cells (live preview only). '
    '"Center on Initial Mask" anchors the grid to where the object '
    'started (its first tracked/segmented position) instead of wherever '
    "it's been edited to since - useful once the mask has moved a lot. "
    '"This Frame Only"/"All Frames" controls which frame(s) the '
    'restriction actually applies to.\n'
    '\n'
    'Reset This Frame / Reset to Tracking:\n'
    '  Discard edits to just the frame on screen, or every frame, back to '
    'the original tracked/segmented mask.')
_MESH_GRID_COLOR = 'cyan'
_MESH_SELECTED_COLOR = np.array([0.0, 0.6, 1.0, 0.35])  # translucent blue


class MaskEditDialog(qtw.QDialog):
    """Modal editor for a single tracked object's (N, H, W) boolean mask
    stack. `exec_()` returns QDialog.Accepted once the user confirms;
    `get_mask_stack()` then returns the edited (N, H, W) array to write back
    into the caller's own dataframe - editing happens on a private copy, so
    Cancel leaves the original stack untouched. Edge Detection here is a
    live, non-destructive preview only (exactly like the main tab) - it is
    never baked into the returned mask, so toggling it never compounds
    across redraws or fine-tuning sessions.

    A Threshold box (method/ROI blur/deviation, mirroring the main tab's own
    controls) is built too, but only when the caller passes
    `recompute_thresh_fn` - i.e. only for threshold-derived masks
    (Tab_Tracking_CV2; SAM2's masks come from the segmentation network, so
    it never applies there). Unlike Edge Detection, changing it really does
    overwrite mask_stack - live, for just the current frame, as each control
    changes; "Apply to All Frames" is the one action that still needs an
    explicit button, since it discards every other frame's edits at once.
    `get_edge_settings()`/`get_thresh_settings()` let the caller read back
    whatever was left set in either box on Save && Close, to sync its own
    main-tab controls to match.
    """

    def __init__(self, parent, mask_stack, bg_stack=None, start_frame=0, logger=None,
                 default_mask_stack=None, edge_settings=None,
                 thresh_settings=None, recompute_thresh_fn=None, mesh_settings=None):
        super().__init__(parent)
        self.setWindowTitle('Fine-Tune Mask')
        # Maximize button too (off by default on a QDialog) - the image
        # needs real screen space to make fine edits legible, and a fixed
        # initial size can't anticipate every navigation-image resolution.
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint
                            | Qt.WindowMinimizeButtonHint)
        screen = qtw.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            h = int(avail.height() * 0.85)
            # Single-column layout (canvas stacked over compact control
            # rows) never needs anywhere near full screen width - tie it to
            # the height instead so this reads as a tall, roughly square
            # window rather than a wide, mostly-empty one.
            w = min(int(avail.width() * 0.55), max(h, 700))
            self.resize(w, h)
        else:
            self.resize(800, 900)
        self.logger = logger
        # (thresh_method, thresh_offset, blur_kernel) -> (N, H, W) bool mask
        # stack, re-thresholding the ROI/nav-image data this mask came from
        # with new settings - only given by callers whose masks are
        # threshold-derived (Tab_Tracking_CV2; SAM2's masks come from the
        # segmentation network instead, so it never passes this). None
        # skips building the Threshold box entirely.
        self.recompute_thresh_fn = recompute_thresh_fn
        self._original_stack = np.asarray(mask_stack).astype(bool)
        self.mask_stack = self._original_stack.copy()
        # The pristine tracking-derived stack (SAM2 output, or a freshly
        # recomputed cv2-tracking threshold mask) - independent of anything
        # edited in this dialog, this session or any previous one. Falls
        # back to this session's opening state if the caller has nothing
        # better to offer (e.g. no tracking result was ever cached).
        self._default_stack = (np.asarray(default_mask_stack).astype(bool)
                               if default_mask_stack is not None else None)
        self.bg_stack = bg_stack
        self.n_frames = self.mask_stack.shape[0]
        self.frame = int(np.clip(start_frame, 0, self.n_frames - 1))

        self._pixel_paint_value = None  # True/False while Ctrl+drag-painting pixels
        self._roi_drag = None  # (x0, y0, value) while Shift+drag-drawing a ROI
        self._roi_rect_artist = None
        self._roi_bg = None

        # Mesh: per-object grid-cell restriction (see _build_mesh_box below)
        # - unlike Edge Detection, there's no main-tab equivalent to sync
        # against, so this dialog is the only place it's ever set/edited;
        # the caller round-trips it via get_mesh_settings() into its own
        # per-object dataframe column instead.
        mesh_settings = mesh_settings or {}
        self._mesh_cells = set(tuple(c) for c in mesh_settings.get('cells', []))
        self._mesh_grid_artists = []   # grid-line contour artists
        self._mesh_cell_artist = None  # selected-cells highlight overlay

        layout = qtw.QVBoxLayout(self)

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(480, 480)
        # Stretch factor 1 (every other widget in this column defaults to
        # 0/fixed-height) - the image is what needs the extra space when
        # the dialog is resized/maximized, not the controls below it.
        layout.addWidget(self.canvas, 1)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        h, w = self.mask_stack.shape[1:]
        bg0 = self.bg_stack[self.frame] if self.bg_stack is not None else np.zeros((h, w))
        self.img_bg = self.ax.imshow(bg0, cmap='gray')
        self.img_mask = self.ax.imshow(self._mask_rgba(self.mask_stack[self.frame]))

        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

        row_frame = qtw.QHBoxLayout()
        layout.addLayout(row_frame)
        self.label_frame = qtw.QLabel()
        row_frame.addWidget(self.label_frame)
        self.slider_frame = qtw.QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, self.n_frames - 1)
        self.slider_frame.setValue(self.frame)
        self.slider_frame.valueChanged.connect(self._on_frame_changed)
        row_frame.addWidget(self.slider_frame)
        # Discoverable help affordance (item 4) - directly under the canvas,
        # same idea as the main tabs' own ribbon "?" button
        # (base_tab.show_shortcuts_dialog), just scoped to this dialog's own
        # controls instead of a whole tab's. A plain QPushButton rather than
        # ribbon.py's build_icon: this dialog doesn't use the RibbonPanel
        # machinery anywhere else, so pulling it in just for one icon isn't
        # worth the coupling.
        self.button_help = qtw.QPushButton('?')
        self.button_help.setFixedSize(24, 24)
        self.button_help.setToolTip('Show mouse/keyboard controls for this dialog')
        self.button_help.clicked.connect(self._show_help_dialog)
        row_frame.addWidget(self.button_help)

        # Everything below the frame slider (D-pad, Threshold/Edge
        # Detection, Mesh, ...) sits in an independently-scrolling area
        # instead of the dialog's own top-level layout (see bug fix in the
        # class docstring / item 0 of the originating request): a QDialog
        # can't be resized smaller than the sum of its children's minimum
        # size hints, and the canvas's own 480px minimum plus every
        # groupbox's natural height can together exceed a smaller/laptop
        # screen's available height, pushing row_buttons (Save && Close/
        # Cancel) off-screen with no way to reach it. QScrollArea's own
        # minimumSizeHint is small (frame + scrollbar allowance) regardless
        # of how tall its contents are, so wrapping them here - instead of
        # adding them to `layout` directly - guarantees the canvas (top) and
        # row_buttons (bottom, added straight to `layout` below) always both
        # fit, however many control groupboxes exist or however short the
        # screen is; only the controls in between ever need to scroll.
        scroll_controls = qtw.QScrollArea()
        scroll_controls.setWidgetResizable(True)
        scroll_controls.setFrameShape(qtw.QFrame.NoFrame)
        scroll_controls.setMinimumHeight(160)
        scroll_content = qtw.QWidget()
        scroll_layout = qtw.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_controls.setWidget(scroll_content)
        layout.addWidget(scroll_controls)

        #%% D-pad grow/shrink buttons - arranged spatially (top/left/right/
        # bottom of a 3x3 grid) instead of a plain list, arrows pointing away
        # from center = grow, toward center = shrink.
        box_directional = qtw.QGroupBox('Grow / Shrink Mask (1 px per click)')
        scroll_layout.addWidget(box_directional)
        grid = qtw.QGridLayout()
        box_directional.setLayout(grid)

        def dpad_button(symbol, tooltip, angle, grow):
            btn = qtw.QPushButton(symbol)
            btn.setFixedSize(34, 28)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda _, a=angle, g=grow: self._grow_shrink(a, grow=g))
            return btn

        row_top = qtw.QHBoxLayout()
        row_top.addWidget(dpad_button('▲', 'Add one row to the top', 270, True))
        row_top.addWidget(dpad_button('▼', 'Remove one row from the top', 270, False))
        grid.addLayout(row_top, 0, 1)

        row_bottom = qtw.QHBoxLayout()
        row_bottom.addWidget(dpad_button('▲', 'Remove one row from the bottom', 90, False))
        row_bottom.addWidget(dpad_button('▼', 'Add one row to the bottom', 90, True))
        grid.addLayout(row_bottom, 2, 1)

        col_left = qtw.QVBoxLayout()
        col_left.addWidget(dpad_button('◀', 'Add one column to the left', 180, True))
        col_left.addWidget(dpad_button('▶', 'Remove one column from the left', 180, False))
        grid.addLayout(col_left, 1, 0)

        col_right = qtw.QVBoxLayout()
        col_right.addWidget(dpad_button('▶', 'Add one column to the right', 0, True))
        col_right.addWidget(dpad_button('◀', 'Remove one column from the right', 0, False))
        grid.addLayout(col_right, 1, 2)

        label_center = qtw.QLabel('Mask')
        label_center.setAlignment(Qt.AlignCenter)
        grid.addWidget(label_center, 1, 1)

        #%% threshold - rebuilds the base mask itself from the ROI. Changing
        # method/blur/deviation applies live to just the current frame (like
        # Edge Detection below, minus the "never baked in" part - a fresh
        # re-threshold from raw ROI data is idempotent w.r.t. its own
        # parameters, so overwriting mask_stack[frame] outright each change
        # doesn't compound); only propagating that to every frame is a
        # deliberate, explicit "Apply to All Frames" action. Mirrors the main
        # tab's own Threshold/ROI Blur/Deviation controls, only shown when
        # the caller's masks are actually threshold-derived
        # (recompute_thresh_fn given - see class docstring). Threshold and
        # Edge Detection share one horizontal row (item 1) instead of each
        # being its own full-width row - saves vertical space, which also
        # helps keep row_buttons on-screen (see the QScrollArea note above).
        # When there's no Threshold box (SAM2's case, recompute_thresh_fn is
        # None), Edge Detection is the row's only widget and naturally
        # stretches to fill it instead of leaving a lopsided empty gap.
        row_thresh_edge = qtw.QHBoxLayout()
        scroll_layout.addLayout(row_thresh_edge)
        if self.recompute_thresh_fn is not None:
            thresh_settings = thresh_settings or {}
            box_thresh = qtw.QGroupBox('Threshold (rebuild mask from ROI)')
            row_thresh_edge.addWidget(box_thresh)
            layout_thresh = qtw.QVBoxLayout()
            box_thresh.setLayout(layout_thresh)

            row_t1 = qtw.QHBoxLayout()
            layout_thresh.addLayout(row_t1)
            row_t1.addWidget(qtw.QLabel('Threshold'))
            self.combo_threshMethod = qtw.QComboBox()
            self.combo_threshMethod.addItems(['li', 'otsu', 'yen', 'mean'])
            self.combo_threshMethod.setCurrentText(thresh_settings.get('method', 'li'))
            row_t1.addWidget(self.combo_threshMethod)
            row_t1.addWidget(qtw.QLabel('ROI Blur'))
            self.combo_threshBlur = qtw.QComboBox()
            self.combo_threshBlur.addItems([str(i) for i in range(1, 23, 2)])
            self.combo_threshBlur.setCurrentText(str(thresh_settings.get('blur', 1)))
            row_t1.addWidget(self.combo_threshBlur)
            row_t1.addStretch(1)

            row_t2 = qtw.QHBoxLayout()
            layout_thresh.addLayout(row_t2)
            row_t2.addWidget(qtw.QLabel('Deviation'))
            self.slider_threshDev = qtw.QSlider(Qt.Horizontal)
            self.slider_threshDev.setRange(0, 200)
            self.slider_threshDev.setValue(int(thresh_settings.get('offset_raw', 100)))
            row_t2.addWidget(self.slider_threshDev)
            self.button_threshReset = qtw.QPushButton('Reset')
            self.button_threshReset.clicked.connect(lambda: self.slider_threshDev.setValue(100))
            row_t2.addWidget(self.button_threshReset)

            for signal in (self.combo_threshMethod.currentIndexChanged,
                          self.combo_threshBlur.currentIndexChanged,
                          self.slider_threshDev.valueChanged):
                signal.connect(self._threshold_live_update)

            row_t3 = qtw.QHBoxLayout()
            layout_thresh.addLayout(row_t3)
            row_t3.addStretch(1)
            self.button_applyThreshAll = qtw.QPushButton('Apply to All Frames')
            self.button_applyThreshAll.setToolTip(
                'Recompute every frame\'s mask from the threshold settings above')
            self.button_applyThreshAll.clicked.connect(self._apply_threshold_all)
            row_t3.addWidget(self.button_applyThreshAll)
        else:
            self.combo_threshMethod = None
            self.combo_threshBlur = None
            self.slider_threshDev = None

        #%% edge detection - live preview only (see class docstring)
        box_edge = qtw.QGroupBox('Edge Detection')
        row_thresh_edge.addWidget(box_edge)
        layout_edge = qtw.QVBoxLayout()
        box_edge.setLayout(layout_edge)

        row1 = qtw.QHBoxLayout()
        layout_edge.addLayout(row1)
        self.checkbox_edgeOnly = qtw.QCheckBox('Edge Detection')
        self.checkbox_edgeOnly.setToolTip('Live preview only - doesn\'t change the returned mask')
        row1.addWidget(self.checkbox_edgeOnly)
        row1.addWidget(qtw.QLabel('Kernel'))
        self.spinbox_edgeKernel = qtw.QSpinBox()
        self.spinbox_edgeKernel.setRange(1, 99)
        self.spinbox_edgeKernel.setValue(3)
        row1.addWidget(self.spinbox_edgeKernel)
        self.checkbox_revertMask = qtw.QCheckBox('Revert Mask')
        row1.addWidget(self.checkbox_revertMask)
        row1.addStretch(1)

        row2 = qtw.QHBoxLayout()
        layout_edge.addLayout(row2)
        self.checkbox_edgeDirectional = qtw.QCheckBox('Directional')
        row2.addWidget(self.checkbox_edgeDirectional)
        row2.addWidget(qtw.QLabel('Angle (°)'))
        self.spinbox_edgeDirection = qtw.QDoubleSpinBox()
        self.spinbox_edgeDirection.setRange(-360, 360)
        self.spinbox_edgeDirection.setSingleStep(5)
        self.spinbox_edgeDirection.setDisabled(True)
        row2.addWidget(self.spinbox_edgeDirection)
        row2.addStretch(1)
        self.checkbox_edgeDirectional.stateChanged.connect(
            lambda: self.spinbox_edgeDirection.setEnabled(self.checkbox_edgeDirectional.isChecked()))

        # Scope (item 3): mirrors the Mesh box's own "This Frame Only"/"All
        # Frames" pattern below exactly (radio_meshFrame/radio_meshAll/
        # spinbox_meshFrame, see _mesh_applies_to_frame) - before this,
        # Edge Detection had no way to apply to just one frame, unlike
        # Mesh's existing per-frame scope. "All Frames" stays the default so
        # existing behavior (uniform edge detection across every frame)
        # doesn't change unless the user opts into a narrower scope.
        row3 = qtw.QHBoxLayout()
        layout_edge.addLayout(row3)
        self.radio_edgeFrame = qtw.QRadioButton('This Frame Only')
        row3.addWidget(self.radio_edgeFrame)
        self.spinbox_edgeFrame = qtw.QSpinBox()
        self.spinbox_edgeFrame.setRange(1, self.n_frames)
        self.spinbox_edgeFrame.setValue(self.frame + 1)
        self.spinbox_edgeFrame.setToolTip('Which frame the Edge Detection preview applies to')
        row3.addWidget(self.spinbox_edgeFrame)
        self.radio_edgeAll = qtw.QRadioButton('All Frames')
        self.radio_edgeAll.setChecked(True)
        row3.addWidget(self.radio_edgeAll)
        row3.addStretch(1)
        self.radio_edgeFrame.toggled.connect(
            lambda checked: self.spinbox_edgeFrame.setEnabled(checked))
        self.spinbox_edgeFrame.setEnabled(False)

        if edge_settings:
            self.checkbox_edgeOnly.setChecked(bool(edge_settings.get('enabled', False)))
            self.spinbox_edgeKernel.setValue(int(edge_settings.get('kernel', 3)))
            self.checkbox_revertMask.setChecked(bool(edge_settings.get('revert', False)))
            self.checkbox_edgeDirectional.setChecked(bool(edge_settings.get('directional', False)))
            self.spinbox_edgeDirection.setValue(float(edge_settings.get('direction', 0)))
            self.spinbox_edgeDirection.setEnabled(self.checkbox_edgeDirectional.isChecked())
            # Missing 'scope' (old edge_settings shape, saved before this
            # option existed) defaults to 'all' - preserves old behavior
            # exactly for any caller/saved-analysis that predates it.
            if edge_settings.get('scope', 'all') == 'frame':
                self.radio_edgeFrame.setChecked(True)
                self.spinbox_edgeFrame.setValue(int(edge_settings.get('frame_idx', self.frame)) + 1)

        # Any change just redraws (non-destructive) - unlike the removed
        # "Apply" button, nothing is ever baked into mask_stack here.
        for signal in (self.checkbox_edgeOnly.stateChanged, self.spinbox_edgeKernel.valueChanged,
                       self.checkbox_revertMask.stateChanged, self.checkbox_edgeDirectional.stateChanged,
                       self.spinbox_edgeDirection.valueChanged, self.radio_edgeFrame.toggled,
                       self.spinbox_edgeFrame.valueChanged):
            signal.connect(lambda *_: self._redraw_mask())

        #%% mesh - live preview only, like Edge Detection above, but the
        # selected cells (not just enabled/angle/cell size) are themselves
        # part of what's previewed/returned - see class docstring and
        # get_mesh_settings(). No main-tab equivalent to sync against (mesh
        # only makes sense relative to one specific object's mask), so this
        # dialog is the only place it's ever edited.
        box_mesh = qtw.QGroupBox('Mesh (restrict extraction to selected cell(s))')
        scroll_layout.addWidget(box_mesh)
        layout_mesh = qtw.QVBoxLayout()
        box_mesh.setLayout(layout_mesh)

        row_m1 = qtw.QHBoxLayout()
        layout_mesh.addLayout(row_m1)
        self.checkbox_meshEnabled = qtw.QCheckBox('Enable Mesh')
        self.checkbox_meshEnabled.setToolTip(
            'Divide the mask into a grid and restrict it to just the cell(s) '
            'clicked below - left-click a cell to toggle it. Live preview '
            "only, like Edge Detection - doesn't change the returned mask.")
        row_m1.addWidget(self.checkbox_meshEnabled)
        row_m1.addWidget(qtw.QLabel('Angle (°)'))
        self.spinbox_meshAngle = qtw.QDoubleSpinBox()
        # +-180 (item 2) - was 0-179.9 (a rotation is only unique mod 180 for
        # an unoriented grid, but the user asked for the full +-180 range,
        # so honor that literally rather than silently wrapping it; +-179.9
        # instead of +-180 avoids the 180/-180 seam being an ambiguous
        # duplicate value at the very ends of the range).
        self.spinbox_meshAngle.setRange(-179.9, 179.9)
        self.spinbox_meshAngle.setSingleStep(5)
        self.spinbox_meshAngle.setToolTip('Grid rotation relative to horizontal.')
        row_m1.addWidget(self.spinbox_meshAngle)
        row_m1.addWidget(qtw.QLabel('Cell Size (px)'))
        self.spinbox_meshCellSize = qtw.QSpinBox()
        self.spinbox_meshCellSize.setRange(1, 9999)
        self.spinbox_meshCellSize.setValue(20)
        row_m1.addWidget(self.spinbox_meshCellSize)
        row_m1.addStretch(1)

        # Item 2: anchor the grid to where the object STARTED (its initial
        # tracked/segmented mask) instead of the live-edited one - see
        # _mesh_origin(). Its own row (rather than squeezing into the
        # already-busy row_m1) since it reads as a standalone toggle, not
        # one more grid-geometry field alongside Angle/Cell Size.
        row_m1b = qtw.QHBoxLayout()
        layout_mesh.addLayout(row_m1b)
        self.checkbox_meshCenterInitial = qtw.QCheckBox('Center on Initial Mask')
        self.checkbox_meshCenterInitial.setToolTip(
            "Anchor the grid to the object's centroid on its initial "
            '(pre-edit) tracked/segmented mask, instead of recentering on '
            "wherever the mask has been edited to since - useful once the "
            "object's position has drifted a lot from where it started.")
        row_m1b.addWidget(self.checkbox_meshCenterInitial)
        row_m1b.addStretch(1)

        row_m2 = qtw.QHBoxLayout()
        layout_mesh.addLayout(row_m2)
        self.label_meshCells = qtw.QLabel()
        row_m2.addWidget(self.label_meshCells)
        row_m2.addStretch(1)
        self.button_meshClear = qtw.QPushButton('Clear Selection')
        self.button_meshClear.clicked.connect(self._clear_mesh_selection)
        row_m2.addWidget(self.button_meshClear)

        # Apply to just one frame, or every frame of the tracked stack - the
        # grid itself is always anchored to *that* frame's own mask
        # (mask_centroid), so the same selected cell(s) stay aligned with
        # the same relative part of the object either way, however much it
        # has moved/tracked between frames.
        row_m3 = qtw.QHBoxLayout()
        layout_mesh.addLayout(row_m3)
        self.radio_meshFrame = qtw.QRadioButton('This Frame Only')
        row_m3.addWidget(self.radio_meshFrame)
        self.spinbox_meshFrame = qtw.QSpinBox()
        self.spinbox_meshFrame.setRange(1, self.n_frames)
        self.spinbox_meshFrame.setValue(self.frame + 1)
        self.spinbox_meshFrame.setToolTip('Which frame the mesh restriction applies to')
        row_m3.addWidget(self.spinbox_meshFrame)
        self.radio_meshAll = qtw.QRadioButton('All Frames')
        self.radio_meshAll.setChecked(True)
        row_m3.addWidget(self.radio_meshAll)
        row_m3.addStretch(1)
        self.radio_meshFrame.toggled.connect(
            lambda checked: self.spinbox_meshFrame.setEnabled(checked))
        self.spinbox_meshFrame.setEnabled(False)

        if mesh_settings:
            self.checkbox_meshEnabled.setChecked(bool(mesh_settings.get('enabled', False)))
            self.spinbox_meshAngle.setValue(float(mesh_settings.get('angle', 0)))
            self.spinbox_meshCellSize.setValue(int(mesh_settings.get('cell_size', 20)))
            self.checkbox_meshCenterInitial.setChecked(bool(mesh_settings.get('center_on_initial', False)))
            if mesh_settings.get('scope', 'all') == 'frame':
                self.radio_meshFrame.setChecked(True)
                self.spinbox_meshFrame.setValue(int(mesh_settings.get('frame_idx', self.frame)) + 1)
        self._update_mesh_cell_label()

        for signal in (self.checkbox_meshEnabled.stateChanged, self.spinbox_meshAngle.valueChanged,
                       self.spinbox_meshCellSize.valueChanged, self.radio_meshFrame.toggled,
                       self.spinbox_meshFrame.valueChanged, self.checkbox_meshCenterInitial.stateChanged):
            signal.connect(lambda *_: self._redraw_mask())

        row_buttons = qtw.QHBoxLayout()
        layout.addLayout(row_buttons)
        self.button_resetFrame = qtw.QPushButton('Reset This Frame')
        self.button_resetFrame.setToolTip('Discard edits made to this frame only')
        self.button_resetFrame.clicked.connect(self._reset_frame)
        row_buttons.addWidget(self.button_resetFrame)
        self.button_resetTracking = qtw.QPushButton('Reset to Tracking')
        self.button_resetTracking.setToolTip('Discard all edits, every frame, restore the original mask')
        self.button_resetTracking.clicked.connect(self._reset_to_tracking)
        row_buttons.addWidget(self.button_resetTracking)
        row_buttons.addStretch(1)
        self.button_ok = qtw.QPushButton('Save && Close')
        self.button_ok.clicked.connect(self.accept)
        row_buttons.addWidget(self.button_ok)
        self.button_cancel_dlg = qtw.QPushButton('Cancel')
        self.button_cancel_dlg.clicked.connect(self.reject)
        row_buttons.addWidget(self.button_cancel_dlg)

        self._update_frame_label()
        self._redraw_mask()

    def _show_help_dialog(self):
        """The ribbon-style "?" button's slot: show this dialog's own
        mouse/keyboard controls reference (_HELP_TEXT). Same QDialog+
        QTextEdit shape as the main tabs' own show_shortcuts_dialog
        (ui_tabs/base_tab.py), just self-contained here since
        MaskEditDialog is a plain QDialog, not a TabBase subclass."""
        dlg = qtw.QDialog(self)
        dlg.setWindowTitle('Shortcuts & Controls')
        layout = qtw.QVBoxLayout(dlg)
        text_edit = qtw.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(_HELP_TEXT)
        layout.addWidget(text_edit)
        button_close = qtw.QPushButton('Close')
        button_close.clicked.connect(dlg.close)
        layout.addWidget(button_close, alignment=Qt.AlignRight)
        dlg.resize(480, 360)
        dlg.exec_()

    def _mask_rgba(self, mask):
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = _MASK_COLOR
        return rgba

    def _update_frame_label(self):
        self.label_frame.setText(f'Frame {self.frame + 1} / {self.n_frames}')

    def _effective_mask(self, frame):
        """The mask as currently displayed: the editable base, plus the Edge
        Detection preview and (if enabled, and Mesh applies to `frame` - see
        _mesh_applies_to_frame) the Mesh cell restriction on top - neither
        is ever written back to mask_stack itself."""
        base = self.mask_stack[frame]
        mask = base
        if self.checkbox_edgeOnly.isChecked() and self._edge_applies_to_frame(frame):
            direction = (self.spinbox_edgeDirection.value()
                        if self.checkbox_edgeDirectional.isChecked() else None)
            mask = io.erode_mask_edge(mask, self.spinbox_edgeKernel.value(),
                                      direction=direction, revert=self.checkbox_revertMask.isChecked())
        if self.checkbox_meshEnabled.isChecked() and self._mesh_applies_to_frame(frame):
            # Origin from _mesh_origin_for_frame (RAW mask centroid, or the
            # initial-mask centroid if "Center on Initial Mask" is checked -
            # see item 2), never from any Edge-Detection-preview mask, so
            # it's stable regardless of whether Edge Detection is toggled,
            # and matches exactly what the overlay/click-toggling below
            # used to build this same selection.
            origin = self._mesh_origin_for_frame(frame)
            mask = io.mesh_restrict_mask(mask, self.spinbox_meshAngle.value(),
                                         self.spinbox_meshCellSize.value(), self._mesh_cells, origin=origin)
        return mask

    def _redraw_mask(self):
        self.img_mask.set_data(self._mask_rgba(self._effective_mask(self.frame)))
        self._redraw_mesh_overlay()
        self.canvas.draw_idle()

    #%% edge detection
    def _edge_applies_to_frame(self, frame):
        """Whether the Edge Detection preview (if enabled) actually applies
        to `frame` - either every frame ("All Frames", the pre-existing
        behavior) or just the one picked in "This Frame Only" (item 3 - see
        get_edge_settings). Mirrors _mesh_applies_to_frame below exactly."""
        if self.radio_edgeAll.isChecked():
            return True
        return frame == self.spinbox_edgeFrame.value() - 1

    #%% mesh
    def _mesh_applies_to_frame(self, frame):
        """Whether the Mesh restriction (if enabled) actually applies to
        `frame` - either every frame ("All Frames") or just the one picked
        in "This Frame Only" (see get_mesh_settings)."""
        if self.radio_meshAll.isChecked():
            return True
        return frame == self.spinbox_meshFrame.value() - 1

    def _rotated_coords(self, origin):
        """(rot_x, rot_y) continuous rotated-coordinate arrays for the
        current mesh angle, centered on `origin` - a pure display concern
        (grid-line contouring), so kept local here rather than in the
        shared io.mesh_cell_ids (which only needs the floored/integer form)."""
        h, w = self.mask_stack.shape[1:]
        y, x = np.mgrid[0:h, 0:w]
        x = x - origin[0]
        y = y - origin[1]
        theta = np.deg2rad(self.spinbox_meshAngle.value())
        rot_x = x * np.cos(theta) + y * np.sin(theta)
        rot_y = -x * np.sin(theta) + y * np.cos(theta)
        return rot_x, rot_y

    def _mesh_origin_for_frame(self, frame):
        """The centroid the grid overlay/click-toggling/_effective_mask are
        built relative to, for `frame`. By default this is the object's own
        centroid on its currently-edited mask (self.mask_stack) - regardless
        of the "Apply To" scope, which only controls which frame(s) the
        selection is later restricted on (see
        _mesh_applies_to_frame/_effective_mask).

        When "Center on Initial Mask" (item 2) is checked instead, the grid
        is anchored to the centroid of the INITIAL mask on this frame -
        `_default_stack` (the pristine tracking/segmentation output) if the
        caller gave one, else `_original_stack` (this session's opening
        state) - rather than `self.mask_stack`, which reflects whatever has
        been edited (grown/shrunk/painted) since. The point is to keep the
        grid anchored to where the object STARTED even after its live mask
        has drifted from that, instead of recentering on every edit."""
        if self.checkbox_meshCenterInitial.isChecked():
            source = self._default_stack if self._default_stack is not None else self._original_stack
            return io.mask_centroid(source[frame])
        return io.mask_centroid(self.mask_stack[frame])

    def _mesh_origin(self):
        """_mesh_origin_for_frame for the currently-displayed frame - used
        by the grid overlay/click-toggling, which always work relative to
        *this* frame regardless of the Mesh box's "Apply To" scope."""
        return self._mesh_origin_for_frame(self.frame)

    def _redraw_mesh_overlay(self):
        """(Re)draw the grid-line + selected-cell overlay - a no-op cleanup
        (removing any previous artists) when Mesh is off."""
        for artist in self._mesh_grid_artists:
            artist.remove()
        self._mesh_grid_artists = []
        if self._mesh_cell_artist is not None:
            self._mesh_cell_artist.remove()
            self._mesh_cell_artist = None

        if not self.checkbox_meshEnabled.isChecked():
            return
        cell_size = self.spinbox_meshCellSize.value()
        origin = self._mesh_origin()
        rot_x, rot_y = self._rotated_coords(origin)
        levels_x = np.arange(np.floor(rot_x.min() / cell_size),
                             np.ceil(rot_x.max() / cell_size) + 1) * cell_size
        levels_y = np.arange(np.floor(rot_y.min() / cell_size),
                             np.ceil(rot_y.max() / cell_size) + 1) * cell_size
        contour_x = self.ax.contour(rot_x, levels=levels_x, colors=_MESH_GRID_COLOR,
                                    linestyles='dashed', linewidths=0.7)
        contour_y = self.ax.contour(rot_y, levels=levels_y, colors=_MESH_GRID_COLOR,
                                    linestyles='dashed', linewidths=0.7)
        self._mesh_grid_artists = [contour_x, contour_y]

        if self._mesh_cells:
            cell_i, cell_j = io.mesh_cell_ids(self.mask_stack.shape[1:],
                                              self.spinbox_meshAngle.value(), cell_size, origin)
            keep = np.zeros(self.mask_stack.shape[1:], dtype=bool)
            for i, j in self._mesh_cells:
                keep |= (cell_i == i) & (cell_j == j)
            rgba = np.zeros((*keep.shape, 4))
            rgba[keep] = _MESH_SELECTED_COLOR
            self._mesh_cell_artist = self.ax.imshow(rgba)

    def _update_mesh_cell_label(self):
        n = len(self._mesh_cells)
        self.label_meshCells.setText(f'{n} cell{"s" if n != 1 else ""} selected')

    def _clear_mesh_selection(self):
        self._mesh_cells.clear()
        self._update_mesh_cell_label()
        self._redraw_mask()

    def _toggle_mesh_cell(self, event):
        """Toggle the mesh cell under the cursor - always relative to the
        object's own position on whichever frame is currently displayed
        (see _mesh_origin), regardless of the "Apply To" scope."""
        cell_i, cell_j = io.mesh_cell_ids(self.mask_stack.shape[1:], self.spinbox_meshAngle.value(),
                                          self.spinbox_meshCellSize.value(), self._mesh_origin())
        row, col = int(round(event.ydata)), int(round(event.xdata))
        h, w = self.mask_stack.shape[1:]
        if not (0 <= row < h and 0 <= col < w):
            return
        cell = (int(cell_i[row, col]), int(cell_j[row, col]))
        if cell in self._mesh_cells:
            self._mesh_cells.discard(cell)
        else:
            self._mesh_cells.add(cell)
        self._update_mesh_cell_label()
        self._redraw_mask()

    def get_mesh_settings(self):
        """Current Mesh box values - the caller round-trips this into its
        own per-object dataframe column (there's no main-tab equivalent to
        sync against, unlike get_edge_settings()). `frame_idx` is only
        meaningful when scope == 'frame' (0-indexed, matching mask_stack)."""
        return {
            'enabled': self.checkbox_meshEnabled.isChecked(),
            'angle': self.spinbox_meshAngle.value(),
            'cell_size': self.spinbox_meshCellSize.value(),
            'cells': [list(c) for c in self._mesh_cells],
            'center_on_initial': self.checkbox_meshCenterInitial.isChecked(),
            'scope': 'frame' if self.radio_meshFrame.isChecked() else 'all',
            'frame_idx': self.spinbox_meshFrame.value() - 1,
        }

    def _on_frame_changed(self, value):
        self.frame = value
        self._update_frame_label()
        if self.bg_stack is not None:
            self.img_bg.set_data(self.bg_stack[self.frame])
        self._cancel_drag()
        self._redraw_mask()

    def _grow_shrink(self, angle, grow):
        self.mask_stack[self.frame] = io.shift_mask_edge(
            self.mask_stack[self.frame], angle, grow=grow)
        self._redraw_mask()

    def _reset_frame(self):
        self.mask_stack[self.frame] = self._original_stack[self.frame].copy()
        self._redraw_mask()

    def _reset_to_tracking(self):
        source = self._default_stack if self._default_stack is not None else self._original_stack
        self.mask_stack = source.copy()
        self._redraw_mask()

    def _recompute_threshold_stack(self):
        """Full (N, H, W) mask stack from the Threshold box's current
        method/blur/deviation, via the caller's recompute_thresh_fn - or
        None if that raised (caller decides how loudly to report it)."""
        method = self.combo_threshMethod.currentText()
        blur = int(self.combo_threshBlur.currentText())
        offset = self.slider_threshDev.value() / 100
        try:
            return np.asarray(self.recompute_thresh_fn(method, offset, blur)).astype(bool)
        except Exception:
            if self.logger:
                self.logger.exception('Failed to recompute threshold mask.')
            return None

    def _threshold_live_update(self, *_):
        """Threshold box method/blur/deviation changed: recompute and
        overwrite just self.frame's mask immediately, like Edge Detection's
        live redraw - unlike Edge Detection this really does overwrite
        mask_stack (a fresh re-threshold from raw ROI data is idempotent
        w.r.t. its own parameters, so it doesn't compound the way repeated
        grow/shrink or erosion would). Silent on failure (e.g. a
        momentarily invalid combination while the user is still adjusting
        the slider) - Apply to All Frames below is the one action that
        surfaces a real error dialog."""
        full = self._recompute_threshold_stack()
        if full is None:
            return
        self.mask_stack[self.frame] = full[self.frame]
        self._redraw_mask()

    def _apply_threshold_all(self):
        """Threshold box "Apply to All Frames": recompute and overwrite the
        whole mask_stack - the one Threshold action that needs an explicit
        button, since (unlike the live per-frame update above) it discards
        every other frame's edits at once."""
        full = self._recompute_threshold_stack()
        if full is None:
            qtw.QMessageBox.critical(self, 'Threshold Failed',
                'Could not recompute the mask with these threshold settings - see log for details.')
            return
        self.mask_stack = full
        self._redraw_mask()

    def get_mask_stack(self):
        return self.mask_stack

    def get_edge_settings(self):
        """Current Edge Detection box values - lets the caller sync its own
        main-tab controls to whatever was left set here on Save && Close."""
        return {
            'enabled': self.checkbox_edgeOnly.isChecked(),
            'kernel': self.spinbox_edgeKernel.value(),
            'revert': self.checkbox_revertMask.isChecked(),
            'directional': self.checkbox_edgeDirectional.isChecked(),
            'direction': self.spinbox_edgeDirection.value(),
        }

    def get_thresh_settings(self):
        """Current Threshold box values, or None if this dialog was opened
        without recompute_thresh_fn (no Threshold box built at all - see
        class docstring)."""
        if self.recompute_thresh_fn is None:
            return None
        return {
            'method': self.combo_threshMethod.currentText(),
            'offset_raw': self.slider_threshDev.value(),
            'blur': int(self.combo_threshBlur.currentText()),
        }

    #%% mouse interaction: Ctrl+Scroll zoom, Ctrl+Click(+drag) pixel paint,
    # Shift+drag rectangular region paint
    def _cancel_drag(self):
        self._pixel_paint_value = None
        self._roi_drag = None
        if self._roi_rect_artist is not None:
            try:
                self._roi_rect_artist.remove()
            except Exception:
                self.logger.debug('ROI drag rectangle already removed.', exc_info=True)
            self._roi_rect_artist = None

    def _on_scroll(self, event):
        """Zoom in/out on Ctrl+scroll wheel, centered on the cursor."""
        if event.inaxes != self.ax or event.xdata is None or 'ctrl' not in event.modifiers:
            return
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])
        self.ax.set_xlim([event.xdata - new_width * (1 - relx), event.xdata + new_width * relx])
        self.ax.set_ylim([event.ydata - new_height * (1 - rely), event.ydata + new_height * rely])
        self.canvas.draw_idle()

    def _paint_pixel(self, event):
        row, col = int(round(event.ydata)), int(round(event.xdata))
        h, w = self.mask_stack.shape[1:]
        if 0 <= row < h and 0 <= col < w:
            self.mask_stack[self.frame, row, col] = self._pixel_paint_value
            self._redraw_mask()

    def _on_press(self, event):
        """Start Ctrl+Click pixel painting (left=add, right=remove) or
        Shift+drag rectangular-region painting - or, while the Mesh box is
        enabled, a plain left-click (no modifier - unused on this canvas
        otherwise) toggles the mesh cell under the cursor instead."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        mods = event.modifiers
        if (self.checkbox_meshEnabled.isChecked() and event.button == 1 and not mods):
            self._toggle_mesh_cell(event)
        elif 'ctrl' in mods and event.button in (1, 3):
            self._pixel_paint_value = (event.button == 1)
            self._paint_pixel(event)
        elif 'shift' in mods and event.button in (1, 3):
            self._roi_drag = (event.xdata, event.ydata, event.button == 1)
            color = 'lime' if event.button == 1 else 'red'
            self._roi_rect_artist = patches.Rectangle(
                (event.xdata, event.ydata), 0, 0, linewidth=1.5,
                edgecolor=color, facecolor='none')
            self.ax.add_patch(self._roi_rect_artist)
            self.canvas.draw()
            self._roi_bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def _on_motion(self, event):
        """Continue whichever paint mode on_press started - matches the rest
        of the app's convention of only gating the modifier key on press,
        not on every subsequent motion event."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self._pixel_paint_value is not None:
            self._paint_pixel(event)
        elif self._roi_drag is not None:
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
        self._pixel_paint_value = None
        if self._roi_drag is None:
            return
        x0, y0, value = self._roi_drag
        self._roi_drag = None
        if self._roi_rect_artist is not None:
            try:
                self._roi_rect_artist.remove()
            except Exception:
                self.logger.debug('ROI drag rectangle already removed.', exc_info=True)
            self._roi_rect_artist = None
        if event.xdata is not None and event.ydata is not None:
            col0, col1 = sorted((int(round(x0)), int(round(event.xdata))))
            row0, row1 = sorted((int(round(y0)), int(round(event.ydata))))
            h, w = self.mask_stack.shape[1:]
            row0, row1 = max(row0, 0), min(row1 + 1, h)
            col0, col1 = max(col0, 0), min(col1 + 1, w)
            if row1 > row0 and col1 > col0:
                self.mask_stack[self.frame, row0:row1, col0:col1] = value
        self._redraw_mask()
