# -*- coding: utf-8 -*-
"""Frame-by-frame manual fine-tuning of a tracked object's per-frame mask
stack: grow/shrink the mask directionally (D-pad buttons), paint single
pixels or rectangular regions in/out with the mouse, and preview the same
Edge Detection post-processing the main tab uses - live, without baking it
into the stored mask. Shared by Tab_SAM2 and Tab_Tracking_CV2 (see their own
open_fine_tune_mask_dialog())."""
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QIntValidator
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import _Mode
import EDyssey.io_utils as io
from .ribbon import RibbonPanel, RibbonTool
from .denoise_widget import DenoiseBox

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
    'Ribbon (under the canvas):\n'
    '  Paint In / Paint Out / Rect In / Rect Out  ->  click (or, for the '
    'Rect tools, drag) directly on the canvas while armed - no modifier '
    'key needed. Click the same button again, or another tool, to '
    'disarm/switch it. Same effect as the Ctrl/Shift shortcuts below, '
    'just without needing to hold a key.\n'
    '  Pan / Zoom (rectangle) / Home  ->  matplotlib\'s own pan/zoom-box/'
    'reset-view, same as the toolbar under the main tabs\' own canvases.\n'
    '  "?"  ->  this reference.\n'
    '\n'
    'Canvas (mouse + modifier keys - always available, regardless of '
    'which ribbon tool is armed):\n'
    '  Hold "Ctrl" + Scroll wheel  ->  Zoom in/out, centered on the cursor\n'
    '  Hold "Ctrl" + Left-Click (or drag)  ->  Paint pixels IN (add to mask)\n'
    '  Hold "Ctrl" + Right-Click (or drag)  ->  Paint pixels OUT (remove from mask)\n'
    '  Hold "Shift" + Left-drag  ->  Paint a rectangular region IN\n'
    '  Hold "Shift" + Right-drag  ->  Paint a rectangular region OUT\n'
    '  Left-Click a mesh cell (while Mesh is enabled and no ribbon tool is '
    'armed, no modifier)  ->  Toggle that cell\n'
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
    'Frame Navigation & Segments:\n'
    '  ◀ / ▶ step one frame at a time; "Go to:" jumps straight to a typed '
    'frame number. Dilate/Erode and Mesh share one timeline of "segments" - '
    'consecutive frame ranges, each with its own independent settings for '
    'both - shown as the colored bar below the slider (click it to jump to '
    'that frame; orange lines mark boundaries, the yellow line is the '
    'current frame). "Split Here" breaks the current segment in two at this '
    'frame - the first half keeps its settings, the new second half starts '
    'back at plain defaults (both disabled); "Reset to Default" puts the '
    'current segment back to that same untouched state without changing '
    'its frame range; "Merge with Previous" removes the boundary at this '
    'frame, folding it back into the previous segment (using the previous '
    'segment\'s settings). ⏮/⏭ Segment jump to the previous/next segment\'s '
    'start. Whichever segment covers the frame on screen is the one '
    'Dilate/Erode\'s and Mesh\'s widgets show/edit - Edge Detection isn\'t '
    'part of this timeline, see below.\n'
    '\n'
    'Dilate / Erode Mask:\n'
    '  Live preview only - never changes the returned mask. Grows or '
    'shrinks the mask uniformly by "Kernel Size" pixels: positive dilates '
    '(grows), negative erodes (shrinks). "Opening Kernel Size" (erode then '
    'dilate) removes small bright specks/thin protrusions without changing '
    'the mask\'s overall size; "Closing Kernel Size" (dilate then erode) '
    'fills small dark holes/gaps the same way - both 0 (off) by default, '
    'applied in that order right after Kernel Size above. All three apply '
    'before Edge Detection below, and are scoped to the current segment '
    '(see Frame Navigation & Segments above).\n'
    '\n'
    'Edge Detection:\n'
    '  Live preview only - never changes the returned mask. Reduces the '
    'mask to its outline (optionally one-sided via "Directional" + Angle). '
    'Applies to the whole stack at once, not scoped to the current segment '
    'like Dilate/Erode and Mesh - anchored to whatever settings the object '
    'already had before this session, so splitting/editing other segments '
    'never changes what Edge Detection itself does.\n'
    '\n'
    'Mesh:\n'
    '  Divides the mask into a rotated grid; left-click cell(s) to '
    'restrict extraction to just those cells (live preview only), scoped '
    'to the current segment. The full mask keeps showing in orange as a '
    'reference - the part of it inside the selected cell(s) is highlighted '
    'brown on top, instead of the mask shrinking down to just the '
    'selection. "Lines Only" restricts to full-width stripes along the '
    'grid angle instead of individual square cells - clicking one keeps '
    'its whole stripe. Switching it clears the current selection. '
    '"Center on Initial Mask" anchors the grid to where the object '
    'started (its first tracked/segmented position) instead of wherever '
    "it's been edited to since - useful once the mask has moved a lot.\n"
    '\n'
    'Reset This Frame / Reset to Tracking:\n'
    '  Discard edits to just the frame on screen, or every frame, back to '
    'the original tracked/segmented mask.\n'
    '\n'
    'Find Tilt Axis:\n'
    '  Estimates the tomography tilt axis from how this object\'s mask '
    'centroid moves across frames, and draws it on the canvas as a dashed '
    'yellow reference line. "Show Details..." opens the underlying '
    'centroid scatter and candidate-angle sweep in a separate window. '
    'Assumes every frame is a tilt of the same specimen about one fixed '
    'in-plane axis, with little translational drift between frames.')
_MESH_GRID_COLOR = 'cyan'
# tab:brown, on top of _MASK_COLOR's tab:orange - highlights just the part
# of the mask that falls inside the selected mesh cell(s) (see
# _redraw_mesh_overlay), so the full mask (still shown underneath,
# unrestricted) stays visible as a reference while picking cells instead of
# being replaced by the cell-restricted view.
_MESH_SELECTED_COLOR = np.array([0.549, 0.337, 0.294, 0.7])
_SEGMENT_BAR_COLORS = [QColor('#3a3a3a'), QColor('#4c4c4c')]  # alternating segment blocks
_SEGMENT_BOUNDARY_COLOR = QColor('orange')
_SEGMENT_PLAYHEAD_COLOR = QColor('yellow')


def _dilate_erode_fields(d):
    """Just the Dilate/Erode-relevant fields out of `d` (a flat settings
    dict, or one entry from a `{'segments': [...]}` list), with defaults
    for anything missing - the per-segment shape MaskEditDialog._segments
    stores under each segment's 'dilate_erode' key. 'open_kernel'/
    'close_kernel' are the box's own Opening/Closing controls - unlike
    'kernel' (signed: grows or shrinks the mask), these are unsigned sizes
    (0 = off) since opening/closing each apply as one fixed erode-then-
    dilate (or reverse) pair, with no "which direction" to pick - see
    io.open_mask/io.close_mask."""
    d = d or {}
    return {'enabled': bool(d.get('enabled', False)), 'kernel': int(d.get('kernel', 0)),
            'open_kernel': int(d.get('open_kernel', 0)), 'close_kernel': int(d.get('close_kernel', 0))}


def _edge_fields(d):
    """Edge-Detection-relevant fields out of `d` - see _dilate_erode_fields."""
    d = d or {}
    return {'enabled': bool(d.get('enabled', False)), 'kernel': int(d.get('kernel', 3)),
            'directional': bool(d.get('directional', False)),
            'direction': float(d.get('direction', 0)), 'revert': bool(d.get('revert', False))}


def _mesh_fields(d):
    """Mesh-relevant fields out of `d` - see _dilate_erode_fields."""
    d = d or {}
    return {'enabled': bool(d.get('enabled', False)), 'angle': float(d.get('angle', 0)),
            'cell_size': int(d.get('cell_size', 20)), 'cells': [list(c) for c in d.get('cells', [])],
            'lines_only': bool(d.get('lines_only', False)),
            'center_on_initial': bool(d.get('center_on_initial', False))}


def _build_initial_segments(n_frames, dilate_erode_settings, edge_settings, mesh_settings):
    """Turn whatever the caller passed into MaskEditDialog.__init__ (plain
    old-shape settings dicts, or new `{'segments': [...]}`-shaped ones from
    a previous Fine-Tune Mask session with this same feature) into this
    dialog's own internal segment list: a sorted, contiguous list of
    {'start', 'end' (both inclusive), 'dilate_erode', 'edge', 'mesh'}
    dicts spanning every frame in [0, n_frames) - one shared timeline all
    three boxes carry their own settings against, instead of one fixed
    setting (Dilate/Erode, Edge Detection) or a single this-frame/all-
    frames toggle (Edge Detection, Mesh) for the whole stack.

    Old-format callers (or no settings at all) collapse to one segment
    spanning every frame, using whatever flat enabled/kernel/angle/etc.
    values they had - any old 'scope'/'frame_idx' field is intentionally
    ignored (there's no single-range equivalent worth preserving once
    multiple named ranges exist)."""
    de_list = (dilate_erode_settings or {}).get('segments')
    edge_list = (edge_settings or {}).get('segments')
    mesh_list = (mesh_settings or {}).get('segments')
    if de_list or edge_list or mesh_list:
        boundaries = sorted({0} | {s['start'] for s in (de_list or [])}
                            | {s['start'] for s in (edge_list or [])}
                            | {s['start'] for s in (mesh_list or [])})
        boundaries = [b for b in boundaries if b < n_frames] or [0]
        segments = []
        for i, start in enumerate(boundaries):
            end = (boundaries[i + 1] - 1) if i + 1 < len(boundaries) else n_frames - 1
            segments.append({
                'start': start, 'end': end,
                'dilate_erode': _dilate_erode_fields(io.segment_for_frame(de_list, start)),
                'edge': _edge_fields(io.segment_for_frame(edge_list, start)),
                'mesh': _mesh_fields(io.segment_for_frame(mesh_list, start)),
            })
        return segments
    return [{
        'start': 0, 'end': n_frames - 1,
        'dilate_erode': _dilate_erode_fields(dilate_erode_settings),
        'edge': _edge_fields(edge_settings),
        'mesh': _mesh_fields(mesh_settings),
    }]


class _SegmentBar(qtw.QWidget):
    """A thin, clickable strip below the frame slider showing each Fine-
    Tune-Mask "segment" (a frame range with its own Dilate/Erode/Edge
    Detection/Mesh settings - see MaskEditDialog) as an alternating-shaded
    block sized to its frame span, with orange lines at segment boundaries
    and a yellow playhead line at the current frame - similar to a video
    editor's timeline. Click anywhere to jump to that frame."""
    frameClicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip('Click to jump to that frame - orange lines mark segment boundaries')
        self._segments = []
        self._n_frames = 1
        self._current_frame = 0

    def set_state(self, segments, current_frame, n_frames):
        self._segments = segments
        self._current_frame = current_frame
        self._n_frames = max(1, n_frames)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        for i, seg in enumerate(self._segments):
            x0 = w * seg['start'] / self._n_frames
            x1 = w * (seg['end'] + 1) / self._n_frames
            painter.fillRect(QRectF(x0, 0, x1 - x0, h), _SEGMENT_BAR_COLORS[i % 2])
            if i > 0:
                painter.setPen(QPen(_SEGMENT_BOUNDARY_COLOR, 2))
                painter.drawLine(int(x0), 0, int(x0), h)
        x_cur = w * (self._current_frame + 0.5) / self._n_frames
        painter.setPen(QPen(_SEGMENT_PLAYHEAD_COLOR, 2))
        painter.drawLine(int(x_cur), 0, int(x_cur), h)
        painter.end()

    def mousePressEvent(self, event):
        frac = event.pos().x() / max(1, self.width())
        frame = int(np.clip(frac * self._n_frames, 0, self._n_frames - 1))
        self.frameClicked.emit(frame)


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

    A Denoise box (see denoise_widget.DenoiseBox) controls the displayed
    background image only - never the mask itself, and never what
    recompute_thresh_fn re-thresholds from (that still reads the caller's
    own already-processed stack) - purely so different denoise methods can
    be compared live while deciding the OTHER settings above. `bg_stack`
    must be contrast-only (not already denoised) so this never double-
    applies denoising on top of whatever the caller's own main-tab setting
    already baked in; it defaults to `denoise_state` (typically the calling
    tab's own current Denoise state, via ContrastScalingBox.box_denoise.
    get_state()) so the preview starts out looking the same either way.

    Dilate/Erode and Mesh share one timeline of "segments" (self._segments -
    see _build_initial_segments) instead of one fixed setting (or a single
    this-frame/all-frames toggle) for the whole stack: the frame range is
    divided into consecutive ranges, each with its own independent settings
    for both boxes at once, navigable/editable via the frame-navigation row
    (prev/next frame, prev/next segment, jump-to-frame, the segment bar,
    Split/Merge) below the canvas. Whichever segment covers the frame
    currently on screen is the one Dilate/Erode's and Mesh's widgets show/
    edit; get_dilate_erode_settings()/get_mesh_settings() each return their
    own `{'segments': [...]}` view of that same shared timeline for the
    caller to persist per-object and use during real extraction.

    Edge Detection is deliberately NOT part of that shared timeline - it's
    one dialog-wide value instead, anchored to whatever the object's own
    Edge Detection settings already were before this session started (its
    `edge_settings` argument - typically the caller's own currently-saved
    per-object value). For more consistent analyses: splitting/editing
    Dilate/Erode or Mesh into more segments never changes what Edge
    Detection itself does, and it's applied uniformly to the whole stack
    rather than potentially drifting per frame range - see
    _write_widgets_to_current_segment (writes it into every segment at
    once) and __init__'s post-_build_initial_segments normalization (in
    case `edge_settings` was itself already segments-shaped from before
    this decoupling existed). get_edge_settings() still returns the same
    `{'segments': [...]}` shape as before for the caller, just with exactly
    one segment spanning the whole stack.
    """

    def __init__(self, parent, mask_stack, bg_stack=None, start_frame=0, logger=None,
                 default_mask_stack=None, edge_settings=None,
                 thresh_settings=None, recompute_thresh_fn=None, mesh_settings=None,
                 dilate_erode_settings=None, denoise_state=None):
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
        # `bg_stack` is contrast-only, NOT the caller's own live Denoise
        # setting baked in - self.box_denoise (below) applies denoising
        # itself, fresh, on top of this - so switching denoise method here
        # never double-applies whatever the main tab already has active.
        # See _bg_frame(), the single place every displayed background
        # image is read from.
        self.bg_stack = bg_stack
        self.n_frames = self.mask_stack.shape[0]
        self.frame = int(np.clip(start_frame, 0, self.n_frames - 1))

        self._pixel_paint_value = None  # True/False while Ctrl+drag-painting pixels
        self._roi_drag = None  # (x0, y0, value) while Shift+drag-drawing a ROI
        self._roi_rect_artist = None
        self._roi_bg = None

        # Segments: the shared Dilate/Erode + Mesh timeline (Edge Detection
        # is NOT per-segment - see below) - see _build_initial_segments/
        # class docstring. _current_segment_idx is whichever segment covers
        # self.frame, kept in sync by _sync_current_segment (called from
        # _on_frame_changed) - Dilate/Erode/Mesh's widgets always show/edit
        # *that* segment's settings.
        self._segments = _build_initial_segments(
            self.n_frames, dilate_erode_settings, edge_settings, mesh_settings)
        # Edge Detection is a single dialog-wide value anchored to whatever
        # the object's own settings already were before this session (the
        # frame-0/default entry - i.e. exactly what the main tab last had
        # saved for it) - for consistent analyses, later splitting/editing
        # Dilate/Erode or Mesh into more segments must never change how
        # Edge Detection itself looks. A caller reopening a session that
        # (from before this decoupling existed) had genuinely different
        # edge_settings per range would otherwise show a different one
        # depending on which segment happens to be current - collapse to
        # the first/default entry here so it's unambiguous from the start.
        if self._segments:
            default_edge = self._segments[0]['edge']
            for seg in self._segments:
                seg['edge'] = dict(default_edge)
        self._current_segment_idx = 0
        for i, seg in enumerate(self._segments):
            if seg['start'] <= self.frame <= seg['end']:
                self._current_segment_idx = i
                break
        self._mesh_grid_artists = []   # grid-line contour artists
        self._mesh_cell_artist = None  # selected-cells highlight overlay

        # Tilt axis: set once "Find Tilt Axis" has been clicked (see
        # _find_tilt_axis) - None until then, so _redraw_tilt_axis_overlay
        # has nothing to draw and "Show Details..." stays disabled.
        self._tilt_axis_angle = None
        self._tilt_axis_pca_angle = None
        self._tilt_axis_centroids = None  # (n_frames, 2) per-frame mask centroids
        self._tilt_axis_sweep = None      # (angles, variances) from the refinement sweep
        self._tilt_axis_line_artist = None
        self._tilt_axis_details_dlg = None  # keeps the non-modal details window alive

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
        # Kept alive (not shown) purely for its view-stack bookkeeping
        # (.update()/.push_current(), seeding the ribbon's Home button) and
        # as the target of the ribbon's own Pan/Zoom/Home actions below -
        # mirrors each main tab's identical (hidden) NavigationToolbar2QT.
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()

        # Ribbon: a horizontal strip of icon buttons directly under the
        # canvas - the same RibbonPanel/RibbonTool machinery each main tab
        # docks vertically to the right of its own canvas (ui_tabs/ribbon.py),
        # just laid out horizontally here since this dialog's single-column
        # layout has no room for a tall side dock. Gives every mouse
        # interaction _on_press already supports via a Ctrl/Shift modifier
        # (paint pixels in/out, paint a rectangular region in/out) a
        # discoverable, click-to-arm button equivalent too - a plain
        # click/drag on the canvas then acts according to whichever tool is
        # armed (see _on_press), exactly like select_roi/add_point do on the
        # main tabs. Deliberately doesn't duplicate the D-pad/Threshold/Edge
        # Detection/Mesh controls below - those already have their own
        # always-visible buttons, unlike these canvas-gesture actions.
        self.ribbon = RibbonPanel([
            RibbonTool('paint_in', 'paint_in', 'Paint pixels IN (add to mask) - '
                      'click/drag on the canvas (same as Ctrl+Left-Click)', 'tool'),
            RibbonTool('paint_out', 'paint_out', 'Paint pixels OUT (remove from mask) - '
                      'click/drag on the canvas (same as Ctrl+Right-Click)', 'tool'),
            RibbonTool('rect_in', 'rect_in', 'Paint a rectangular region IN - '
                      'drag on the canvas (same as Shift+Left-drag)', 'tool'),
            RibbonTool('rect_out', 'rect_out', 'Paint a rectangular region OUT - '
                      'drag on the canvas (same as Shift+Right-drag)', 'tool'),
            RibbonTool('sep1', kind='separator'),
            # Pan/Zoom are 'tool' kind (not 'action') like the paint/rect
            # tools above - checkable and mutually exclusive with every
            # other tool in this same panel (see RibbonPanel's own
            # docstring), so: (a) selecting Pan/Zoom automatically
            # deselects whatever paint/rect tool was armed, instead of both
            # being "active" and firing off the same click/drag at once,
            # and (b) re-clicking Pan/Zoom's own button deselects it - the
            # only way to leave pan/zoom mode before this fix. The actual
            # matplotlib pan()/zoom() toggle calls happen in
            # _on_ribbon_tool_changed (see _sync_pan_zoom_mode), since
            # 'tool' kind doesn't take a callback the way 'action' did.
            RibbonTool('pan', 'pan', 'Toggle pan mode', 'tool'),
            RibbonTool('zoom', 'zoom', 'Toggle rectangle-zoom mode (same as Ctrl+Scroll to zoom)',
                      'tool'),
            RibbonTool('home', 'home', 'Reset the view', 'action', self.toolbar.home),
            RibbonTool('sep2', kind='separator'),
            RibbonTool('help', 'help', 'Show mouse/keyboard controls for this dialog',
                      'action', self._show_help_dialog),
        ], parent=self, orientation='horizontal')
        self.ribbon.toolChanged.connect(self._on_ribbon_tool_changed)
        # Deferred (see _apply_ribbon_cursor's docstring) - reapplies the
        # ribbon cursor after mpl's own NavigationToolbar2 cursor-restore
        # logic (wrapped around every canvas.draw()) has already run.
        self.canvas.mpl_connect(
            'draw_event', lambda evt: QTimer.singleShot(0, self._apply_ribbon_cursor))
        layout.addWidget(self.ribbon)

        # Frame slider (row 0) and Segment bar (row 1, added further below)
        # share one QGridLayout instead of two independent QHBoxLayouts, so
        # the bar renders at exactly the same width as the slider above it:
        # Qt unifies each column's width across every row of a shared grid,
        # so column 1 (the stretchy slider/bar itself) ends up the same
        # width in both rows regardless of how each row's own flanking
        # buttons/labels differ - matching flanking widths by hand would
        # drift out of sync the moment either row's own content changes.
        grid_slider = qtw.QGridLayout()
        layout.addLayout(grid_slider)
        grid_slider.setColumnStretch(1, 1)

        self.label_frame = qtw.QLabel()
        layout_frame_left = qtw.QHBoxLayout()
        layout_frame_left.setContentsMargins(0, 0, 0, 0)
        layout_frame_left.addWidget(self.label_frame)
        # Prev/Next Frame sit together, right before the slider itself,
        # rather than flanking it on both sides.
        self.button_prevFrame = qtw.QPushButton('◀')
        self.button_prevFrame.setFixedWidth(28)
        self.button_prevFrame.setToolTip('Previous frame')
        self.button_prevFrame.clicked.connect(lambda: self._step_frame(-1))
        layout_frame_left.addWidget(self.button_prevFrame)
        self.button_nextFrame = qtw.QPushButton('▶')
        self.button_nextFrame.setFixedWidth(28)
        self.button_nextFrame.setToolTip('Next frame')
        self.button_nextFrame.clicked.connect(lambda: self._step_frame(1))
        layout_frame_left.addWidget(self.button_nextFrame)
        grid_slider.addLayout(layout_frame_left, 0, 0)

        self.slider_frame = qtw.QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, self.n_frames - 1)
        self.slider_frame.setValue(self.frame)
        self.slider_frame.valueChanged.connect(self._on_frame_changed)
        grid_slider.addWidget(self.slider_frame, 0, 1)

        layout_frame_right = qtw.QHBoxLayout()
        layout_frame_right.setContentsMargins(0, 0, 0, 0)
        layout_frame_right.addWidget(qtw.QLabel('Go to:'))
        self.lineedit_frameJump = qtw.QLineEdit()
        self.lineedit_frameJump.setFixedWidth(50)
        self.lineedit_frameJump.setValidator(QIntValidator(1, self.n_frames))
        self.lineedit_frameJump.setToolTip('Type a frame number and press Enter to jump to it')
        self.lineedit_frameJump.returnPressed.connect(self._jump_to_frame_lineedit)
        layout_frame_right.addWidget(self.lineedit_frameJump)
        grid_slider.addLayout(layout_frame_right, 0, 2)

        # Denoise: background-image display only (see class docstring) -
        # seeded to whatever the caller's own main-tab Denoise control
        # currently has set, so this preview starts out looking identical
        # to what's already on screen there, but can be changed locally
        # (e.g. to compare methods while deciding Dilate/Erode/Edge
        # Detection/Mesh) without touching the main tab's own setting.
        # Constructed here (built before the Segments row below, which
        # needs self._segments etc. already initialized) but placed into
        # the groupbox grid further down, alongside Grow/Shrink Mask - see
        # grid_boxes.addWidget(self.box_denoise, ...) there.
        self.box_denoise = DenoiseBox(title='Denoise (preview only)', show_apply_all=False)
        self.box_denoise.set_state(denoise_state)
        self.box_denoise.settingsChanged.connect(self._on_denoise_changed)
        self.box_denoise.checkMethodsRequested.connect(self._show_denoise_check_methods)
        # img_bg was already created above (before this box existed) from
        # the plain, undenoised bg0 - refresh it now that denoise_state has
        # actually been applied to box_denoise, so the seeded state shows
        # immediately instead of only after the first frame/setting change.
        if self.bg_stack is not None:
            self.img_bg.set_data(self._bg_frame(self.frame))

        # Segments: the frame-range timeline Dilate/Erode, Edge Detection,
        # and Mesh all share (see class docstring/_build_initial_segments).
        # The bar visualizes it (click to jump, like the slider above);
        # Split/Merge edit the boundaries, and Prev/Next Segment jump
        # straight to a range's start without hunting for it by hand.
        self.button_prevSegment = qtw.QPushButton('⏮ Segment')
        self.button_prevSegment.setToolTip('Jump to the start of the previous segment')
        self.button_prevSegment.clicked.connect(lambda: self._jump_to_segment(-1))
        grid_slider.addWidget(self.button_prevSegment, 1, 0)
        self.segment_bar = _SegmentBar()
        self.segment_bar.frameClicked.connect(self.slider_frame.setValue)
        grid_slider.addWidget(self.segment_bar, 1, 1)
        self.button_nextSegment = qtw.QPushButton('Segment ⏭')
        self.button_nextSegment.setToolTip('Jump to the start of the next segment')
        self.button_nextSegment.clicked.connect(lambda: self._jump_to_segment(1))
        grid_slider.addWidget(self.button_nextSegment, 1, 2)

        row_segment2 = qtw.QHBoxLayout()
        layout.addLayout(row_segment2)
        self.label_segment = qtw.QLabel()
        row_segment2.addWidget(self.label_segment)
        row_segment2.addStretch(1)
        self.button_splitSegment = qtw.QPushButton('Split Here')
        self.button_splitSegment.setToolTip(
            'Split the current segment into two at this frame - the first half keeps '
            'its settings, the new second half starts back at plain defaults')
        self.button_splitSegment.clicked.connect(self._split_segment_here)
        row_segment2.addWidget(self.button_splitSegment)
        self.button_mergeSegment = qtw.QPushButton('Merge with Previous')
        self.button_mergeSegment.setToolTip(
            'Remove the boundary at this frame, extending the previous segment '
            "to cover this one too (using the previous segment's settings)")
        self.button_mergeSegment.clicked.connect(self._merge_with_previous_segment)
        row_segment2.addWidget(self.button_mergeSegment)
        self.button_resetSegment = qtw.QPushButton('Reset to Default')
        self.button_resetSegment.setToolTip(
            "Put this segment's Dilate/Erode/Mesh settings back to plain defaults "
            "(both disabled) without changing its frame range - doesn't affect Edge "
            'Detection, which applies uniformly to the whole stack, not per-segment')
        self.button_resetSegment.clicked.connect(self._reset_current_segment_to_default)
        row_segment2.addWidget(self.button_resetSegment)

        # Tilt axis: one button, kept deliberately simple - the estimate
        # (PCA + angle-sweep refinement on the object's own per-frame mask
        # centroid, see io.estimate_tilt_axis_pca/sweep_tilt_axis_angle)
        # runs entirely behind this one click, and its result is drawn
        # straight onto the canvas as a reference line rather than needing
        # its own dedicated plot to be read first. "Show Details..." is the
        # opt-in path to the full centroid-scatter/angle-sweep plot, for
        # when the user actually wants to sanity-check the estimate itself,
        # in its own (non-modal) window rather than cluttering this one.
        row_tilt = qtw.QHBoxLayout()
        layout.addLayout(row_tilt)
        self.button_findTiltAxis = qtw.QPushButton('Find Tilt Axis')
        self.button_findTiltAxis.setToolTip(
            "Estimate the tomography tilt axis from how this object's mask "
            "centroid moves across frames, and draw it on the canvas as a "
            "reference line. Assumes every frame is a tilt of the same "
            "specimen about one fixed in-plane axis with little "
            "translational drift.")
        self.button_findTiltAxis.clicked.connect(self._find_tilt_axis)
        row_tilt.addWidget(self.button_findTiltAxis)
        self.button_tiltAxisDetails = qtw.QPushButton('Show Details...')
        self.button_tiltAxisDetails.setToolTip(
            'Open the frame centroid scatter and candidate-angle variance '
            'sweep behind this estimate in a separate window')
        self.button_tiltAxisDetails.setEnabled(False)
        self.button_tiltAxisDetails.clicked.connect(self._show_tilt_axis_details)
        row_tilt.addWidget(self.button_tiltAxisDetails)
        self.label_tiltAxis = qtw.QLabel('')
        row_tilt.addWidget(self.label_tiltAxis)
        row_tilt.addStretch(1)

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

        # Every control groupbox below (Grow/Shrink Mask, Threshold, Edge
        # Detection, Dilate/Erode, Mesh) is placed into this shared 2-column
        # grid instead of one after another - halves the vertical space they
        # take up, which also helps keep row_buttons on-screen (see the
        # QScrollArea note above). Row 0: Denoise | Grow/Shrink Mask. Row 1:
        # Threshold, if this tab has one (spans both columns - its own row
        # controls, unlike the others below, aren't naturally column-paired
        # with anything). Row 2: Edge Detection (stacked above Dilate/
        # Erode, in one shared column) | Mesh.
        grid_boxes = qtw.QGridLayout()
        grid_boxes.setHorizontalSpacing(8)
        grid_boxes.setVerticalSpacing(8)
        scroll_layout.addLayout(grid_boxes)

        grid_boxes.addWidget(self.box_denoise, 0, 0)

        #%% D-pad grow/shrink buttons - arranged spatially (top/left/right/
        # bottom of a 3x3 grid) instead of a plain list, arrows pointing away
        # from center = grow, toward center = shrink.
        box_directional = qtw.QGroupBox('Grow / Shrink Mask (1 px per click)')
        grid_boxes.addWidget(box_directional, 0, 1)
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
        # (recompute_thresh_fn given - see class docstring). Grid-placed
        # below alongside Dilate/Erode (row 1) - box_thresh stays None (see
        # else branch) when this tab has no Threshold box at all, so that
        # row just starts with Dilate/Erode instead of leaving a gap.
        if self.recompute_thresh_fn is not None:
            thresh_settings = thresh_settings or {}
            box_thresh = qtw.QGroupBox('Threshold (rebuild mask from ROI)')
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
            self.spinbox_threshBlur = qtw.QDoubleSpinBox()
            self.spinbox_threshBlur.setFixedWidth(60)
            # Same Gaussian-blur-sigma convention as the "Adjust Contrast"
            # box's own Denoise control - 0 means no blur.
            self.spinbox_threshBlur.setRange(0.0, 20.0)
            self.spinbox_threshBlur.setSingleStep(0.1)
            self.spinbox_threshBlur.setDecimals(1)
            self.spinbox_threshBlur.setValue(thresh_settings.get('blur', 0))
            row_t1.addWidget(self.spinbox_threshBlur)
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
                          self.spinbox_threshBlur.valueChanged,
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
            box_thresh = None
            self.combo_threshMethod = None
            self.spinbox_threshBlur = None
            self.slider_threshDev = None

        #%% edge detection - live preview only (see class docstring). Not
        # grid-placed directly - stacked above Dilate/Erode into one shared
        # column below (see the "Row 2" comment further down).
        box_edge = qtw.QGroupBox('Edge Detection')
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

        # No per-box scope control anymore - which frame(s) these settings
        # apply to is governed by the shared Segments timeline instead (see
        # class docstring/the frame-navigation row above). Any change
        # writes back into the currently-active segment then redraws
        # (non-destructive) - see _on_segment_widgets_changed.
        for signal in (self.checkbox_edgeOnly.stateChanged, self.spinbox_edgeKernel.valueChanged,
                       self.checkbox_revertMask.stateChanged, self.checkbox_edgeDirectional.stateChanged,
                       self.spinbox_edgeDirection.valueChanged):
            signal.connect(lambda *_: self._on_segment_widgets_changed())

        #%% dilate/erode - live preview only, like Edge Detection above (see
        # io.dilate_erode_mask/io.open_mask/io.close_mask). One "Enable"
        # checkbox governs all three kernel-size spinboxes below it: the
        # signed one dilates (positive) or erodes (negative) the mask
        # uniformly on every side; unsigned "Opening"/"Closing" clean it up
        # instead (remove small specks / fill small holes) without changing
        # its overall size - applied in that fixed order (Kernel Size, then
        # Opening, then Closing), before Edge Detection's own outline
        # extraction (see _effective_mask). Per-ROI/object (like Mesh below,
        # unlike Threshold, which lives on the caller's own main-tab
        # controls) - this dialog is the only place any of it is ever
        # set/edited; the caller round-trips it via
        # get_dilate_erode_settings() into its own per-object column.
        box_dilate = qtw.QGroupBox('Dilate / Erode Mask')
        layout_dilate = qtw.QVBoxLayout()
        box_dilate.setLayout(layout_dilate)

        row_dilate = qtw.QHBoxLayout()
        layout_dilate.addLayout(row_dilate)
        self.checkbox_dilateErode = qtw.QCheckBox('Enable')
        self.checkbox_dilateErode.setToolTip('Live preview only - doesn\'t change the returned mask '
                                             '- governs Opening/Closing below too')
        row_dilate.addWidget(self.checkbox_dilateErode)
        row_dilate.addWidget(qtw.QLabel('Kernel Size'))
        self.spinbox_dilateErode = qtw.QSpinBox()
        self.spinbox_dilateErode.setRange(-99, 99)
        self.spinbox_dilateErode.setToolTip('Positive = dilate (grow the mask); Negative = erode (shrink it)')
        row_dilate.addWidget(self.spinbox_dilateErode)
        row_dilate.addStretch(1)

        # Opening (erode then dilate) and Closing (dilate then erode) -
        # unlike the signed Dilate/Erode control above, these don't grow or
        # shrink the mask overall; they clean it up instead (see
        # io.open_mask/io.close_mask), so a single unsigned kernel size
        # each is enough - 0 (the default) is a no-op. Applied in
        # _effective_mask right after Dilate/Erode, in Opening-then-Closing
        # order: Opening first strips small bright specks/thin protrusions
        # before Closing fills small dark holes/gaps, so Closing doesn't
        # end up bridging noise Opening would otherwise have removed.
        row_open = qtw.QHBoxLayout()
        layout_dilate.addLayout(row_open)
        row_open.addWidget(qtw.QLabel('Opening Kernel Size'))
        self.spinbox_openKernel = qtw.QSpinBox()
        self.spinbox_openKernel.setRange(0, 99)
        self.spinbox_openKernel.setToolTip(
            'Erode then dilate by this much (0 = off) - removes small bright '
            "specks/thin protrusions from the mask's edge without changing its "
            'overall size.')
        row_open.addWidget(self.spinbox_openKernel)
        row_open.addStretch(1)

        row_close = qtw.QHBoxLayout()
        layout_dilate.addLayout(row_close)
        row_close.addWidget(qtw.QLabel('Closing Kernel Size'))
        self.spinbox_closeKernel = qtw.QSpinBox()
        self.spinbox_closeKernel.setRange(0, 99)
        self.spinbox_closeKernel.setToolTip(
            'Dilate then erode by this much (0 = off) - fills small dark holes/'
            "gaps inside the mask without changing its overall size.")
        row_close.addWidget(self.spinbox_closeKernel)
        row_close.addStretch(1)

        # No per-box scope here either - see the Edge Detection box's own
        # comment above.
        for signal in (self.checkbox_dilateErode.stateChanged, self.spinbox_dilateErode.valueChanged,
                       self.spinbox_openKernel.valueChanged, self.spinbox_closeKernel.valueChanged):
            signal.connect(lambda *_: self._on_segment_widgets_changed())

        # Row 1: Threshold, if this tab has one, spanning both columns -
        # its own controls aren't naturally paired with anything else here.
        if box_thresh is not None:
            grid_boxes.addWidget(box_thresh, 1, 0, 1, 2)

        # Row 2, column 0: Edge Detection stacked directly above
        # Dilate/Erode (in front of/left of Mesh - column 1, below) - both
        # apply to the same mask, in that order (see _effective_mask), so
        # reads more naturally grouped together than paired with Mesh.
        box_edge_dilate = qtw.QWidget()
        layout_edge_dilate = qtw.QVBoxLayout(box_edge_dilate)
        layout_edge_dilate.setContentsMargins(0, 0, 0, 0)
        layout_edge_dilate.addWidget(box_edge)
        layout_edge_dilate.addWidget(box_dilate)
        grid_boxes.addWidget(box_edge_dilate, 2, 0)

        #%% mesh - live preview only, like Edge Detection above, but the
        # selected cells (not just enabled/angle/cell size) are themselves
        # part of what's previewed/returned - see class docstring and
        # get_mesh_settings(). No main-tab equivalent to sync against (mesh
        # only makes sense relative to one specific object's mask), so this
        # dialog is the only place it's ever edited.
        box_mesh = qtw.QGroupBox('Mesh (restrict extraction to selected cell(s))')
        grid_boxes.addWidget(box_mesh, 2, 1)
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
        self.checkbox_meshLinesOnly = qtw.QCheckBox('Lines Only')
        self.checkbox_meshLinesOnly.setToolTip(
            "Restrict to full-width stripes along the grid's rotated angle "
            'instead of individual square cells - a selected cell keeps its '
            'whole row of cells, not just that one. Switching this clears '
            'the current selection (a square-cell and a stripe selection '
            "aren't interchangeable).")
        self.checkbox_meshLinesOnly.stateChanged.connect(self._on_mesh_lines_only_toggled)
        row_m1b.addWidget(self.checkbox_meshLinesOnly)
        row_m1b.addStretch(1)

        row_m2 = qtw.QHBoxLayout()
        layout_mesh.addLayout(row_m2)
        self.label_meshCells = qtw.QLabel()
        row_m2.addWidget(self.label_meshCells)
        row_m2.addStretch(1)
        self.button_meshClear = qtw.QPushButton('Clear Selection')
        self.button_meshClear.clicked.connect(self._clear_mesh_selection)
        row_m2.addWidget(self.button_meshClear)

        # No per-box scope here either - see the Edge Detection box's own
        # comment above; the grid itself is still always anchored to
        # *whichever* frame is on screen's own mask (mask_centroid), so the
        # same selected cell(s) stay aligned with the same relative part of
        # the object however much it's moved/tracked by then.
        for signal in (self.checkbox_meshEnabled.stateChanged, self.spinbox_meshAngle.valueChanged,
                       self.spinbox_meshCellSize.valueChanged, self.checkbox_meshCenterInitial.stateChanged):
            signal.connect(lambda *_: self._on_segment_widgets_changed())

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

        self._load_segment_into_widgets()
        self._update_frame_label()
        self._update_segment_ui()
        self._redraw_mask()
        # Seeds the toolbar's view-stack with this initial view, so the
        # ribbon's Home button has something to reset to - without this,
        # NavigationToolbar2's stack starts completely empty and Home is a
        # silent no-op (it only ever pops/returns to a *previously pushed*
        # view). Must come after the canvas has its final initial extent
        # set (everything above), not before.
        self.toolbar.update()
        self.toolbar.push_current()

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

    def _segment_for_frame(self, frame):
        """The segment (see class docstring/_build_initial_segments)
        covering `frame` - a thin wrapper around io.segment_for_frame over
        self._segments, which is always sorted/contiguous/covers every
        frame by construction (see _build_initial_segments,
        _split_segment_here, _merge_with_previous_segment)."""
        return io.segment_for_frame(self._segments, frame)

    def _effective_mask(self, frame):
        """The mask as currently displayed: the editable base, plus the
        Dilate/Erode and Edge Detection previews stacked on top, in that
        order, using whichever segment covers `frame` (see
        _segment_for_frame) - neither is ever written back to mask_stack
        itself.

        Deliberately does NOT apply the Mesh cell restriction, even when
        Mesh is enabled - unlike Dilate/Erode and Edge Detection, Mesh is a
        spatial *selection* the user is actively picking cells against, so
        replacing the mask shown here with the already-restricted result
        would hide the very thing (the mask's full extent) they need to see
        to pick the right cells. The restriction itself is instead drawn as
        a separate highlight overlay on top - see _redraw_mesh_overlay."""
        base = self.mask_stack[frame]
        mask = base
        seg = self._segment_for_frame(frame)
        de = seg['dilate_erode']
        if de['enabled']:
            if de['kernel'] != 0:
                mask = io.dilate_erode_mask(mask, de['kernel'])
            if de['open_kernel'] != 0:
                mask = io.open_mask(mask, de['open_kernel'])
            if de['close_kernel'] != 0:
                mask = io.close_mask(mask, de['close_kernel'])
        edge = seg['edge']
        if edge['enabled']:
            direction = edge['direction'] if edge['directional'] else None
            mask = io.erode_mask_edge(mask, edge['kernel'], direction=direction, revert=edge['revert'])
        return mask

    def _redraw_mask(self):
        self.img_mask.set_data(self._mask_rgba(self._effective_mask(self.frame)))
        self._redraw_mesh_overlay()
        self._redraw_tilt_axis_overlay()
        self.canvas.draw_idle()

    #%% segments
    def _load_segment_into_widgets(self):
        """Populate the Dilate/Erode, Edge Detection, and Mesh widgets from
        self._segments[self._current_segment_idx] - called on init and
        whenever _sync_current_segment finds the current frame now belongs
        to a different segment than before. Signals blocked throughout so
        this doesn't loop back into _on_segment_widgets_changed and
        overwrite the very segment it's loading from."""
        seg = self._segments[self._current_segment_idx]
        de, edge, mesh = seg['dilate_erode'], seg['edge'], seg['mesh']
        widgets = (self.checkbox_dilateErode, self.spinbox_dilateErode, self.spinbox_openKernel,
                  self.spinbox_closeKernel, self.checkbox_edgeOnly,
                  self.spinbox_edgeKernel, self.checkbox_edgeDirectional, self.spinbox_edgeDirection,
                  self.checkbox_revertMask, self.checkbox_meshEnabled, self.spinbox_meshAngle,
                  self.spinbox_meshCellSize, self.checkbox_meshLinesOnly, self.checkbox_meshCenterInitial)
        for wid in widgets:
            wid.blockSignals(True)
        self.checkbox_dilateErode.setChecked(de['enabled'])
        self.spinbox_dilateErode.setValue(de['kernel'])
        self.spinbox_openKernel.setValue(de['open_kernel'])
        self.spinbox_closeKernel.setValue(de['close_kernel'])
        self.checkbox_edgeOnly.setChecked(edge['enabled'])
        self.spinbox_edgeKernel.setValue(edge['kernel'])
        self.checkbox_edgeDirectional.setChecked(edge['directional'])
        self.spinbox_edgeDirection.setValue(edge['direction'])
        self.spinbox_edgeDirection.setEnabled(edge['directional'])
        self.checkbox_revertMask.setChecked(edge['revert'])
        self.checkbox_meshEnabled.setChecked(mesh['enabled'])
        self.spinbox_meshAngle.setValue(mesh['angle'])
        self.spinbox_meshCellSize.setValue(mesh['cell_size'])
        self.checkbox_meshLinesOnly.setChecked(mesh['lines_only'])
        self.checkbox_meshCenterInitial.setChecked(mesh['center_on_initial'])
        for wid in widgets:
            wid.blockSignals(False)
        self._update_mesh_cell_label()

    def _write_widgets_to_current_segment(self):
        """The inverse of _load_segment_into_widgets - called whenever a
        Dilate/Erode/Edge Detection/Mesh control changes, so the currently-
        active segment's own stored Dilate/Erode/Mesh settings (not just the
        live widget state) reflect the edit - Edge Detection is written into
        every segment at once instead, since it's dialog-wide, not per-
        segment (see below). Doesn't touch mesh['cells'] - that's kept in
        sync separately via the _mesh_cells property."""
        seg = self._segments[self._current_segment_idx]
        seg['dilate_erode'] = {'enabled': self.checkbox_dilateErode.isChecked(),
                               'kernel': self.spinbox_dilateErode.value(),
                               'open_kernel': self.spinbox_openKernel.value(),
                               'close_kernel': self.spinbox_closeKernel.value()}
        edge = {'enabled': self.checkbox_edgeOnly.isChecked(),
                'kernel': self.spinbox_edgeKernel.value(),
                'directional': self.checkbox_edgeDirectional.isChecked(),
                'direction': self.spinbox_edgeDirection.value(),
                'revert': self.checkbox_revertMask.isChecked()}
        # Edge Detection is a single dialog-wide value, not per-segment like
        # Dilate/Erode/Mesh - anchored to whatever the object's own settings
        # already were before this session (see class docstring/_HELP_TEXT),
        # so it doesn't drift depending on how many segments Dilate/Erode or
        # Mesh end up split into. Every segment's own 'edge' entry is kept
        # in sync here (not just the current one) so _load_segment_into_
        # widgets shows the same value regardless of which segment happens
        # to be current when it runs.
        for s in self._segments:
            s['edge'] = dict(edge)
        seg['mesh'].update({'enabled': self.checkbox_meshEnabled.isChecked(),
                            'angle': self.spinbox_meshAngle.value(),
                            'cell_size': self.spinbox_meshCellSize.value(),
                            'lines_only': self.checkbox_meshLinesOnly.isChecked(),
                            'center_on_initial': self.checkbox_meshCenterInitial.isChecked()})

    def _on_segment_widgets_changed(self):
        """Any Dilate/Erode/Edge Detection/Mesh control changed (except
        mesh cell clicks/Lines Only, which go through the _mesh_cells
        property/_on_mesh_lines_only_toggled instead)."""
        self._write_widgets_to_current_segment()
        self._redraw_mask()

    def _sync_current_segment(self):
        """Update self._current_segment_idx to whichever segment now covers
        self.frame, reloading the Dilate/Erode/Edge Detection/Mesh widgets
        (see _load_segment_into_widgets) if that's actually a different
        segment than before - called from _on_frame_changed, so those boxes
        always show/edit the range currently on screen."""
        for i, seg in enumerate(self._segments):
            if seg['start'] <= self.frame <= seg['end']:
                if i != self._current_segment_idx:
                    self._current_segment_idx = i
                    self._load_segment_into_widgets()
                return

    def _update_segment_ui(self):
        """Refresh the segment bar's painted state, the "Segment k/M"
        label, and the Prev/Next/Split/Merge buttons' enabled state -
        called whenever the current frame or the segment list changes."""
        self.segment_bar.set_state(self._segments, self.frame, self.n_frames)
        idx = self._current_segment_idx
        seg = self._segments[idx]
        self.label_segment.setText(
            f'Segment {idx + 1}/{len(self._segments)}: frames {seg["start"] + 1}-{seg["end"] + 1}')
        self.button_prevSegment.setEnabled(idx > 0)
        self.button_nextSegment.setEnabled(idx < len(self._segments) - 1)
        self.button_mergeSegment.setEnabled(idx > 0 and self.frame == seg['start'])
        self.button_splitSegment.setEnabled(self.frame > seg['start'])

    def _step_frame(self, delta):
        """Prev/Next Frame buttons: move the slider by one frame."""
        self.slider_frame.setValue(int(np.clip(self.frame + delta, 0, self.n_frames - 1)))

    def _jump_to_segment(self, delta):
        """Prev/Next Segment buttons: jump straight to the start of the
        previous/next segment."""
        idx = int(np.clip(self._current_segment_idx + delta, 0, len(self._segments) - 1))
        self.slider_frame.setValue(self._segments[idx]['start'])

    def _jump_to_frame_lineedit(self):
        """"Go to:" field (Enter pressed): jump to the typed 1-indexed
        frame number, clamped to the valid range - the field's own
        QIntValidator already keeps stray non-numeric input out, so this
        only has to handle an empty field."""
        text = self.lineedit_frameJump.text()
        if not text:
            return
        frame = int(np.clip(int(text) - 1, 0, self.n_frames - 1))
        self.slider_frame.setValue(frame)

    def _split_segment_here(self):
        """"Split Here": break the current segment into two at self.frame.
        The first half keeps every setting the segment already had; the new
        second half starts back at plain defaults for Dilate/Erode/Mesh
        (both disabled) rather than inheriting a copy of the first half's
        settings - a segment nobody has actually configured yet should read
        as "untouched", not as a hidden duplicate of whatever segment it
        was split off from. Use "Reset to Default" to put an already-
        configured segment back to this same state. Edge Detection is
        excluded from this reset - it isn't per-segment at all (see class
        docstring/_write_widgets_to_current_segment), so the new segment
        just inherits the one dialog-wide value like every other segment
        does. A no-op if self.frame is already this segment's own start
        (nothing to split)."""
        idx = self._current_segment_idx
        seg = self._segments[idx]
        if self.frame <= seg['start']:
            return
        new_seg = {
            'start': self.frame, 'end': seg['end'],
            'dilate_erode': _dilate_erode_fields(None),
            'edge': dict(seg['edge']),
            'mesh': _mesh_fields(None),
        }
        seg['end'] = self.frame - 1
        self._segments.insert(idx + 1, new_seg)
        self._current_segment_idx = idx + 1
        self._load_segment_into_widgets()
        self._redraw_mask()
        self._update_segment_ui()

    def _reset_current_segment_to_default(self):
        """"Reset to Default": put the current segment's Dilate/Erode/Mesh
        settings back to plain defaults (both disabled) - the same state a
        freshly-split, never-touched segment starts in (see
        _split_segment_here) - without changing its frame range or touching
        any other segment. Edge Detection is untouched here - it isn't
        per-segment (see class docstring/_write_widgets_to_current_segment),
        so there's nothing segment-scoped about it to reset."""
        seg = self._segments[self._current_segment_idx]
        seg['dilate_erode'] = _dilate_erode_fields(None)
        seg['mesh'] = _mesh_fields(None)
        self._load_segment_into_widgets()
        self._redraw_mask()
        self._update_segment_ui()

    def _merge_with_previous_segment(self):
        """"Merge with Previous": remove the boundary at self.frame,
        extending the previous segment to cover this one too (using the
        previous segment's own settings - this one's are discarded). Only
        meaningful (and only enabled - see _update_segment_ui) when
        self.frame is exactly a segment's own start, and it isn't the very
        first segment."""
        idx = self._current_segment_idx
        if idx == 0 or self.frame != self._segments[idx]['start']:
            return
        prev_seg = self._segments[idx - 1]
        prev_seg['end'] = self._segments[idx]['end']
        del self._segments[idx]
        self._current_segment_idx = idx - 1
        self._load_segment_into_widgets()
        self._update_segment_ui()
        self._redraw_mask()

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

    @property
    def _mesh_cells(self):
        """The current segment's selected mesh cells, as a set of (i, j)
        tuples - backed by self._segments[self._current_segment_idx]
        ['mesh']['cells'] (a plain list-of-lists, JSON/dataframe-friendly),
        converted on the fly. Returns a fresh set each read, so mutating it
        (.add()/.discard()/.clear()) does NOT persist - reassign the
        property instead (`self._mesh_cells = cells`) after mutating a
        local copy, same as _toggle_mesh_cell/_clear_mesh_selection do."""
        return set(tuple(c) for c in self._segments[self._current_segment_idx]['mesh']['cells'])

    @_mesh_cells.setter
    def _mesh_cells(self, value):
        self._segments[self._current_segment_idx]['mesh']['cells'] = [list(c) for c in value]

    def _mesh_origin_for_frame(self, frame):
        """The centroid the grid overlay/click-toggling/_effective_mask are
        built relative to, for `frame`. By default this is the object's own
        centroid on its currently-edited mask (self.mask_stack).

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
            # Delegate to io.mesh_restrict_mask - the same function real
            # extraction uses (tab_tracking_cv2.py/tab_sam2.py's own
            # apply_edge_mask) - rather than re-deriving the cell-membership
            # logic here, so "Lines Only" (or anything else about how cells
            # translate to kept pixels) can never drift out of sync between
            # this preview and the real thing.
            keep = io.mesh_restrict_mask(
                self._effective_mask(self.frame), self.spinbox_meshAngle.value(), cell_size,
                self._mesh_cells, origin=origin, lines_only=self.checkbox_meshLinesOnly.isChecked())
            rgba = np.zeros((*keep.shape, 4))
            rgba[keep] = _MESH_SELECTED_COLOR
            self._mesh_cell_artist = self.ax.imshow(rgba)

    def _update_mesh_cell_label(self):
        n = len(self._mesh_cells)
        unit = 'line' if self.checkbox_meshLinesOnly.isChecked() else 'cell'
        self.label_meshCells.setText(f'{n} {unit}{"s" if n != 1 else ""} selected')

    def _clear_mesh_selection(self):
        self._mesh_cells = set()
        self._update_mesh_cell_label()
        self._redraw_mask()

    def _on_mesh_lines_only_toggled(self):
        """A square-cell selection and a stripe selection aren't
        interchangeable (same (i, j) values, different meaning) - clear
        whatever was picked rather than silently reinterpreting it. Also
        writes the toggle itself back into the current segment, same as
        every other Mesh control (see _on_segment_widgets_changed) -
        _clear_mesh_selection's own _redraw_mask alone won't do that."""
        self._write_widgets_to_current_segment()
        self._clear_mesh_selection()

    def _toggle_mesh_cell(self, event):
        """Toggle the mesh cell (or, with "Lines Only" checked, the whole
        stripe sharing the clicked cell's `cell_i`) under the cursor -
        always relative to the object's own position on whichever frame is
        currently displayed (see _mesh_origin). In "Lines Only" mode, a
        stripe is stored/toggled as a single representative (i, 0) entry in
        _mesh_cells rather than one entry per (i, j) pair actually on
        screen - io.mesh_restrict_mask's own `lines_only` flag is what
        makes that (i, 0) entry mean "every cell_i == i pixel" instead of
        just that one cell. Reads/mutates/reassigns _mesh_cells (a property
        backed by the current segment - see its own docstring) rather than
        mutating it in place, since each read returns a fresh set."""
        cell_i, cell_j = io.mesh_cell_ids(self.mask_stack.shape[1:], self.spinbox_meshAngle.value(),
                                          self.spinbox_meshCellSize.value(), self._mesh_origin())
        row, col = int(round(event.ydata)), int(round(event.xdata))
        h, w = self.mask_stack.shape[1:]
        if not (0 <= row < h and 0 <= col < w):
            return
        i, j = int(cell_i[row, col]), int(cell_j[row, col])
        cells = self._mesh_cells
        if self.checkbox_meshLinesOnly.isChecked():
            existing = [c for c in cells if c[0] == i]
            if existing:
                for c in existing:
                    cells.discard(c)
            else:
                cells.add((i, 0))
        else:
            cell = (i, j)
            if cell in cells:
                cells.discard(cell)
            else:
                cells.add(cell)
        self._mesh_cells = cells
        self._update_mesh_cell_label()
        self._redraw_mask()

    def get_mesh_settings(self):
        """The shared Segments timeline's own Mesh values, one entry per
        segment - the caller round-trips this into its own per-object
        dataframe column (there's no main-tab equivalent to sync against,
        unlike get_edge_settings()). Each entry's 'start'/'end' (inclusive
        frame indices) is what the caller resolves per-frame via
        io.segment_for_frame at extraction time (see
        tab_tracking_cv2.py/tab_sam2.py's own apply_edge_mask). 'lines_only'
        - see io.mesh_restrict_mask - is what the caller must pass that same
        function alongside 'cells' for this to mean what it looked like in
        this dialog's own preview."""
        return {'segments': [{'start': s['start'], 'end': s['end'], **s['mesh']} for s in self._segments]}

    #%% tilt axis
    def _find_tilt_axis(self):
        """"Find Tilt Axis" button: estimate the tomography tilt axis from
        how this object's mask centroid moves frame-to-frame - PCA on the
        per-frame centroid scatter, refined by a direct angle sweep (see
        io.estimate_tilt_axis_pca/sweep_tilt_axis_angle for the underlying
        projection-slice-theorem argument). Uses the currently-displayed
        (edge-detection/mesh-previewed) mask per frame via _effective_mask,
        so the estimate matches what's actually on screen right now."""
        centroids = np.array([io.mask_centroid(self._effective_mask(f))
                              for f in range(self.n_frames)])
        if self.n_frames < 3 or np.allclose(centroids, centroids[0]):
            qtw.QMessageBox.warning(self, 'Find Tilt Axis',
                'Not enough centroid movement across frames to estimate a '
                'tilt axis - the object needs to visibly shift position '
                'from frame to frame.')
            return
        pca_angle, centered = io.estimate_tilt_axis_pca(centroids)
        angles, variances, swept_angle = io.sweep_tilt_axis_angle(centered)
        self._tilt_axis_angle = swept_angle
        self._tilt_axis_pca_angle = pca_angle
        self._tilt_axis_centroids = centroids
        self._tilt_axis_sweep = (angles, variances)
        self.label_tiltAxis.setText(f'Tilt axis: {self._tilt_axis_angle:.1f}°')
        self.button_tiltAxisDetails.setEnabled(True)
        self._redraw_mask()

    def _redraw_tilt_axis_overlay(self):
        """(Re)draw the estimated tilt-axis reference line, centered on the
        current frame's own mask centroid - a no-op cleanup (removing any
        previous line) until "Find Tilt Axis" has been clicked at least
        once."""
        if self._tilt_axis_line_artist is not None:
            self._tilt_axis_line_artist.remove()
            self._tilt_axis_line_artist = None
        if self._tilt_axis_angle is None:
            return
        h, w = self.mask_stack.shape[1:]
        cx, cy = io.mask_centroid(self._effective_mask(self.frame))
        phi = np.radians(self._tilt_axis_angle)
        length = 0.6 * min(w, h)
        dx, dy = np.cos(phi) * length, np.sin(phi) * length
        # add_artist(), NOT plot()/add_line() - the line's endpoints can
        # fall outside the image (the mask centroid it's centered on isn't
        # necessarily the image center), and any artist added via plot()/
        # add_line() registers its own extent into the axes' dataLim
        # regardless of scalex/scaley=False (those only skip autoscaling
        # *at this call*, they don't exclude the artist from dataLim) - so
        # a later, unrelated autoscale trigger (e.g. toggling Mesh, whose
        # own contour() call autoscales) would still stretch the view to
        # include this line's true out-of-bounds extent. add_artist()
        # never registers the line's extent at all, so it can never affect
        # autoscale, no matter what triggers it or when.
        line = Line2D([cx - dx, cx + dx], [cy - dy, cy + dy],
                      linestyle='--', color='yellow', lw=1.5)
        self.ax.add_artist(line)
        self._tilt_axis_line_artist = line

    def _show_tilt_axis_details(self):
        """"Show Details..." button: the full centroid-scatter + candidate-
        angle variance sweep behind the estimate, in their own non-modal
        window - the reference line drawn on the main canvas is this
        distilled down to just the answer; this is for actually
        sanity-checking that answer."""
        if self._tilt_axis_angle is None or self._tilt_axis_centroids is None:
            return
        dlg = qtw.QDialog(self)
        dlg.setWindowTitle('Tilt Axis Details')
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.resize(900, 480)
        dlg_layout = qtw.QVBoxLayout(dlg)
        figure = Figure(constrained_layout=True, figsize=(9, 4.5))
        canvas = FigureCanvas(figure)
        dlg_layout.addWidget(canvas)
        ax_img, ax_sweep = figure.subplots(1, 2)

        h, w = self.mask_stack.shape[1:]
        bg = self._bg_frame(self.frame) if self.bg_stack is not None else np.zeros((h, w))
        ax_img.imshow(bg, cmap='gray')
        cx, cy = io.mask_centroid(self._effective_mask(self.frame))
        phi = np.radians(self._tilt_axis_angle)
        length = 0.6 * min(w, h)
        dx, dy = np.cos(phi) * length, np.sin(phi) * length
        ax_img.plot([cx - dx, cx + dx], [cy - dy, cy + dy], '--', color='yellow', lw=1.5,
                   label=f'Tilt axis ({self._tilt_axis_angle:.1f}°)')
        ax_img.plot(self._tilt_axis_centroids[:, 0], self._tilt_axis_centroids[:, 1],
                   'o-', color='orange', ms=3, lw=1, alpha=0.7, label='Frame centroids')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(f'Frame {self.frame + 1} / {self.n_frames}')
        ax_img.legend(loc='lower right', fontsize=7)

        angles, variances = self._tilt_axis_sweep
        ax_sweep.plot(angles, variances)
        ax_sweep.axvline(self._tilt_axis_angle, color='red', ls='--',
                         label=f'sweep min @ {self._tilt_axis_angle:.1f}°')
        ax_sweep.axvline(self._tilt_axis_pca_angle, color='cyan', ls=':',
                         label=f'PCA @ {self._tilt_axis_pca_angle:.1f}°')
        ax_sweep.set_xlabel('Candidate tilt-axis angle (°)')
        ax_sweep.set_ylabel('Along-axis centroid variance')
        ax_sweep.set_title('Angle sweep')
        ax_sweep.legend(fontsize=7)

        button_close = qtw.QPushButton('Close')
        button_close.clicked.connect(dlg.close)
        dlg_layout.addWidget(button_close, alignment=Qt.AlignRight)
        self._tilt_axis_details_dlg = dlg
        dlg.show()

    def _on_frame_changed(self, value):
        self.frame = value
        self._sync_current_segment()
        self._update_frame_label()
        self._update_segment_ui()
        if self.bg_stack is not None:
            self.img_bg.set_data(self._bg_frame(self.frame))
        self._cancel_drag()
        self._redraw_mask()

    def _bg_frame(self, frame):
        """The displayed background image for `frame` - box_denoise's
        current method/parameter applied fresh on top of the caller's
        contrast-only bg_stack (see class docstring/__init__). The single
        place every background-image read in this dialog goes through."""
        return self.box_denoise.apply(self.bg_stack[frame])

    def _on_denoise_changed(self):
        """box_denoise's method/parameter changed: refresh just the
        current frame's displayed background (cheap - same reasoning as
        the main tab's own live Denoise preview)."""
        if self.bg_stack is None:
            return
        self.img_bg.set_data(self._bg_frame(self.frame))
        self.canvas.draw_idle()

    def _show_denoise_check_methods(self):
        """box_denoise's "Check Methods..." button: compare every method on
        the current frame's raw (contrast-only) background."""
        if self.bg_stack is None:
            return
        self._check_methods_dlg = self.box_denoise.open_check_methods_dialog(
            self.bg_stack[self.frame], parent=self)

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
        blur = self.spinbox_threshBlur.value()
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
        """Edge Detection's one dialog-wide value (see class docstring),
        wrapped in the same `{'segments': [...]}` shape
        get_dilate_erode_settings()/get_mesh_settings() use - a single
        segment spanning the whole stack, since it isn't actually per-
        segment. The caller round-trips this into its own per-object
        dataframe column and resolves it per-frame at extraction time via
        io.segment_for_frame, same as the other two."""
        edge = self._segments[0]['edge'] if self._segments else _edge_fields(None)
        return {'segments': [{'start': 0, 'end': self.n_frames - 1, **edge}]}

    def get_thresh_settings(self):
        """Current Threshold box values, or None if this dialog was opened
        without recompute_thresh_fn (no Threshold box built at all - see
        class docstring)."""
        if self.recompute_thresh_fn is None:
            return None
        return {
            'method': self.combo_threshMethod.currentText(),
            'offset_raw': self.slider_threshDev.value(),
            'blur': self.spinbox_threshBlur.value(),
        }

    def get_dilate_erode_settings(self):
        """The shared Segments timeline's own Dilate/Erode values, one
        entry per segment - see get_edge_settings()/get_mesh_settings()."""
        return {'segments': [{'start': s['start'], 'end': s['end'], **s['dilate_erode']}
                             for s in self._segments]}

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

    #%% ribbon
    def _on_ribbon_tool_changed(self, tool_id):
        self._sync_pan_zoom_mode(tool_id)
        self._apply_ribbon_cursor()

    def _sync_pan_zoom_mode(self, tool_id):
        """Keep matplotlib's own pan/zoom navigation mode (self.toolbar.mode
        - NavigationToolbar2's real state machine, independent of the
        ribbon's own `active_tool`) in lockstep with the ribbon's 'pan'/
        'zoom' tool selection now that they're unified into one mutually-
        exclusive choice (see the RibbonTool entries in __init__): selecting
        'pan'/'zoom' turns the matching matplotlib mode on; selecting
        anything else (including None, i.e. re-clicking the active one to
        deselect it) turns whichever mode is currently on back off.
        NavigationToolbar2 has no direct "set mode to NONE" - pan()/zoom()
        are each their own on/off toggle, so leaving pan/zoom mode calls
        whichever of the two is still active, once, to flip it off."""
        current = self.toolbar.mode
        if tool_id == 'pan':
            if current != _Mode.PAN:
                self.toolbar.pan()
        elif tool_id == 'zoom':
            if current != _Mode.ZOOM:
                self.toolbar.zoom()
        elif current == _Mode.PAN:
            self.toolbar.pan()
        elif current == _Mode.ZOOM:
            self.toolbar.zoom()

    def _apply_ribbon_cursor(self):
        """Set the canvas cursor to match the ribbon's active tool - mirrors
        each main tab's identical _apply_ribbon_cursor exactly, including
        the deferred reapplication on every 'draw_event' (see its
        connection in __init__): NavigationToolbar2's own cursor-restore
        logic, wrapped around every canvas.draw() call, would otherwise
        silently overwrite this right after it's set. Pan/Zoom are excluded
        here - NavigationToolbar2 already sets its own cursor for those two
        modes (a hand/crosshair), which this would otherwise stomp on right
        back to a plain arrow."""
        tool = self.ribbon.active_tool
        if tool in ('pan', 'zoom'):
            return
        cursor = {'paint_in': Qt.PointingHandCursor, 'paint_out': Qt.PointingHandCursor,
                  'rect_in': Qt.CrossCursor, 'rect_out': Qt.CrossCursor}.get(tool)
        self.canvas.setCursor(cursor if cursor is not None else Qt.ArrowCursor)

    def _on_press(self, event):
        """Start pixel painting or rectangular-region painting - via
        Ctrl/Shift+Click(+drag) as before, or (new) a plain click/drag while
        the matching ribbon tool ('paint_in'/'paint_out'/'rect_in'/
        'rect_out') is armed, which encodes add-vs-remove in the tool itself
        rather than left-vs-right mouse button. Or, while the Mesh box is
        enabled and no ribbon tool is armed, a plain left-click (no
        modifier) toggles the mesh cell under the cursor instead - gated on
        `tool is None` so an armed paint/rect tool always takes priority
        over mesh-cell toggling. Entirely skipped while Pan/Zoom is the
        active tool - NavigationToolbar2 already owns click/drag on the
        canvas in that mode, so nothing here should also react to the same
        gesture (this used to be possible, since Pan/Zoom were "action"
        buttons independent of `active_tool` - see the ribbon's own
        docstring in __init__)."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        mods = event.modifiers
        tool = self.ribbon.active_tool
        if tool in ('pan', 'zoom'):
            return
        plain_left = event.button == 1 and not mods

        if plain_left and tool is None and self.checkbox_meshEnabled.isChecked():
            self._toggle_mesh_cell(event)
            return

        if 'ctrl' in mods and event.button in (1, 3):
            paint_value = event.button == 1
        elif plain_left and tool in ('paint_in', 'paint_out'):
            paint_value = tool == 'paint_in'
        else:
            paint_value = None
        if paint_value is not None:
            self._pixel_paint_value = paint_value
            self._paint_pixel(event)
            return

        if 'shift' in mods and event.button in (1, 3):
            rect_value = event.button == 1
        elif plain_left and tool in ('rect_in', 'rect_out'):
            rect_value = tool == 'rect_in'
        else:
            rect_value = None
        if rect_value is not None:
            self._roi_drag = (event.xdata, event.ydata, rect_value)
            color = 'lime' if rect_value else 'red'
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
            # A real drag only - col0==col1/row0==row1 means press and
            # release landed on the same pixel (a plain click, no actual
            # drag), which should paint nothing at all rather than the
            # single pixel the +1 below would otherwise inflate it to.
            if row1 > row0 and col1 > col0:
                h, w = self.mask_stack.shape[1:]
                row0, row1 = max(row0, 0), min(row1 + 1, h)
                col0, col1 = max(col0, 0), min(col1 + 1, w)
                if row1 > row0 and col1 > col0:
                    self.mask_stack[self.frame, row0:row1, col0:col1] = value
        self._redraw_mask()
