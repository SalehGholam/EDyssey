# -*- coding: utf-8 -*-
"""Frame-by-frame manual fine-tuning of a tracked object's per-frame mask
stack: grow/shrink the mask directionally (D-pad buttons), paint single
pixels or rectangular regions in/out with the mouse, and preview the same
Edge Detection post-processing the main tab uses - live, without baking it
into the stored mask. Shared by Tab_SAM2 and Tab_Tracking_CV2 (see their own
open_fine_tune_mask_dialog())."""
import copy
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QIntValidator, QKeySequence
from PyQt5.QtWidgets import QShortcut
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import _Mode
import EDyssey.io_utils as io
from .ribbon import RibbonPanel, RibbonTool
from .denoise_widget import DenoiseBox

_DEFAULT_MASK_ALPHA = 0.28  # starting value for both spinbox_initialMaskAlpha/spinbox_tunedMaskAlpha
# tab:brown - the same hue as the Mesh box's own selected-cell highlight
# (_MESH_SELECTED_COLOR below, which shares this exact RGB). By request,
# used for the INITIAL (pre-edit) mask rather than the current/tuned one -
# _TUNED_MASK_RGB below is that one's own color instead.
_MASK_RGB = (0.549, 0.337, 0.294)
# A different hue (blue, not brown) for the current/tuned mask, drawn ON
# TOP of the initial-mask reference layer below it (item 5) - so the two are
# distinguishable at a glance even at the same opacity. Each mask's actual
# alpha is instance state (self._initial_mask_alpha/_tuned_mask_alpha, set
# from spinbox_initialMaskAlpha/spinbox_tunedMaskAlpha - see _mask_rgba/
# _initial_mask_rgba), not baked into a module-level constant, since it's
# now a live, per-session user preference rather than a fixed look.
_TUNED_MASK_RGB = (0.15, 0.45, 0.95)
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
    '  Undo / Redo  ->  step back/forward through mask edits (paint/rect '
    'paint, D-pad grow/shrink, Reset Frame, Threshold changes, Reset to '
    'Tracking\'s mask) - same as Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) below. '
    'Jumps to whichever frame the undone/redone edit was on if it isn\'t '
    'already on screen. Does NOT cover Dilate/Erode, Edge Detection, Mesh, '
    'Blob Selection (live previews only, never touch the saved mask, so '
    'reversing them is just changing the control back), or any segment '
    'boundary/setting Reset Frame/Reset Segment/Reset to Tracking change - '
    'those need a manual Split/Merge/re-tweak, or (for a full wipe) '
    'confirming Reset to Tracking again is the only way back.\n'
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
    'frame number. Dilate/Erode, Edge Detection, and Mesh share one '
    'timeline of "segments" - consecutive frame ranges, each with its own '
    'fully independent settings for all three - shown as the colored bar '
    'below the slider (click it to jump to that frame; orange lines mark '
    'boundaries, the yellow line is the current frame). "Split Here" '
    'breaks the current segment in two at this frame - the first half '
    'keeps its settings, the new second half starts back at plain '
    'defaults (all disabled); "Merge with Previous" removes the boundary '
    'at this frame, folding it back into the previous segment (using the '
    'previous segment\'s settings). ⏮/⏭ Segment jump to the previous/next '
    'segment\'s start. Whichever segment covers the frame on screen is the '
    'one Dilate/Erode\'s, Edge Detection\'s, and Mesh\'s widgets show/edit.\n'
    '\n'
    'Edit Scope (above the canvas):\n'
    '  The "Single Frame"/"Segment" radios govern what a Dilate/Erode, '
    'Edge Detection, or Mesh change (a control tweak or a mesh cell click) '
    'actually applies to - "Segment" (the default) applies it to the whole '
    'segment currently on screen; "Single Frame" instead silently splits '
    'the current segment down to just this one frame first, inheriting '
    'whatever settings already covered it, then applies the change to '
    'only that frame. Either way, '
    'the result is an ordinary segment - scrubbing back to an already '
    'frame-isolated frame later shows its own settings regardless of '
    'which mode is currently selected.\n'
    '  Three reset actions cover every scope, narrowest to widest: '
    '"Reset Frame" discards this frame\'s painted mask AND (isolating it '
    'first if needed) its Dilate/Erode/Edge Detection/Mesh settings, '
    'without touching neighboring frames - undo-able (Ctrl+Z). "Reset '
    'Segment" (below the slider) puts the whole current segment\'s '
    'Dilate/Erode/Edge Detection/Mesh settings back to plain defaults '
    'without changing its frame range or touching the mask. "Reset to '
    'Tracking" (confirmed first, not undo-able) discards everything at '
    'once - the mask on every frame, back to the original tracked/'
    'segmented result, and every segment boundary/setting, back to one '
    'plain-default segment spanning the whole stack.\n'
    '  "Exclude Manual Edits from Effects" (checked by default) - pixels '
    'you painted, rect-painted, or grew/shrunk by hand keep exactly that '
    'value even while Dilate/Erode or Edge Detection is on, instead of '
    'those effects reprocessing them along with the rest of the mask. '
    'Uncheck to go back to effects applying uniformly everywhere, manual '
    'edits included.\n'
    '\n'
    'Dilate / Erode Mask:\n'
    '  Live preview only - never changes the returned mask. Grows or '
    'shrinks the mask uniformly by "Kernel Size" pixels: positive dilates '
    '(grows), negative erodes (shrinks) - optionally one-sided instead via '
    '"Directional" + Angle (0 = grow/shrink the right-hand edge, 90 = '
    'bottom, 180 = left, 270 = top), leaving the mask\'s other sides '
    'untouched. "Opening Kernel Size" (erode then dilate) removes small '
    'bright specks/thin protrusions without changing the mask\'s overall '
    'size; "Closing Kernel Size" (dilate then erode) fills small dark '
    'holes/gaps the same way - both 0 (off) by default, always isotropic '
    '(no direction), applied in that order right after Kernel Size above. '
    'All three apply before Edge Detection below, scoped to either the '
    'current frame or segment depending on Edit Scope (see above), and '
    'skip any manually painted pixel unless "Exclude Manual Edits from '
    'Effects" is off.\n'
    '\n'
    'Edge Detection:\n'
    '  Live preview only - never changes the returned mask. Reduces the '
    'mask to its outline (optionally one-sided via "Directional" + Angle). '
    'Scoped by Edit Scope like Dilate/Erode and Mesh (see above) - each '
    'segment has its own independent Edge Detection settings, and (like '
    'Dilate/Erode) skips manually painted pixels unless "Exclude Manual '
    'Edits from Effects" is off.\n'
    '\n'
    'Mesh:\n'
    '  A segment can hold several independent meshes - pick which one to '
    'edit from the list (its own inline checkbox is that mesh\'s "enabled", '
    'independent of just being selected there); "Add Mesh"/"Delete Mesh" '
    'manage the list. Left-click cell(s) on the canvas to restrict '
    'extraction to just those cells in whichever mesh is selected (live '
    'preview only), scoped by Edit Scope like the other two (see above). '
    'The full mask keeps showing in orange as a reference - the part of it '
    'kept by every ENABLED mesh at once (their INTERSECTION, when more '
    'than one is enabled - each additional enabled mesh narrows the '
    'selection further) is highlighted brown on top, instead of the mask '
    'shrinking down to just the selection. "Lines Only" restricts to '
    'full-width stripes along the grid angle instead of individual square '
    'cells - clicking one keeps its whole stripe. Switching it clears the '
    'current selection. "Fixed Mesh" anchors the selected mesh\'s grid to '
    'one fixed point - by default, captured from where the object started '
    '(its centroid on the segment\'s initial, pre-edit mask) - instead of '
    'recentering on wherever the mask has been edited to since; "Center '
    'Grid (Click)" instead lets you pick that point yourself: click it, '
    'then click the canvas where you want the grid anchored (also turns on '
    'Fixed Mesh). While NOT Fixed, "Center on:" picks which mask the '
    "grid's per-frame recentering tracks - the object's live/edited mask "
    '(the default) or its initial/pre-edit one (same source Fixed Mesh '
    "itself captures its point from), useful when you'd rather the grid "
    'keep following the original tracked/segmented position instead of '
    'wherever the mask has since been painted/grown/shrunk to.\n'
    '\n'
    'Find Tilt Axis:\n'
    '  Estimates the tomography tilt axis from how this object\'s mask '
    'centroid moves across frames, and draws it on the canvas as a dashed '
    'yellow reference line. "Show Details..." opens the underlying '
    'centroid scatter and candidate-angle sweep in a separate window. '
    'Assumes every frame is a tilt of the same specimen about one fixed '
    'in-plane axis, with little translational drift between frames.')
_MESH_GRID_COLOR = 'cyan'
# Same hue as _MASK_COLOR (_MASK_RGB), at a much higher alpha - drawn on top
# of it, highlighting just the part of the mask that falls inside the
# selected mesh cell(s) (see _redraw_mesh_overlay), so the full mask (still
# shown underneath, unrestricted) stays visible as a reference while
# picking cells instead of being replaced by the cell-restricted view.
_MESH_SELECTED_COLOR = np.array([*_MASK_RGB, 0.7])
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
    io.open_mask/io.close_mask. 'directional'/'direction' apply only to the
    signed 'kernel' step (see io.dilate_erode_mask's own `direction`
    argument) - Opening/Closing stay isotropic, same as Edge Detection's
    own "Directional" only ever applying to its one erosion step."""
    d = d or {}
    return {'enabled': bool(d.get('enabled', False)), 'kernel': int(d.get('kernel', 0)),
            'open_kernel': int(d.get('open_kernel', 0)), 'close_kernel': int(d.get('close_kernel', 0)),
            'directional': bool(d.get('directional', False)), 'direction': float(d.get('direction', 0))}


def _edge_fields(d):
    """Edge-Detection-relevant fields out of `d` - see _dilate_erode_fields."""
    d = d or {}
    return {'enabled': bool(d.get('enabled', False)), 'kernel': int(d.get('kernel', 3)),
            'directional': bool(d.get('directional', False)),
            'direction': float(d.get('direction', 0)), 'revert': bool(d.get('revert', False))}


def _mesh_fields(d):
    """A single mesh's own fields out of `d` - see _dilate_erode_fields.
    'fixed'/'origin' (item 3, "Fixed Mesh") replace the old single
    'center_on_initial' boolean: 'fixed' still means "don't recenter on the
    object's live position every frame", but the actual anchor point is now
    captured once into 'origin' ((x, y), None until first set) instead of
    being recomputed from the initial mask on every redraw - see
    MaskEditDialog._mesh_origin_for_mesh for why (the old flag was silently
    ignored at real extraction time, since apply_edge_mask always recomputed
    a fresh live centroid - this is what made "Center on Initial Mask" look
    like it did nothing). Old saved dicts only ever have 'center_on_initial'
    (no 'origin') - read as fixed-but-not-yet-anchored, same as freshly
    checking the box now.

    'center_basis' (item 4) is which mask the DYNAMIC (non-'fixed') per-frame
    centroid tracking is computed from - 'edited' (self.mask_stack, the
    default, same as the old always-on behavior) or 'initial' (this
    segment's initial/pre-edit mask, same source 'fixed' captures its own
    one-time origin from - _default_stack if the caller gave one, else
    _original_stack). Only matters while 'fixed' is off - a fixed mesh
    always uses its own captured 'origin' regardless."""
    d = d or {}
    origin = d.get('origin')
    return {'enabled': bool(d.get('enabled', False)), 'angle': float(d.get('angle', 0)),
            'cell_size': int(d.get('cell_size', 20)), 'cells': [list(c) for c in d.get('cells', [])],
            'lines_only': bool(d.get('lines_only', False)),
            'fixed': bool(d.get('fixed', d.get('center_on_initial', False))),
            'origin': [float(origin[0]), float(origin[1])] if origin is not None else None,
            'center_basis': d.get('center_basis', 'edited') if d.get('center_basis') in ('edited', 'initial') else 'edited'}


def _meshes_from_raw(raw):
    """Normalize whatever a segment's raw stored mesh settings looked like
    into this dialog's own internal shape: a list of mesh dicts (item 2,
    "multiple meshes per segment") - `raw` is either already
    `{'meshes': [...]}` (this feature's own shape), a single old-style flat
    mesh dict (pre-item-2 saved data - one mesh, its fields directly on the
    segment/settings dict), or None/empty (no mesh at all)."""
    if not raw:
        return []
    if isinstance(raw, dict) and 'meshes' in raw:
        return [_mesh_fields(m) for m in raw['meshes']]
    return [_mesh_fields(raw)]


def _build_initial_segments(n_frames, dilate_erode_settings, edge_settings, mesh_settings):
    """Turn whatever the caller passed into MaskEditDialog.__init__ (plain
    old-shape settings dicts, or new `{'segments': [...]}`-shaped ones from
    a previous Fine-Tune Mask session with this same feature) into this
    dialog's own internal segment list: a sorted, contiguous list of
    {'start', 'end' (both inclusive), 'dilate_erode', 'edge', 'meshes'}
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
                'meshes': _meshes_from_raw(io.segment_for_frame(mesh_list, start)),
            })
        return segments
    return [{
        'start': 0, 'end': n_frames - 1,
        'dilate_erode': _dilate_erode_fields(dilate_erode_settings),
        'edge': _edge_fields(edge_settings),
        'meshes': _meshes_from_raw(mesh_settings),
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


def format_frame_list(frame_indices, max_shown=10):
    """'3, 5, 12, ... (+7 more)'-style summary of `frame_indices`, for a
    one-line log message - see MaskEditDialog.get_edited_frame_indices()."""
    frame_indices = list(frame_indices)
    if not frame_indices:
        return 'none'
    shown = ', '.join(str(int(i)) for i in frame_indices[:max_shown])
    extra = len(frame_indices) - max_shown
    return f'{shown} (+{extra} more)' if extra > 0 else shown


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
    background image, and - for a threshold-derived caller (Tab_Tracking_
    CV2, via recompute_thresh_fn's own `denoised_imgs` argument - see
    _denoised_bg_stack) - what the Threshold box actually rebuilds the mask
    from too, so choosing a denoise method here isn't just cosmetic; it
    still never touches mask_stack itself directly, the same as Dilate/
    Erode/Edge Detection/Mesh. `bg_stack` must be contrast-only (not already
    denoised) so this never double-applies denoising on top of whatever the
    caller's own main-tab setting already baked in; it defaults to
    `denoise_state` (typically the calling tab's own current Denoise state,
    via ContrastScalingBox.box_denoise.get_state()) so the preview starts
    out looking the same either way.

    Dilate/Erode, Edge Detection, and Mesh all share one timeline of
    "segments" (self._segments - see _build_initial_segments) instead of one
    fixed setting for the whole stack: the frame range is divided into
    consecutive ranges, each with its own fully independent settings for all
    three boxes at once, navigable/editable via the frame-navigation row
    (prev/next frame, prev/next segment, jump-to-frame, the segment bar,
    Split/Merge) below the canvas. Whichever segment covers the frame
    currently on screen is the one Dilate/Erode's, Edge Detection's, and
    Mesh's widgets show/edit; get_dilate_erode_settings()/get_edge_settings()/
    get_mesh_settings() each return their own `{'segments': [...]}` view of
    that same shared timeline for the caller to persist per-object and use
    during real extraction.
    """

    def __init__(self, parent, mask_stack, bg_stack=None, start_frame=0, logger=None,
                 default_mask_stack=None, edge_settings=None,
                 thresh_settings=None, recompute_thresh_fn=None, mesh_settings=None,
                 dilate_erode_settings=None, denoise_state=None, manual_edit_settings=None):
        super().__init__(parent)
        self.setWindowTitle('Fine-Tune Mask')
        # Maximize button too (off by default on a QDialog) - the image
        # needs real screen space to make fine edits legible, and a fixed
        # initial size can't anticipate every navigation-image resolution.
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint
                            | Qt.WindowMinimizeButtonHint)
        screen = qtw.QApplication.primaryScreen()
        # Two-column layout (options on the left, canvas+slider+segments on
        # the right - see the layout construction below) needs real width
        # alongside the height-driven, roughly-square canvas, unlike the
        # old single-column arrangement this replaced.
        self._LEFT_PANEL_WIDTH = 360
        if screen is not None:
            avail = screen.availableGeometry()
            h = int(avail.height() * 0.85)
            w = min(int(avail.width() * 0.85), h + self._LEFT_PANEL_WIDTH + 60)
            self.resize(w, h)
        else:
            self.resize(1160, 900)
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
        # Item 1: which pixels were set by a manual tool (paint/rect paint,
        # D-pad grow/shrink) rather than Dilate/Erode/Edge Detection's own
        # live-preview recompute - see _effective_mask/_paint_pixel/
        # _on_release/_grow_shrink. Restored from a previous session's own
        # get_manual_edit_settings() when the shape still matches (same
        # object, same frame count/size); otherwise starts empty (nothing
        # manually touched yet).
        manual_edit_settings = manual_edit_settings or {}
        stored_manual = manual_edit_settings.get('mask')
        if stored_manual is not None and np.asarray(stored_manual).shape == self.mask_stack.shape:
            self._manual_mask_stack = np.asarray(stored_manual).astype(bool).copy()
        else:
            self._manual_mask_stack = np.zeros_like(self.mask_stack, dtype=bool)
        self._initial_protect_manual_edits = bool(manual_edit_settings.get('protect', True))
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
        # Cache for _denoised_bg_stack() - see its own docstring.
        self._denoised_threshold_stack = None
        # Live per-session opacity for each mask layer - see spinbox_
        # initialMaskAlpha/spinbox_tunedMaskAlpha and _mask_rgba/
        # _initial_mask_rgba. Set here (before the widgets exist) so
        # img_initial_mask/img_mask's very first construction below already
        # uses the right value, not a stale module-level default.
        self._initial_mask_alpha = _DEFAULT_MASK_ALPHA
        self._tuned_mask_alpha = _DEFAULT_MASK_ALPHA
        self.n_frames = self.mask_stack.shape[0]
        self.frame = int(np.clip(start_frame, 0, self.n_frames - 1))

        self._pixel_paint_value = None  # True/False while Ctrl+drag-painting pixels
        self._roi_drag = None  # (x0, y0, value) while Shift+drag-drawing a ROI
        self._roi_rect_artist = None

        # Undo/redo history for destructive mask_stack edits - paint/rect
        # paint, D-pad grow/shrink, Reset This Frame, Threshold (live and
        # Apply to All Frames), Reset to Tracking - see _push_undo/_do_undo/
        # _do_redo. NOT for the non-destructive live-preview controls
        # (Dilate/Erode, Edge Detection, Mesh, Blob Selection where
        # applicable): those never touch mask_stack itself, so they're
        # already fully reversible just by changing the control back.
        # Each entry is (frame_idx, prev_array) - frame_idx is the single
        # frame prev_array replaces, or None for a whole-stack snapshot.
        # Capped so a long editing session can't grow this unboundedly -
        # each entry can be a full (H, W) or (N, H, W) boolean array.
        self._UNDO_MAX_DEPTH = 50
        self._undo_stack = []
        self._redo_stack = []
        self._threshold_undo_frame = None  # see _threshold_live_update
        self._roi_bg = None
        # Cached blit background for _redraw_mask (see _blit_mask_display) -
        # None means "needs (re)capture". Separate from _roi_bg above,
        # which is the ROI-drag preview's own independent blit cache.
        self._mask_frame_bg = None

        # See _on_frame_changed's own comment - coalesces the expensive
        # per-frame redraw (denoise + mask recompute) so dragging the frame
        # slider doesn't run it once per intermediate value.
        self._frame_redraw_timer = QTimer(self)
        self._frame_redraw_timer.setSingleShot(True)
        self._frame_redraw_timer.setInterval(20)
        self._frame_redraw_timer.timeout.connect(self._redraw_frame_content)

        # Segments: the shared Dilate/Erode + Mesh timeline (Edge Detection
        # is NOT per-segment - see below) - see _build_initial_segments/
        # class docstring. _current_segment_idx is whichever segment covers
        # self.frame, kept in sync by _sync_current_segment (called from
        # _on_frame_changed) - Dilate/Erode/Mesh's widgets always show/edit
        # *that* segment's settings.
        self._segments = _build_initial_segments(
            self.n_frames, dilate_erode_settings, edge_settings, mesh_settings)
        self._current_segment_idx = 0
        for i, seg in enumerate(self._segments):
            if seg['start'] <= self.frame <= seg['end']:
                self._current_segment_idx = i
                break
        self._mesh_grid_artists = []   # grid-line contour artists
        self._mesh_cell_artist = None  # selected-cells highlight overlay
        # Item 2: which of the current segment's self._segments[i]['meshes']
        # list is shown/edited by the Mesh box's own widgets - reset to 0
        # whenever the segment changes (see _load_segment_into_widgets).
        self._active_mesh_idx = 0
        # Item 3: True while "Center Grid (Click)" is armed, waiting for the
        # next canvas click to set the active mesh's fixed origin - see
        # _arm_mesh_center_pick/_on_press.
        self._picking_mesh_center = False

        # Tilt axis: set once "Find Tilt Axis" has been clicked (see
        # _find_tilt_axis) - None until then, so _redraw_tilt_axis_overlay
        # has nothing to draw and "Show Details..." stays disabled.
        self._tilt_axis_angle = None
        self._tilt_axis_pca_angle = None
        self._tilt_axis_centroids = None  # (n_frames, 2) per-frame mask centroids
        self._tilt_axis_sweep = None      # (angles, variances) from the refinement sweep
        self._tilt_axis_line_artist = None
        self._tilt_axis_details_dlg = None  # keeps the non-modal details window alive

        # Top-level: a left (options) / right (canvas + slider + segments)
        # split, with Save&&Close/Cancel spanning the full width at the
        # very bottom - see the class docstring's layout note. `layout`
        # (the right column) is what nearly every widget below still adds
        # itself to, same as when this was the dialog's only column;
        # `left_layout` is the new one, used only by the option groupboxes
        # further down (Denoise, Grow/Shrink, Threshold, Edge Detection,
        # Dilate/Erode, Mesh, Tilt Axis).
        outer_layout = qtw.QVBoxLayout(self)
        main_row = qtw.QHBoxLayout()
        outer_layout.addLayout(main_row, 1)
        left_panel = qtw.QWidget()
        left_panel.setFixedWidth(self._LEFT_PANEL_WIDTH)
        left_layout = qtw.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        main_row.addWidget(left_panel, 0)
        layout = qtw.QVBoxLayout()
        main_row.addLayout(layout, 1)

        # Edit Scope: whether the NEXT Dilate/Erode/Edge Detection/Mesh
        # edit (a checkbox/spinbox change, or a mesh cell click) applies
        # to just this one frame or to the whole segment currently
        # covering it - see _write_widgets_to_current_segment/_mesh_cells'
        # setter, both of which isolate the current frame into its own
        # segment first (inheriting whatever settings already cover it,
        # not blank defaults - unlike a manual "Split Here") when this is
        # set to "Single Frame". Purely a "how do I want my next edit
        # committed" mode, not itself saved anywhere - the RESULT of an
        # edit (a new 1-frame segment, or a wider one left as-is) is
        # exactly the same segment structure this dialog already saves,
        # so navigating back to an already-isolated frame later shows its
        # own settings regardless of which mode is selected here.
        #
        # Reset Frame/Reset to Tracking share this row too (moved here from
        # the old full-width bottom row, and "Reset This Frame" folded into
        # "Reset Frame" - see _reset_current_frame_to_default) - grouping
        # every "throw away edits" action along one edge, from narrowest
        # scope (this frame) up to widest (everything, "Reset to
        # Tracking"), with "Reset Segment" below the slider (row_segment2)
        # in between.
        row_edit_scope = qtw.QHBoxLayout()
        row_edit_scope.addWidget(qtw.QLabel('Edit Scope:'))
        self.radio_scopeFrame = qtw.QRadioButton('Single Frame')
        self.radio_scopeFrame.setToolTip(
            'Dilate/Erode, Edge Detection, and Mesh changes apply to just this one frame')
        row_edit_scope.addWidget(self.radio_scopeFrame)
        self.radio_scopeSegment = qtw.QRadioButton('Segment')
        self.radio_scopeSegment.setChecked(True)
        self.radio_scopeSegment.setToolTip(
            'Dilate/Erode, Edge Detection, and Mesh changes apply to the whole current '
            'segment (the frame range shown below the slider)')
        row_edit_scope.addWidget(self.radio_scopeSegment)
        # Belt-and-suspenders exclusivity - the two radios already exclude
        # each other via their shared parent widget, but an explicit group
        # doesn't rely on that layout detail staying true.
        self._scope_group = qtw.QButtonGroup(self)
        self._scope_group.addButton(self.radio_scopeFrame)
        self._scope_group.addButton(self.radio_scopeSegment)
        # Visual break between the Edit Scope radios and the two unrelated
        # checkboxes that follow in this same row - a plain addSpacing()
        # alone still read as "one long group" at a glance; a vertical
        # divider line makes the grouping unambiguous.
        row_edit_scope.addSpacing(12)
        sep_edit_scope = qtw.QFrame()
        sep_edit_scope.setFrameShape(qtw.QFrame.VLine)
        sep_edit_scope.setFrameShadow(qtw.QFrame.Sunken)
        row_edit_scope.addWidget(sep_edit_scope)
        row_edit_scope.addSpacing(12)
        # Item 1: pixels set by a manual tool (paint/rect paint, D-pad grow/
        # shrink) used to get re-processed by Dilate/Erode/Edge Detection
        # right along with everything else on every redraw (see
        # _effective_mask) - inaccurate for a region the user just placed by
        # hand. Checked by default: those pixels now keep exactly the value
        # the manual tool gave them, regardless of what Dilate/Erode/Edge
        # Detection do to the rest of the mask. Unchecking restores the old
        # behavior (effects apply uniformly, manual edits included).
        self.checkbox_protectManualEdits = qtw.QCheckBox('Exclude Manual Edits from Effects')
        self.checkbox_protectManualEdits.setChecked(self._initial_protect_manual_edits)
        self.checkbox_protectManualEdits.setToolTip(
            'Pixels painted, rect-painted, or grown/shrunk by hand keep '
            'exactly that manually-set value even when Dilate/Erode or Edge '
            'Detection is enabled, instead of those effects reprocessing '
            'them too. Uncheck to let effects apply uniformly across the '
            'whole mask again, manual edits included.')
        self.checkbox_protectManualEdits.stateChanged.connect(lambda *_: self._redraw_mask())
        row_edit_scope.addWidget(self.checkbox_protectManualEdits)
        # Item 5's always-visible initial-mask reference layer (img_initial_
        # mask) can get in the way once it's served its purpose for a given
        # edit - this hides it (set_visible only, the layer itself keeps
        # updating per-frame underneath) without touching whether it's ever
        # drawn at all.
        self.checkbox_showInitialMask = qtw.QCheckBox('Show Initial Mask')
        self.checkbox_showInitialMask.setChecked(True)
        self.checkbox_showInitialMask.setToolTip(
            "Show the mask as it was before any edit this session (see the color legend "
            "at the bottom-left) underneath the current/tuned mask - uncheck to hide it")
        self.checkbox_showInitialMask.stateChanged.connect(self._on_show_initial_mask_toggled)
        row_edit_scope.addWidget(self.checkbox_showInitialMask)
        # Live opacity for each mask layer (see _mask_rgba/_initial_mask_
        # rgba) - a per-session preference, not persisted anywhere, same as
        # every other control in this row.
        row_edit_scope.addWidget(qtw.QLabel('Initial α'))
        self.spinbox_initialMaskAlpha = qtw.QSpinBox()
        self.spinbox_initialMaskAlpha.setRange(0, 100)
        self.spinbox_initialMaskAlpha.setSuffix(' %')
        self.spinbox_initialMaskAlpha.setValue(round(_DEFAULT_MASK_ALPHA * 100))
        self.spinbox_initialMaskAlpha.setToolTip("Opacity of the initial (pre-edit) mask layer")
        self.spinbox_initialMaskAlpha.valueChanged.connect(self._on_mask_alpha_changed)
        row_edit_scope.addWidget(self.spinbox_initialMaskAlpha)
        row_edit_scope.addWidget(qtw.QLabel('Tuned α'))
        self.spinbox_tunedMaskAlpha = qtw.QSpinBox()
        self.spinbox_tunedMaskAlpha.setRange(0, 100)
        self.spinbox_tunedMaskAlpha.setSuffix(' %')
        self.spinbox_tunedMaskAlpha.setValue(round(_DEFAULT_MASK_ALPHA * 100))
        self.spinbox_tunedMaskAlpha.setToolTip("Opacity of the current/tuned mask layer")
        self.spinbox_tunedMaskAlpha.valueChanged.connect(self._on_mask_alpha_changed)
        row_edit_scope.addWidget(self.spinbox_tunedMaskAlpha)
        row_edit_scope.addStretch(1)
        self.button_resetFrame = qtw.QPushButton('Reset Frame')
        self.button_resetFrame.setToolTip(
            "Discard edits to just this frame - both its painted mask and (isolating "
            "it into its own segment first if it isn't already one) its Dilate/Erode/"
            'Edge Detection/Mesh settings')
        self.button_resetFrame.clicked.connect(self._reset_current_frame_to_default)
        row_edit_scope.addWidget(self.button_resetFrame)
        self.button_resetTracking = qtw.QPushButton('Reset to Tracking')
        self.button_resetTracking.setToolTip(
            'Discard every edit in this dialog - the mask on every frame (back to the '
            'original tracked/segmented result) and every segment boundary/Dilate/'
            'Erode/Edge Detection/Mesh setting')
        self.button_resetTracking.clicked.connect(self._reset_to_tracking)
        row_edit_scope.addWidget(self.button_resetTracking)
        layout.addLayout(row_edit_scope)

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
        # Just the coordinate, not matplotlib's own default "x=.. y=.. [val]"
        # (which also appends the underlying pixel value) - see
        # label_cursorCoords/_mirror_cursor_coords_to_label: a fixed, short,
        # single-line format keeps that label's own size constant as the
        # mouse moves, instead of varying with however long the value
        # happened to format to (which was forcing this whole row - and,
        # via the dialog's own layout minimum width, the canvas above it -
        # to subtly resize on every hover).
        self.ax.format_coord = lambda x, y: f'x={x:.1f}, y={y:.1f}'
        h, w = self.mask_stack.shape[1:]
        bg0 = self.bg_stack[self.frame] if self.bg_stack is not None else np.zeros((h, w))
        self.img_bg = self.ax.imshow(bg0, cmap='gray')
        # Item 5: the mask as it was BEFORE any edit in this session
        # (_original_stack), always shown - fainter than the live/tuned mask
        # below it (see _INITIAL_MASK_COLOR) - as a constant reference for
        # how much has changed, instead of disappearing the moment the first
        # edit is made. Never affected by Dilate/Erode/Edge Detection (it's
        # not run through _effective_mask) - it's a static snapshot, not a
        # preview. See _redraw_frame_content/_on_frame_changed, the only
        # other place its data is updated (once per frame change).
        self.img_initial_mask = self.ax.imshow(
            self._initial_mask_rgba(self._original_stack[self.frame]))
        self.img_mask = self.ax.imshow(self._mask_rgba(self.mask_stack[self.frame]))

        # Any real view change - Ctrl+scroll (_on_scroll) or the ribbon's
        # own Pan/Zoom tools driving self.toolbar.pan()/.zoom() below -
        # shifts what's baked into the cached blit background (see
        # _blit_mask_display); catching it here, on the axes' own
        # xlim/ylim callbacks, covers both without hunting down every
        # call site that can move the view.
        self.ax.callbacks.connect('xlim_changed', lambda ax: setattr(self, '_mask_frame_bg', None))
        self.ax.callbacks.connect('ylim_changed', lambda ax: setattr(self, '_mask_frame_bg', None))
        # Resizing/maximizing this dialog (it has both hints - see
        # setWindowFlags above) changes the canvas's own pixel dimensions
        # without necessarily touching xlim/ylim, so the two callbacks
        # above alone don't catch it - restoring a background cached at
        # the OLD, smaller canvas size then leaves a visible stale
        # rectangle (the old canvas's own edge) in a corner of the new,
        # bigger one, since restore_region() only repaints the region it
        # was captured over.
        self.canvas.mpl_connect('resize_event', lambda evt: setattr(self, '_mask_frame_bg', None))

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
            RibbonTool('undo', 'undo', 'Undo the last mask edit (Ctrl+Z)',
                      'action', self._do_undo),
            RibbonTool('redo', 'redo', 'Redo the last undone mask edit (Ctrl+Y)',
                      'action', self._do_redo),
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
        QShortcut(QKeySequence('Ctrl+Z'), self, self._do_undo)
        QShortcut(QKeySequence('Ctrl+Y'), self, self._do_redo)
        QShortcut(QKeySequence('Ctrl+Shift+Z'), self, self._do_redo)
        self._update_undo_redo_buttons()
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
        self.button_resetSegment = qtw.QPushButton('Reset Segment')
        self.button_resetSegment.setToolTip(
            "Put this segment's Dilate/Erode/Edge Detection/Mesh settings back to plain "
            'defaults (all disabled) without changing its frame range - see also "Reset '
            'Frame" above, which does the same for just the current frame')
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
        # Tilt Axis is an "option" (a control the user clicks to run/toggle
        # a preview), not a slider/segment - it lives in the left panel
        # alongside the other option groupboxes below, not stacked under
        # the canvas with the frame slider/segment bar.
        box_tilt = qtw.QGroupBox('Tilt Axis')
        left_layout.addWidget(box_tilt)
        box_tilt_layout = qtw.QVBoxLayout(box_tilt)
        # Item 4: the two buttons side by side, not stacked - they're a
        # single "run the estimate, then optionally inspect it" pair, not an
        # ordered sequence of independent actions worth a full row each.
        row_tilt = qtw.QHBoxLayout()
        box_tilt_layout.addLayout(row_tilt)
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
        box_tilt_layout.addWidget(self.label_tiltAxis)
        box_tilt_layout.addStretch(1)

        # The left panel's own controls (Denoise, Grow/Shrink Mask,
        # Threshold, Edge Detection, Dilate/Erode, Mesh) go straight into
        # left_layout, stretch-factor 1 so they claim any leftover height
        # below Tilt Axis above - no independent scroll area around them
        # (an earlier version had one, to protect against a short screen
        # pushing row_buttons off-screen, but it also kicked in - showing
        # an unwanted scrollbar and clipping labels against it - on
        # perfectly tall-enough screens; removed at the user's request).
        grid_boxes = qtw.QVBoxLayout()
        grid_boxes.setContentsMargins(0, 0, 0, 0)
        grid_boxes.setSpacing(8)
        left_layout.addLayout(grid_boxes, 1)

        grid_boxes.addWidget(self.box_denoise)

        #%% D-pad grow/shrink buttons - arranged spatially (top/left/right/
        # bottom of a 3x3 grid) instead of a plain list, arrows pointing away
        # from center = grow, toward center = shrink.
        box_directional = qtw.QGroupBox('Grow / Shrink Mask (1 px per click)')
        grid_boxes.addWidget(box_directional)
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
        # A shared QGridLayout, not stacked QHBoxLayouts, so every row's
        # label/spinbox column lines up with every other row's - both here
        # and in Dilate/Erode/Mesh below, which follow the exact same
        # column convention: 0 = the row's own enable/mode checkbox,
        # 1 = a field label, 2 = its control, 3 = a second checkbox where
        # one exists, with a trailing stretch column soaking up the rest of
        # the box's width.
        box_edge = qtw.QGroupBox('Edge Detection')
        layout_edge = qtw.QGridLayout()
        layout_edge.setColumnStretch(4, 1)
        box_edge.setLayout(layout_edge)

        self.checkbox_edgeOnly = qtw.QCheckBox('Enable')
        self.checkbox_edgeOnly.setToolTip('Live preview only - doesn\'t change the returned mask')
        layout_edge.addWidget(self.checkbox_edgeOnly, 0, 0)
        layout_edge.addWidget(qtw.QLabel('Kernel'), 0, 1)
        self.spinbox_edgeKernel = qtw.QSpinBox()
        self.spinbox_edgeKernel.setRange(1, 99)
        self.spinbox_edgeKernel.setValue(3)
        layout_edge.addWidget(self.spinbox_edgeKernel, 0, 2)
        self.checkbox_revertMask = qtw.QCheckBox('Revert Mask')
        layout_edge.addWidget(self.checkbox_revertMask, 0, 3)

        self.checkbox_edgeDirectional = qtw.QCheckBox('Directional')
        layout_edge.addWidget(self.checkbox_edgeDirectional, 1, 0)
        layout_edge.addWidget(qtw.QLabel('Angle (°)'), 1, 1)
        self.spinbox_edgeDirection = qtw.QDoubleSpinBox()
        self.spinbox_edgeDirection.setRange(-360, 360)
        self.spinbox_edgeDirection.setSingleStep(5)
        self.spinbox_edgeDirection.setDisabled(True)
        layout_edge.addWidget(self.spinbox_edgeDirection, 1, 2)
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
        layout_dilate = qtw.QGridLayout()
        layout_dilate.setColumnStretch(4, 1)
        box_dilate.setLayout(layout_dilate)

        self.checkbox_dilateErode = qtw.QCheckBox('Enable')
        self.checkbox_dilateErode.setToolTip('Live preview only - doesn\'t change the returned mask '
                                             '- governs Opening/Closing below too')
        layout_dilate.addWidget(self.checkbox_dilateErode, 0, 0)
        layout_dilate.addWidget(qtw.QLabel('Kernel Size'), 0, 1)
        self.spinbox_dilateErode = qtw.QSpinBox()
        self.spinbox_dilateErode.setRange(-99, 99)
        self.spinbox_dilateErode.setToolTip('Positive = dilate (grow the mask); Negative = erode (shrink it)')
        layout_dilate.addWidget(self.spinbox_dilateErode, 0, 2)

        # Directional (item: user request) - restricts the signed Kernel
        # Size step above to grow/shrink from just one side instead of
        # uniformly all around, same "Directional" + Angle convention as
        # Edge Detection above (see io.dilate_erode_mask's own `direction`
        # argument) - Opening/Closing stay isotropic (no direction to pick,
        # same as Edge Detection's own Directional only affecting its own
        # one erosion step, not anything else in the box).
        self.checkbox_dilateDirectional = qtw.QCheckBox('Directional')
        self.checkbox_dilateDirectional.setToolTip(
            'Grow/shrink the mask from just one side (the angle below) '
            'instead of uniformly all around - applies to Kernel Size only, '
            'not Opening/Closing.')
        layout_dilate.addWidget(self.checkbox_dilateDirectional, 1, 0)
        layout_dilate.addWidget(qtw.QLabel('Angle (°)'), 1, 1)
        self.spinbox_dilateDirection = qtw.QDoubleSpinBox()
        self.spinbox_dilateDirection.setRange(-360, 360)
        self.spinbox_dilateDirection.setSingleStep(5)
        self.spinbox_dilateDirection.setDisabled(True)
        layout_dilate.addWidget(self.spinbox_dilateDirection, 1, 2)
        self.checkbox_dilateDirectional.stateChanged.connect(
            lambda: self.spinbox_dilateDirection.setEnabled(self.checkbox_dilateDirectional.isChecked()))

        # Opening (erode then dilate) and Closing (dilate then erode) -
        # unlike the signed Dilate/Erode control above, these don't grow or
        # shrink the mask overall; they clean it up instead (see
        # io.open_mask/io.close_mask), so a single unsigned kernel size
        # each is enough - 0 (the default) is a no-op. Applied in
        # _effective_mask right after Dilate/Erode, in Opening-then-Closing
        # order: Opening first strips small bright specks/thin protrusions
        # before Closing fills small dark holes/gaps, so Closing doesn't
        # end up bridging noise Opening would otherwise have removed.
        layout_dilate.addWidget(qtw.QLabel('Opening Kernel Size'), 2, 1)
        self.spinbox_openKernel = qtw.QSpinBox()
        self.spinbox_openKernel.setRange(0, 99)
        self.spinbox_openKernel.setToolTip(
            'Erode then dilate by this much (0 = off) - removes small bright '
            "specks/thin protrusions from the mask's edge without changing its "
            'overall size.')
        layout_dilate.addWidget(self.spinbox_openKernel, 2, 2)

        layout_dilate.addWidget(qtw.QLabel('Closing Kernel Size'), 3, 1)
        self.spinbox_closeKernel = qtw.QSpinBox()
        self.spinbox_closeKernel.setRange(0, 99)
        self.spinbox_closeKernel.setToolTip(
            'Dilate then erode by this much (0 = off) - fills small dark holes/'
            "gaps inside the mask without changing its overall size.")
        layout_dilate.addWidget(self.spinbox_closeKernel, 3, 2)

        # No per-box scope here either - see the Edge Detection box's own
        # comment above.
        for signal in (self.checkbox_dilateErode.stateChanged, self.spinbox_dilateErode.valueChanged,
                       self.spinbox_openKernel.valueChanged, self.spinbox_closeKernel.valueChanged,
                       self.checkbox_dilateDirectional.stateChanged, self.spinbox_dilateDirection.valueChanged):
            signal.connect(lambda *_: self._on_segment_widgets_changed())

        # Threshold, if this tab has one.
        if box_thresh is not None:
            grid_boxes.addWidget(box_thresh)

        # Edge Detection directly above Dilate/Erode - both apply to the
        # same mask, in that order (see _effective_mask), so reads more
        # naturally grouped together.
        grid_boxes.addWidget(box_edge)
        grid_boxes.addWidget(box_dilate)

        #%% mesh - live preview only, like Edge Detection above, but the
        # selected cells (not just enabled/angle/cell size) are themselves
        # part of what's previewed/returned - see class docstring and
        # get_mesh_settings(). No main-tab equivalent to sync against (mesh
        # only makes sense relative to one specific object's mask), so this
        # dialog is the only place it's ever edited.
        #
        # Item 2: a segment can hold several independent meshes (its own
        # differently angled/sized/positioned grid and cell selection each) -
        # self.list_meshes picks which one the controls below show/edit;
        # each row's own inline checkbox is that mesh's "enabled" (whether it
        # actually restricts extraction, independent of just being present in
        # the list). When more than one mesh is enabled, the final kept
        # region is their INTERSECTION (see _redraw_mesh_overlay) - each
        # additional enabled mesh narrows the selection further, letting one
        # mesh's selection be refined by another rather than replaced by it.
        box_mesh = qtw.QGroupBox('Mesh (restrict extraction to selected cell(s))')
        grid_boxes.addWidget(box_mesh)
        layout_mesh = qtw.QVBoxLayout()
        box_mesh.setLayout(layout_mesh)

        row_m0 = qtw.QHBoxLayout()
        layout_mesh.addLayout(row_m0)
        row_m0.addWidget(qtw.QLabel('Meshes (checked = enabled; when several are checked, the '
                                    'kept region is their intersection):'))
        self.label_meshCount = qtw.QLabel()
        row_m0.addWidget(self.label_meshCount)
        row_m0.addStretch(1)

        self.list_meshes = qtw.QListWidget()
        self.list_meshes.setFixedHeight(84)
        self.list_meshes.setToolTip(
            "Select a mesh to edit its own Angle/Cell Size/Lines Only/Fixed "
            "settings and cell selection below - click a checkbox to enable/"
            "disable that mesh without selecting it.")
        self.list_meshes.currentRowChanged.connect(self._on_mesh_selected)
        self.list_meshes.itemChanged.connect(self._on_mesh_item_changed)
        layout_mesh.addWidget(self.list_meshes)

        row_m0b = qtw.QHBoxLayout()
        layout_mesh.addLayout(row_m0b)
        row_m0b.addStretch(1)
        self.button_meshAdd = qtw.QPushButton('Add Mesh')
        self.button_meshAdd.setToolTip('Add a new, independent mesh to this segment')
        self.button_meshAdd.clicked.connect(self._add_mesh)
        row_m0b.addWidget(self.button_meshAdd)
        self.button_meshDelete = qtw.QPushButton('Delete Mesh')
        self.button_meshDelete.setToolTip('Remove the selected mesh from this segment')
        self.button_meshDelete.clicked.connect(self._delete_mesh)
        row_m0b.addWidget(self.button_meshDelete)

        # The active mesh's own geometry/anchoring controls, in one grid -
        # same "line every row's fields up in columns" idea as Edge
        # Detection/Dilate-Erode above, adapted to this box's own mix of
        # paired fields, a checkbox+button row, and a labeled radio pair.
        grid_mesh = qtw.QGridLayout()
        grid_mesh.setColumnStretch(4, 1)
        layout_mesh.addLayout(grid_mesh)

        grid_mesh.addWidget(qtw.QLabel('Angle (°)'), 0, 0)
        self.spinbox_meshAngle = qtw.QDoubleSpinBox()
        # +-180 (item 2) - was 0-179.9 (a rotation is only unique mod 180 for
        # an unoriented grid, but the user asked for the full +-180 range,
        # so honor that literally rather than silently wrapping it; +-179.9
        # instead of +-180 avoids the 180/-180 seam being an ambiguous
        # duplicate value at the very ends of the range).
        self.spinbox_meshAngle.setRange(-179.9, 179.9)
        self.spinbox_meshAngle.setSingleStep(5)
        self.spinbox_meshAngle.setToolTip('Grid rotation relative to horizontal.')
        grid_mesh.addWidget(self.spinbox_meshAngle, 0, 1)
        grid_mesh.addWidget(qtw.QLabel('Cell Size (px)'), 0, 2)
        self.spinbox_meshCellSize = qtw.QSpinBox()
        self.spinbox_meshCellSize.setRange(1, 9999)
        self.spinbox_meshCellSize.setValue(20)
        grid_mesh.addWidget(self.spinbox_meshCellSize, 0, 3)

        # Item 3: "Fixed Mesh" (renamed from "Center on Initial Mask") - see
        # _mesh_fields'/_on_mesh_fixed_toggled's own docstrings for why the
        # anchor is now a captured (x, y) 'origin' rather than a flag
        # recomputed from the initial mask on every redraw. "Center Grid
        # (Click)" is the alternative, manual way to set that same origin -
        # see _arm_mesh_center_pick.
        self.checkbox_meshFixed = qtw.QCheckBox('Fixed Mesh')
        self.checkbox_meshFixed.setToolTip(
            "Anchor this mesh's grid to one fixed point - by default, "
            "captured now from the object's centroid on this segment's "
            "initial (pre-edit) mask - instead of recentering on wherever "
            "the mask has been edited to since. Use \"Center Grid (Click)\" "
            "to instead pick that point yourself.")
        self.checkbox_meshFixed.stateChanged.connect(self._on_mesh_fixed_toggled)
        grid_mesh.addWidget(self.checkbox_meshFixed, 1, 0)
        self.button_meshCenterClick = qtw.QPushButton('Center Grid (Click)')
        self.button_meshCenterClick.setCheckable(True)
        self.button_meshCenterClick.setToolTip(
            'Click this, then click a point on the canvas to move this '
            "mesh's fixed grid there (also turns on Fixed Mesh).")
        self.button_meshCenterClick.clicked.connect(self._arm_mesh_center_pick)
        grid_mesh.addWidget(self.button_meshCenterClick, 1, 1, 1, 2)

        # Item 4: while NOT "Fixed Mesh" (the grid recentering every frame),
        # which mask that per-frame centroid comes from - the object's
        # live/edited mask (the old, and still default, behavior) or its
        # initial/pre-edit one (same source "Fixed Mesh" itself captures its
        # one-time origin from - see _mesh_origin_for_mesh). Belongs to the
        # active mesh, same as every other control in this box.
        grid_mesh.addWidget(qtw.QLabel('Center on:'), 2, 0)
        self.radio_meshCenterEdited = qtw.QRadioButton('Edited Mask')
        self.radio_meshCenterEdited.setChecked(True)
        self.radio_meshCenterEdited.setToolTip(
            "While not Fixed, recenter this mesh's grid every frame on the "
            "object's current, live-edited mask.")
        grid_mesh.addWidget(self.radio_meshCenterEdited, 2, 1)
        self.radio_meshCenterInitial = qtw.QRadioButton('Initial Mask')
        self.radio_meshCenterInitial.setToolTip(
            "While not Fixed, recenter this mesh's grid every frame on the "
            "object's initial (pre-edit) mask instead - keeps it tracking "
            "the original tracked/segmented position even as the live mask "
            "is painted/grown/shrunk away from it.")
        grid_mesh.addWidget(self.radio_meshCenterInitial, 2, 2)
        self._mesh_center_basis_group = qtw.QButtonGroup(self)
        self._mesh_center_basis_group.addButton(self.radio_meshCenterEdited)
        self._mesh_center_basis_group.addButton(self.radio_meshCenterInitial)
        for radio in (self.radio_meshCenterEdited, self.radio_meshCenterInitial):
            radio.toggled.connect(lambda checked: checked and self._on_mesh_widgets_changed())

        self.checkbox_meshLinesOnly = qtw.QCheckBox('Lines Only')
        self.checkbox_meshLinesOnly.setToolTip(
            "Restrict to full-width stripes along the grid's rotated angle "
            'instead of individual square cells - a selected cell keeps its '
            'whole row of cells, not just that one. Switching this clears '
            'the current selection (a square-cell and a stripe selection '
            "aren't interchangeable).")
        self.checkbox_meshLinesOnly.stateChanged.connect(self._on_mesh_lines_only_toggled)
        grid_mesh.addWidget(self.checkbox_meshLinesOnly, 3, 0)

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
        # the object however much it's moved/tracked by then (unless Fixed
        # Mesh is on for this mesh).
        for signal in (self.spinbox_meshAngle.valueChanged, self.spinbox_meshCellSize.valueChanged):
            signal.connect(lambda *_: self._on_mesh_widgets_changed())

        # Soaks up any leftover height in the left panel (e.g. once the
        # dialog is maximized) as blank space at the bottom, instead of the
        # group boxes above stretching to fill it - grid_boxes itself
        # already has stretch-factor 1 within left_layout (see its own
        # comment above), but with nothing here to claim that extra space
        # a Qt layout distributes it across the boxes/their inner widgets
        # instead of leaving it empty.
        grid_boxes.addStretch(1)

        # Full dialog width, below both columns - the window's own true
        # bottom-left/bottom-right corners (not just the right column's own
        # bottom, which sits above the left panel's option boxes when
        # they're taller - see class docstring's layout note): a small
        # legend for the two mask colors always on screen (item 3/5 -
        # initial vs. tuned mask) and a live cursor position readout at
        # bottom-left (via the main tabs' own (hidden) navigation toolbar's
        # hover mechanism - see _mirror_cursor_coords_to_label - just
        # surfaced here instead of the main window's status bar, since this
        # dialog is its own separate top-level window), Save&&Close/Cancel
        # at bottom-right.
        row_buttons = qtw.QHBoxLayout()
        outer_layout.addLayout(row_buttons)
        status_col = qtw.QVBoxLayout()
        self.label_colorLegend = qtw.QLabel(
            'Mask colors: initial (pre-edit) mask = brown (same as a selected Mesh cell, '
            'just fainter), current/tuned mask = blue')
        self.label_colorLegend.setStyleSheet('color: gray; font-size: 9pt;')
        status_col.addWidget(self.label_colorLegend)
        self.label_cursorCoords = qtw.QLabel('')
        self.label_cursorCoords.setStyleSheet('font-size: 9pt;')
        # No wrapping, and a fixed height matching exactly one line - a
        # wrapped second line (however briefly, as the text's own width
        # fluctuates) would change this label's height just as visibly as
        # its width changing would. Combined with the fixed width below and
        # ax.format_coord's own short, fixed-format "x=.., y=.." (no pixel
        # value - see its own comment above), this label's size never
        # changes at all as the mouse moves - otherwise row_buttons (which
        # spans the full dialog width, in outer_layout) reflows on every
        # hover, and since Qt won't let the window shrink below its layout's
        # own minimum size, that was pulling the whole dialog (and the
        # canvas's own share of it) very slightly bigger/smaller on every
        # hover - a visible canvas "jitter".
        self.label_cursorCoords.setWordWrap(False)
        self.label_cursorCoords.setFixedHeight(self.label_cursorCoords.fontMetrics().height())
        self.label_cursorCoords.setMinimumWidth(
            self.label_cursorCoords.fontMetrics().horizontalAdvance('x=-9999.9, y=-9999.9'))
        status_col.addWidget(self.label_cursorCoords)
        row_buttons.addLayout(status_col)
        self._mirror_cursor_coords_to_label()
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

    def _on_mask_alpha_changed(self):
        """Either mask-opacity spinbox changed: update the live alpha and
        redraw both layers - img_initial_mask directly (its own data is
        only ever refreshed by _redraw_frame_content/here, not _redraw_mask,
        since it's frame-driven, not edit-driven), img_mask via the normal
        _redraw_mask() path."""
        self._initial_mask_alpha = self.spinbox_initialMaskAlpha.value() / 100
        self._tuned_mask_alpha = self.spinbox_tunedMaskAlpha.value() / 100
        self.img_initial_mask.set_data(self._initial_mask_rgba(self._original_stack[self.frame]))
        self._redraw_mask()

    def _on_show_initial_mask_toggled(self):
        """"Show Initial Mask" checkbox: just a visibility toggle on
        img_initial_mask - the layer itself keeps being updated per-frame
        underneath regardless (see _redraw_frame_content), so re-checking
        it immediately shows the correct frame's initial mask again."""
        self.img_initial_mask.set_visible(self.checkbox_showInitialMask.isChecked())
        self._blit_mask_display()

    def _mirror_cursor_coords_to_label(self):
        """self.toolbar's own hover coordinate/pixel-value readout (its
        locLabel) is invisible along with the rest of it (see .hide() in
        __init__) - mirror its set_message() calls into self.label_
        cursorCoords instead, same trick as TabBase.
        _mirror_toolbar_coords_to_statusbar (base_tab.py), just targeting a
        label in THIS dialog's own bottom-left corner rather than the main
        window's status bar, since this dialog is its own separate
        top-level window, not embedded in the main window."""
        orig_set_message = self.toolbar.set_message
        def _set_message(s, _orig=orig_set_message):
            _orig(s)
            self.label_cursorCoords.setText(s)
        self.toolbar.set_message = _set_message

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
        rgba[mask] = (*_TUNED_MASK_RGB, self._tuned_mask_alpha)
        return rgba

    def _initial_mask_rgba(self, mask):
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = (*_MASK_RGB, self._initial_mask_alpha)
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
                direction = de['direction'] if de['directional'] else None
                mask = io.dilate_erode_mask(mask, de['kernel'], direction=direction)
            if de['open_kernel'] != 0:
                mask = io.open_mask(mask, de['open_kernel'])
            if de['close_kernel'] != 0:
                mask = io.close_mask(mask, de['close_kernel'])
        edge = seg['edge']
        if edge['enabled']:
            direction = edge['direction'] if edge['directional'] else None
            mask = io.erode_mask_edge(mask, edge['kernel'], direction=direction, revert=edge['revert'])
        # Item 1: pixels set by a manual tool keep that exact value, instead
        # of whatever Dilate/Erode/Edge Detection just did to them above -
        # see checkbox_protectManualEdits/_paint_pixel/_on_release/
        # _grow_shrink for where self._manual_mask_stack gets marked.
        if self.checkbox_protectManualEdits.isChecked():
            manual = self._manual_mask_stack[frame]
            if manual.any():
                mask = np.where(manual, base, mask)
        return mask

    def _redraw_mask(self):
        """Recompute and show the mask overlay (+ Mesh/tilt-axis overlays
        on top) for the current frame - called on every single mouse-move
        tick while pixel-painting (_paint_pixel/_on_motion), so the actual
        screen update is blitted (_blit_mask_display) rather than a full
        canvas.draw_idle(): a full redraw here (re-running constrained_
        layout, repainting every axis) on every one of those made painting
        feel sluggish for anything but a tiny mask."""
        self.img_mask.set_data(self._mask_rgba(self._effective_mask(self.frame)))
        self._redraw_mesh_overlay()
        self._redraw_tilt_axis_overlay()
        self._blit_mask_display()

    def _blit_mask_display(self):
        """Blit the background image, mask overlay, and Mesh/tilt-axis
        overlays onto the canvas - see TabBase._blit_canvas (duplicated
        here, in miniature, since this dialog doesn't inherit TabBase).
        img_bg is included even though most _redraw_mask callers (paint/
        mesh/tilt-axis edits) never actually change it - it's cheap to
        redraw and this way the same cache also serves _redraw_frame_
        content's per-frame calls (where it DOES change), without a
        separate code path. Mesh/tilt-axis overlays are freshly recreated
        artists every call (see _redraw_mesh_overlay/_redraw_tilt_axis_
        overlay - a contour set can't just have its data updated), so
        they're read fresh here rather than cached, the same convention
        as the main tabs' own per-frame ROI-rectangle overlays."""
        dynamic = [self.img_bg, self.img_initial_mask, self.img_mask] + self._mesh_grid_artists
        if self._mesh_cell_artist is not None:
            dynamic.append(self._mesh_cell_artist)
        if self._tilt_axis_line_artist is not None:
            dynamic.append(self._tilt_axis_line_artist)
        self._blit_canvas(self.canvas, self.figure, '_mask_frame_bg', dynamic,
                          hide_for_background=[self.img_bg, self.img_initial_mask, self.img_mask])

    def _blit_canvas(self, canvas, figure, bg_attr, artists,
                     hide_for_background=(), titles_for_background=()):
        """See TabBase._blit_canvas (ui_tabs/base_tab.py) - identical
        logic, duplicated here since MaskEditDialog is a QDialog, not a
        TabBase subclass."""
        if getattr(self, bg_attr) is None:
            prev_visible = [a.get_visible() for a in hide_for_background]
            for a in hide_for_background:
                a.set_visible(False)
            prev_titles = [ax.get_title() for ax in titles_for_background]
            for ax in titles_for_background:
                ax.set_title('')

            canvas.draw()
            setattr(self, bg_attr, canvas.copy_from_bbox(figure.bbox))

            for a, v in zip(hide_for_background, prev_visible):
                a.set_visible(v)
            for ax, t in zip(titles_for_background, prev_titles):
                ax.set_title(t)

        canvas.restore_region(getattr(self, bg_attr))
        for artist in artists:
            artist.axes.draw_artist(artist)
        canvas.blit(figure.bbox)

    #%% segments
    def _load_segment_into_widgets(self):
        """Populate the Dilate/Erode and Edge Detection widgets, and the
        Mesh list/active-mesh widgets, from
        self._segments[self._current_segment_idx] - called on init and
        whenever _sync_current_segment finds the current frame now belongs
        to a different segment than before. Signals blocked throughout so
        this doesn't loop back into _on_segment_widgets_changed and
        overwrite the very segment it's loading from."""
        seg = self._segments[self._current_segment_idx]
        de, edge = seg['dilate_erode'], seg['edge']
        widgets = (self.checkbox_dilateErode, self.spinbox_dilateErode, self.spinbox_openKernel,
                  self.spinbox_closeKernel, self.checkbox_dilateDirectional, self.spinbox_dilateDirection,
                  self.checkbox_edgeOnly,
                  self.spinbox_edgeKernel, self.checkbox_edgeDirectional, self.spinbox_edgeDirection,
                  self.checkbox_revertMask)
        for wid in widgets:
            wid.blockSignals(True)
        self.checkbox_dilateErode.setChecked(de['enabled'])
        self.spinbox_dilateErode.setValue(de['kernel'])
        self.spinbox_openKernel.setValue(de['open_kernel'])
        self.spinbox_closeKernel.setValue(de['close_kernel'])
        self.checkbox_dilateDirectional.setChecked(de['directional'])
        self.spinbox_dilateDirection.setValue(de['direction'])
        self.spinbox_dilateDirection.setEnabled(de['directional'])
        self.checkbox_edgeOnly.setChecked(edge['enabled'])
        self.spinbox_edgeKernel.setValue(edge['kernel'])
        self.checkbox_edgeDirectional.setChecked(edge['directional'])
        self.spinbox_edgeDirection.setValue(edge['direction'])
        self.spinbox_edgeDirection.setEnabled(edge['directional'])
        self.checkbox_revertMask.setChecked(edge['revert'])
        for wid in widgets:
            wid.blockSignals(False)
        # New segment - start showing its first mesh (if any) again, rather
        # than whatever index happened to be selected in the previous one.
        self._active_mesh_idx = 0
        self._refresh_mesh_list_widget()
        self._load_active_mesh_into_widgets()

    def _isolate_current_frame_as_segment(self):
        """Split the segment currently covering self.frame down to exactly
        that one frame, inheriting its Dilate/Erode/Edge Detection/Mesh
        settings (deep-copied, not blank defaults - unlike a manual "Split
        Here") into the pieces before/after it, so the rest of the
        original range is left exactly as it was. Called before writing an
        edit while Edit Scope is "Single Frame" (see
        _isolate_if_single_frame_scope). A no-op if the current segment is
        already exactly this one frame."""
        idx = self._current_segment_idx
        seg = self._segments[idx]
        if seg['start'] == seg['end'] == self.frame:
            return
        before = None
        if self.frame > seg['start']:
            before = {'start': seg['start'], 'end': self.frame - 1,
                     'dilate_erode': copy.deepcopy(seg['dilate_erode']),
                     'edge': copy.deepcopy(seg['edge']),
                     'meshes': copy.deepcopy(seg['meshes'])}
        after = None
        if self.frame < seg['end']:
            after = {'start': self.frame + 1, 'end': seg['end'],
                    'dilate_erode': copy.deepcopy(seg['dilate_erode']),
                    'edge': copy.deepcopy(seg['edge']),
                    'meshes': copy.deepcopy(seg['meshes'])}
        seg['start'] = seg['end'] = self.frame
        insert_at = idx
        if before is not None:
            self._segments[idx] = before
            self._segments.insert(idx + 1, seg)
            insert_at = idx + 1
        if after is not None:
            self._segments.insert(insert_at + 1, after)
        self._current_segment_idx = insert_at

    def _reset_current_frame_to_default(self):
        """"Reset Frame": discard every edit to just this frame - both its
        painted mask (back to the original tracked/segmented result, same
        as the old "Reset This Frame" button this folds in - undo-able,
        see _push_undo) and, isolating it into its own segment first if it
        isn't already one (see _isolate_current_frame_as_segment), its
        Dilate/Erode/Edge Detection/Mesh settings back to plain defaults -
        without touching neighboring frames."""
        self._push_undo(self.frame)
        self.mask_stack[self.frame] = self._original_stack[self.frame].copy()
        self._manual_mask_stack[self.frame] = False
        self._isolate_current_frame_as_segment()
        seg = self._segments[self._current_segment_idx]
        seg['dilate_erode'] = _dilate_erode_fields(None)
        seg['edge'] = _edge_fields(None)
        seg['meshes'] = []
        self._load_segment_into_widgets()
        self._redraw_mask()
        self._update_segment_ui()

    def _reset_segments_to_single_default(self):
        """Collapse self._segments back to one plain-default segment
        spanning the whole stack, reload the widgets/segment UI from it -
        the segment-list half of "Reset to Tracking" (which also resets
        self.mask_stack itself - see _reset_to_tracking). Does not redraw -
        the caller does that once, after its own other state changes."""
        self._segments = [{
            'start': 0, 'end': self.n_frames - 1,
            'dilate_erode': _dilate_erode_fields(None),
            'edge': _edge_fields(None),
            'meshes': [],
        }]
        self._current_segment_idx = 0
        self._load_segment_into_widgets()
        self._update_segment_ui()

    def _isolate_if_single_frame_scope(self):
        """Isolate the current frame into its own segment (see
        _isolate_current_frame_as_segment) when Edit Scope is "Single
        Frame" - shared by every Dilate/Erode/Edge Detection/Mesh-editing
        method (_write_widgets_to_current_segment, the _mesh_cells setter,
        and every per-mesh mutator below) so they all honor Edit Scope the
        same way. A no-op under "Segment" scope, or if the current segment
        is already exactly this one frame."""
        if self.radio_scopeFrame.isChecked():
            self._isolate_current_frame_as_segment()

    def _write_widgets_to_current_segment(self):
        """The inverse of _load_segment_into_widgets - called whenever a
        Dilate/Erode/Edge Detection control changes, so the currently-
        active segment's own stored Dilate/Erode/Edge settings (not just the
        live widget state) reflect the edit. If Edit Scope is "Single
        Frame", isolates the current frame into its own segment first (see
        _isolate_if_single_frame_scope) so the edit lands on just this
        frame, not the whole range. Mesh is handled separately (item 2 -
        several independent meshes per segment) - see
        _on_mesh_widgets_changed/_on_mesh_fixed_toggled/_add_mesh/
        _delete_mesh/_on_mesh_enabled_toggled/the _mesh_cells property."""
        self._isolate_if_single_frame_scope()
        seg = self._segments[self._current_segment_idx]
        seg['dilate_erode'] = {'enabled': self.checkbox_dilateErode.isChecked(),
                               'kernel': self.spinbox_dilateErode.value(),
                               'open_kernel': self.spinbox_openKernel.value(),
                               'close_kernel': self.spinbox_closeKernel.value(),
                               'directional': self.checkbox_dilateDirectional.isChecked(),
                               'direction': self.spinbox_dilateDirection.value()}
        seg['edge'] = {'enabled': self.checkbox_edgeOnly.isChecked(),
                       'kernel': self.spinbox_edgeKernel.value(),
                       'directional': self.checkbox_edgeDirectional.isChecked(),
                       'direction': self.spinbox_edgeDirection.value(),
                       'revert': self.checkbox_revertMask.isChecked()}

    def _on_segment_widgets_changed(self):
        """Any Dilate/Erode/Edge Detection/Mesh control changed (except
        mesh cell clicks/Lines Only, which go through the _mesh_cells
        property/_on_mesh_lines_only_toggled instead)."""
        self._write_widgets_to_current_segment()
        self._update_segment_ui()
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
        second half starts back at plain defaults for Dilate/Erode/Edge
        Detection/Mesh (all disabled) rather than inheriting a copy of the
        first half's settings - a segment nobody has actually configured yet
        should read as "untouched", not as a hidden duplicate of whatever
        segment it was split off from. Use "Reset Segment" to put an
        already-configured segment back to this same state. A no-op if
        self.frame is already this segment's own start (nothing to
        split)."""
        idx = self._current_segment_idx
        seg = self._segments[idx]
        if self.frame <= seg['start']:
            return
        new_seg = {
            'start': self.frame, 'end': seg['end'],
            'dilate_erode': _dilate_erode_fields(None),
            'edge': _edge_fields(None),
            'meshes': [],
        }
        seg['end'] = self.frame - 1
        self._segments.insert(idx + 1, new_seg)
        self._current_segment_idx = idx + 1
        self._load_segment_into_widgets()
        self._redraw_mask()
        self._update_segment_ui()

    def _reset_current_segment_to_default(self):
        """"Reset Segment": put the current segment's Dilate/Erode/Edge
        Detection/Mesh settings back to plain defaults (all disabled) - the
        same state a freshly-split, never-touched segment starts in (see
        _split_segment_here) - without changing its frame range or touching
        any other segment. See also "Reset Frame"
        (_reset_current_frame_to_default), which does the same for just
        the current frame."""
        seg = self._segments[self._current_segment_idx]
        seg['dilate_erode'] = _dilate_erode_fields(None)
        seg['edge'] = _edge_fields(None)
        seg['meshes'] = []
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

    @property
    def _active_mesh(self):
        """The mesh dict currently shown/edited by the Mesh box's widgets -
        self._segments[self._current_segment_idx]['meshes'][self.
        _active_mesh_idx] (item 2), or None if the current segment has no
        meshes at all yet."""
        meshes = self._segments[self._current_segment_idx]['meshes']
        if not meshes:
            return None
        return meshes[min(self._active_mesh_idx, len(meshes) - 1)]

    @property
    def _mesh_cells(self):
        """The active mesh's selected cells, as a set of (i, j) tuples -
        backed by its own ['cells'] (a plain list-of-lists, JSON/dataframe-
        friendly), converted on the fly. Empty (not an error) if there's no
        active mesh. Returns a fresh set each read, so mutating it
        (.add()/.discard()/.clear()) does NOT persist - reassign the
        property instead (`self._mesh_cells = cells`) after mutating a
        local copy, same as _toggle_mesh_cell/_clear_mesh_selection do."""
        mesh = self._active_mesh
        return set(tuple(c) for c in mesh['cells']) if mesh is not None else set()

    @_mesh_cells.setter
    def _mesh_cells(self, value):
        self._isolate_if_single_frame_scope()
        mesh = self._active_mesh
        if mesh is not None:
            mesh['cells'] = [list(c) for c in value]

    def _mesh_origin_for_mesh(self, mesh, frame):
        """The centroid `mesh`'s grid overlay/click-toggling/extraction are
        built relative to, for `frame`.

        When `mesh['fixed']` is set (item 3, "Fixed Mesh") and it already
        has a captured `mesh['origin']`, that fixed (x, y) point is used,
        on every frame, regardless of where the object has since moved to -
        captured once (see _on_mesh_fixed_toggled/_arm_mesh_center_pick),
        not recomputed here. This is also what tab_sam2.py/
        tab_tracking_cv2.py's own apply_edge_mask must do with the same
        mesh dict at real extraction time for "Fixed Mesh" to actually take
        effect there too, not just in this dialog's preview.

        Otherwise (dynamic, per-frame tracking), `mesh['center_basis']`
        (item 4) picks which mask that per-frame centroid comes from -
        'edited' (self.mask_stack, the default - the grid recenters onto
        wherever the object has been edited to since) or 'initial' (the
        segment's initial/pre-edit mask - `_default_stack` if the caller
        gave one, else `_original_stack` - so the grid keeps tracking the
        object's ORIGINAL tracked/segmented position on this frame even as
        the live mask is painted/grown/shrunk away from it)."""
        if mesh is not None and mesh.get('fixed') and mesh.get('origin') is not None:
            return tuple(mesh['origin'])
        if mesh is not None and mesh.get('center_basis') == 'initial':
            source = self._default_stack if self._default_stack is not None else self._original_stack
            return io.mask_centroid(source[frame])
        return io.mask_centroid(self.mask_stack[frame])

    def _mesh_origin_for_frame(self, frame):
        """_mesh_origin_for_mesh for the ACTIVE mesh - used by the grid
        overlay/click-toggling, which always edit whichever mesh is
        currently selected in self.list_meshes."""
        return self._mesh_origin_for_mesh(self._active_mesh, frame)

    def _mesh_origin(self):
        """_mesh_origin_for_frame for the currently-displayed frame."""
        return self._mesh_origin_for_frame(self.frame)

    def _refresh_mesh_list_widget(self):
        """Rebuild self.list_meshes's rows from the current segment's own
        ['meshes'] list - called whenever that list's length/order changes
        (segment switch, Add/Delete Mesh) - see _load_segment_into_widgets/
        _add_mesh/_delete_mesh. Each row is a plain, natively-checkable
        QListWidgetItem (its check state is that mesh's 'enabled',
        independent of which row is SELECTED/being edited - self.
        _active_mesh_idx) rather than a custom row widget - a widget filling
        the whole row (an earlier version of this used one, via
        setItemWidget) swallows every mouse click before the list view ever
        sees it, so clicking a row never actually selected it; a native
        checkable item's own checkbox indicator is the one small area that
        does NOT change selection on click, while the rest of the row still
        does, exactly as a checkable list is expected to behave."""
        seg = self._segments[self._current_segment_idx]
        self.list_meshes.blockSignals(True)
        self.list_meshes.clear()
        for i, mesh in enumerate(seg['meshes']):
            item = qtw.QListWidgetItem(f'Mesh {i + 1}')
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if mesh['enabled'] else Qt.Unchecked)
            self.list_meshes.addItem(item)
        if seg['meshes']:
            self.list_meshes.setCurrentRow(min(self._active_mesh_idx, len(seg['meshes']) - 1))
        self.list_meshes.blockSignals(False)
        self.button_meshDelete.setEnabled(bool(seg['meshes']))
        self.label_meshCount.setText(f'({len(seg["meshes"])})')

    def _load_active_mesh_into_widgets(self):
        """Populate the Angle/Cell Size/Lines Only/Fixed Mesh/Center-on
        widgets (and the cell-count label) from the active mesh - or, if the
        current segment has none yet, just disable them (nothing to edit)."""
        mesh = self._active_mesh
        widgets = (self.spinbox_meshAngle, self.spinbox_meshCellSize,
                  self.checkbox_meshLinesOnly, self.checkbox_meshFixed,
                  self.radio_meshCenterEdited, self.radio_meshCenterInitial)
        for wid in widgets:
            wid.blockSignals(True)
        editable = mesh is not None
        for wid in widgets + (self.button_meshCenterClick, self.button_meshClear):
            wid.setEnabled(editable)
        if mesh is not None:
            self.spinbox_meshAngle.setValue(mesh['angle'])
            self.spinbox_meshCellSize.setValue(mesh['cell_size'])
            self.checkbox_meshLinesOnly.setChecked(mesh['lines_only'])
            self.checkbox_meshFixed.setChecked(mesh['fixed'])
            if mesh['center_basis'] == 'initial':
                self.radio_meshCenterInitial.setChecked(True)
            else:
                self.radio_meshCenterEdited.setChecked(True)
        for wid in widgets:
            wid.blockSignals(False)
        self._update_mesh_cell_label()

    def _on_mesh_selected(self, row):
        """self.list_meshes' currentRowChanged: switch which mesh the
        widgets below show/edit - a row deselecting entirely (row < 0, e.g.
        the list just got cleared/rebuilt with nothing left) is a no-op,
        not "select nothing"."""
        if row < 0:
            return
        self._active_mesh_idx = row
        self._load_active_mesh_into_widgets()
        self._redraw_mask()

    def _add_mesh(self):
        """"Add Mesh": append a new, independent, plain-default mesh to the
        current segment and select it - see class docstring/item 2. Honors
        Edit Scope like every other Mesh edit (see
        _isolate_if_single_frame_scope)."""
        self._isolate_if_single_frame_scope()
        seg = self._segments[self._current_segment_idx]
        seg['meshes'].append(_mesh_fields(None))
        self._active_mesh_idx = len(seg['meshes']) - 1
        self._refresh_mesh_list_widget()
        self._load_active_mesh_into_widgets()
        self._update_segment_ui()
        self._redraw_mask()

    def _delete_mesh(self):
        """"Delete Mesh": remove the active mesh from the current segment -
        a no-op if there isn't one."""
        self._isolate_if_single_frame_scope()
        seg = self._segments[self._current_segment_idx]
        if not seg['meshes']:
            return
        idx = min(self._active_mesh_idx, len(seg['meshes']) - 1)
        del seg['meshes'][idx]
        self._active_mesh_idx = max(0, idx - 1)
        self._refresh_mesh_list_widget()
        self._load_active_mesh_into_widgets()
        self._update_segment_ui()
        self._redraw_mask()

    def _on_mesh_item_changed(self, item):
        """A mesh row's own native check state (its 'enabled') changed -
        by row index (self.list_meshes.row(item)), independent of
        self._active_mesh_idx (the row being EDITED, not necessarily the
        one just checked/unchecked). Guarded by blockSignals during
        _refresh_mesh_list_widget's own rebuild, so this only ever fires
        from an actual user click."""
        idx = self.list_meshes.row(item)
        seg_before = self._segments[self._current_segment_idx]
        if idx < 0 or idx >= len(seg_before['meshes']):
            return
        checked = item.checkState() == Qt.Checked
        self._isolate_if_single_frame_scope()
        seg = self._segments[self._current_segment_idx]
        seg['meshes'][idx]['enabled'] = checked
        self._update_segment_ui()
        self._redraw_mask()

    def _on_mesh_widgets_changed(self):
        """Angle/Cell Size/"Center on" changed for the active mesh."""
        self._isolate_if_single_frame_scope()
        mesh = self._active_mesh
        if mesh is None:
            return
        mesh['angle'] = self.spinbox_meshAngle.value()
        mesh['cell_size'] = self.spinbox_meshCellSize.value()
        mesh['center_basis'] = 'initial' if self.radio_meshCenterInitial.isChecked() else 'edited'
        self._update_segment_ui()
        self._redraw_mask()

    def _on_mesh_fixed_toggled(self):
        """Item 3: "Fixed Mesh" checked/unchecked for the active mesh. The
        first time it's checked with no origin captured yet, this captures
        one now - the centroid of the object's INITIAL (pre-edit) mask
        (`_default_stack` if the caller gave one, else `_original_stack`)
        on this SEGMENT's own start frame (its earliest frame, not
        necessarily the one on screen) - so re-checking later (after
        unchecking) doesn't silently drop a manually-clicked point (see
        _arm_mesh_center_pick) by recomputing over it."""
        self._isolate_if_single_frame_scope()
        mesh = self._active_mesh
        if mesh is None:
            return
        checked = self.checkbox_meshFixed.isChecked()
        mesh['fixed'] = checked
        if checked and mesh.get('origin') is None:
            seg = self._segments[self._current_segment_idx]
            source = self._default_stack if self._default_stack is not None else self._original_stack
            cx, cy = io.mask_centroid(source[seg['start']])
            mesh['origin'] = [float(cx), float(cy)]
        self._update_segment_ui()
        self._redraw_mask()

    def _arm_mesh_center_pick(self):
        """"Center Grid (Click)": arm/disarm picking a new fixed origin for
        the active mesh by clicking the canvas - see _on_press, which
        completes the pick (also resetting this button/the cursor) on the
        next canvas click while self._picking_mesh_center is True."""
        if self._active_mesh is None:
            self.button_meshCenterClick.setChecked(False)
            return
        self._picking_mesh_center = self.button_meshCenterClick.isChecked()
        if self._picking_mesh_center:
            self.canvas.setCursor(Qt.CrossCursor)
        else:
            self._apply_ribbon_cursor()

    def _rotated_coords(self, origin):
        """(rot_x, rot_y) continuous rotated-coordinate arrays for the
        active mesh's angle, centered on `origin` - a pure display concern
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

    def _redraw_mesh_overlay(self):
        """(Re)draw the active mesh's own grid lines (only the active one -
        drawing every mesh's grid at once would be unreadable clutter), plus
        one combined selected-cell highlight for the current segment's
        every ENABLED mesh that actually has cells picked - their
        INTERSECTION (item 2: each additional enabled mesh narrows the kept
        region further, rather than adding to it), matching what
        tab_sam2.py/tab_tracking_cv2.py's own apply_edge_mask computes at
        real extraction time."""
        for artist in self._mesh_grid_artists:
            artist.remove()
        self._mesh_grid_artists = []
        if self._mesh_cell_artist is not None:
            self._mesh_cell_artist.remove()
            self._mesh_cell_artist = None

        active = self._active_mesh
        if active is not None:
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

        # Delegate to io.mesh_restrict_mask - the same function real
        # extraction uses (tab_tracking_cv2.py/tab_sam2.py's own
        # apply_edge_mask) - rather than re-deriving the cell-membership
        # logic here, so "Lines Only" (or anything else about how cells
        # translate to kept pixels) can never drift out of sync between
        # this preview and the real thing.
        effective = self._effective_mask(self.frame)
        keep = None
        for mesh in self._segments[self._current_segment_idx]['meshes']:
            if not mesh['enabled'] or not mesh['cells']:
                continue
            origin_m = self._mesh_origin_for_mesh(mesh, self.frame)
            keep_m = io.mesh_restrict_mask(
                effective, mesh['angle'], mesh['cell_size'],
                set(tuple(c) for c in mesh['cells']), origin=origin_m, lines_only=mesh['lines_only'])
            keep = keep_m if keep is None else (keep & keep_m)
        if keep is not None:
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
        self._update_segment_ui()
        self._redraw_mask()

    def _on_mesh_lines_only_toggled(self):
        """A square-cell selection and a stripe selection aren't
        interchangeable (same (i, j) values, different meaning) - clear
        whatever was picked rather than silently reinterpreting it. Also
        writes the toggle itself back into the active mesh first, same as
        every other Mesh control - _clear_mesh_selection's own _redraw_mask
        alone won't do that."""
        self._isolate_if_single_frame_scope()
        mesh = self._active_mesh
        if mesh is not None:
            mesh['lines_only'] = self.checkbox_meshLinesOnly.isChecked()
        self._clear_mesh_selection()

    def _toggle_mesh_cell(self, event):
        """Toggle the active mesh's cell (or, with "Lines Only" checked, the
        whole stripe sharing the clicked cell's `cell_i`) under the cursor -
        always relative to the object's own position on whichever frame is
        currently displayed (see _mesh_origin). In "Lines Only" mode, a
        stripe is stored/toggled as a single representative (i, 0) entry in
        _mesh_cells rather than one entry per (i, j) pair actually on
        screen - io.mesh_restrict_mask's own `lines_only` flag is what
        makes that (i, 0) entry mean "every cell_i == i pixel" instead of
        just that one cell. Reads/mutates/reassigns _mesh_cells (a property
        backed by the active mesh - see its own docstring) rather than
        mutating it in place, since each read returns a fresh set."""
        if self._active_mesh is None:
            return
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
        self._update_segment_ui()
        self._redraw_mask()

    def get_mesh_settings(self):
        """The shared Segments timeline's own Mesh values, one entry per
        segment - the caller round-trips this into its own per-object
        dataframe column (there's no main-tab equivalent to sync against,
        unlike get_edge_settings()). Each entry's 'start'/'end' (inclusive
        frame indices) is what the caller resolves per-frame via
        io.segment_for_frame at extraction time (see
        tab_tracking_cv2.py/tab_sam2.py's own apply_edge_mask). Each
        segment's 'meshes' (item 2) is a LIST of independent mesh dicts, not
        one flat mesh - combined by intersection wherever more than one is
        enabled (see _redraw_mesh_overlay/apply_edge_mask). 'lines_only' -
        see io.mesh_restrict_mask - is what the caller must pass that same
        function alongside 'cells' for this to mean what it looked like in
        this dialog's own preview."""
        return {'segments': [{'start': s['start'], 'end': s['end'], 'meshes': s['meshes']}
                             for s in self._segments]}

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
        self._cancel_drag()
        self._threshold_undo_frame = None  # a new frame starts a fresh undo-coalescing group
        # The actual redraw (re-denoising the background - can be genuinely
        # slow for e.g. non-local-means - plus recomputing Dilate/Erode/Edge
        # Detection/Mesh) is coalesced onto a short timer rather than run
        # synchronously here: dragging the slider fires this once per pixel
        # of travel, and re-running that whole pipeline on every single one
        # of those intermediate values is what made dragging feel sluggish.
        # Restarting the timer on every call means only the LAST frame
        # value in a fast drag actually triggers the expensive work, a few
        # ms after the drag settles - a plain click/single step still feels
        # instant (the delay is imperceptibly short), while a fast drag no
        # longer stalls trying to redraw every frame it passes through.
        self._frame_redraw_timer.start()

    def _redraw_frame_content(self):
        """The actual per-frame redraw _on_frame_changed defers onto
        self._frame_redraw_timer (see its own comment) - re-denoises the
        background image, if any, then redraws the mask/mesh/tilt-axis
        overlay on top."""
        if self.bg_stack is not None:
            self.img_bg.set_data(self._bg_frame(self.frame))
        self.img_initial_mask.set_data(self._initial_mask_rgba(self._original_stack[self.frame]))
        self._redraw_mask()

    def _bg_frame(self, frame):
        """The displayed background image for `frame` - box_denoise's
        current method/parameter applied fresh on top of the caller's
        contrast-only bg_stack (see class docstring/__init__). The single
        place every background-image read in this dialog goes through."""
        return self.box_denoise.apply(self.bg_stack[frame])

    def _denoised_bg_stack(self):
        """The WHOLE bg_stack run through box_denoise's current method/
        parameter, cached (see _on_denoise_changed) - what
        _recompute_threshold_stack hands recompute_thresh_fn so the
        Threshold box's rebuild actually reflects the Denoise box's choice
        instead of always re-thresholding the raw, undenoised images (a
        past inconsistency - Denoise only ever affected what was DISPLAYED,
        never what got thresholded). None if this dialog has no bg_stack at
        all. Cached (not recomputed on every Threshold slider tick, unlike
        _bg_frame's own single-frame, always-fresh read) because denoising
        the FULL-RESOLUTION stack for every frame at once - unlike the
        Threshold box's own cheap, ROI-cropped "ROI Blur" - can be
        genuinely slow for a slower method (e.g. non-local means)."""
        if self.bg_stack is None:
            return None
        if self._denoised_threshold_stack is None:
            self._denoised_threshold_stack = np.stack(
                [self.box_denoise.apply(img) for img in self.bg_stack])
        return self._denoised_threshold_stack

    def _on_denoise_changed(self):
        """box_denoise's method/parameter changed: refresh just the
        current frame's displayed background (cheap - same reasoning as
        the main tab's own live Denoise preview), and drop the cached
        whole-stack denoise _denoised_bg_stack used for Threshold rebuilds -
        it's now stale, recomputed fresh (once) the next time a Threshold
        control changes.

        For a threshold-derived mask (recompute_thresh_fn given), also
        re-runs that Threshold rebuild right now (_threshold_live_update) -
        otherwise the DISPLAYED background updates immediately but the
        actual mask silently keeps reflecting whatever Denoise setting was
        active the last time a Threshold control itself was touched, until
        the user happens to nudge one - Denoise not visibly affecting the
        mask at all until then reads as "the threshold isn't using the
        denoised image", even though the very next Threshold tweak would
        have picked it up automatically."""
        self._denoised_threshold_stack = None
        if self.bg_stack is None:
            return
        self.img_bg.set_data(self._bg_frame(self.frame))
        if self.recompute_thresh_fn is not None:
            self._threshold_live_update()
        else:
            self._blit_mask_display()

    def _show_denoise_check_methods(self):
        """box_denoise's "Check Methods..." button: compare every method on
        the current frame's raw (contrast-only) background."""
        if self.bg_stack is None:
            return
        self._check_methods_dlg = self.box_denoise.open_check_methods_dialog(
            self.bg_stack[self.frame], parent=self)

    #%% undo/redo - see the history init in __init__ for what this does and
    # does not cover.
    def _push_undo(self, frame_idx):
        """Snapshot mask_stack[frame_idx] (the whole stack if frame_idx is
        None) onto the undo history, before a destructive edit overwrites
        it - call this immediately before the mutation itself, not after.
        Clears the redo branch (a fresh edit invalidates whatever was
        undone before it) and refreshes the ribbon's Undo/Redo buttons."""
        prev = self.mask_stack.copy() if frame_idx is None else self.mask_stack[frame_idx].copy()
        self._undo_stack.append((frame_idx, prev))
        if len(self._undo_stack) > self._UNDO_MAX_DEPTH:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_redo_buttons()

    def _apply_undo_entry(self, frame_idx, array, other_stack):
        """Swap `array` into mask_stack at `frame_idx` (the whole stack if
        None), pushing the piece it replaces onto `other_stack` (the
        opposite history - redo's own stack when called from _do_undo, and
        vice versa) - shared by _do_undo/_do_redo, which are otherwise
        exact mirror images of each other. Jumps the frame slider to
        `frame_idx` first if it isn't already the one on screen, so the
        user actually sees what just got restored - that alone already
        redraws (see _on_frame_changed), so this only calls _redraw_mask()
        itself for a whole-stack entry or one that's already on screen."""
        if frame_idx is None:
            other_stack.append((None, self.mask_stack.copy()))
            self.mask_stack = array
        else:
            other_stack.append((frame_idx, self.mask_stack[frame_idx].copy()))
            self.mask_stack[frame_idx] = array
            if frame_idx != self.frame:
                self.slider_frame.setValue(frame_idx)
                return
        self._redraw_mask()

    def _do_undo(self):
        if not self._undo_stack:
            return
        frame_idx, array = self._undo_stack.pop()
        self._apply_undo_entry(frame_idx, array, self._redo_stack)
        self._threshold_undo_frame = None  # see _threshold_live_update
        self._update_undo_redo_buttons()

    def _do_redo(self):
        if not self._redo_stack:
            return
        frame_idx, array = self._redo_stack.pop()
        self._apply_undo_entry(frame_idx, array, self._undo_stack)
        self._threshold_undo_frame = None
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self):
        btn = self.ribbon.get_button('undo')
        if btn is not None:
            btn.setEnabled(bool(self._undo_stack))
        btn = self.ribbon.get_button('redo')
        if btn is not None:
            btn.setEnabled(bool(self._redo_stack))

    def _grow_shrink(self, angle, grow):
        self._push_undo(self.frame)
        old = self.mask_stack[self.frame]
        new = io.shift_mask_edge(old, angle, grow=grow)
        # Item 1: every pixel this actually changed counts as a manual edit
        # too, same as paint/rect paint - see checkbox_protectManualEdits.
        self._manual_mask_stack[self.frame] |= (old != new)
        self.mask_stack[self.frame] = new
        self._redraw_mask()

    def _reset_to_tracking(self):
        """"Reset to Tracking": discard every edit in this dialog - the
        mask on every frame (back to the original tracked/segmented
        result) AND every segment boundary/Dilate/Erode/Edge Detection/
        Mesh setting (back to one plain-default segment spanning the whole
        stack) - confirmed first since it's the single most destructive
        action here, and (unlike "Reset Frame"/painted edits) not
        undo-able: _push_undo only ever snapshots self.mask_stack, not
        self._segments, so Ctrl+Z can't bring a wiped segment list back."""
        if qtw.QMessageBox.question(
                self, 'Reset to Tracking',
                'Discard every edit in this dialog - the mask on every frame (back to '
                'the original tracked/segmented result) and every segment boundary/'
                'Dilate/Erode/Edge Detection/Mesh setting? This cannot be undone.',
                qtw.QMessageBox.Yes | qtw.QMessageBox.No, qtw.QMessageBox.No
        ) != qtw.QMessageBox.Yes:
            return
        self._push_undo(None)
        source = self._default_stack if self._default_stack is not None else self._original_stack
        self.mask_stack = source.copy()
        self._manual_mask_stack[:] = False
        self._reset_segments_to_single_default()
        self._redraw_mask()

    def _recompute_threshold_stack(self):
        """Full (N, H, W) mask stack from the Threshold box's current
        method/blur/deviation, via the caller's recompute_thresh_fn - or
        None if that raised (caller decides how loudly to report it).
        Passes the Denoise box's own current whole-stack output too (see
        _denoised_bg_stack) - a threshold-derived caller (ROI Tracker) uses
        it instead of its own raw images when re-thresholding, so the
        Denoise box's choice actually affects the binarized mask, not just
        what's displayed."""
        method = self.combo_threshMethod.currentText()
        blur = self.spinbox_threshBlur.value()
        offset = self.slider_threshDev.value() / 100
        try:
            return np.asarray(self.recompute_thresh_fn(
                method, offset, blur, denoised_imgs=self._denoised_bg_stack())).astype(bool)
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
        surfaces a real error dialog.

        Undo: a slider drag fires this many times in a row - snapshotting
        every single one would flood the undo history with near-duplicates
        for what's really one edit. self._threshold_undo_frame tracks
        which frame already has a pending snapshot for the CURRENT
        uninterrupted tweak; only the first call after that changes (i.e.
        the frame itself changed - see _on_frame_changed) pushes a fresh
        one, so one Ctrl+Z undoes the whole tweak, however many times the
        slider fired along the way."""
        full = self._recompute_threshold_stack()
        if full is None:
            return
        if self._threshold_undo_frame != self.frame:
            self._push_undo(self.frame)
            self._threshold_undo_frame = self.frame
        self.mask_stack[self.frame] = full[self.frame]
        # A fresh re-threshold discards whatever was painted here before it.
        self._manual_mask_stack[self.frame] = False
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
        self._push_undo(None)
        self.mask_stack = full
        self._manual_mask_stack[:] = False
        self._redraw_mask()

    def get_mask_stack(self):
        return self.mask_stack

    def get_manual_edit_settings(self):
        """Item 1: which pixels were set by a manual tool (see
        self._manual_mask_stack) and whether the Exclude Manual Edits from
        Effects checkbox is on - the caller round-trips this into its own
        per-object column (like get_mesh_settings() etc.) and passes it back
        in as `manual_edit_settings` next time this dialog opens for the
        same object, AND threads it through its own apply_edge_mask so
        Dilate/Erode/Edge Detection keep excluding these pixels at real
        extraction time too, not just in this dialog's own live preview."""
        return {'protect': self.checkbox_protectManualEdits.isChecked(),
                'mask': self._manual_mask_stack}

    def get_edited_frame_indices(self):
        """Frame indices whose mask actually differs from `_original_stack`
        (this session's opening state) - a direct before/after array
        comparison, not the undo stack (which is depth-capped and not
        meant as a change log - see _push_undo). Used only for a one-line
        summary log when the caller accepts this dialog, not to drive any
        editing behavior itself."""
        changed = np.any(self.mask_stack != self._original_stack, axis=tuple(range(1, self.mask_stack.ndim)))
        return np.nonzero(changed)[0]

    def get_edge_settings(self):
        """The shared Segments timeline's own Edge Detection values, one
        entry per segment - see get_dilate_erode_settings()/
        get_mesh_settings()."""
        return {'segments': [{'start': s['start'], 'end': s['end'], **s['edge']}
                             for s in self._segments]}

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
            self._manual_mask_stack[self.frame, row, col] = True
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
        # Item 3: "Center Grid (Click)" armed - this click sets the active
        # mesh's fixed origin instead of anything else below (paint/rect/
        # mesh-cell-toggle), then disarms itself regardless of what was
        # clicked on.
        if self._picking_mesh_center:
            self._isolate_if_single_frame_scope()
            mesh = self._active_mesh
            if mesh is not None:
                mesh['origin'] = [float(event.xdata), float(event.ydata)]
                mesh['fixed'] = True
                self.checkbox_meshFixed.blockSignals(True)
                self.checkbox_meshFixed.setChecked(True)
                self.checkbox_meshFixed.blockSignals(False)
                self._update_segment_ui()
                self._redraw_mask()
            self._picking_mesh_center = False
            self.button_meshCenterClick.setChecked(False)
            self._apply_ribbon_cursor()
            return
        mods = event.modifiers
        tool = self.ribbon.active_tool
        if tool in ('pan', 'zoom'):
            return
        plain_left = event.button == 1 and not mods

        if plain_left and tool is None and self._active_mesh is not None:
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
            # One undo entry per press-drag-release gesture, not per pixel
            # touched along the way - snapshotted here (drag start), before
            # _paint_pixel's own repeated mask_stack writes below.
            self._push_undo(self.frame)
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
                    self._push_undo(self.frame)
                    self.mask_stack[self.frame, row0:row1, col0:col1] = value
                    self._manual_mask_stack[self.frame, row0:row1, col0:col1] = True
        self._redraw_mask()
