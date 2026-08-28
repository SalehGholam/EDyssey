# -*- coding: utf-8 -*-
"""matplotlib plot-overlay helpers shared across the app: a readable
(outlined, background-free) scale bar, and reciprocal-space calibration
rings drawn on a diffraction-pattern axis.
"""
import numpy as np
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
from matplotlib_scalebar.scalebar import ScaleBar

#%% scale bar readability
class ReadableScaleBar(ScaleBar):
    """A matplotlib_scalebar ScaleBar with its background fully removed
    (box_alpha=0) and its label given a stroked outline so it stays legible
    over arbitrary image content underneath, regardless of `color`.

    matplotlib_scalebar rebuilds its Text artist from scratch inside
    draw() on every redraw and doesn't expose a path_effects hook, so this
    patches the module-level `TextArea` name it calls for the duration of
    our own draw() call, restoring it immediately after.
    """
    _OUTLINE = [patheffects.withStroke(linewidth=2.5, foreground='black')]

    def draw(self, renderer, *args, **kwargs):
        import matplotlib_scalebar.scalebar as _sb_mod
        original_text_area = _sb_mod.TextArea
        outline = self._OUTLINE

        def _outlined_text_area(s, textprops=None, **kw):
            textprops = dict(textprops or {})
            textprops['path_effects'] = outline
            return original_text_area(s, textprops=textprops, **kw)

        _sb_mod.TextArea = _outlined_text_area
        try:
            super().draw(renderer, *args, **kwargs)
        finally:
            _sb_mod.TextArea = original_text_area


def add_readable_scalebar(ax, dx, units='m', **kwargs):
    """Remove any existing ScaleBar on `ax` and add a `ReadableScaleBar` -
    transparent background, outlined text - in its place. `kwargs` are
    forwarded to ReadableScaleBar (e.g. `dimension`, `location`); `color`
    defaults to white, which reads well (with the outline) against this
    app's dark colormaps (viridis/inferno/gray).
    """
    kwargs.setdefault('dimension', 'si-length')
    kwargs.setdefault('location', 'lower left')
    kwargs.setdefault('color', 'w')
    kwargs.setdefault('box_alpha', 0)
    for artist in list(ax.artists):
        if isinstance(artist, ScaleBar):
            artist.remove()
    scalebar = ReadableScaleBar(dx, units, **kwargs)
    ax.add_artist(scalebar)
    return scalebar

def remove_scalebar(ax):
    """Remove any existing ScaleBar artist(s) from `ax`, if present."""
    for artist in list(ax.artists):
        if isinstance(artist, ScaleBar):
            artist.remove()

def nonzero_display_min(array):
    """The lowest non-zero value in `array`, for use as a display vmin -
    navigation images commonly have literal 0-count background/unscanned
    positions that would otherwise dominate the low end of the displayed
    intensity range, washing out the real (non-zero) contrast. Falls back
    to the array's true minimum if every value is zero (or the array is
    empty), so a blank image still gets a usable number instead of raising."""
    array = np.asarray(array)
    nonzero = array[array != 0]
    if nonzero.size == 0:
        return array.min() if array.size else 0
    return nonzero.min()

def draw_reciprocal_scale_circles(ax, scale_recip, shape, center=None, old_artists=None):
    """Draw concentric dashed circles marking whole-1/A radii on a
    diffraction-pattern axis, in place of a conventional scale bar (which
    doesn't read naturally on a radially-symmetric reciprocal-space image).
    Circles are spaced every 1 1/A out to whatever fits in the image; if
    even the first one wouldn't fit (coarse reciprocal-space resolution), a
    single 0.5 1/A circle is drawn instead, in a visually distinct (dotted)
    style so it can't be mistaken for a whole-1/A ring. Each ring is also
    labeled with its 1/A value near its top-right edge.

    Args:
        ax: matplotlib Axes the diffraction-pattern image is displayed on.
        scale_recip: Reciprocal-space calibration in 1/Angstrom per pixel.
            No circles are drawn if this is None, 0, or unparseable.
        shape: (height, width) of the displayed image, in pixels.
        center: (x, y) center to draw circles around, in pixel coordinates.
            Defaults to the image's own geometric center.
        old_artists: Circle/label artists returned by a previous call,
            removed before drawing the new ones - so repeated calls (e.g. on
            every redraw) don't accumulate stale circles.

    Returns:
        List of the newly-added circle/label artists; pass this back in as
        `old_artists` on the next call.
    """
    # A mistaken/misunderstood-units entry (e.g. 1 instead of 0.01 1/A per
    # pixel) can make max_r_recip huge - without a hard cap this would try
    # to create thousands of patches and one or more full-figure redraws for
    # them on every keystroke, which is what actually froze the app rather
    # than the value itself being "wrong": the cap keeps the ring spacing
    # concept intact (still one ring per 1/A) while bounding the worst case.
    MAX_RINGS = 20

    if old_artists:
        for artist in old_artists:
            try:
                artist.remove()
            except Exception:
                pass  # already removed (e.g. axes was cleared since the last call)
    new_artists = []
    try:
        scale_recip = float(scale_recip)
    except (TypeError, ValueError):
        return new_artists
    if not scale_recip or scale_recip <= 0:
        return new_artists

    h, w = shape[:2]
    cx, cy = center if center is not None else (w / 2, h / 2)
    max_r_px = min(cx, cy, w - cx, h - cy)
    max_r_recip = max_r_px * scale_recip

    # Top-right point of each ring (45 deg up from horizontal) - y grows
    # downward in image/pixel coordinates, so "up" is a negative y offset.
    label_dx, label_dy = np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))
    label_outline = [patheffects.withStroke(linewidth=2, foreground='black')]

    def _add_ring_label(r_px, value, color):
        label = ax.text(cx + r_px * label_dx, cy - r_px * label_dy,
                        f'{value:g}' r' $\AA^{-1}$',
                        color=color, fontsize=7, ha='center', va='center',
                        path_effects=label_outline)
        new_artists.append(label)

    n = 1
    while n <= max_r_recip and len(new_artists) < MAX_RINGS:
        r_px = n / scale_recip
        circle = patches.Circle((cx, cy), r_px, fill=False, edgecolor='cyan',
                                linestyle='--', linewidth=1.2, alpha=0.6)
        ax.add_patch(circle)
        new_artists.append(circle)
        _add_ring_label(r_px, n, 'cyan')
        n += 1

    if not new_artists and max_r_recip > 0:
        # Not even the first 1/A ring fits - fall back to a finer 0.5/A
        # ring, in a different (dotted, white) style so it reads as
        # distinct from the (absent) whole-1/A rings.
        r_px = 0.5 / scale_recip
        if r_px <= max_r_px:
            circle = patches.Circle((cx, cy), r_px, fill=False, edgecolor='white',
                                    linestyle=':', linewidth=1.4, alpha=0.6)
            ax.add_patch(circle)
            new_artists.append(circle)
            _add_ring_label(r_px, 0.5, 'white')

    return new_artists
