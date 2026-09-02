# -*- coding: utf-8 -*-
"""8-bit contrast normalization for navigation-signal display and
downstream processing (tracking/SAM2), plus a small general-purpose blur
helper used by the same tracking-prep code paths.
"""
import numpy as np
import cv2

CONTRAST_METHODS = ('percentile', 'minmax', 'std')

def _contrast_bounds(data, axis, method='percentile', plow=1.0, phigh=99.0, n_std=3.0):
    """Compute the (lo, hi) value bounds contrast is stretched between, for
    the given method - shared by convert_to_8bit/convert_img_to_8bit.

    Args:
        data: float array.
        axis: axis (or tuple of axes) reduced over - the trailing two axes
            for a per-frame stack, or None for a single 2-D image.
        method: One of CONTRAST_METHODS.
            'percentile' (default): stretch between the plow/phigh
                percentiles - robust to the handful of hot/dead pixels
                common on electron-counting detectors, which would
                otherwise dominate a raw min/max range and wash out
                everything else (a real cause of SAM2 producing inaccurate
                masks on effectively-flat-looking 8-bit input).
            'minmax': stretch between the raw min and max - simple, but
                exactly as vulnerable to those outlier pixels as the name
                implies.
            'std': stretch between mean ± n_std standard deviations -
                another common way to reject outliers, tunable via n_std.
        plow, phigh: 'percentile' method's low/high percentiles (0-100).
        n_std: 'std' method's clip half-width, in standard deviations.

    Returns:
        (lo, hi), each broadcastable against `data`.
    """
    if method == 'minmax':
        lo = data.min(axis=axis, keepdims=True)
        hi = data.max(axis=axis, keepdims=True)
    elif method == 'std':
        mean = data.mean(axis=axis, keepdims=True)
        std = data.std(axis=axis, keepdims=True)
        lo = mean - n_std * std
        hi = mean + n_std * std
    elif method == 'percentile':
        lo = np.percentile(data, plow, axis=axis, keepdims=True)
        hi = np.percentile(data, phigh, axis=axis, keepdims=True)
    else:
        raise ValueError(f'Unknown contrast method {method!r}; expected one of {CONTRAST_METHODS}')
    return lo, hi

def convert_to_8bit(s, method='percentile', plow=1.0, phigh=99.0, n_std=3.0,
                     clip_low=None, clip_high=None):
    """Normalise each frame of a HyperSpy signal to 8-bit (0-255).

    See `_contrast_bounds` for the available methods and their tunable
    parameters.

    Args:
        s: HyperSpy Signal2D with arbitrary numeric dtype.
        method: Contrast method - one of CONTRAST_METHODS.
        plow, phigh: 'percentile' method's low/high percentiles (per frame).
        n_std: 'std' method's clip half-width, in standard deviations.
        clip_low, clip_high: Optional raw-value thresholds - values below
            clip_low/above clip_high are clamped to that threshold *before*
            the method's own stretch is computed, e.g. to knock a saturated
            beam stop or a dead-pixel border out of consideration. None
            (default) skips this pre-clip entirely.

    Returns:
        A deep copy of `s` with data converted to uint8, each frame independently normalised.
    """
    data = s.data.astype(np.float32)
    if clip_low is not None or clip_high is not None:
        data = np.clip(data, clip_low, clip_high)
    # Normalise per-frame using broadcasting (vectorised, no Python loop)
    axis = tuple(range(data.ndim - 2, data.ndim))
    los, his = _contrast_bounds(data, axis, method, plow, phigh, n_std)
    data = np.clip((data - los) / (his - los + 1e-8), 0.0, 1.0)
    data = (data * 255.0).astype(np.uint8)
    s_8bit = s._deepcopy_with_new_data(data)
    return s_8bit

def convert_img_to_8bit(img, method='percentile', plow=1.0, phigh=99.0, n_std=3.0,
                         clip_low=None, clip_high=None):
    """Normalise a single 2-D numpy array to uint8 - see `convert_to_8bit`/
    `_contrast_bounds` for the available methods and their tunable parameters.

    Args:
        img: 2-D numpy array of any numeric dtype.
        method: Contrast method - one of CONTRAST_METHODS.
        plow, phigh: 'percentile' method's low/high percentiles.
        n_std: 'std' method's clip half-width, in standard deviations.
        clip_low, clip_high: Optional raw-value thresholds - see
            `convert_to_8bit`. None (default) skips this pre-clip entirely.

    Returns:
        numpy.ndarray of dtype uint8.
    """
    img = np.asarray(img, dtype=np.float32)
    if clip_low is not None or clip_high is not None:
        img = np.clip(img, clip_low, clip_high)
    lo, hi = _contrast_bounds(img, None, method, plow, phigh, n_std)
    img_8bit = np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0) * 255.0
    return img_8bit.astype(np.uint8)

def gaussian_blur(img, kernel_size=3):
    """Apply a square Gaussian blur kernel to a 2-D image.

    Args:
        img: 2-D numpy array.
        kernel_size: Side length of the kernel (must be odd). Default is 3.

    Returns:
        Blurred numpy.ndarray of the same shape and dtype as `img`.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def threshold_ignore_zero(threshold_func, img):
    """Apply a skimage threshold function (threshold_otsu/threshold_li/...)
    considering only pixels with a value above 0.

    True-zero pixels are background/never-visited scan positions (always the
    case for unvisited positions in a reconstructed smart-scan array - see
    `loaders._reconstruct_smart_scan` - and common at the frame border on
    dense scans too), not real signal - letting them dominate the histogram
    skews the computed threshold low. Falls back to thresholding the whole
    image if there are no nonzero pixels at all (an all-zero `img` has
    nothing meaningful to threshold either way).

    Args:
        threshold_func: A skimage.filters threshold function (e.g.
            `threshold_otsu`), called with a 1-D array of the nonzero pixels.
        img: 2-D (or N-D) numpy array.

    Returns:
        The scalar threshold value.
    """
    nonzero = img[img > 0]
    if nonzero.size == 0:
        return threshold_func(img)
    return threshold_func(nonzero)

def _directional_kernel(length, angle_deg):
    """Build a one-sided ray-shaped erosion kernel: a `length`-pixel line
    from the kernel's anchor point out to angle `angle_deg` (0 = +x/right,
    increasing clockwise to match image row-down/col-right pixel axes).

    Used by `erode_mask_edge` for directional erosion: `cv2.erode` with this
    kernel and anchor keeps a pixel only if the mask extends the full ray
    length from it in that direction, so pixels within `length` of the
    boundary *as seen looking along `angle_deg`* get eroded away - i.e. the
    boundary crossed by walking in that direction (its "far side"). A
    kernel_size square (the isotropic default) is just every angle's ray
    combined, hence eroding uniformly from all sides.

    Returns:
        (kernel, anchor) - kernel is a uint8 numpy.ndarray, anchor an
        (x, y) tuple for cv2.erode's own `anchor` argument.
    """
    length = max(1, int(round(length)))
    size = 2 * length + 1
    anchor = (length, length)
    kernel = np.zeros((size, size), np.uint8)
    theta = np.deg2rad(angle_deg)
    x1 = anchor[0] + length * np.cos(theta)
    y1 = anchor[1] + length * np.sin(theta)
    cv2.line(kernel, anchor, (int(round(x1)), int(round(y1))), 1, thickness=1)
    kernel[anchor[1], anchor[0]] = 1
    return kernel, anchor

def erode_mask_edge(mask, kernel_size=3, direction=None, revert=False):
    """Reduce a binary mask to just its edge/outline, `kernel_size` pixels wide.

    Erodes the mask, then subtracts the eroded mask from the original -
    what's left is the boundary band the erosion ate away. Used as an
    optional post-processing step on segmentation/threshold masks (ROI on
    4D, ROI Tracker, SAM2 Tracker) so extraction/display can be restricted
    to a particle's edge instead of its whole area.

    Args:
        mask: 2-D array, truthy where the mask is set (any dtype).
        kernel_size: Erosion kernel size, in pixels (the square kernel's
            side when `direction` is None, the ray length otherwise).
            Larger values eat further into the mask before subtracting,
            producing a wider edge band. Must be >= 1.
        direction: None (default) erodes uniformly from every side with a
            square kernel, keeping the mask's whole outline - matches prior
            behaviour. A float angle in degrees (0 = +x/right, increasing
            clockwise) instead erodes only along that one direction (see
            `_directional_kernel`), keeping just the boundary band facing
            that angle - e.g. 0 keeps the mask's right-hand edge, 90 its
            bottom edge, 180 its left edge, 270 its top edge.
        revert: If False (default), returns just the edge band (prior
            behaviour). If True, returns its complement *within the
            original mask* instead - i.e. the mask's interior (and, with
            `direction` set, its other sides too), with only the one
            detected edge band cut out.

    Returns:
        numpy.ndarray of dtype bool, same shape as `mask` - True on the
        mask's edge band (or, if `revert`, everywhere in the original mask
        except that edge band).
    """
    if direction is None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        anchor = (-1, -1)  # cv2's own "center" sentinel
    else:
        kernel, anchor = _directional_kernel(kernel_size, direction)
    mask_u8 = mask.astype('uint8')
    mask_eroded = cv2.erode(mask_u8, kernel, anchor=anchor, iterations=1)
    edge_mask = (mask_u8 & ~mask_eroded).astype(bool)
    if revert:
        return mask_u8.astype(bool) & ~edge_mask
    return edge_mask

def shift_mask_edge(mask, direction, grow=True):
    """Grow or shrink a binary mask by one pixel along one direction -
    manual, directional single-pixel mask fine-tuning (see mask_edit_dialog.py).

    Reuses `_directional_kernel`'s one-sided ray kernel (see `erode_mask_edge`
    for the angle convention: 0 = +x/right, 90 = +y/down, 180 = left,
    270 = up), dilating to grow the mask outward in that direction or
    eroding to pull its edge inward from that side.

    Args:
        mask: 2-D array, truthy where the mask is set (any dtype).
        direction: Angle in degrees (0/90/180/270 for right/bottom/left/top).
        grow: If True, dilate (extend outward); if False, erode (pull inward).

    Returns:
        numpy.ndarray of dtype bool, same shape as `mask`.
    """
    mask_u8 = mask.astype('uint8')
    if grow:
        # Dilating with the *same* kernel/anchor erosion uses to shrink from
        # `direction` actually extends the mask the opposite way (erosion's
        # "keep only if supported looking towards `direction`" becomes, for
        # dilation's max-over-neighbourhood, "pull in support from
        # `direction`" instead of "push out towards it") - the 180-degree
        # rotated kernel is what actually grows the mask towards `direction`.
        kernel, anchor = _directional_kernel(1, direction + 180)
        return cv2.dilate(mask_u8, kernel, anchor=anchor, iterations=1).astype(bool)
    kernel, anchor = _directional_kernel(1, direction)
    return cv2.erode(mask_u8, kernel, anchor=anchor, iterations=1).astype(bool)


def dilate_erode_mask(mask, kernel_size):
    """Uniformly grow or shrink a binary mask with a plain isotropic square
    structuring element - unlike shift_mask_edge (one direction, one pixel
    per call) or erode_mask_edge (reduces the mask to a boundary band
    instead of keeping its interior), this dilates/erodes the whole mask
    by `kernel_size` on every side at once. Used by the "Dilate / Erode"
    control shared by ROI Tracker, SAM2 Tracker and mask_edit_dialog.py's
    Fine-Tune Mask dialog.

    Args:
        mask: 2-D array, truthy where the mask is set (any dtype).
        kernel_size: Signed kernel size, in pixels - sign picks dilate
            (positive, grows the mask) vs. erode (negative, shrinks it);
            0 is a no-op. Same `np.ones((k, k))` square-kernel convention
            as erode_mask_edge's own `kernel_size`.

    Returns:
        numpy.ndarray of dtype bool, same shape as `mask`.
    """
    kernel_size = int(round(kernel_size))
    if kernel_size == 0:
        return mask.astype(bool)
    kernel = np.ones((abs(kernel_size), abs(kernel_size)), np.uint8)
    mask_u8 = mask.astype('uint8')
    op = cv2.dilate if kernel_size > 0 else cv2.erode
    return op(mask_u8, kernel, anchor=(-1, -1), iterations=1).astype(bool)


def mask_centroid(mask):
    """(x, y) center of mass of `mask` - the origin mesh_cell_ids/
    mesh_restrict_mask anchor their rotated grid to by default, so a mesh
    cell selection made on one frame stays aligned with the same relative
    position *on the object* in every other frame of a tracked stack, even
    as the object itself moves - rather than a fixed grid over the full
    frame, which a tracked object drifts across from frame to frame. Falls
    back to the array's own center if `mask` is empty (nothing to center on)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        h, w = mask.shape
        return w / 2, h / 2
    return float(xs.mean()), float(ys.mean())


def select_blob_by_centroid(mask, seed_centroid=None):
    """Restrict `mask` to just one of its connected components ("blobs") -
    the one whose own centroid is closest to `seed_centroid` - for a
    tracked ROI whose threshold mask actually contains more than one
    separate object (e.g. two nearby particles), letting the caller keep
    just one of them for extraction. Used by ROI Tracker's "Blob
    Selection" (see Tab_Tracking_CV2.apply_edge_mask/_resolve_blob_mask) -
    runs before Dilate/Erode/Edge Detection/Mesh, on the raw (possibly
    multi-blob) threshold mask.

    Args:
        mask: 2-D array, truthy where the mask is set (any dtype).
        seed_centroid: (x, y) to match against, e.g. where the user last
            clicked, or the previous frame's own chosen centroid (for
            frame-to-frame "auto-follow" - see _resolve_blob_mask). None
            picks the largest-area blob instead - a reasonable default the
            first time a mask is seen, before any seed exists yet.

    Returns:
        (restricted_mask, chosen_centroid) - `restricted_mask` is a bool
        array the same shape as `mask`, True only where the chosen blob
        is; `chosen_centroid` is that blob's own (x, y) centroid (for the
        caller to seed the *next* frame's call with, continuing the
        follow), or None if `mask` has no blobs at all (nothing to
        choose - `restricted_mask` is then just `mask` itself, unchanged).
    """
    mask_u8 = mask.astype('uint8')
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:  # label 0 is background - no real blobs at all
        return mask.astype(bool), None
    blob_labels = np.arange(1, num_labels)
    if seed_centroid is None:
        areas = stats[1:, cv2.CC_STAT_AREA]
        chosen = blob_labels[int(np.argmax(areas))]
    else:
        blob_centroids = centroids[1:]  # (x, y) per blob, matching blob_labels
        dists = np.hypot(blob_centroids[:, 0] - seed_centroid[0],
                         blob_centroids[:, 1] - seed_centroid[1])
        chosen = blob_labels[int(np.argmin(dists))]
    # Plain floats (not numpy.float64) - this ends up stored in a per-ROI
    # settings dict that gets serialized straight to JSON (see
    # Tab_Tracking_CV2.save_results), which numpy scalar types can trip up.
    chosen_centroid = (float(centroids[chosen][0]), float(centroids[chosen][1]))
    return labels == chosen, chosen_centroid


def estimate_tilt_axis_pca(centroids):
    """PCA-based tomography tilt-axis angle estimate (degrees, 0-180, from
    the +x axis via np.arctan2 - same convention as the Mesh box's own
    `angle_deg` above) from a stack of per-frame (x, y) mask centroids
    (mask_centroid) of one object tracked across a tilt series.

    By the projection-slice theorem, a specimen's centroid coordinate
    *along* the true tilt axis stays fixed as tilt angle changes, while
    the perpendicular coordinate sweeps back and forth - so the tilt axis
    is the minimum-variance direction of the centroid scatter, found here
    via PCA (eigendecomposition of the covariance matrix). Ported from
    the standalone tilt_axis_finder.py prototype (which estimates this
    from raw per-frame intensity images instead) - mask_edit_dialog.py's
    "Find Tilt Axis" button uses the already-tracked mask centroids
    instead, since that's what's on hand there.

    Returns:
        (angle, centered) - the estimated angle, and the mean-centered
        centroids (feed straight into sweep_tilt_axis_angle for a
        refined/cross-checked estimate).
    """
    centroids = np.asarray(centroids, dtype=float)
    centered = centroids - centroids.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    axis_dir = eigvecs[:, 0]  # smallest-variance direction = tilt axis
    angle = np.degrees(np.arctan2(axis_dir[1], axis_dir[0])) % 180
    return float(angle), centered


def sweep_tilt_axis_angle(centered, angle_range=(0, 180), step=0.5):
    """Refinement/diagnostic for estimate_tilt_axis_pca: variance of the
    along-axis centroid component for every candidate angle in
    `angle_range` - the minimum should land close to the PCA estimate
    (`centered` is that same function's second return value), and is a
    bit more numerically direct since it doesn't depend on eigenvector
    sign/degeneracy.

    Returns:
        (angles, variances, best_angle).
    """
    angles = np.arange(*angle_range, step)
    variances = np.empty_like(angles)
    for i, phi in enumerate(angles):
        axis_dir = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
        variances[i] = (centered @ axis_dir).var()
    best_angle = float(angles[np.argmin(variances)])
    return angles, variances, best_angle


def mesh_cell_ids(shape, angle_deg, cell_size, origin):
    """Integer mesh-cell coordinates for every pixel in `shape`, from a grid
    rotated `angle_deg` from horizontal (0 = axis-aligned, increasing
    clockwise - same convention as erode_mask_edge's `direction`), centered
    on `origin`, with square `cell_size`-px cells - used by
    mask_edit_dialog.py's "Mesh" box to let the user restrict a mask to
    specific cell(s) for extraction (see mesh_restrict_mask).

    Args:
        shape: (height, width) of the mask/image the mesh is laid over.
        angle_deg: Grid rotation, in degrees.
        cell_size: Cell side length, in pixels. Must be > 0.
        origin: (x, y) pixel coordinate the grid is centered on - normally
            the object's own mask_centroid(), so cell (0, 0) always sits on
            the object regardless of where it is in the frame.

    Returns:
        (cell_i, cell_j) - two int arrays, shape `shape`, each pixel's
        column/row index in the rotated grid.
    """
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    x = x - origin[0]
    y = y - origin[1]
    theta = np.deg2rad(angle_deg)
    rot_x = x * np.cos(theta) + y * np.sin(theta)
    rot_y = -x * np.sin(theta) + y * np.cos(theta)
    cell_i = np.floor(rot_x / cell_size).astype(int)
    cell_j = np.floor(rot_y / cell_size).astype(int)
    return cell_i, cell_j


def mesh_restrict_mask(mask, angle_deg, cell_size, cells, origin=None, lines_only=False):
    """`mask` AND the union of `cells` (an iterable of (cell_i, cell_j)
    tuples from mesh_cell_ids) - restricts a mask to just the mesh cell(s)
    the user picked in mask_edit_dialog.py's "Mesh" box. `mask` returned
    unchanged if `cells` is empty/None (nothing picked yet, so there's
    nothing to restrict to).

    `origin` defaults to mask_centroid(mask) - i.e. the grid is anchored to
    *this* mask's own position, so the same selected cell(s) stay aligned
    with the same relative part of the object across every frame of a
    tracked stack, however much the object itself has moved by then. Pass
    an explicit `origin` instead only to match a grid anchored elsewhere
    (e.g. mask_edit_dialog.py's live overlay, which shares one origin
    across the whole redraw rather than recomputing it twice).

    `lines_only`: if True, each cell's `cell_j` (row-along-the-grid) index
    is ignored - a selected (i, j) keeps every pixel with that same `i`,
    regardless of `j`, i.e. a full-width band/stripe running the whole
    length of the grid's rotated y-axis instead of one square cell. Lets
    the Mesh box restrict to parallel stripes (mask_edit_dialog.py's
    "Lines Only" checkbox) instead of a 2-D grid of squares."""
    if not cells:
        return mask
    if origin is None:
        origin = mask_centroid(mask)
    cell_i, cell_j = mesh_cell_ids(mask.shape, angle_deg, cell_size, origin)
    keep = np.zeros(mask.shape, dtype=bool)
    if lines_only:
        for i in {c[0] for c in cells}:
            keep |= (cell_i == i)
    else:
        for i, j in cells:
            keep |= (cell_i == i) & (cell_j == j)
    return mask & keep


def segment_for_frame(segments, frame):
    """The segment (a dict with at least 'start'/'end', inclusive frame
    indices) covering `frame`, from a sorted, contiguous list of segments -
    mask_edit_dialog.py's Fine-Tune Mask "segments": per-object frame
    ranges that can each carry their own Dilate/Erode, Edge Detection, and
    Mesh settings, instead of one fixed setting for every frame (see
    MaskEditDialog's own _segment_for_frame, and get_dilate_erode_settings/
    get_edge_settings/get_mesh_settings's `{'segments': [...]}` shape that
    ROI Tracker/SAM2 Tracker's own apply_edge_mask reads via this same
    function). Falls back to the last segment if `frame` is past every
    listed range (e.g. a stack that grew since these segments were last
    edited), and to None if `segments` is empty."""
    if not segments:
        return None
    for seg in segments:
        if seg['start'] <= frame <= seg['end']:
            return seg
    return segments[-1]
