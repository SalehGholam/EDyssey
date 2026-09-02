# -*- coding: utf-8 -*-
"""Segmentation methods for ROI Tracker's Blob Selection (see contrast.py's
select_blob_by_centroid, and Tab_Tracking_CV2's own "Blob" object-list
column/blob_segmentation_dialog.py) - splitting a ROI's raw threshold mask
into individual blobs BEFORE one is picked by centroid, so touching or
partially-overlapping particles (which plain connected-components analysis
would merge into a single blob) can still be told apart and selected
individually.

Each method takes the ROI's binary threshold mask (and, for intensity-
based methods, the matching raw-intensity crop - see
Tab_Tracking_CV2._current_intensity_crop) and returns an int-labeled array
the same shape as `mask` (0 = background/outside any blob, 1..N = one
label per detected blob) - the same shape select_blob_by_centroid's own
plain-connected-components default already produces, so neither it nor the
"which blob did the user click"/contour-overlay code needs to know which
method actually produced it.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

_STRUCT8 = np.ones((3, 3), dtype=int)  # 8-connectivity, matching the cv2
                                        # connectivity=8 the 'connected'
                                        # method itself uses


def _peak_markers(elevation, mask, min_distance, threshold_rel):
    """Shared marker step for both watershed methods below: find local
    maxima of `elevation`, at least `min_distance` px apart and at least
    `threshold_rel` of the way from a local peak's own pre-split
    connected-component's min to max (rejects weak/noise peaks that would
    otherwise over-segment a single blob into spurious extra pieces -
    evaluated per pre-existing connected component, not globally across
    the whole ROI crop, so one dim particle sharing a crop with one bright
    touching pair isn't penalized by the bright pair's own scale), then
    label each surviving peak as one marker. Returns None if no peak
    survives (caller should fall back to treating `mask` as one unsplit
    blob)."""
    components, _ = ndi.label(mask, structure=_STRUCT8)
    coords = peak_local_max(elevation, min_distance=max(1, int(min_distance)),
                            labels=components, threshold_rel=threshold_rel,
                            exclude_border=False)
    if len(coords) == 0:
        return None
    peak_mask = np.zeros(elevation.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask, structure=_STRUCT8)
    return markers


def label_watershed_distance(mask, min_distance=7, threshold_rel=0.3, **_ignored):
    """Marker-controlled watershed on the mask's own distance transform -
    splits touching/partially-overlapping blobs at their narrowest
    connecting "neck", the standard approach for round-ish particles
    (nanoparticles, beads, cells) whose shape alone already implies where
    one ends and the next begins, regardless of their brightness.

    `min_distance`: minimum px separation between two accepted blob
    centers - raise this if one real particle is being split into more
    than one piece. `threshold_rel`: minimum peak height (fraction of its
    own component's own deepest point) - raise this to require a more
    pronounced "neck" before splitting (fewer, more confident splits),
    lower it to accept shallower ones (more aggressive splitting, more
    prone to false splits on irregular/elongated shapes)."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=int)
    distance = ndi.distance_transform_edt(mask)
    markers = _peak_markers(distance, mask, min_distance, threshold_rel)
    if markers is None:
        return mask.astype(int)
    return watershed(-distance, markers, mask=mask)


def label_watershed_intensity(mask, img_cut, min_distance=7, threshold_rel=0.2, **_ignored):
    """Marker-controlled watershed on the ROI's own raw intensity crop -
    splits blobs at their dimmest connecting boundary instead of their
    narrowest shape "neck", for particles whose overlap still leaves a
    visible dip in Z-contrast/intensity between neighboring cores even
    where their outer thresholded shape has already merged into one blob
    (common in HAADF-STEM). Falls back to the distance-transform method
    automatically if `img_cut` isn't usable. Same `min_distance`/
    `threshold_rel` meaning as label_watershed_distance, just measured on
    intensity peaks instead of distance-transform peaks."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=int)
    if img_cut is None or img_cut.shape != mask.shape:
        return label_watershed_distance(mask, min_distance, threshold_rel)
    img = img_cut.astype(float)
    markers = _peak_markers(img, mask, min_distance, threshold_rel)
    if markers is None:
        return mask.astype(int)
    return watershed(-img, markers, mask=mask)


def _cluster_pixel_coords(mask, n_clusters):
    """Shared setup for label_kmeans/label_gmm: `mask`'s own True pixels
    as an (N, 2) array of (x, y) coordinates, plus `n_clusters` clamped to
    [1, N] (clustering into more groups than there are points to cluster
    isn't meaningful). Returns (coords, ys, xs, n_clusters) - ys/xs are
    np.where(mask)'s own row/col index arrays, reused to scatter each
    pixel's cluster assignment back into a 2-D labeled array."""
    ys, xs = np.where(mask)
    n_clusters = max(1, min(int(n_clusters), len(xs)))
    coords = np.column_stack([xs, ys]).astype(float)
    return coords, ys, xs, n_clusters


def label_kmeans(mask, n_clusters=2, **_ignored):
    """K-Means clustering of the mask's own True-pixel (x, y) coordinates
    into `n_clusters` spatial groups - splits touching/overlapping blobs
    by "which cluster center is this pixel closest to" instead of the
    Watershed methods' own shape/intensity-peak flood-fill, useful when
    neither finds a clean split (no shape "neck" or intensity dip between
    the particles) but you know roughly how many there are. Always
    produces exactly `n_clusters` blobs (unlike Watershed, which can
    produce anywhere from 1 to several depending on how many peaks clear
    their own threshold) - assumes roughly round, similarly-sized blobs,
    since K-Means clusters are always round/isotropic; see label_gmm for
    an elongated/anisotropic-shape-aware alternative.

    Args:
        mask: 2-D array, truthy where the mask is set (any dtype).
        n_clusters: How many blobs to split the mask into.

    Returns:
        Int-labeled array (0 = background, 1..n_clusters one per blob),
        same shape as `mask`.
    """
    labels = np.zeros(mask.shape, dtype=int)
    if not mask.any():
        return labels
    coords, ys, xs, n_clusters = _cluster_pixel_coords(mask, n_clusters)
    if n_clusters == 1:
        labels[ys, xs] = 1
        return labels
    fit = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(coords)
    labels[ys, xs] = fit.labels_ + 1  # 0 is reserved for background
    return labels


def label_gmm(mask, n_clusters=2, **_ignored):
    """Gaussian Mixture Model clustering of the mask's own True-pixel
    (x, y) coordinates into `n_clusters` spatial groups - same idea as
    label_kmeans, but each blob gets its own full-covariance ellipse
    instead of K-Means' fixed round one, so it copes better with
    elongated/anisotropic particle shapes (e.g. needle-like crystals) at
    the cost of needing more pixels per blob to fit reliably. Falls back
    to label_kmeans (numerically more robust - no covariance matrix to
    invert) if the GMM fit itself fails, e.g. too few pixels for the
    requested `n_clusters`, or a degenerate (near-collinear) point cloud.
    Args/Returns: see label_kmeans."""
    labels = np.zeros(mask.shape, dtype=int)
    if not mask.any():
        return labels
    coords, ys, xs, n_clusters = _cluster_pixel_coords(mask, n_clusters)
    if n_clusters == 1:
        labels[ys, xs] = 1
        return labels
    try:
        fit = GaussianMixture(n_components=n_clusters, random_state=0).fit(coords)
        assignments = fit.predict(coords)
    except ValueError:
        return label_kmeans(mask, n_clusters)
    labels[ys, xs] = assignments + 1
    return labels


# id -> method metadata, used both to dispatch (label_blobs) and to build
# blob_segmentation_dialog.py's method combo/parameter form generically
# instead of hardcoding a form per method. param_specs entries are
# (key, label, low, high, step, decimals, tooltip).
BLOB_SEGMENTATION_METHODS = {
    'connected': {
        'label': 'Connected Components (No Splitting)',
        # None means "use plain connected-components" - the original/
        # default behavior, kept as a literal no-op case (see label_blobs)
        # rather than routed through a function of its own, so a ROI that
        # never touches Blob Selection's segmentation options at all keeps
        # producing pixel-identical results to before this feature existed.
        'func': None,
        'needs_intensity': False,
        'default_params': {},
        'param_specs': [],
        'description': (
            'Each separate connected region of the threshold mask is its own '
            'blob. Simplest and fastest - use this unless two particles are '
            'actually touching/overlapping in the threshold mask (in which '
            'case they show up as one merged blob here, and one of the '
            'Watershed methods below is needed to tell them apart).'),
    },
    'watershed_distance': {
        'label': 'Watershed - Shape',
        'func': label_watershed_distance,
        'needs_intensity': False,
        'default_params': {'min_distance': 7, 'threshold_rel': 0.3},
        'param_specs': [
            ('min_distance', 'Min. Peak Separation (px)', 1, 50, 1, 0,
             'Minimum distance (pixels) between two blob centers - raise this '
             'if one real particle is being split into more than one piece.'),
            ('threshold_rel', 'Min. Peak Prominence', 0.0, 1.0, 0.05, 2,
             'How pronounced the "neck" between two particles must be '
             '(relative to the deepest point) to count as a split - raise '
             'this if there are false splits, lower it if touching particles '
             "aren't being split at all."),
        ],
        'description': (
            'Splits blobs at their narrowest connecting "neck" in shape '
            'alone, via a distance-transform watershed - the standard choice '
            'for round-ish touching/overlapping particles (nanoparticles, '
            'beads, cells).'),
    },
    'watershed_intensity': {
        'label': 'Watershed - Intensity (Z-Contrast)',
        'func': label_watershed_intensity,
        'needs_intensity': True,
        'default_params': {'min_distance': 7, 'threshold_rel': 0.2},
        'param_specs': [
            ('min_distance', 'Min. Peak Separation (px)', 1, 50, 1, 0,
             'Minimum distance (pixels) between two blob centers - raise this '
             'if one real particle is being split into more than one piece.'),
            ('threshold_rel', 'Min. Peak Prominence', 0.0, 1.0, 0.05, 2,
             'How pronounced the intensity dip between two particles must be '
             '(relative to their own brightest point) to count as a split - '
             'raise this if there are false splits, lower it if touching '
             "particles aren't being split at all."),
        ],
        'description': (
            'Splits blobs at their dimmest connecting boundary in raw '
            'intensity, via an intensity watershed - use this instead of '
            'Watershed - Shape when overlapping particles still show a '
            'visible Z-contrast dip between their cores even though their '
            'outer threshold shape has already merged into one blob.'),
    },
    'kmeans': {
        'label': 'K-Means Clustering',
        'func': label_kmeans,
        'needs_intensity': False,
        'default_params': {'n_clusters': 2},
        'param_specs': [
            ('n_clusters', 'Number of Blobs', 1, 20, 1, 0,
             'How many blobs to split the mask into - always exactly this '
             'many, unlike the Watershed methods above. Raise it if there are '
             'more overlapping particles than currently detected, lower it if '
             "the mask is being split into more pieces than there really are "
             'particles.'),
        ],
        'description': (
            'Splits the mask into exactly this many roughly round, similarly-'
            'sized spatial clusters (K-Means on pixel coordinates) - a good '
            'choice when you know roughly how many particles overlap here and '
            'neither Watershed method above finds a clean split (no shape '
            '"neck" or intensity dip between them).'),
    },
    'gmm': {
        'label': 'Gaussian Mixture (GMM)',
        'func': label_gmm,
        'needs_intensity': False,
        'default_params': {'n_clusters': 2},
        'param_specs': [
            ('n_clusters', 'Number of Blobs', 1, 20, 1, 0,
             "How many blobs to split the mask into - same as K-Means "
             "Clustering's own Number of Blobs."),
        ],
        'description': (
            'Like K-Means Clustering above, but each blob gets its own '
            'elliptical (not fixed-round) shape, fit via a Gaussian Mixture '
            'Model - better for elongated/anisotropic particles, at the cost '
            'of needing more pixels per blob to fit reliably.'),
    },
}

DEFAULT_BLOB_METHOD = 'connected'


def default_blob_params(method):
    """A fresh copy of `method`'s own default parameters (falling back to
    DEFAULT_BLOB_METHOD's for an unrecognized id) - callers should never
    mutate BLOB_SEGMENTATION_METHODS[...]['default_params'] itself."""
    return dict(BLOB_SEGMENTATION_METHODS.get(
        method, BLOB_SEGMENTATION_METHODS[DEFAULT_BLOB_METHOD])['default_params'])


def label_blobs(mask, img_cut, method, params):
    """Dispatch to the labeling function for `method` (see
    BLOB_SEGMENTATION_METHODS) - the single entry point contrast.
    select_blob_by_centroid's callers (see Tab_Tracking_CV2._label_blobs_for)
    go through, whichever method a ROI has chosen. Returns None for
    'connected' (or an unrecognized method) - signaling the caller to fall
    back to select_blob_by_centroid's own built-in plain-connected-
    components default, keeping that case exactly as it always was."""
    spec = BLOB_SEGMENTATION_METHODS.get(method)
    if spec is None or spec['func'] is None:
        return None
    if not mask.any():
        return np.zeros(mask.shape, dtype=int)
    kwargs = {**spec['default_params'], **(params or {})}
    if spec['needs_intensity']:
        return spec['func'](mask, img_cut, **kwargs)
    return spec['func'](mask, **kwargs)
