"""
Estimate the tilt axis of a TEM tomography tilt series and plot it.

Methods
-------
Both estimators below report the tilt-axis angle in the same convention
(degrees, 0-180, 0 = image +x axis, measured the way np.arctan2 does), so
they can be directly compared/cross-checked against each other.

1. PCA (estimate_tilt_axis_pca) / explicit angle sweep
   (sweep_tilt_axis_angle) - by the projection-slice theorem, the specimen
   centroid's coordinate *along* the true tilt axis is invariant with tilt
   angle, while the *perpendicular* coordinate sweeps back and forth. So
   the tilt axis is the minimum-variance direction of the centroid scatter
   - found via PCA (eigendecomposition of the covariance matrix) and cross-
   checked by directly sweeping candidate angles and measuring variance.
   Fast, and the closest thing to a "default" here, but reduces each frame
   to a single (x, y) number (its centroid), discarding everything else
   about its shape.

2. Radon-transform / common-line consistency (estimate_tilt_axis_radon) -
   generalizes the exact same invariant from the centroid alone to the
   *entire* projected 1-D intensity profile (one Radon-transform row per
   candidate angle per frame): for the correct axis, that whole profile -
   not just its first moment - should stay nearly identical across tilts.
   More robust to noise/asymmetric or low-contrast specimens than the
   centroid-only methods above, since it compares the full projected shape
   instead of one summary number per frame, at the cost of being much
   slower (one Radon transform per frame per candidate angle) - see its
   docstring for ways to keep the runtime reasonable. A genuinely
   independent cross-check: agreement between it and PCA/sweep is much
   stronger evidence than either alone.

Not implemented here (needs actual fiducial markers, unlike everything
above): multi-marker bundle adjustment, e.g. IMOD's tiltalign/AreTomo -
track several distinct gold-bead (or other high-contrast point)
trajectories across the series (each one traces a cosine of the *actual*
tilt angle, amplitude/phase set by its 3-D offset from the axis - the
classic "sinusoidal fiducial trajectory" model) and jointly solve for one
shared tilt axis plus per-marker 3-D position by least squares. This is far
more constrained/robust than anything above, since it uses many independent
trajectories at once instead of just one - but doing this with only a
*single* feature (e.g. the whole specimen's own centroid as a stand-in
marker) doesn't actually work: a lone point's trajectory is confined to one
line through the origin no matter the true axis, and projecting that line
onto any candidate direction is still a perfect cosine of tilt angle
regardless of whether the candidate is right - i.e. there's no signal left
to discriminate the correct angle from a single trajectory alone, which is
exactly why real fiducial-based alignment needs *several* markers at
different 3-D positions, not one. This is the production-grade approach;
everything actually implemented above is a coarse, marker-free estimate for
a quick look/initial guess.

Both methods above assume the specimen rotates rigidly about a fixed
in-plane axis with no frame-to-frame stage/beam translational drift - real
data will have some drift, which adds noise to the centroid/profile
trajectory and therefore to both estimates equally.

Usage
-----
    python tilt_axis_finder.py path/to/tiltseries.mrc [--tlt path/to/series.rawtlt]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import hyperspy.api as hs


def load_tilt_series(path, tlt_file=None):
    """Load a tomography tilt series with HyperSpy.

    Returns
    -------
    data : (n_tilts, ny, nx) ndarray
    tilt_angles : (n_tilts,) ndarray of degrees (falls back to the
        navigation axis values, or plain frame index, if no .tlt/.rawtlt
        file is supplied)
    signal : the loaded HyperSpy signal
    """
    s = hs.load(path)

    nav_axes = s.axes_manager.navigation_axes
    if len(nav_axes) != 1:
        raise ValueError(
            f"Expected a tilt series with one navigation axis, found {len(nav_axes)}."
        )

    data = s.data.astype(float)

    if tlt_file is not None:
        tilt_angles = np.loadtxt(tlt_file)
    else:
        tilt_angles = nav_axes[0].axis.astype(float)

    if len(tilt_angles) != data.shape[0]:
        raise ValueError(
            f"Got {len(tilt_angles)} tilt angles for {data.shape[0]} images."
        )

    return data, tilt_angles, s


def compute_centroids(stack):
    """Intensity-weighted centroid (x, y) of every image in the stack."""
    n, ny, nx = stack.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    centroids = np.empty((n, 2))
    for i in range(n):
        img = np.clip(stack[i] - np.percentile(stack[i], 1), 0, None)
        total = img.sum()
        if total <= 0:
            centroids[i] = (nx / 2, ny / 2)
            continue
        centroids[i, 0] = (img * xs).sum() / total
        centroids[i, 1] = (img * ys).sum() / total
    return centroids


def estimate_tilt_axis_pca(centroids):
    """PCA-based tilt-axis angle estimate (degrees, 0-180, from +x axis)."""
    centered = centroids - centroids.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    axis_dir = eigvecs[:, 0]  # smallest-variance direction = tilt axis
    angle = np.degrees(np.arctan2(axis_dir[1], axis_dir[0])) % 180
    return angle, centered


def sweep_tilt_axis_angle(centered, angle_range=(0, 180), step=0.5):
    """Diagnostic sweep: variance of the along-axis centroid component vs angle."""
    angles = np.arange(*angle_range, step)
    variances = np.empty_like(angles)
    for i, phi in enumerate(angles):
        axis_dir = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
        variances[i] = (centered @ axis_dir).var()
    best_angle = angles[np.argmin(variances)]
    return angles, variances, best_angle


def estimate_tilt_axis_radon(stack, angle_range=(0, 180), step=2.0):
    """Radon-transform / common-line tilt-axis estimate: generalizes the
    same projection-slice invariant estimate_tilt_axis_pca/
    sweep_tilt_axis_angle exploit via the centroid (first moment) alone to
    the *entire* projected 1-D intensity profile - for the correct axis,
    that whole profile (not just its first moment) should stay nearly
    identical across tilts, by the same projection-slice-theorem argument
    (see module docstring). More robust to noise, asymmetric shapes, or
    low-contrast specimens than the centroid-only methods above, since it
    compares the full projected shape frame-to-frame instead of reducing
    each frame to one summary number - at the cost of being much slower
    (one Radon transform per candidate angle per frame).

    For each candidate axis angle, every frame's profile is projected along
    the axis-parallel direction (skimage.transform.radon, called once per
    frame with the whole `angles` array as `theta` - much cheaper than one
    call per (frame, angle) pair), then each frame's profile is centered
    and L2-normalized (so raw intensity/background differences between
    frames don't count against an otherwise-correct angle) before summing
    the frame-to-frame squared deviation from the across-tilt mean profile.
    The correct angle minimizes that deviation.

    To keep runtime down on a long/large-frame series: pass a subsampled
    stack (e.g. `stack[::5]`), a coarser `step`, and/or downsample each
    frame's spatial resolution first (e.g. via a simple block-average) -
    none of that changes which angle comes out on top by more than a
    fraction of a degree in practice, since it's the same invariant either
    way, just measured more finely.

    Args:
        stack: (n_tilts, ny, nx) array, e.g. from load_tilt_series.
        angle_range, step: candidate tilt-axis angles to try, in degrees.

    Returns:
        (angles, scores, best_angle) - candidate angles, each one's
        frame-to-frame profile-inconsistency score, and the best
        (minimum-score) angle.
    """
    from skimage.transform import radon

    angles = np.arange(*angle_range, step)
    # This module's own angle convention is 0 = image +x axis, measured via
    # np.arctan2 on (row, col) = (y, x) array coordinates (same as
    # estimate_tilt_axis_pca/sweep_tilt_axis_angle). We want, for each
    # candidate *axis* angle, the profile obtained by summing *along the
    # axis-perpendicular direction* (i.e. angle + 90) - that's the one whose
    # profile stays invariant across tilts for the correct axis (see
    # function docstring/module docstring), not the profile of summing
    # along the axis itself. skimage.transform.radon(image, theta=T) sums
    # along direction T measured from the same array's y axis instead of x,
    # which combined with row-major (y-down) indexing empirically gives
    # T = 90 - direction (verified with a line drawn at a known angle: that
    # skimage theta is where the line's own profile is sharpest/most
    # concentrated, i.e. summing along the line's own direction). Composing
    # both: skimage_theta = 90 - (angle + 90) = -angle.
    skimage_thetas = (-angles) % 180

    # One radon() call per frame (covering every candidate angle at once via
    # `theta=skimage_thetas`) rather than one call per (frame, angle) pair.
    per_angle_profiles = [[] for _ in angles]
    for frame in stack:
        sino = radon(frame, theta=skimage_thetas, circle=False)  # (profile_len, n_angles)
        for j in range(len(angles)):
            per_angle_profiles[j].append(sino[:, j])

    scores = np.empty_like(angles)
    for i in range(len(angles)):
        profiles = np.stack(per_angle_profiles[i])  # (n_tilts, profile_len)
        profiles = profiles - profiles.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(profiles, axis=1, keepdims=True)
        norm[norm == 0] = 1
        profiles = profiles / norm
        mean_profile = profiles.mean(axis=0)
        scores[i] = ((profiles - mean_profile) ** 2).sum()

    best_angle = angles[np.argmin(scores)]
    return angles, scores, best_angle


def plot_tilt_axis(image, tilt_axis_angle, centroids=None, ax=None, title=None):
    """Overlay the estimated tilt-axis line (and centroid trajectory) on an image."""
    ny, nx = image.shape
    cx, cy = nx / 2, ny / 2

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(image, cmap="gray")

    phi = np.radians(tilt_axis_angle)
    length = 0.6 * min(nx, ny)
    dx, dy = np.cos(phi) * length, np.sin(phi) * length
    ax.plot(
        [cx - dx, cx + dx], [cy - dy, cy + dy],
        "-", color="cyan", lw=2, label=f"Tilt axis ({tilt_axis_angle:.1f} deg)",
    )

    if centroids is not None:
        ax.plot(
            centroids[:, 0], centroids[:, 1],
            "o-", color="orange", ms=3, lw=1, alpha=0.6, label="Frame centroids",
        )

    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.set_title(title or f"Estimated tilt axis: {tilt_axis_angle:.1f} deg")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def main(path, tlt_file=None):
    data, tilt_angles, _signal = load_tilt_series(path, tlt_file)
    print(
        f"Loaded tilt series: {data.shape[0]} images, "
        f"tilt range {tilt_angles.min():.1f} to {tilt_angles.max():.1f} deg"
    )

    centroids = compute_centroids(data)
    pca_angle, centered = estimate_tilt_axis_pca(centroids)
    angles, variances, swept_angle = sweep_tilt_axis_angle(centered)

    print(f"PCA tilt-axis angle estimate:  {pca_angle:.2f} deg")
    print(f"Sweep-refined tilt-axis angle: {swept_angle:.2f} deg")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ref_idx = len(data) // 2
    plot_tilt_axis(
        data[ref_idx], swept_angle, centroids=centroids, ax=axes[0],
        title=f"Tilt axis on frame {ref_idx} ({tilt_angles[ref_idx]:.1f} deg)",
    )

    axes[1].plot(angles, variances)
    axes[1].axvline(swept_angle, color="red", ls="--", label=f"min var @ {swept_angle:.1f} deg")
    axes[1].set_xlabel("Candidate tilt-axis angle (deg)")
    axes[1].set_ylabel("Along-axis centroid variance")
    axes[1].set_title("Tilt-axis angle sweep")
    axes[1].legend()

    fig.tight_layout()
    plt.show()

    return swept_angle, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Tilt series file (.mrc, .st, .dm4, .tif stack, ...)")
    parser.add_argument("--tlt", dest="tlt_file", default=None,
                         help="Optional .tlt/.rawtlt file with tilt angles (one per line)")
    args = parser.parse_args()
    main(args.path, args.tlt_file)
