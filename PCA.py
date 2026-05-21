import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def _reshape_to_2d(data, scan_shape=None):
    """
    Reshape ED data to (n_patterns, n_pixels) and return scan_shape.

    Conventions handled automatically:
      4D (nx, ny, dx, dy) : 2D scan + 2D detector
      3D (n, dx, dy)      : flat scan + 2D detector  [dx == dy or dx*dy >> n]
      3D (nx, ny, n_pix)  : 2D scan + flat detector  [n_pix << nx*ny]

    Pass scan_shape=(nx, ny) explicitly to override the heuristic.
    """
    if scan_shape is not None:
        n_scan = scan_shape[0] * scan_shape[1]
        n_pix = data.size // n_scan
        return data.reshape(n_scan, n_pix), scan_shape

    if data.ndim == 4:
        n0, n1, n2, n3 = data.shape
        return data.reshape(n0 * n1, n2 * n3), (n0, n1)

    if data.ndim == 3:
        n0, n1, n2 = data.shape
        # Treat as (n_frames, det_x, det_y) when last two dims look like a 2D detector
        # (equal dims, or much larger than n_frames)
        if n1 == n2 or n1 * n2 > n0:
            sq = int(np.round(np.sqrt(n0)))
            inferred_scan = (sq, sq) if sq * sq == n0 else (n0, 1)
            return data.reshape(n0, n1 * n2), inferred_scan
        else:
            # (scan_x, scan_y, n_pixels)
            return data.reshape(n0 * n1, n2), (n0, n1)

    raise ValueError(f"Expected 3D or 4D array, got {data.ndim}D.")


def apply_pca(data, n_components=10, normalize=True, scan_shape=None):
    """
    Apply PCA to a 3D/4D ED dataset.

    Parameters
    ----------
    data : ndarray, shape (nx, ny, npx) or (nx, ny, det_x, det_y)
    n_components : int
    normalize : bool
        Standardize each pixel's intensity across patterns before PCA.

    Returns
    -------
    dict with keys:
        scores      : (nx, ny, n_components) — loading maps in scan space
        components  : (n_components, n_pixels) — eigenpatterns
        explained   : (n_components,) — explained variance ratio
        pca         : fitted PCA object
        scan_shape  : (nx, ny)
    """
    X, scan_shape = _reshape_to_2d(data, scan_shape)
    X = X.astype(np.float64)

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    scores_flat = pca.fit_transform(X)

    nx, ny = scan_shape
    scores = scores_flat.reshape(nx, ny, n_components)

    return {
        "scores": scores,
        "components": pca.components_,
        "explained": pca.explained_variance_ratio_,
        "pca": pca,
        "scan_shape": scan_shape,
    }


def apply_ica(data, n_components=5, normalize=True, random_state=42, scan_shape=None):
    """
    Apply ICA to a 3D/4D ED dataset.

    Parameters
    ----------
    data : ndarray, shape (nx, ny, npx) or (nx, ny, det_x, det_y)
    n_components : int
    normalize : bool
    random_state : int

    Returns
    -------
    dict with keys:
        sources     : (nx, ny, n_components) — source maps in scan space
        components  : (n_components, n_pixels) — independent component patterns
        mixing      : (n_components, n_components) — mixing matrix
        ica         : fitted FastICA object
        scan_shape  : (nx, ny)
    """
    X, scan_shape = _reshape_to_2d(data, scan_shape)
    X = X.astype(np.float64)

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    n_components = min(n_components, X.shape[0], X.shape[1])
    ica = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
    sources_flat = ica.fit_transform(X)

    nx, ny = scan_shape
    sources = sources_flat.reshape(nx, ny, n_components)

    return {
        "sources": sources,
        "components": ica.components_,
        "mixing": ica.mixing_,
        "ica": ica,
        "scan_shape": scan_shape,
    }


def visualize_pca(result, det_shape=None, max_components=6, cmap_map="RdBu_r", cmap_pat="viridis"):
    """
    Visualize PCA results: scree plot, score maps, and eigenpatterns.

    Parameters
    ----------
    result     : dict returned by apply_pca()
    det_shape  : (det_x, det_y) to reshape components back to 2D detector image.
                 If None, plotted as a 1D line.
    max_components : how many components to show
    """
    scores = result["scores"]
    components = result["components"]
    explained = result["explained"]
    n_comp = min(max_components, components.shape[0])

    fig = plt.figure(figsize=(4 + n_comp * 2.5, 9))
    gs = GridSpec(3, n_comp + 1, figure=fig, hspace=0.4, wspace=0.35)

    # --- Scree plot ---
    ax_scree = fig.add_subplot(gs[0, :])
    ax_scree.bar(range(1, len(explained) + 1), explained * 100, color="steelblue", alpha=0.8)
    ax_scree.plot(range(1, len(explained) + 1), np.cumsum(explained) * 100,
                  "o-", color="tomato", label="Cumulative")
    ax_scree.set_xlabel("Component")
    ax_scree.set_ylabel("Explained variance (%)")
    ax_scree.set_title("PCA Scree Plot")
    ax_scree.legend()

    # --- Score maps (loading maps in scan space) ---
    for i in range(n_comp):
        ax = fig.add_subplot(gs[1, i])
        img = scores[:, :, i].squeeze()  # collapse (n,1) → (n,) for 1D scans
        vmax = np.percentile(np.abs(img), 99)
        if img.ndim == 1:
            ax.plot(img)
        else:
            ax.imshow(img, cmap=cmap_map, vmin=-vmax, vmax=vmax)
            ax.axis("off")
        ax.set_title(f"PC{i+1}\n({explained[i]*100:.1f}%)")

    # --- Eigenpatterns ---
    for i in range(n_comp):
        ax = fig.add_subplot(gs[2, i])
        comp = components[i]
        if det_shape is not None:
            ax.imshow(comp.reshape(det_shape), cmap=cmap_pat)
        else:
            ax.plot(comp, lw=0.8)
        ax.set_title(f"Pattern PC{i+1}")
        ax.axis("off") if det_shape is not None else ax.set_xlabel("Pixel")

    fig.suptitle("PCA Decomposition", fontsize=14, fontweight="bold")
    plt.show()
    return fig


def visualize_ica(result, det_shape=None, max_components=6, cmap_map="RdBu_r", cmap_pat="viridis"):
    """
    Visualize ICA results: source maps and independent component patterns.

    Parameters
    ----------
    result     : dict returned by apply_ica()
    det_shape  : (det_x, det_y) to reshape components to 2D. If None, plotted as 1D line.
    max_components : how many components to show
    """
    sources = result["sources"]
    components = result["components"]
    n_comp = min(max_components, components.shape[0])

    fig, axes = plt.subplots(2, n_comp, figsize=(n_comp * 2.8, 6))
    if n_comp == 1:
        axes = axes[:, np.newaxis]

    for i in range(n_comp):
        # Source map
        ax_map = axes[0, i]
        img = sources[:, :, i].squeeze()
        vmax = np.percentile(np.abs(img), 99)
        if img.ndim == 1:
            ax_map.plot(img)
        else:
            ax_map.imshow(img, cmap=cmap_map, vmin=-vmax, vmax=vmax)
            ax_map.axis("off")
        ax_map.set_title(f"IC{i+1} map")

        # Component pattern
        ax_pat = axes[1, i]
        comp = components[i]
        if det_shape is not None:
            ax_pat.imshow(comp.reshape(det_shape), cmap=cmap_pat)
            ax_pat.axis("off")
        else:
            ax_pat.plot(comp, lw=0.8)
            ax_pat.set_xlabel("Pixel")
        ax_pat.set_title(f"IC{i+1} pattern")

    axes[0, 0].set_ylabel("Source maps", fontsize=9)
    axes[1, 0].set_ylabel("IC patterns", fontsize=9)
    fig.suptitle("ICA Decomposition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig


def separate_domains(data, result, n_components=2):
    """
    Cluster frames into 2 domains using K-means on PCA/ICA scores.

    Parameters
    ----------
    data        : ndarray (n_frames, det_x, det_y)  — original frames
    result      : dict from apply_pca() or apply_ica()
    n_components: number of leading components to use for clustering

    Returns
    -------
    dict with keys:
        frames_A    : frames assigned to domain A
        frames_B    : frames assigned to domain B
        labels      : (n_frames,) array of 0/1 cluster labels
        domain_mask : (nx, ny) spatial map of labels
        indices_A   : original frame indices for domain A
        indices_B   : original frame indices for domain B
    """
    scores = result.get("sources", result.get("scores"))  # ICA or PCA
    scan_shape = result["scan_shape"]
    nx, ny = scan_shape

    n_comp = min(n_components, scores.shape[2])
    features = scores[:, :, :n_comp].reshape(nx * ny, n_comp)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # Reshape data for indexing: (n_frames, det_x, det_y) where n_frames = nx*ny
    flat_data = data.reshape(nx * ny, *data.shape[1:]) if data.ndim > 3 else data

    mask_A = labels == 0
    mask_B = labels == 1

    return {
        "frames_A":   flat_data[mask_A],
        "frames_B":   flat_data[mask_B],
        "labels":     labels,
        "domain_mask": labels.reshape(nx, ny),
        "indices_A":  np.where(mask_A)[0],
        "indices_B":  np.where(mask_B)[0],
    }


def visualize_domain_separation(sep_result, det_shape=None, cmap_pat="viridis"):
    """
    Visualize domain separation: spatial domain map + mean pattern per domain.

    Parameters
    ----------
    sep_result : dict returned by separate_domains()
    det_shape  : (det_x, det_y) to display mean patterns as 2D images
    """
    domain_mask = sep_result["domain_mask"]
    frames_A = sep_result["frames_A"]
    frames_B = sep_result["frames_B"]

    mean_A = frames_A.mean(axis=0)
    mean_B = frames_B.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Domain map
    axes[0].imshow(domain_mask, cmap="bwr", interpolation="nearest")
    axes[0].set_title(f"Domain map\nA: {len(frames_A)} frames  B: {len(frames_B)} frames")
    axes[0].axis("off")

    # Mean pattern A
    img_A = mean_A.reshape(det_shape) if det_shape and mean_A.ndim == 1 else mean_A
    axes[1].imshow(img_A, cmap=cmap_pat)
    axes[1].set_title("Mean pattern — Domain A")
    axes[1].axis("off")

    # Mean pattern B
    img_B = mean_B.reshape(det_shape) if det_shape and mean_B.ndim == 1 else mean_B
    axes[2].imshow(img_B, cmap=cmap_pat)
    axes[2].set_title("Mean pattern — Domain B")
    axes[2].axis("off")

    fig.suptitle("Domain Separation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Example / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tifffile
    from glob import glob
    import os
    from tqdm import tqdm
    path = r'C:\My Files\Microscope Data\Tecnai\24-09-11_TiO2_200 kV\S2\5DED Analysis\2025-05-03__15-49-37\roi No 4\frames'
    fns = glob(os.path.join(path,'*.tif'))
    data = np.zeros((len(fns),512,512), dtype='uint16')
    for i, fn in enumerate(tqdm(fns)):
        data[i] = tifffile.imread(fn)
        
    det_shape = (512, 512)

    pca_result = apply_pca(data, n_components=4)
    ica_result = apply_ica(data, n_components=4)

    visualize_pca(pca_result, det_shape=det_shape, max_components=4)
    visualize_ica(ica_result, det_shape=det_shape, max_components=4)

    sep = separate_domains(data, ica_result, n_components=2)
    print(f"Domain A: {len(sep['frames_A'])} frames")
    print(f"Domain B: {len(sep['frames_B'])} frames")
    visualize_domain_separation(sep, det_shape=det_shape)
