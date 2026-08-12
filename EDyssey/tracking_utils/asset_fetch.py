# -*- coding: utf-8 -*-
"""On-demand download of the large model-weight files EDyssey needs for the
SAM2 tab and the Nano/DaSiamRPN OpenCV trackers, but doesn't bundle in the
installer (offline installers with ~1GB of model weights risk GitHub
Releases' 2GB-per-asset cap, and most installs never touch the SAM2/Nano/
DaSiamRPN features at all). See THIRD_PARTY_NOTICES.md for what each file
actually is, its license, and where it comes from.

Split out from tracking_utils_ui.py/tab_sam2.py so both the CV2-tracker and
SAM2 call sites share one download/verify/atomic-rename implementation
rather than duplicating it.
"""
import hashlib
import logging
import os
import sys
import urllib.request
import urllib.error

_logger = logging.getLogger(__name__)


class AssetDownloadError(Exception):
    """Raised when a required model file can't be downloaded or fails
    integrity verification - callers should catch this and show the user an
    actionable message rather than letting a raw urllib exception surface."""


def _base_dir():
    """EDyssey/tracking_utils/ - the PyInstaller-extracted temp dir when
    frozen (same pattern as worker_dispatch._base_dir()), this file's own
    directory otherwise."""
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'EDyssey', 'tracking_utils')
    return os.path.dirname(os.path.abspath(__file__))


SAM2_CHECKPOINT_DIR = os.path.join(_base_dir(), 'SAM2_checkpoints')
SAM2_CHECKPOINT_FILENAME = 'sam2.1_hiera_large.pt'
# Meta's own official download host - the same URL checkpoints/
# download_ckpts.sh (part of facebookresearch/sam2, not shipped/tracked in
# this repo) uses for this exact checkpoint. Apache-2.0.
SAM2_CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'
SAM2_CHECKPOINT_SIZE = 898083611  # bytes; Meta publishes no official checksum to verify against

OPENCV_MODELS_DIR = os.path.join(_base_dir(), 'opencv_models')

# opencv/opencv_zoo pinned to the last commit before DaSiamRPN was removed
# there ("Remove DaSiamRPN since we have its superseder VitTrack now (#213)",
# commit b0fe2136) - Apache-2.0 (models/object_tracking_dasiamrpn/LICENSE at
# that commit, confirmed). Pinning to a commit SHA keeps the URL stable even
# though the folder no longer exists on the default branch. These are
# Git-LFS objects on GitHub, so downloads have to go through
# media.githubusercontent.com - raw.githubusercontent.com only returns the
# ~130-byte LFS pointer text for files this size.
_DASIAMRPN_REF = 'fef72f8fa7c52eaf116d3df358d24e6e959ada0e'
_DASIAMRPN_BASE = (
    f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{_DASIAMRPN_REF}'
    '/models/object_tracking_dasiamrpn/'
)
DASIAMRPN_MODELS = {
    'dasiamrpn_model.onnx': {
        'url': _DASIAMRPN_BASE + 'object_tracking_dasiamrpn_model_2021nov.onnx',
        'sha256': 'e88370b85cbad914a5eb414d9d9e0820f87fd0cd89b65205a766174206c35719',
        'size': 91040894,
    },
    'dasiamrpn_kernel_cls1.onnx': {
        'url': _DASIAMRPN_BASE + 'object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx',
        'sha256': 'd85b03e2aeded6cc9be945dfdc3ed6b8f4151f101e485037b6c5d5b36a6c4204',
        'size': 23603598,
    },
    'dasiamrpn_kernel_r1.onnx': {
        'url': _DASIAMRPN_BASE + 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx',
        'sha256': '082c85d231b88b97a1b2a50e73b640a332c5d98d7c1d80b5da9ab534fa7a9e5b',
        'size': 47206788,
    },
}

# NanoTrack's models aren't hosted by opencv_zoo at all (never were - only
# DaSiamRPN was). OpenCV's own samples/python/tracker.py points at
# HonglinChu/SiamTrackers as the source, which is what this pulls from.
#
# IMPORTANT - unlike SAM2 and DaSiamRPN above, this repository carries NO
# LICENSE file (checked via the GitHub API and a full repo-root listing) -
# there is no clear grant of redistribution rights for these two files.
# They're wired up here because the app's 'nano' tracker option needs them
# to function at all, but this is a real licensing gap, not a cleared
# dependency - see THIRD_PARTY_NOTICES.md. Confirm redistribution rights
# (or drop the 'nano' tracker option) before shipping this to anyone else.
_NANOTRACK_BASE = (
    'https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master'
    '/NanoTrack/models/nanotrackv2/'
)
NANOTRACK_MODELS = {
    'backbone.onnx': {
        'url': _NANOTRACK_BASE + 'nanotrack_backbone_sim.onnx',
        'sha256': '530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4',
        'size': 1056849,
    },
    'neckhead.onnx': {
        'url': _NANOTRACK_BASE + 'nanotrack_head_sim.onnx',
        'sha256': '0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9',
        'size': 726198,
    },
}


def _download(url, dest_path, expected_size=None, expected_sha256=None, progress_callback=None):
    """Stream `url` to `dest_path`, verifying size/hash before an atomic
    rename into place - a cancelled or failed download this way never leaves
    a corrupt file at `dest_path` for a later run to mistake for the real
    thing.

    Args:
        progress_callback: optional `callback(bytes_done, total_bytes)`,
            called after each chunk - `total_bytes` is 0 if the server
            didn't send Content-Length.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    part_path = dest_path + '.part'
    hasher = hashlib.sha256() if expected_sha256 else None

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            total = int(resp.headers.get('Content-Length', 0)) or (expected_size or 0)
            done = 0
            with open(part_path, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    if hasher is not None:
                        hasher.update(chunk)
                    done += len(chunk)
                    if progress_callback is not None:
                        progress_callback(done, total)
    except (urllib.error.URLError, OSError) as exc:
        try:
            os.remove(part_path)
        except OSError:
            pass
        raise AssetDownloadError(f'Could not download {url!r}: {exc}') from exc

    actual_size = os.path.getsize(part_path)
    if expected_size and actual_size != expected_size:
        os.remove(part_path)
        raise AssetDownloadError(
            f'Download of {url!r} was {actual_size} bytes, expected {expected_size} - '
            'likely truncated or corrupted, try again.')
    if expected_sha256 and hasher.hexdigest() != expected_sha256:
        os.remove(part_path)
        raise AssetDownloadError(
            f'Download of {url!r} failed a checksum check - likely corrupted, try again.')

    os.replace(part_path, dest_path)
    _logger.info('Downloaded %s (%d bytes) -> %s', url, actual_size, dest_path)


def ensure_sam2_checkpoint(progress_callback=None):
    """Return the SAM2 checkpoint's path, downloading it first if missing.
    Matches the layout `worker_sam.py` already expects
    (EDyssey/tracking_utils/SAM2_checkpoints/sam2.1_hiera_large.pt)."""
    dest = os.path.join(SAM2_CHECKPOINT_DIR, SAM2_CHECKPOINT_FILENAME)
    if os.path.isfile(dest) and os.path.getsize(dest) == SAM2_CHECKPOINT_SIZE:
        return dest
    _logger.info('SAM2 checkpoint not found locally, downloading (~%.0f MB)...',
                 SAM2_CHECKPOINT_SIZE / 1e6)
    _download(SAM2_CHECKPOINT_URL, dest, expected_size=SAM2_CHECKPOINT_SIZE,
              progress_callback=progress_callback)
    return dest


def tracker_needs_download(tracking_method):
    """True if `tracking_method` ('csrt'/'mil'/'nano'/'dasiamrpn'/xcorr
    methods) has any model file missing from EDyssey/tracking_utils/
    opencv_models/. csrt/mil/xcorr methods need no external weights at all,
    so this is always False for them."""
    models = _models_for(tracking_method)
    return any(not os.path.isfile(os.path.join(OPENCV_MODELS_DIR, fn)) for fn in models)


def _models_for(tracking_method):
    if tracking_method == 'nano':
        return NANOTRACK_MODELS
    if tracking_method == 'dasiamrpn':
        return DASIAMRPN_MODELS
    return {}


def ensure_tracker_models(tracking_method, progress_callback=None):
    """Download whichever of `tracking_method`'s model file(s) are missing
    from EDyssey/tracking_utils/opencv_models/ - a no-op for 'csrt'/'mil'/
    the xcorr methods, which need no external weights.

    Args:
        progress_callback: optional `callback(bytes_done, total_bytes)` -
            for a multi-file tracker (dasiamrpn), this fires per-file with
            that file's own byte counts, not a combined total.
    """
    for filename, info in _models_for(tracking_method).items():
        dest = os.path.join(OPENCV_MODELS_DIR, filename)
        if os.path.isfile(dest) and os.path.getsize(dest) == info['size']:
            continue
        _logger.info('Tracker model %s not found locally, downloading (~%.1f MB)...',
                     filename, info['size'] / 1e6)
        _download(info['url'], dest, expected_size=info['size'],
                  expected_sha256=info['sha256'], progress_callback=progress_callback)
