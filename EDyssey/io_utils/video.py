# -*- coding: utf-8 -*-
"""Frame/video export: per-frame TIFF stacks and animated clips (DP,
tracking, SAM2-mask overlays), piped to FFmpeg when available and falling
back to a GIF otherwise. Also the small array-shape munging helpers used to
prepare frames for these exports.
"""
import contextlib
import numpy as np
import os
import shutil
import subprocess
import sys
import tempfile
import cv2
import tifffile
import hyperspy.api as hs
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .plotting import ReadableScaleBar, draw_reciprocal_scale_circles
from .progress import _log_or_print

_ffmpeg_cache = None
_ffmpeg_checked = False


def _find_ffmpeg():
    """The on-demand-downloaded ffmpeg (Help > Download ffmpeg..., see
    EDyssey_MainWindow.py) if present, else whatever's on PATH, else None
    (callers fall back to a GIF export). asset_fetch is imported here, not
    at module level - EDyssey.tracking_utils's own __init__.py imports back
    into this same EDyssey.io_utils package (video.py's own), which is
    still mid-initialization the first time video.py itself gets imported
    (via io_utils_ui.py) - a circular import if done eagerly."""
    global _ffmpeg_cache, _ffmpeg_checked
    if not _ffmpeg_checked:
        from EDyssey.tracking_utils.asset_fetch import resolve_ffmpeg_exe
        _ffmpeg_cache = resolve_ffmpeg_exe() or shutil.which('ffmpeg')
        if _ffmpeg_cache:
            mpl.rcParams['animation.ffmpeg_path'] = _ffmpeg_cache
        _ffmpeg_checked = True
    return _ffmpeg_cache

def convert_to_rgb(imgs):
    """Convert a stack of grayscale images to an RGBA/RGB array suitable for video encoding.

    Args:
        imgs: numpy.ndarray of shape (N, W, H).

    Returns:
        numpy.ndarray of shape (W, H, 3, N) with the same dtype as `imgs`.
    """
    l, w, h = imgs.shape
    # imgs_8bit = convert_to_8bit(imgs)
    imgs_rgb = np.zeros((w,h,3,l), dtype=imgs.dtype)
    for i, img in enumerate(imgs):
        imgs_rgb[:,:,:,i] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return imgs_rgb

def create_tracking_result(s, rois):
    """Overlay ROI bounding boxes and index labels on every frame of a signal.

    Args:
        s: HyperSpy Signal2D (N, H, W) serving as the background images.
        rois: List of ROI arrays, each of shape (N, 4) with columns (x, y, w, h).

    Returns:
        HyperSpy Signal2D with white rectangles and numeric labels drawn on each frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.25
    font_thickness = 1
    font_color = (255, 255, 255)  # BGR color format (blue, green, red)
    s_roi = np.zeros(s.data.shape, dtype=np.uint16)
    for i_img, img in enumerate(tqdm(s.data)):
        img = img.copy()  # lightweight copy instead of deepcopy
        for i_r, roi in enumerate(rois):
            (x,y,w,h) = roi[i_img]
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 1)
            cv2.putText(img, str(i_r+1), (x-3, y-3), font, font_scale, font_color, font_thickness)
        s_roi[i_img] = img
    s_roi = hs.signals.Signals2D(s_roi)
    return s_roi

def create_array_from_dissimilar_imgs(imgs, mode='constant', signal=True):
    """Pad a list of differently-shaped 2-D images into a uniform 3-D array.

    Args:
        imgs: List of 2-D numpy arrays with potentially different (H, W) shapes.
        mode: `numpy.pad` mode for fill values. Default is `'constant'` (zero-padding).
        signal: If True, wrap the result in a HyperSpy Signal2D. Default is True.

    Returns:
        numpy.ndarray or HyperSpy Signal2D of shape (N, H_max, W_max) with dtype uint8.
    """
    shapes = [[img.shape[0], img.shape[1]] for img in imgs]
    shapes = np.array(shapes)
    w = shapes[:,0].max()
    h = shapes[:,1].max()
    new_arr = np.zeros((len(imgs), w, h), dtype=np.uint8)
    for i, img in enumerate(imgs):
        w2 = w - img.shape[0]
        h2 = h - img.shape[1]
        new_img = np.pad(img, pad_width=[(0,w2), (0,h2)], mode=mode)
        new_arr[i] = new_img
        # new_arr[i][:img.shape[0], :img.shape[1]] = img
    if signal:
        new_arr = hs.signals.Signal2D(new_arr)
    return new_arr

def _log_error(logger, msg):
    if logger is not None:
        logger.error(msg)
    else:
        print(msg, file=sys.stderr)


def create_frames(pathSave, frames, logger=None):
    """Save each frame in `frames` as a numbered TIFF file under `pathSave`.

    A frame that fails to write is logged and skipped; the remaining frames are still written
    and a RuntimeError summarising the failures is raised at the end.
    """
    if frames is None:
        return
    n_frames = len(frames)
    _log_or_print(logger, f'Writing {n_frames} TIFF frames to {pathSave}...')
    os.makedirs(pathSave, exist_ok=True)
    for name in os.listdir(pathSave):
        old = os.path.join(pathSave, name)
        if os.path.isfile(old) and name.lower().endswith(('.tif', '.tiff')):
            os.remove(old)
    failures = []
    for i, fr in enumerate(tqdm(frames)):
        path = os.path.join(pathSave, f'{i+1:04d}.tif')
        try:
            tifffile.imwrite(path, fr)
        except Exception as exc:
            failures.append((path, exc))
            if len(failures) <= 3:
                _log_error(logger, f'Could not write {path}: {type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError(f'{len(failures)} of {n_frames} TIFF frames failed to write to '
                           f'{pathSave} (first: {type(failures[0][1]).__name__}: {failures[0][1]})')
    _log_or_print(logger, f'Wrote {n_frames} TIFF frames.')


def _offscreen_figure(dpi):
    """(fig, ax) on a plain Agg canvas, not pyplot, whose subplots() builds a Qt window under the
    app's Qt backend, which is unsafe on a worker thread."""
    fig = Figure(dpi=dpi)
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


@contextlib.contextmanager
def _export_guard(label, logger):
    """Log start/end of a clip export, and its failure (with the stage markers logged before it)."""
    _log_or_print(logger, f'Making {label}...')
    try:
        yield
    except Exception as exc:
        _log_error(logger, f'{label} failed: {type(exc).__name__}: {exc}')
        raise
    _log_or_print(logger, f'{label} is created!')


def _start_ffmpeg(cmd, err_file):
    """Start ffmpeg reading frames from a pipe, with every standard handle given explicitly.

    Popen otherwise inherits this process's stdout/stderr, and
    redirect_console_to_logger swaps those under other threads, which can leave a stale handle
    (OSError WinError 6) here.
    """
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err_file)


def _encode_clip(fig, n_frames, draw_frame, fn, fps, artists, blit, label, logger):
    """Render `n_frames` of `fig` (updated by `draw_frame(i)`) to `fn`.mp4 through ffmpeg, or to
    `fn`.gif when no ffmpeg is found. ffmpeg's own error output is kept and reported on failure."""
    path_ffmpeg = _find_ffmpeg()
    if not path_ffmpeg:
        _log_or_print(logger, f'{label}: no ffmpeg found, writing a GIF instead...')

        def _animate(fr_no):
            draw_frame(fr_no)
            return artists

        ani = FuncAnimation(fig, _animate, frames=range(n_frames), blit=blit)
        ani.save(fn + '.gif', writer='pillow', fps=fps)
        return
    _log_or_print(logger, f'{label}: encoding {n_frames} frames with ffmpeg ({path_ffmpeg})...')
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    cmd = [
        path_ffmpeg, '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx264',
        '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
        '-pix_fmt', 'yuv420p', '-crf', '18', fn + '.mp4',
    ]
    log_every = max(1, n_frames // 4)
    with tempfile.TemporaryFile() as err:
        proc = _start_ffmpeg(cmd, err)

        def _ffmpeg_error():
            err.seek(0)
            return err.read().decode(errors='replace').strip()[-1500:]

        try:
            for fr_no in tqdm(range(n_frames)):
                if fr_no % log_every == 0:
                    _log_or_print(logger, f'{label}: frame {fr_no + 1}/{n_frames}')
                draw_frame(fr_no)
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                proc.stdin.write(buf[..., :3].tobytes())
            proc.stdin.close()
            proc.wait()
        except Exception as exc:
            proc.kill()
            proc.wait()
            raise RuntimeError(f'ffmpeg stopped while encoding {fn}.mp4 '
                               f'({type(exc).__name__}: {exc}). ffmpeg said: {_ffmpeg_error()}') from exc
        if proc.returncode != 0:
            raise RuntimeError(f'ffmpeg exited with code {proc.returncode} writing {fn}.mp4. '
                               f'ffmpeg said: {_ffmpeg_error()}')


def create_clip_dp(fn, s, scale=None, center=None, dpi=150, fps=5, vmin=None, vmax=None,
                   cmap='inferno', logger=None):
    """Save an animated video of diffraction patterns with optional reciprocal-space
    scale rings, matching the live UI (concentric dashed circles rather than a
    conventional straight scale bar - see `draw_reciprocal_scale_circles`).

    Pipes raw RGB frames directly to an FFmpeg subprocess for fast encoding.
    Falls back to a GIF via FuncAnimation when FFmpeg is not on PATH.

    Args:
        fn: Output file path without extension (`.mp4` appended, `.gif` as fallback).
        s: Array-like of shape (N, H, W) containing diffraction pattern frames.
        scale: Reciprocal-space calibration in 1/Angstrom per pixel. If None (or
            unparseable), no scale rings are drawn.
        center: (x, y) center to draw the scale rings around, in pixel coordinates.
            Defaults to the image's own geometric center.
        dpi: Dots per inch for the matplotlib figure (controls output resolution).
        fps: Frames per second. Auto-set to max(1, N // 20) when None.
        vmin: Minimum value for the colour normalisation.
        vmax: Maximum value for the colour normalisation.
        cmap: Matplotlib colormap name.
        logger: Optional logger to report status through instead of stdout.
    """
    with _export_guard('DP clip', logger):
        _log_or_print(logger, 'DP clip: building figure...')
        fig, ax = _offscreen_figure(dpi)
        img_plot = ax.imshow(s[0], cmap=cmap, norm=mcolors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=1))
        fig.colorbar(img_plot, ax=ax)
        fig.tight_layout()
        draw_reciprocal_scale_circles(ax, scale, s[0].shape, center=center)
        if fps is None:
            fps = max(1, len(s) // 20)

        def _update_dp(fr_no):
            img_plot.set_data(s[fr_no])
            img_plot.set_clim(vmin=s[fr_no].min(), vmax=s[fr_no].max())

        _encode_clip(fig, len(s), _update_dp, fn, fps, (img_plot,), True, 'DP clip', logger)


def create_clip_tracking(fn, imgs, rois=None, ref_rois=None, scale=None, dpi=150,
                         fps=5, duration=None, cmap='viridis', logger=None):
    """Save an animated video of navigation images with an optional tracking ROI overlay.

    Pipes raw RGB frames directly to an FFmpeg subprocess for fast encoding.
    Falls back to a GIF via FuncAnimation when FFmpeg is not on PATH.

    Args:
        fn: Output file path without extension (`.mp4` appended, `.gif` as fallback).
        imgs: numpy.ndarray of shape (N, H, W) with navigation image frames.
        rois: Optional array/list of shape (N, 4) with (x, y, w, h) ROI per frame.
        ref_rois: Optional array/list of shape (N, 4) with (x, y, w, h) per
            frame for a second, reference ROI - drawn in a distinct dashed
            yellow box alongside `rois`' own red one. For a ROI-in-ROI
            object's own clip specifically (see Tab_Tracking_CV2's
            _save_results_impl), this is its parent/reference ROI's tracked
            box, so the clip still shows where the inner ROI sits relative
            to the outer one it was tracked against, not just the inner box
            in isolation. None (default) draws no second box.
        scale: Real-space pixel size in nm. If None, no scale bar is drawn.
        dpi: Dots per inch for the matplotlib figure (controls output resolution).
        fps: Frames per second. Auto-set from `duration` or max(1, N // 20) when None.
        duration: Target clip duration in seconds used to derive `fps` when provided.
        cmap: Matplotlib colormap name.
        logger: Optional logger to report status through instead of stdout.
    """
    with _export_guard('tracking clip', logger):
        _log_or_print(logger, 'tracking clip: building figure...')
        fig, ax = _offscreen_figure(dpi)
        img_plot = ax.imshow(imgs[0], cmap=cmap)
        fig.colorbar(img_plot, ax=ax)
        if scale is not None:
            scalebar = ReadableScaleBar(scale, 'nm', dimension='si-length',
                                location='lower left', box_alpha=0, color='w')
            ax.add_artist(scalebar)
        active_rect = [None]
        ref_rect = [None]
        if rois is not None:
            x, y, w, h = rois[0]
            active_rect[0] = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(active_rect[0])
        if ref_rois is not None:
            x, y, w, h = ref_rois[0]
            ref_rect[0] = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='yellow',
                                            linestyle='--', facecolor='none')
            ax.add_patch(ref_rect[0])
        fig.tight_layout()
        if fps is None:
            denom = duration if duration else 20
            fps = max(1, len(imgs) // denom)

        def _update_tr(fr_no):
            img_plot.set_data(imgs[fr_no])
            img_plot.set_clim(vmin=imgs[fr_no].min(), vmax=imgs[fr_no].max())
            if rois is not None:
                if active_rect[0] is not None:
                    active_rect[0].remove()
                x, y, w, h = rois[fr_no]
                active_rect[0] = patches.Rectangle((x, y), w, h, linewidth=1,
                                                   edgecolor='r', facecolor='none')
                ax.add_patch(active_rect[0])
            if ref_rois is not None:
                if ref_rect[0] is not None:
                    ref_rect[0].remove()
                x, y, w, h = ref_rois[fr_no]
                ref_rect[0] = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='yellow',
                                                linestyle='--', facecolor='none')
                ax.add_patch(ref_rect[0])

        _encode_clip(fig, len(imgs), _update_tr, fn, fps, (img_plot,), False, 'tracking clip', logger)


def create_clip_tracking_with_mask(fn, imgs, masks, obj_id=1, scale=None,
                                   dpi=150, fps=5, cmap='viridis', logger=None):
    """Save an animated video of navigation images with a SAM2 segmentation mask overlay.

    Pipes raw RGB frames directly to an FFmpeg subprocess for fast encoding.
    Falls back to a GIF via FuncAnimation when FFmpeg is not on PATH.

    Args:
        fn: Output file path without extension (`.mp4` appended, `.gif` as fallback).
        imgs: numpy.ndarray of shape (N, H, W) with navigation image frames.
        masks: Boolean array of shape (N, H, W) with the segmentation mask per frame.
        obj_id: Object ID used to select the overlay colour from the `tab10` colormap.
        scale: Real-space pixel size in nm. If None, no scale bar is drawn.
        dpi: Dots per inch for the matplotlib figure (controls output resolution).
        fps: Frames per second. Auto-set to max(1, N // 20) when None.
        cmap: Matplotlib colormap name for the background image.
        logger: Optional logger to report status through instead of stdout.
    """
    with _export_guard('tracking clip', logger):
        _log_or_print(logger, 'tracking clip: building figure...')
        fig, ax = _offscreen_figure(dpi)
        img_plot = ax.imshow(imgs[0], cmap=cmap)
        fig.colorbar(img_plot, ax=ax)
        img_mask_plot = ax.imshow(np.zeros((*imgs[0].shape, 4)))
        if scale is not None:
            scalebar = ReadableScaleBar(scale, 'nm', dimension='si-length',
                                location='lower left', box_alpha=0, color='w')
            ax.add_artist(scalebar)
        fig.tight_layout()
        color = np.array([*mpl.colormaps['tab10'](obj_id)[:3], 0.6])

        def _make_mask_rgba(mask):
            h, w = mask.shape[-2:]
            return mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        def _update_mask(fr_no):
            img_plot.set_data(imgs[fr_no])
            img_plot.set_clim(vmin=imgs[fr_no].min(), vmax=imgs[fr_no].max())
            img_mask_plot.set_data(_make_mask_rgba(masks[fr_no]))

        img_mask_plot.set_data(_make_mask_rgba(masks[0]))
        if fps is None:
            fps = max(1, len(imgs) // 20)
        _encode_clip(fig, len(imgs), _update_mask, fn, fps, (img_plot, img_mask_plot), True,
                     'tracking clip', logger)
