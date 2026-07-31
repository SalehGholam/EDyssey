# -*- coding: utf-8 -*-
"""Frame/video export: per-frame TIFF stacks and animated clips (DP,
tracking, SAM2-mask overlays), piped to FFmpeg when available and falling
back to a GIF otherwise. Also the small array-shape munging helpers used to
prepare frames for these exports.
"""
import numpy as np
import os
import shutil
import subprocess
import cv2
import tifffile
import hyperspy.api as hs
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from .plotting import ReadableScaleBar, draw_reciprocal_scale_circles
from .progress import _log_or_print

_ffmpeg = shutil.which('ffmpeg')
if _ffmpeg:
    plt.rcParams['animation.ffmpeg_path'] = _ffmpeg

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

def create_frames(pathSave, frames):
    """Save each frame in `frames` as a numbered TIFF file under `pathSave`."""
    if frames is None:
        return
    if os.path.isdir(pathSave):
        [os.remove(os.path.join(pathSave, fn)) for fn in os.listdir(pathSave)]
    else:
        os.mkdir(pathSave)
    for i, fr in enumerate(tqdm(frames)):
        tifffile.imwrite(os.path.join(pathSave, f'{i+1:04d}.tif'), fr)

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
    _log_or_print(logger, 'Making DP clip...')
    plt.ioff()
    fig, ax = plt.subplots(dpi=dpi)
    img_plot = ax.imshow(s[0], cmap=cmap, norm=mcolors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=1))
    fig.colorbar(img_plot, ax=ax)
    fig.tight_layout()
    draw_reciprocal_scale_circles(ax, scale, s[0].shape, center=center)
    if fps is None:
        fps = max(1, len(s) // 20)
    path_ffmpeg = shutil.which('ffmpeg')
    if path_ffmpeg:
        fig.canvas.draw()
        W, H = fig.canvas.get_width_height()
        cmd = [
            path_ffmpeg, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
            '-i', '-', '-an', '-vcodec', 'libx264',
            '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
            '-pix_fmt', 'yuv420p', '-crf', '18', fn + '.mp4',
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr_no in tqdm(range(len(s))):
            img_plot.set_data(s[fr_no])
            img_plot.set_clim(vmin=s[fr_no].min(), vmax=s[fr_no].max())
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            proc.stdin.write(buf[..., :3].tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        def _update_dp(fr_no):
            img_plot.set_data(s[fr_no])
            img_plot.set_clim(vmin=s[fr_no].min(), vmax=s[fr_no].max())
            return (img_plot,)
        ani = FuncAnimation(fig, _update_dp, frames=range(s.shape[0]), blit=True)
        ani.save(fn + '.gif', fps=fps)
        _log_or_print(logger, 'No ffmpeg found, saved as GIF')
    plt.close(fig)
    plt.ion()
    _log_or_print(logger, 'DP clip is created!')

def create_clip_tracking(fn, imgs, rois=None, scale=None, dpi=150,
                         fps=5, duration=None, cmap='viridis', logger=None):
    """Save an animated video of navigation images with an optional tracking ROI overlay.

    Pipes raw RGB frames directly to an FFmpeg subprocess for fast encoding.
    Falls back to a GIF via FuncAnimation when FFmpeg is not on PATH.

    Args:
        fn: Output file path without extension (`.mp4` appended, `.gif` as fallback).
        imgs: numpy.ndarray of shape (N, H, W) with navigation image frames.
        rois: Optional array/list of shape (N, 4) with (x, y, w, h) ROI per frame.
        scale: Real-space pixel size in nm. If None, no scale bar is drawn.
        dpi: Dots per inch for the matplotlib figure (controls output resolution).
        fps: Frames per second. Auto-set from `duration` or max(1, N // 20) when None.
        duration: Target clip duration in seconds used to derive `fps` when provided.
        cmap: Matplotlib colormap name.
        logger: Optional logger to report status through instead of stdout.
    """
    _log_or_print(logger, 'Making tracking clip...')
    plt.ioff()
    fig, ax = plt.subplots(dpi=dpi)
    img_plot = ax.imshow(imgs[0], cmap=cmap)
    fig.colorbar(img_plot, ax=ax)
    if scale is not None:
        scalebar = ReadableScaleBar(scale, 'nm', dimension='si-length',
                            location='lower left', box_alpha=0, color='w')
        ax.add_artist(scalebar)
    active_rect = [None]
    if rois is not None:
        x, y, w, h = rois[0]
        active_rect[0] = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(active_rect[0])
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

    path_ffmpeg = shutil.which('ffmpeg')
    if path_ffmpeg:
        fig.canvas.draw()
        W, H = fig.canvas.get_width_height()
        cmd = [
            path_ffmpeg, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
            '-i', '-', '-an', '-vcodec', 'libx264',
            '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
            '-pix_fmt', 'yuv420p', '-crf', '18', fn + '.mp4',
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr_no in tqdm(range(len(imgs))):
            _update_tr(fr_no)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            proc.stdin.write(buf[..., :3].tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        def _anim_tr(fr_no):
            _update_tr(fr_no)
            return (img_plot,)
        ani = FuncAnimation(fig, _anim_tr, frames=range(imgs.shape[0]), blit=False)
        ani.save(fn + '.gif', fps=fps)
        _log_or_print(logger, 'No ffmpeg found, saved as GIF')
    plt.close(fig)
    plt.ion()
    _log_or_print(logger, 'Tracking clip is created!')

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
    _log_or_print(logger, 'Making tracking clip...')
    plt.ioff()
    fig, ax = plt.subplots(dpi=dpi)
    img_plot = ax.imshow(imgs[0], cmap=cmap)
    fig.colorbar(img_plot, ax=ax)
    img_mask_plot = ax.imshow(np.zeros((*imgs[0].shape, 4)))
    if scale is not None:
        scalebar = ReadableScaleBar(scale, 'nm', dimension='si-length',
                            location='lower left', box_alpha=0, color='w')
        ax.add_artist(scalebar)
    fig.tight_layout()
    cmap_tab10 = plt.get_cmap('tab10')
    color = np.array([*cmap_tab10(obj_id)[:3], 0.6])

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
    path_ffmpeg = shutil.which('ffmpeg')
    if path_ffmpeg:
        fig.canvas.draw()
        W, H = fig.canvas.get_width_height()
        cmd = [
            path_ffmpeg, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
            '-i', '-', '-an', '-vcodec', 'libx264',
            '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
            '-pix_fmt', 'yuv420p', '-crf', '18', fn + '.mp4',
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr_no in tqdm(range(len(imgs))):
            _update_mask(fr_no)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            proc.stdin.write(buf[..., :3].tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        def _anim_mask(fr_no):
            _update_mask(fr_no)
            return (img_plot,)
        ani = FuncAnimation(fig, _anim_mask, frames=range(imgs.shape[0]), blit=True)
        ani.save(fn + '.gif', fps=fps)
        _log_or_print(logger, 'No ffmpeg found, saved as GIF')
    plt.close(fig)
    plt.ion()
    _log_or_print(logger, 'Tracking clip is created!')
