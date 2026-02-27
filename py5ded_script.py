# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 19:05:22 2026

@author: sgholam
"""

import os
import numpy as np
import hyperspy.api as hs
import py4DTomo.io_utils as io
import py4DTomo.tracking_utils as trk
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RectangleSelector, Button, TextBox
from collections import defaultdict
#%% entry
dtype = 'hdf5'
path_4d = r'C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\py5DED\other_scripts\test_data\4D'
fns_4d = io.get_4d_files(path_4d, dtype)

# fns_4d = fns_4d[:5]
#%% load one file to check
num = 0
s = io.load_signal(fns_4d[num], lazy=True)
s, f = s
s.plot(norm='symlog', cmap='inferno')
# f.close() # closing the file
#%% calculate navigation images
shape_x, shape_y = io.get_scan_size(fns_4d[0])
arr_nav = np.zeros((len(fns_4d), shape_x, shape_y), dtype='uint64')
for i, fn in enumerate(tqdm(fns_4d)):
    arr_nav[i] = io.calculate_nav_img(fn)
#%% plot navigation images
s_nav = hs.signals.Signal2D(arr_nav)
s_nav.plot(cmap='viridis')
#%% plot rois
def roi_browser(images, *, title="ROI browser", cmap="gray", vmin=None, vmax=None):
    imgs = np.asarray(images)
    if imgs.ndim == 2:
        imgs = imgs[None, ...]
    if imgs.ndim != 3:
        raise ValueError("images must be shape (H,W) or (N,H,W)")

    n, h, w = imgs.shape

    backend = matplotlib.get_backend().lower()
    if "inline" in backend:
        print(
            "WARNING: Inline Matplotlib backend detected; widgets won't work.\n"
            "In Spyder: Preferences → IPython console → Graphics → Backend = Qt5 (or QtAgg), then restart."
        )

    current_index = 0
    current_roi_id = 1
    store = defaultdict(list)   # roi_id -> list of entries
    history = []                # (roi_id, idx_in_store_list)
    done = {"flag": False}

    fig = plt.figure(figsize=(10.5, 6))
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

    ax = fig.add_axes([0.08, 0.18, 0.72, 0.76])
    im_artist = ax.imshow(imgs[current_index], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    def _set_title():
        ax.set_title(f"Image {current_index+1}/{n}  |  ROI ID: {current_roi_id}")

    _set_title()

    # Slider
    ax_slider = fig.add_axes([0.08, 0.08, 0.72, 0.04])
    slider = Slider(ax_slider, "Image", 0, n - 1, valinit=0, valstep=1)

    # Buttons (right column)
    ax_previd = fig.add_axes([0.82, 0.86, 0.08, 0.06])
    ax_nextid = fig.add_axes([0.91, 0.86, 0.08, 0.06])
    btn_previd = Button(ax_previd, "Prev ID")
    btn_nextid = Button(ax_nextid, "Next ID")

    ax_idbox = fig.add_axes([0.82, 0.78, 0.17, 0.06])
    id_box = TextBox(ax_idbox, "Go ID", initial=str(current_roi_id))

    ax_newid  = fig.add_axes([0.82, 0.70, 0.17, 0.07])
    ax_undo   = fig.add_axes([0.82, 0.60, 0.17, 0.07])
    ax_finish = fig.add_axes([0.82, 0.50, 0.17, 0.07])
    btn_newid  = Button(ax_newid, "New ID")
    btn_undo   = Button(ax_undo, "Undo")
    btn_finish = Button(ax_finish, "Finish")

    status = fig.text(
        0.08, 0.02,
        "Draw ROI: click-drag. IDs: Prev/Next or type in Go ID. Keys: n=new, [ / ] id, u=undo, enter=finish, g=go id box.",
        fontsize=10
    )

    def _show_only_current_image_rois():
        for rid, entries in store.items():
            for e in entries:
                if not e.get("alive", True):
                    continue
                vis = (e["img"] == current_index)
                e["rect"].set_visible(vis)
                e["text"].set_visible(vis)

    def _set_id(new_id: int):
        nonlocal current_roi_id
        if new_id < 1:
            return
        current_roi_id = int(new_id)
        id_box.set_val(str(current_roi_id))  # updates box + triggers callback; safe but can recurse
        # To prevent recursion issues, we keep callback very light.
        _set_title()
        fig.canvas.draw_idle()

    # Make TextBox submit jump IDs
    _idbox_internal_lock = {"busy": False}
    def _on_id_submit(txt):
        if _idbox_internal_lock["busy"]:
            return
        try:
            val = int(txt)
        except Exception:
            status.set_text("Go ID: please type an integer >= 1.")
            fig.canvas.draw_idle()
            return
        if val < 1:
            status.set_text("Go ID must be >= 1.")
            fig.canvas.draw_idle()
            return
        nonlocal_current = val  # just clarity
        # Avoid infinite loop because _set_id calls set_val which triggers submit
        _idbox_internal_lock["busy"] = True
        try:
            nonlocal current_roi_id
            current_roi_id = int(nonlocal_current)
            _set_title()
            fig.canvas.draw_idle()
        finally:
            _idbox_internal_lock["busy"] = False

    id_box.on_submit(_on_id_submit)

    def _prev_id():
        nonlocal current_roi_id
        if current_roi_id > 1:
            current_roi_id -= 1
            _idbox_internal_lock["busy"] = True
            try:
                id_box.set_val(str(current_roi_id))
            finally:
                _idbox_internal_lock["busy"] = False
            _set_title()
            fig.canvas.draw_idle()

    def _next_id():
        nonlocal current_roi_id
        current_roi_id += 1
        _idbox_internal_lock["busy"] = True
        try:
            id_box.set_val(str(current_roi_id))
        finally:
            _idbox_internal_lock["busy"] = False
        _set_title()
        fig.canvas.draw_idle()

    def _new_id():
        _next_id()

    def _undo():
        if not history:
            status.set_text("Nothing to undo.")
            fig.canvas.draw_idle()
            return
        rid, idx = history.pop()
        e = store[rid][idx]
        if not e.get("alive", True):
            status.set_text("Last ROI was already removed.")
            fig.canvas.draw_idle()
            return

        e["alive"] = False
        try: e["rect"].remove()
        except Exception: pass
        try: e["text"].remove()
        except Exception: pass

        status.set_text(f"Undid ROI id={rid} on image={e['img']}.")
        fig.canvas.draw_idle()

    def _finish():
        done["flag"] = True
        plt.close(fig)

    def _add_roi(x0, y0, x1, y1):
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return

        xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
        ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)

        xmin = float(np.clip(xmin, -0.5, w - 0.5))
        xmax = float(np.clip(xmax, -0.5, w - 0.5))
        ymin = float(np.clip(ymin, -0.5, h - 0.5))
        ymax = float(np.clip(ymax, -0.5, h - 0.5))

        ww = xmax - xmin
        hh = ymax - ymin
        if ww <= 0 or hh <= 0:
            return

        rect = plt.Rectangle((xmin, ymin), ww, hh, fill=False, linewidth=2)
        ax.add_patch(rect)
        txt = ax.text(
            xmin, ymin, str(current_roi_id),
            va="bottom", ha="left", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.15", alpha=0.7)
        )

        entry = {"img": int(current_index), "xywh": [xmin, ymin, ww, hh], "rect": rect, "text": txt, "alive": True}
        store[current_roi_id].append(entry)
        history.append((current_roi_id, len(store[current_roi_id]) - 1))

        _show_only_current_image_rois()
        fig.canvas.draw_idle()

    def _on_select(eclick, erelease):
        if eclick.inaxes != ax or erelease.inaxes != ax:
            return
        _add_roi(eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)

    rect_selector = RectangleSelector(
        ax, _on_select,
        useblit=True,
        button=[1],
        interactive=False,
        spancoords="data",
        minspanx=2, minspany=2,
        props=dict(fill=False)
    )

    def _on_slider(val):
        nonlocal current_index
        current_index = int(val)
        im_artist.set_data(imgs[current_index])
        _set_title()
        _show_only_current_image_rois()
        fig.canvas.draw_idle()
    
    def _step_image(delta):
        nonlocal current_index
        new_idx = int(np.clip(current_index + delta, 0, n - 1))
        if new_idx != current_index:
            slider.set_val(new_idx)  # this will call _on_slider internally

    
    slider.on_changed(_on_slider)

    # Wire buttons
    btn_previd.on_clicked(lambda event: _prev_id())
    btn_nextid.on_clicked(lambda event: _next_id())
    btn_newid.on_clicked(lambda event: _new_id())
    btn_undo.on_clicked(lambda event: _undo())
    btn_finish.on_clicked(lambda event: _finish())

    # Keys
    def _on_key(event):
        if event.key == "n":
            _new_id()
        elif event.key == "[":
            _prev_id()
        elif event.key == "]":
            _next_id()
        elif event.key == "u":
            _undo()
        elif event.key in ("enter", "return"):
            _finish()
        elif event.key == "g":
            try:
                id_box.cursor_index = len(id_box.text)
                id_box.capturekeystrokes = True
                id_box.begin_typing()
            except Exception:
                pass
        elif event.key == "left":
            _step_image(-1)
        elif event.key == "right":
            _step_image(+1)


    fig.canvas.mpl_connect("key_press_event", _on_key)

    # Keep strong references
    fig._roi_browser_refs = (slider, btn_previd, btn_nextid, btn_newid, btn_undo, btn_finish, id_box, rect_selector)

    # Spyder/Qt: show is non-blocking, so block manually
    plt.show(block=False)
    while plt.fignum_exists(fig.number) and not done["flag"]:
        plt.pause(0.05)

    out = {}
    for rid, entries in store.items():
        coords = []
        for e in entries:
            if not e.get("alive", True):
                continue
            coords.append((int(e["img"]), [float(v) for v in e["xywh"]]))
        if coords:
            out[int(rid)] = coords

    return out


test = roi_browser(arr_nav)
#%% get points
def point_browser_pn(images, *, title="Point browser (+/-)", cmap="gray", vmin=None, vmax=None):
    imgs = np.asarray(images)
    if imgs.ndim == 2:
        imgs = imgs[None, ...]
    if imgs.ndim != 3:
        raise ValueError("images must be shape (H,W) or (N,H,W)")

    n, h, w = imgs.shape

    backend = matplotlib.get_backend().lower()
    if "inline" in backend:
        print(
            "WARNING: Inline Matplotlib backend detected; widgets won't work.\n"
            "In Spyder: Preferences → IPython console → Graphics → Backend = Qt5 (or QtAgg), then restart."
        )

    current_index = 0
    current_point_id = 1
    store = defaultdict(list)  # pid -> entries
    history = []
    done = {"flag": False}

    fig = plt.figure(figsize=(10.5, 6))
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

    ax = fig.add_axes([0.08, 0.18, 0.72, 0.76])
    im_artist = ax.imshow(imgs[current_index], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    def _set_title():
        ax.set_title(f"Image {current_index+1}/{n}  |  Point ID: {current_point_id}")

    _set_title()

    ax_slider = fig.add_axes([0.08, 0.08, 0.72, 0.04])
    slider = Slider(ax_slider, "Image", 0, n - 1, valinit=0, valstep=1)

    # ID controls
    ax_previd = fig.add_axes([0.82, 0.86, 0.08, 0.06])
    ax_nextid = fig.add_axes([0.91, 0.86, 0.08, 0.06])
    btn_previd = Button(ax_previd, "Prev ID")
    btn_nextid = Button(ax_nextid, "Next ID")

    ax_idbox = fig.add_axes([0.82, 0.78, 0.17, 0.06])
    id_box = TextBox(ax_idbox, "Go ID", initial=str(current_point_id))

    ax_newid  = fig.add_axes([0.82, 0.70, 0.17, 0.07])
    ax_undo   = fig.add_axes([0.82, 0.60, 0.17, 0.07])
    ax_finish = fig.add_axes([0.82, 0.50, 0.17, 0.07])
    btn_newid  = Button(ax_newid, "New ID")
    btn_undo   = Button(ax_undo, "Undo")
    btn_finish = Button(ax_finish, "Finish")

    status = fig.text(
        0.08, 0.02,
        "Left click = +, Right click = -. IDs: Prev/Next or Go ID. Keys: n=new, [ / ] id, u=undo, enter=finish, g=go id box.",
        fontsize=10
    )

    def _show_only_current_image_points():
        for pid, entries in store.items():
            for e in entries:
                if not e.get("alive", True):
                    continue
                vis = (e["img"] == current_index)
                e["artist"].set_visible(vis)
                e["label"].set_visible(vis)

    _idbox_internal_lock = {"busy": False}
    def _on_id_submit(txt):
        if _idbox_internal_lock["busy"]:
            return
        try:
            val = int(txt)
        except Exception:
            status.set_text("Go ID: please type an integer >= 1.")
            fig.canvas.draw_idle()
            return
        if val < 1:
            status.set_text("Go ID must be >= 1.")
            fig.canvas.draw_idle()
            return
        nonlocal current_point_id
        current_point_id = int(val)
        _set_title()
        fig.canvas.draw_idle()

    id_box.on_submit(_on_id_submit)

    def _prev_id():
        nonlocal current_point_id
        if current_point_id > 1:
            current_point_id -= 1
            _idbox_internal_lock["busy"] = True
            try:
                id_box.set_val(str(current_point_id))
            finally:
                _idbox_internal_lock["busy"] = False
            _set_title()
            fig.canvas.draw_idle()

    def _next_id():
        nonlocal current_point_id
        current_point_id += 1
        _idbox_internal_lock["busy"] = True
        try:
            id_box.set_val(str(current_point_id))
        finally:
            _idbox_internal_lock["busy"] = False
        _set_title()
        fig.canvas.draw_idle()

    def _new_id():
        _next_id()

    def _undo():
        if not history:
            status.set_text("Nothing to undo.")
            fig.canvas.draw_idle()
            return
        pid, idx = history.pop()
        e = store[pid][idx]
        if not e.get("alive", True):
            status.set_text("Last point was already removed.")
            fig.canvas.draw_idle()
            return

        e["alive"] = False
        try: e["artist"].remove()
        except Exception: pass
        try: e["label"].remove()
        except Exception: pass

        status.set_text(f"Undid point id={pid} on image={e['img']} (sign={e['sign']}).")
        fig.canvas.draw_idle()

    def _finish():
        done["flag"] = True
        plt.close(fig)

    def _add_point(x, y, sign):
        if x is None or y is None:
            return
        x = float(np.clip(x, -0.5, w - 0.5))
        y = float(np.clip(y, -0.5, h - 0.5))

        marker = "+" if sign > 0 else "x"
        artist = ax.plot([x], [y], marker=marker, linestyle="None", markersize=10)[0]
        label = ax.text(
            x, y, f"{current_point_id}",
            fontsize=10, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", alpha=0.7)
        )

        entry = {
            "img": int(current_index),
            "sign": int(sign),
            "x": x,
            "y": y,
            "artist": artist,
            "label": label,
            "alive": True
        }
        store[current_point_id].append(entry)
        history.append((current_point_id, len(store[current_point_id]) - 1))

        _show_only_current_image_points()
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            _add_point(event.xdata, event.ydata, +1)
        elif event.button == 3:
            _add_point(event.xdata, event.ydata, 0)

    fig.canvas.mpl_connect("button_press_event", _on_click)

    def _on_slider(val):
        nonlocal current_index
        current_index = int(val)
        im_artist.set_data(imgs[current_index])
        _set_title()
        _show_only_current_image_points()
        fig.canvas.draw_idle()
    
    def _step_image(delta):
        nonlocal current_index
        new_idx = int(np.clip(current_index + delta, 0, n - 1))
        if new_idx != current_index:
            slider.set_val(new_idx)

    
    slider.on_changed(_on_slider)

    # Buttons
    btn_previd.on_clicked(lambda event: _prev_id())
    btn_nextid.on_clicked(lambda event: _next_id())
    btn_newid.on_clicked(lambda event: _new_id())
    btn_undo.on_clicked(lambda event: _undo())
    btn_finish.on_clicked(lambda event: _finish())

    # Keys
    def _on_key(event):
        if event.key == "n":
            _new_id()
        elif event.key == "[":
            _prev_id()
        elif event.key == "]":
            _next_id()
        elif event.key == "u":
            _undo()
        elif event.key in ("enter", "return"):
            _finish()
        elif event.key == "g":
            try:
                id_box.cursor_index = len(id_box.text)
                id_box.capturekeystrokes = True
                id_box.begin_typing()
            except Exception:
                pass
        elif event.key == "left":
            _step_image(-1)
        elif event.key == "right":
            _step_image(+1)


    fig.canvas.mpl_connect("key_press_event", _on_key)

    fig._point_browser_refs = (slider, btn_previd, btn_nextid, btn_newid, btn_undo, btn_finish, id_box)

    plt.show(block=False)
    while plt.fignum_exists(fig.number) and not done["flag"]:
        plt.pause(0.05)

    out = {}
    for pid, entries in store.items():
        pts = []
        for e in entries:
            if not e.get("alive", True):
                continue
            pts.append((int(e["img"]), int(e["sign"]), float(e["x"]), float(e["y"])))
        if pts:
            out[int(pid)] = pts

    return out

test = point_browser_pn(arr_nav)
#%%
