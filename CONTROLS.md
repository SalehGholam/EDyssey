# Canvas mouse controls

Applies to the **SAM2 Seg.**, **Tracking by CV2**, and **ROI on 4D** tabs.
Each embeds a matplotlib navigation toolbar (Home / Pan / Zoom-rectangle /
Save) above its canvas(es) — use it, or `Ctrl` + scroll wheel below, to
navigate images. A plain click or click+drag no longer adds points/ROIs, so
it's free for panning and zooming.

## Diffraction-pattern center (all three tabs)

Every diffraction-pattern (DP) plot's reciprocal-space 1/A rings need a
center. The "Auto-center" checkbox next to the reciprocal-space scale field
controls how it's found:

| Action | Effect |
|---|---|
| "Auto-center" checked (default) | The center is re-found automatically (large-sigma Gaussian blur, robust to hot pixels) after every redraw |
| "Auto-center" unchecked, then `Ctrl` + Left Click on the DP plot | Sets the ring center manually to the clicked point |

The current mode and, once found, the center itself are shown as the DP
axis's x-axis label.

## SAM2 Seg. tab (nav. image panel)

| Action | Effect |
|---|---|
| `Ctrl` + Left Click | Add a **positive** point |
| `Ctrl` + Right Click | Add a **negative** point |
| `Ctrl` + `Shift` + Click | Add the point to the **currently selected object** instead of starting a new one |
| Middle Click | Delete the last added point |
| `Ctrl` + Scroll Wheel | Zoom in/out, centered on the cursor |
| Click + Drag (no `Ctrl`) | Pan/zoom via the navigation toolbar's active tool |

## Tracking by CV2 tab (nav. + tracking-results panels)

| Action | Effect |
|---|---|
| `Ctrl` + Left Click + Drag | Draw a **new ROI** |
| `Ctrl` + Right Click | Add the current frame as an **init point** to the selected/last ROI |
| `Ctrl` + Left/Right Click + Drag (on the "Tracking Results" panel) | Draw a **ROI-in-ROI** against the selected reference ROI |
| `Ctrl` + Scroll Wheel | Zoom in/out, centered on the cursor |
| Click + Drag (no `Ctrl`) | Pan/zoom via the navigation toolbar's active tool |

## ROI on 4D tab (nav. image panel)

| Action | Effect |
|---|---|
| `Ctrl` + Left Click + Drag | Draw a **new ROI** (also usable as a SAM2 box prompt) |
| `Shift` + Click | Add a SAM2 point (left = positive, right = negative) |
| Middle Click | Delete the last added SAM2 point |
| `Ctrl` + Scroll Wheel | Zoom in/out, centered on the cursor |
| Click + Drag (no `Ctrl`) | Pan/zoom via the navigation toolbar's active tool |

Zoom persists across frame changes and redraws. Use the toolbar's "Home"
button to reset back to the full image view.
