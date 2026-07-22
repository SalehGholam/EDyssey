# Canvas mouse controls

Applies to the **SAM2 Seg.** and **Tracking by CV2** tabs. Both embed a
matplotlib navigation toolbar (Home / Pan / Zoom-rectangle / Save) above
each canvas — use it, or the scroll wheel below, to navigate images. A
plain click or click+drag no longer adds points/ROIs, so it's free for
panning and zooming.

## SAM2 Seg. tab (nav. image panel)

| Action | Effect |
|---|---|
| `Ctrl` + Left Click | Add a **positive** point |
| `Ctrl` + Right Click | Add a **negative** point |
| `Ctrl` + `Shift` + Click | Add the point to the **currently selected object** instead of starting a new one |
| Middle Click | Delete the last added point |
| Scroll Wheel | Zoom in/out, centered on the cursor |
| Click + Drag (no `Ctrl`) | Pan/zoom via the navigation toolbar's active tool |

## Tracking by CV2 tab (nav. + tracking-results panels)

| Action | Effect |
|---|---|
| `Ctrl` + Left Click + Drag | Draw a **new ROI** |
| `Ctrl` + Right Click | Add the current frame as an **init point** to the selected/last ROI |
| `Ctrl` + Left/Right Click + Drag (on the "Tracking Results" panel) | Draw a **ROI-in-ROI** against the selected reference ROI |
| Scroll Wheel | Zoom in/out, centered on the cursor |
| Click + Drag (no `Ctrl`) | Pan/zoom via the navigation toolbar's active tool |

Zoom persists across frame changes and redraws. Use the toolbar's "Home"
button to reset back to the full image view.
