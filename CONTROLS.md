# Canvas mouse controls

Applies to all four tabs: **ROI on 4D**, **Navigator**, **ROI Tracker**, and
**SAM2 Tracker**. Each embeds the standard matplotlib navigation toolbar
(Home / Pan / Zoom-rectangle / Save) - reachable either via the vertical
ribbon of icons docked to the right of the canvas, or `Ctrl` + scroll wheel
to zoom. A plain click or click+drag pans/zooms; adding points or drawing
ROIs on the canvas requires holding a modifier key (see below), so the two
don't conflict.

Every tab also has a **"?" (help) button** on its ribbon that opens a
"Shortcuts & Controls" popup listing that tab's own mouse/keyboard controls -
the same information as this file, but always up to date with that tab's
actual current wiring. Use this file for a quick overview across tabs; use
the in-app "?" button as the authoritative reference for one tab.

## Diffraction-pattern center (all tabs with a DP plot)

Every diffraction-pattern (DP) plot's reciprocal-space 1/A rings need a
center. Centering is manual: it persists across redraws until you change it
via one of:

| Action | Effect |
|---|---|
| Click "Center" (near the reciprocal-space scale field, or the ribbon's bullseye icon) | Finds the beam center automatically (large-sigma Gaussian blur, robust to hot pixels) and jumps the rings there |
| `Ctrl` + Click on the DP plot | Sets the ring center manually to the clicked point |

## ROI on 4D tab

| Action | Effect |
|---|---|
| `Ctrl` + Left Click + Drag (Nav. Image) | Draw a **new rectangular ROI** (its diffraction pattern loads automatically) |
| `Shift` + Click (Nav. Image) | Add a SAM2 point (left = positive, right = negative) |
| Middle Click (Nav. Image) | Delete the last added SAM2 point |
| Click "Center" (Scale bars), or `Ctrl` + Click (DP) | Find/set the reciprocal-space rings' center |
| `Ctrl` + Drag the virtual-detector mask's center "+" or a circle edge | Move/resize the virtual detector |
| `Ctrl` + Scroll Wheel | Zoom in/out, centered on the cursor |

## Navigator tab

| Action | Effect |
|---|---|
| `Ctrl` + Drag (Nav. Image) | Draw a scan-space ROI (its Summed DP loads automatically) |
| Right Click (Nav. Image) | Remove the drawn ROI |
| Click "Center" (Files), or `Ctrl` + Click away from the mask (Summed DP) | Find/set the reciprocal-space rings' center |
| Click "Auto Center", or `Ctrl` + Drag the mask's center "+"/a circle edge (Summed DP) | Find/move/resize the virtual detector mask |
| `Ctrl` + Scroll Wheel | Zoom in/out, centered on the cursor |

## ROI Tracker tab

| Action | Effect |
|---|---|
| `Ctrl` + Left Click + Drag (Nav. Image) | Draw a **new ROI** |
| `Ctrl` + Right Click (Nav. Image) | Add the current frame as an **init point** to the selected/last ROI |
| `Ctrl` + Drag (Track Image, after selecting a reference ROI) | Draw a **ROI-in-ROI** |
| Click "Center" (Input Parameters), or `Ctrl` + Click (DP) | Find/set the reciprocal-space rings' center |
| `Ctrl` + Scroll Wheel | Zoom in/out, centered on the cursor |

## SAM2 Tracker tab

| Action | Effect |
|---|---|
| `Ctrl` + Left Click | Add a **positive** point |
| `Ctrl` + Right Click | Add a **negative** point |
| `Ctrl` + `Shift` + Click | Add the point to the **currently selected object** instead of starting a new one |
| Middle Click | Delete the last added point |
| Click "Center" (Input Parameters), or `Ctrl` + Click (DP) | Find/set the reciprocal-space rings' center |
| `Ctrl` + Scroll Wheel | Zoom in/out, centered on the cursor |

Zoom persists across frame changes and redraws. Use the ribbon's "Home" icon
(or the matplotlib toolbar) to reset back to the full image view.
