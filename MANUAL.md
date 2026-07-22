# py5DED User Manual

py5DED is a PyQt5 desktop application for processing and analyzing 4D-STEM
(scanning electron diffraction) datasets: computing navigation images,
inspecting individual ROIs, tracking objects across a scan series (via
classical computer vision or Meta's SAM2 AI segmentation model), and
extracting per-object 3D electron diffraction (3DED) data.

## Launching

Run `5DED_MainWidow.py`. The main window has four tabs (described below)
and a live log console along the bottom.

If you re-run the script in the same Python console/kernel (e.g. Spyder's
"Run File") without restarting it, the app reuses the window that's
already open instead of creating a new one — it will just be brought to
the front.

## Common UI elements

- **Directory/file fields** with a `...` button open a picker dialog.
- **Scale bar fields** ("Real (nm)" / "Recip. (Å⁻¹)") add a calibrated
  scale bar to the canvas once a numeric value is entered.
- **Canvas controls**: each tab embeds the standard matplotlib toolbar
  (Home / Pan / Zoom-rectangle / Save) above its plots, plus scroll-wheel
  zoom centered on the cursor. Adding points or drawing ROIs on the canvas
  requires holding **Ctrl** — a plain click or drag is reserved for
  panning/zooming. See [CONTROLS.md](CONTROLS.md) for the full mouse
  control reference for the two tracking tabs.
- **Log console** (bottom of the main window): live, color-coded messages
  (errors in red, warnings in yellow) tagged by which tab produced them.
  The full session is also written to disk, one file per tab, under a
  `logs/` folder next to the app (plus `logs/app.log` for uncaught
  errors) — check there first if something goes wrong and the console
  message isn't enough.

## Tabs

### 1. ROI on 4D

Quick, single-file 4D-STEM explorer. Load one raw 4D signal directly, view
its navigation image, and drag a single rectangular ROI on it to see the
averaged diffraction pattern for that region update live. There's no
tracking and no save/load here — it's meant for fast inspection or
calibration checks before committing to a full tracking workflow.

**Workflow:** enter the 4D signal path (and scan size / dwell time if not
auto-detected) → **Load Signal** → drag a box on the navigation image to
see its diffraction pattern.

### 2. Make Nav. Sig.

Batch-builds a navigation-image *signal* — a stack with one navigation
image per input file — from a folder containing many raw acquisition
files. This is the prep step that produces the file the two tracking tabs
below load via their own "Load Signal" button.

**Workflow:** point at the folder of raw files → pick the file type (or
select all) → set scan size/dwell time → **Calculate** (parallelized
across CPU cores, count configurable) → the resulting navigation signal
(`.hspy`) and a navigation video clip are written to the save directory.

**Features:** select-all vs. manual file selection, adjustable worker
process count, adjustable output clip frame rate, **Stop** to cancel an
in-progress calculation.

### 3. Tracking by CV2

Tracks one or more drawn ROIs across a navigation signal using classical
OpenCV trackers, then extracts per-object 3D electron diffraction data.

**Workflow:** **Load Signal** (or **Load Saved Analysis** to resume) →
Ctrl+drag to draw a ROI on the navigation panel (or use **Auto Detector**,
below) → pick a tracking algorithm → **Track!** → adjust the
blur/threshold controls to refine the per-frame mask shown in the "ROI
with Threshold" panel → **Extract!** to pull the 3DED data for every
enabled object → **Save Results**.

**Features:**
- Tracking algorithms: CSRT, MIL, Nano, DaSiamRPN.
- **Auto Detector**: opens a popup that automatically detects candidate
  objects on the current frame and adds each one as a ROI, instead of
  drawing them by hand one at a time.
- Thresholding methods for mask generation: Li, Otsu, Yen, mean — with
  adjustable blur kernel and threshold offset.
- **ROI-in-ROI**: track a smaller region relative to a reference ROI
  (e.g. a feature moving within a larger tracked object).
- Per-object enable/disable, end-frame, and delete controls in the object
  tree, with tracked/extracted status icons; **Reset ROIs** clears them
  all.
- Adjustable extraction thread count and **Autosave** on completion.
- **Save Results** / **Load Saved Analysis** (see below).

### 4. SAM2 Seg.

Same overall goal as the CV2 tab, but segmentation is driven by Meta's
SAM2 AI model via point prompts instead of drawn boxes, and multiple
objects are tracked together in a single pass — generally more accurate
masks, especially for irregular shapes.

**Workflow:** **Load Signal** (or **Load Saved Analysis** to resume) →
Ctrl+left-click to add a positive point (Ctrl+right-click for negative);
add Shift to add the point to the currently-selected object instead of
starting a new one → **Track** (runs SAM2 across all frames for every
object) → **Extract!** → **Save Results**.

**Features:**
- Positive/negative point prompts, middle-click to delete the last point.
- Per-object end-frame and stack-size (frames processed per SAM2 call)
  control.
- **Stop** to cancel an in-progress tracking run.
- Adjustable extraction thread count and **Autosave** on completion.
- **Save Results** / **Load Saved Analysis** (see below).

## Saving & resuming analyses (Tracking by CV2 / SAM2 Seg.)

**Save Results** writes a timestamped folder (under your chosen save
directory) containing:
- the navigation signal itself (`navigation_signal.hspy`), saved once per
  session;
- per-object/ROI tracking metadata (JSON: points/init frames, labels,
  end frame, use flag);
- masks and ROI arrays (`.npy`);
- extracted diffraction patterns, saved both as a raw array (`3DED.npy`)
  and as a hyperspy signal (`3DED.hspy`);
- rendered frame images and video clips for both the tracking result and
  the diffraction data.

**Load Saved Analysis** picks one of these saved folders and restores the
navigation signal, every tracked object/ROI (with correct
tracked/extracted status icons), and any extracted diffraction patterns —
so you can review or continue a previous session without redoing the
tracking.

## Keyboard shortcuts (Tracking by CV2 / SAM2 Seg.)

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Load Signal |
| `Ctrl+Shift+O` | Load Saved Analysis |
| `Ctrl+T` | Track |
| `Ctrl+E` | Extract 3DED |
| `Ctrl+S` | Save Results |

## Troubleshooting

- Check the log console at the bottom of the window first — errors are
  shown in red with the tab name that produced them.
- For more detail (or if the app already closed), check `logs/` next to
  the app: one file per tab, plus `logs/app.log` for uncaught exceptions.
- Re-running the app in the same console/kernel brings back the existing
  window rather than opening a second one — if you want a completely
  fresh state, restart the Python console/kernel first.
