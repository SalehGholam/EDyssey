# EDyssey User Manual

EDyssey is a PyQt5 desktop application for processing and analyzing 4D-STEM
(scanning electron diffraction) datasets: computing navigation images,
inspecting individual ROIs, tracking objects across a scan series (via
classical computer vision or Meta's SAM2 AI segmentation model), and
extracting per-object 3D electron diffraction (3DED) data.

EDyssey implements the post-acquisition half of the 4D-STEM tomography
workflow described in Gholam et al., *"A 4D-STEM Tomographic Framework
Assisted by Object Tracking for Nanoparticle Structure Determination"*
(arXiv:2602.09768): a tilt series of 4D-STEM scans (collected with fine
tilt steps and a slightly convergent probe, e.g. via the `evenTem`
acquisition suite) → per-scan navigation images → object tracking/
segmentation of a ROI on those images (this app's ROI Tracker/SAM2 Tracker
tabs) → per-object 3DED extraction, ready for data reduction/structure
solution in PETS2 and Jana2020. This approach targets samples that
challenge conventional 3D ED/CRED tracking - agglomerated or multi-domain
particles, beam-sensitive samples needing minimal fluence, and particles
as small as ~30 nm - and, because tracking happens post-acquisition
instead of live at the microscope, lets you revisit particles or regions
you didn't even notice during the session, straight from the saved
tomogram.

## Launching

See [INSTALL.md](INSTALL.md) for installing the app first (Windows
installer, or run from source). Launch `EDyssey.exe` (installer) or
`EDyssey_MainWindow.py` (from source). The main window has four tabs
(described below), each with its own log console.

If you re-run the script in the same Python console/kernel (e.g. Spyder's
"Run File") without restarting it, the app reuses the window that's
already open instead of creating a new one — it will just be brought to
the front.

## Working with tabs

- **Duplicate Current Tab** (File menu, `Ctrl+Shift+D`): opens a second,
  independent tab of the same type - e.g. a second "ROI on 4D" tab, useful
  for comparing two signals side by side. If the tab you duplicate already
  has an analysis in progress (a loaded signal, computed images,
  tracked/segmented objects, ...), that state is copied into the new tab
  too, as an independent deep copy - editing one tab never affects the
  other. Duplicating an empty tab just opens another empty one. A
  computation still running in the background at the moment of duplication
  is not copied - the new tab gets the state as of the last completed step,
  and the original tab's job keeps running independently.
- **Close Current Tab** (File menu, `Ctrl+W`, or the tab's own "×" button):
  closes a duplicated tab. The 4 original tabs can't be closed this way.

## Common UI elements

- **Directory/file fields** with a `...` button open a picker dialog.
- **"Data Type" dropdown** (ROI on 4D / Navigator: above the file list;
  ROI Tracker / SAM2 Tracker: beside the 4D Signal field): filters the file
  list/folder scan to one format, and - for `.hdf5` files specifically -
  picks which of two loaders to use, since both commonly share the same
  on-disk `.hdf5` extension:
  - `.hdf5 (eventem)`: eventem's own raw export layout.
  - `.hdf5`: a conventional/third-party HDF5 4D-STEM file, loaded via
    HyperSpy.

  Also supported: `.tpx3`, `.hspy`, `.zspy`, `.mib`, `.blo` (loaded via
  HyperSpy, same as `.hspy`/`.zspy`), and (Navigator only) `.tif`.
- **Scale bar fields** ("Real (nm)" / "Recip. (Å⁻¹)") add a calibrated
  scale bar/rings to the canvas once a numeric value is entered.
- **Canvas controls**: each tab embeds the standard matplotlib navigation
  toolbar (Home / Pan / Zoom-rectangle / Save), reachable via the vertical
  ribbon of icons docked to the right of the canvas, plus `Ctrl` + scroll
  wheel to zoom. Adding points or drawing ROIs on the canvas requires
  holding a modifier key (usually `Ctrl`) - a plain click or drag is
  reserved for panning/zooming. See [CONTROLS.md](CONTROLS.md) for the full
  mouse control reference, or click the ribbon's **"?"** icon for the same
  reference specific to the tab you're on.
- **Resizable layout**: the top parameter ribbon, the left file/object-list
  panel, the canvas, and the log console are all separated by drag handles
  - resize any of them to taste by dragging the divider between them.
- **Log console** (bottom of each tab's own canvas area, not shared across
  tabs): live, color-coded messages (errors in red, warnings in yellow) for
  that tab. The full session is also written to disk, one file per tab,
  under a `logs/` folder (plus `logs/app.log` for uncaught errors) — check
  there first if something goes wrong and the console message isn't enough.
  Next to the app when running from source; `%LocalAppData%\EDyssey\logs`
  for an installed build, since the install location itself (particularly
  the default `Program Files`) isn't guaranteed to be writable.
- **Adjust Contrast** (ROI Tracker / SAM2 Tracker, above the object list):
  converts the raw navigation stack to 8-bit for display/tracking/SAM2
  input, via Percentile, Min-Max, or Std. Dev. stretch, plus an optional
  **Denoise** step (Gaussian Blur, Median Filter, Bilateral, Non-Local
  Means, Total Variation, or Wavelet). A parameter tweak previews instantly
  on the current frame; **Apply to All Images** runs it across the whole
  stack. **Test Methods** compares every denoise method side by side on
  the current frame, each with its own retunable parameter. ROI on 4D has
  a standalone **Denoise** box (no separate contrast stretch) that live-
  denoises the Nav. Image display and feeds the same denoised image into
  SAM2 segmentation.
- **Diagnostic console window**: the installer builds also open a plain
  black console window alongside the main app (pushed behind it on
  startup, so it won't cover anything - check the taskbar/Alt+Tab for
  "EDyssey Console"). This is separate from the log console above: it
  shows raw output that doesn't go through EDyssey's own logging - e.g.
  `torch`/CUDA's own messages on the SAM2 Tracker tab. Safe to ignore
  during normal use; check it if something's gone wrong and the log
  console above isn't explaining it. Closing it also closes the app.

## Tabs

### 1. ROI on 4D

Quick, single-file 4D-STEM explorer, with virtual imaging and segmentation
built in. Load one raw 4D signal directly, compute its navigation image
(optionally through one or more virtual detectors, in Sum or Variance
mode), then either drag a rectangular ROI on it or segment a region (SAM2
points, or a real-space threshold) to see the corresponding diffraction
pattern. There's no cross-file tracking here - it's meant for fast
inspection, calibration checks, or one-off virtual-detector/DP extraction
before committing to a full tracking workflow (ROI Tracker / SAM2 Tracker,
below).

**Workflow:** enter the 4D signal path (and scan size / dwell time / smart
scan pattern if not auto-detected) → optionally configure one or more
virtual detectors → **Compute Virtual Image** (`Ctrl+O`) → drag a
rectangular ROI on the Nav. Image, or add SAM2 points and **Segment Image**
(`Ctrl+T`)/set a real-space threshold, to see the corresponding diffraction
pattern.

**Features:**
- Sum or Variance virtual-imaging mode, one or more annular/disk virtual
  detectors (center + inner/outer radius), shown live as an overlay on the
  diffraction pattern.
- SAM2 point-prompt segmentation, or "Summed DP from Threshold" (real-space
  thresholding), either refinable with edge detection (kernel size,
  directional).
- Reciprocal-space beam-center finding/manual setting (see
  [CONTROLS.md](CONTROLS.md)), independent of the virtual-detector mask's
  own center.
- **Denoise** (left panel, above the file list): live-denoises the Nav.
  Image display and feeds the same denoised image into SAM2 segmentation -
  see "Adjust Contrast"/"Denoise" under Common UI elements above.
- Ribbon's **clear_roi** icon: removes the drawn ROI/box so diffraction-
  pattern extraction goes back to using the full frame.
- **Cancel** to stop a running computation.

### 2. Navigator

Batch-builds a navigation-image *signal* — a stack with one navigation
image per input file — from a folder containing many raw acquisition
files. This is the prep step that produces the file the two tracking tabs
below load via their own "Load Signal" button.

**Workflow:** point at the folder of raw files → pick the file type (or
select all) → set scan size/dwell time → **Calculate All** (parallelized
across CPU cores, count configurable) → the resulting navigation signal
(`.hspy`) and a navigation video clip are written to the save directory.

**Features:** select-all vs. manual file selection, virtual detectors (Sum
or Variance mode), adjustable worker process count, adjustable output clip
frame rate, **Stop** to cancel an in-progress calculation. On one
representative test file: **Compute Summed DP** (whole scan), **Summed DP
from ROI** (ribbon's `select_roi` tool - restrict to a drawn scan-space
region, `clear_roi` removes it), or **Summed DP from Threshold...**
(restrict to a real-space-thresholded region instead) - all placing the
same virtual-detector mask preview.

### 3. ROI Tracker

Tracks one or more drawn ROIs across a navigation signal using classical
OpenCV trackers, then extracts per-object 3D electron diffraction data.

**Workflow:** **Load Signal** (`Ctrl+O`, or **Load Saved Analysis**,
`Ctrl+Shift+O`, to resume) → Ctrl+drag to draw a ROI on the navigation
panel (or use **Auto Detector**, above the object list) → pick a tracking
algorithm → **Track!** (`Ctrl+T`) → adjust the blur/threshold controls to
refine the per-frame mask shown in the "ROI with Threshold" panel →
**Extract!** (`Ctrl+E`) to pull the 3DED data for every enabled object →
**Save Results** (`Ctrl+S`).

**Features:**
- Tracking algorithms: CSRT, MIL, Nano, DaSiamRPN. The first time you use
  Nano or DaSiamRPN, their model files (not bundled with the app) download
  automatically - this needs an internet connection once; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for what these are.
- **Auto Detector** (above the object list): opens a popup that
  automatically detects candidate objects on the current frame and adds
  each one as a ROI, instead of drawing them by hand one at a time.
  **Reset Objects** next to it clears the object list.
- Thresholding methods for mask generation: Li, Otsu, Yen, mean — with
  adjustable blur kernel and threshold offset. Restricting extraction to
  the segmented mask (rather than the whole ROI box) improves
  signal-to-noise by excluding contributions from the supporting
  membrane/grid outside the particle.
- **ROI-in-ROI**: track a smaller region relative to a reference ROI
  (e.g. a feature moving within a larger tracked object).
- **Blob Selection** (a "Blob" column in the object list): for a ROI whose
  threshold mask contains more than one particle, pick out just one to
  track/extract. Checking it opens a segmentation dialog (also reachable
  any time via **Blob Settings...**, below the object list) with a live
  contour/number preview and a choice of method for splitting
  touching/overlapping particles before one is picked: Connected
  Components (no splitting, the default), Watershed - Shape, Watershed -
  Intensity (Z-contrast), K-Means Clustering, or Gaussian Mixture (GMM) -
  click a blob in the preview (or directly on the "ROI with Threshold"
  panel) to choose it; with none chosen, the largest is used.
- **Fine-Tune Mask...** (below the object list): manually edit a tracked
  ROI's per-frame mask - see "Fine-Tune Mask dialog" below.
- The object list is transposed (one column per ROI, property names down
  the fixed left column) so adding a ROI adds a column instead of a row;
  long row titles are abbreviated with a tooltip for the full word.
- Per-object enable/disable, end-frame, reference, blob, and delete
  controls in the object list, with tracked/extracted status icons.
- Adjustable extraction thread count and **Autosave** on completion.
- **Make \*.pts2**: writes a PETS2 project file into each ROI's saved
  folder alongside the extracted 3DED data, pre-filled from the signal's
  own metadata (voltage, exposure, Å⁻¹/px scale, beam center) - so
  downstream data reduction/structure solution in PETS2 can start straight
  from Save Results without manually setting up a project file first.
- **Save Results** / **Load Saved Analysis** (see below).

### 4. SAM2 Tracker

Same overall goal as the ROI Tracker tab, but segmentation is driven by
Meta's SAM2 AI model via point prompts instead of drawn boxes, and multiple
objects are tracked together in a single pass — generally more accurate
masks, especially for irregular shapes.

Needs `torch`/`sam2` installed - see INSTALL.md's "Enabling SAM2" section
if this tab's buttons show a "SAM2 Dependencies Not Installed" message. The
SAM2 checkpoint itself (~900MB) downloads automatically on first use.

**Workflow:** **Load Signal** (`Ctrl+O`, or **Load Saved Analysis**,
`Ctrl+Shift+O`, to resume) → Ctrl+left-click to add a positive point
(Ctrl+right-click for negative); add Shift to add the point to the
currently-selected object instead of starting a new one - or use **Auto
Detector**, above the object list, the same way as ROI Tracker's → **Track**
(runs SAM2 across all frames for every object) → **Extract!** (`Ctrl+E`) →
**Save Results** (`Ctrl+S`).

**Features:**
- Positive/negative point prompts, middle-click to delete the last point.
- **Auto Detector** (above the object list) and **Reset Objects** next to
  it, same as ROI Tracker.
- **Fine-Tune Mask...**: manually edit a tracked object's per-frame mask -
  see "Fine-Tune Mask dialog" below (same dialog as ROI Tracker's, minus
  Blob Selection - SAM2 already tracks each object as its own mask).
- The object list is transposed (one column per object, property names
  down the fixed left column), same as ROI Tracker's.
- Per-object end-frame and stack-size (frames processed per SAM2 call)
  control.
- **Stop** to cancel an in-progress tracking run.
- Adjustable extraction thread count and **Autosave** on completion.
- **Make \*.pts2** (see ROI Tracker above) - same PETS2 project-file export.
- **Save Results** / **Load Saved Analysis** (see below).
- **Help > Set Up SAM2...**: automates installing `torch`/`sam2` if this
  tab's buttons show a "SAM2 Dependencies Not Installed" message (not
  needed with the offline installer, which bundles them already).

## Fine-Tune Mask dialog (ROI Tracker / SAM2 Tracker)

Frame-by-frame manual editing of a tracked object's per-frame mask, opened
via **Fine-Tune Mask...**. Every edit here is a live preview, never baked
into the saved/extracted mask until you close the dialog - **Reset This
Frame** / **Reset to Tracking** discard edits back to the original
tracked/segmented result at any point.

- **Paint**: `Ctrl`+click/drag to add/remove single pixels, `Shift`+drag
  for a rectangular region - or arm the equivalent Paint In/Paint
  Out/Rect In/Rect Out ribbon tool to do the same without holding a key.
- **Grow/Shrink Mask**: D-pad buttons add/remove one row/column of pixels
  on a side.
- **Segments**: Dilate/Erode and Mesh each apply per frame-range
  "segment" instead of to the whole stack at once - **Split Here**/
  **Merge with Previous**/**Reset to Default** manage the boundaries,
  shown as a colored bar under the frame slider.
- **Dilate/Erode Mask**: grow/shrink the mask by a signed kernel size
  (positive dilates, negative erodes); **Opening**/**Closing** kernel
  sizes additionally remove small specks or fill small holes without
  changing the mask's overall size.
- **Edge Detection**: reduces the mask to its outline (optionally
  one-sided via Directional + Angle) - applies to the whole stack, not
  per-segment. Extracting 3DED from just a particle's edge instead of its
  whole volume can improve data quality for thick particles (less
  dynamical/multiple scattering along a shorter path length), and is
  useful for edge/surface-specific structural questions - catalysts (bulk
  vs. surface), core-shell structures, or following surface transformations
  in an in-situ/ex-situ series.
- **Mesh**: divides the mask into a rotated grid; click cell(s) to
  restrict extraction to just those (**Lines Only** selects full-width
  stripes instead of individual cells; **Center on Initial Mask** anchors
  the grid to the object's starting position).
- **Threshold** (ROI Tracker only, when the mask is threshold-derived):
  rebuild the mask live from Method/Blur/Deviation - **Apply to All
  Frames** propagates it to the whole stack.
- **Find Tilt Axis**: estimates a tomography tilt series' tilt axis from
  how this object's mask centroid moves across frames (PCA on the
  centroid scatter, cross-checked by a candidate-angle sweep), drawn as a
  dashed reference line - **Show Details...** opens the underlying
  scatter/sweep plot. Assumes a single specimen tilting about one fixed
  in-plane axis with little translational drift between frames.

See the dialog's own **"?"** ribbon icon for the full mouse/keyboard
reference.

## Saving & resuming analyses (ROI Tracker / SAM2 Tracker)

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

## Keyboard shortcuts

| Shortcut | Tab(s) | Action |
|---|---|---|
| `Ctrl+O` | ROI on 4D | Compute Virtual Image |
| `Ctrl+T` | ROI on 4D | Segment Image (SAM2) |
| `Ctrl+O` | ROI Tracker / SAM2 Tracker | Load Signal |
| `Ctrl+Shift+O` | ROI Tracker / SAM2 Tracker | Load Saved Analysis |
| `Ctrl+T` | ROI Tracker / SAM2 Tracker | Track |
| `Ctrl+E` | ROI Tracker / SAM2 Tracker | Extract 3DED |
| `Ctrl+S` | ROI Tracker / SAM2 Tracker | Save Results |
| `Ctrl+Shift+D` | Any | Duplicate Current Tab |
| `Ctrl+W` | Any | Close Current Tab |

## Troubleshooting

- Check the log console at the bottom of the active tab's canvas area
  first — errors are shown in red.
- For more detail (or if the app already closed), check `logs/` next to
  the app: one file per tab, plus `logs/app.log` for uncaught exceptions.
- Re-running the app in the same console/kernel brings back the existing
  window rather than opening a second one — if you want a completely
  fresh state, restart the Python console/kernel first.
