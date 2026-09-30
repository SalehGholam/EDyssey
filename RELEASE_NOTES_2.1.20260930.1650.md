# EDyssey 2.1.20260930.1650

## New Features
- **Display Preferences: "Mask Transparency"** option - controls the opacity of the tracked-object/segmentation mask overlays shown on ROI Tracker's, SAM2 Tracker's, and ROI on 4D's own main canvases (not the separate Fine-Tune Mask dialog, which already has its own independent opacity controls). Replaces 5 previously-independent hardcoded alphas (0.3-0.85) with one shared, live-adjustable value.
- **Display Preferences: Colormaps simplified** - the two shared "Navigation Image"/"Diffraction Pattern" colormap dropdowns and the separate "Per-Plot Colormaps..." dialog are gone; all 11 individually-recolorable plots (across all 4 tabs) now get their own combo box directly in the Colormaps list, with a "Reset All Colormaps" button.

## Fixes
- **Crash fix**: two SAM2 "new object" paths (Ctrl+click, and the auto-detector's results) still built a mask-row one column short after `manual_edits` was added earlier - `pandas` raised `ValueError: cannot set a row with mismatched columns` the moment either ran.
- **Fixed a real Blob Selection mismatch**: opening Fine-Tune Mask could show a different blob than the one actually selected live in the main ROI Tracker UI, because its own blob-resolution pre-pass recomputed the whole auto-follow chain from scratch instead of reusing what the main UI had already resolved for frames you'd actually looked at.
- **Fixed 2 real packaging gaps**: "New eventem" and "pyeventem" were selectable backends in the UI but were never actually bundled into the frozen installer at all (both existed only in dev/venv, not the shipped build) - both are now correctly bundled for the app's own Python version.
- **"Load Saved Analysis" round-trip audit** (both ROI Tracker and SAM2 Tracker): several settings were being saved to disk but silently dropped, or never saved at all, on reload:
  - SAM2: a reloaded object's Dilate/Erode/Edge Detection/Mesh settings were being written to its saved JSON but never actually read back on load (dead-on-arrival data).
  - SAM2: the mask actually written back into memory on reload was the already-processed (Dilate/Erode/Edge Detection/Mesh baked in) deliverable, not the raw, still-further-editable one Fine-Tune Mask expects - re-editing a reloaded object risked double-applying those effects.
  - SAM2: manual paint/grow-shrink edits (the "Exclude Manual Edits from Effects" feature) were never saved for SAM2 objects at all, unlike ROI Tracker, which already saved and restored them correctly.
  - SAM2: the pristine "Reset to Tracking" target (`mask_default`) was never saved on its own - after any save made following a Fine-Tune Mask edit, reloading and then "Reset to Tracking" silently reset to the edited mask instead of the true original SAM2 result. All 4 of the above are now saved to (and correctly restored from) their own files, with backward-compatible fallbacks for analyses saved by an older EDyssey version.
  - ROI Tracker: the Threshold method/ROI Blur/Deviation settings used to produce a saved analysis were being written to disk but never read back - the main tab's own Threshold controls now restore to match on "Load Saved Analysis".

---
*Full changelog since `2.1.20260929.2211`.*
