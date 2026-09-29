# EDyssey 2.1.20260929.2211

## New Features
- **Per-plot colormap overrides**: Display Preferences' "Colormaps" group now has a "Per-Plot Colormaps..." button, opening a list of every individually-recolorable image plot across all 4 tabs (11 in total, including mask/segmentation-overlay backgrounds previously fixed to gray) - each can be pinned to its own colormap independent of the shared Navigation Image/Diffraction Pattern settings, with a "Reset All to Global Colormaps" button to clear every override at once.
- **Fine-Tune Mask: directional Dilate/Erode**: a new "Directional" + Angle option on the Dilate/Erode box's own Kernel Size step, matching Edge Detection's existing convention - grows/shrinks the mask from just one side instead of uniformly all around.
- **Fine-Tune Mask: Denoise now affects Threshold rebuilds**: the Denoise box's current method/parameter is now actually applied before re-thresholding (ROI Tracker), not just to the displayed preview image as before - cached per Denoise-setting change, not recomputed on every Threshold slider tick, to keep it fast.
- **Fine-Tune Mask: cursor position/pixel-value readout** and a **mask-color legend**, both truly at the dialog window's own bottom-left corner, alongside Save&&Close/Cancel (mirroring the main tabs' status-bar readout, since this dialog is its own separate window).
- **Fine-Tune Mask: "Show Initial Mask" checkbox** above the canvas - hides the always-visible initial-mask reference layer when it's in the way.
- **Fine-Tune Mask: tidier Edge Detection/Dilate-Erode/Mesh layout** - each box's controls now line up in a shared grid instead of loosely stacked rows.
- **ROI Tracker & SAM2 Tracker: "Reset" button** in the object list - resets an individual object/ROI straight back to its latest tracking result (mask, and every Dilate/Erode/Edge Detection/Mesh/manual-edit setting), without needing to open Fine-Tune Mask first.
- **Fine-Tune Mask: mesh centering basis** - a "Center on: Edited Mask / Initial Mask" radio pair per mesh, picking which mask a (non-"Fixed") mesh's per-frame recentering tracks.

## Fixes
- **Crash fix**: "Extract All" in ROI Tracker crashed for any ROI tracked over only part of the navigation signal (not the full stack) - re-thresholding looped over every frame including the untracked ones' `(0,0,0,0)` placeholder ROI, and thresholding an empty 0×0 crop raised inside skimage. Untracked frames are now correctly skipped (left as an empty mask), matching the single-frame "Extract!" path, which was already safe.
- Fixed 3 (sometimes more) duplicate "Blob Selection enabled/disabled for ROI N" log lines per single actual change - inserting a brand-new checkbox cell into the object list's table, and `setCheckState()`'s own redundant `setFlags()` call, each fired the table's `itemChanged` signal again even when nothing actually changed; every real per-object checkbox setup/toggle in both tabs' object lists is affected, not just Blob Selection.
- Fine-Tune Mask's mesh list (item 2 from the previous release) was unselectable by mouse - a custom row widget swallowed clicks before the list could register a row selection. Rebuilt on native checkable list items.
- Fine-Tune Mask's Edit Scope now defaults to "Segment" instead of "Single Frame".
- Fine-Tune Mask's current/tuned mask and its always-visible initial-mask reference layer now use different colors (brown vs. blue) instead of relying on opacity alone to distinguish them - brown chosen to match the Mesh box's own selected-cell highlight color, per request.
- Fine-Tune Mask's left panel no longer stretches its control groups to fill extra vertical space when the dialog is maximized - a trailing stretch now absorbs it instead.

---
*Full changelog since `2.1.20260929.1648`.*
