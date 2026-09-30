# EDyssey 2.1.20260930.1920

## New Features

- **Blob Selection in Fine-Tune Mask**: ROI Tracker and SAM2 Tracker's "Fine-Tune Mask" dialog can now directly pick which connected blob a frame/segment's mask should keep, instead of only being configurable from the main tab's separate "Blob Settings..." dialog (ROI Tracker) or not at all (SAM2, new).
  - A ribbon icon ("Pick Blob") under the canvas - armed, then click a blob to restrict the frame/segment on screen to it. Only ever acts while explicitly armed, never on a free click.
  - Scoped by the existing "Edit Scope" (Single Frame/Segment) radios, against its own independent segment timeline - different frame ranges can keep their own blob choice.
  - A "Blob Selection" box with its own Method combo (Connected Components, Watershed - Shape/Intensity, K-Means, GMM) and per-method parameters, plus "Show Blobs" (outlines every candidate, highlights the kept one) and an "Active" master checkbox.
  - For ROI Tracker, candidates come from the current Denoise+Threshold settings within the ROI (not a stale saved mask); for SAM2, from the raw SAM2 segmentation mask.
- **Denoise + Threshold merged** (Fine-Tune Mask, ROI Tracker): the separate "ROI Blur" spinbox is gone - it was a second, always-Gaussian blur stacked on top of Denoise, applied only behind the scenes for thresholding. A new "Apply denoising to:" choice in the merged box picks whether the selected Denoise method feeds the displayed image or the threshold recompute (the default) - not silently both.
- **Denoise "Apply to Segment"** (Fine-Tune Mask): pre-denoises every frame in the current segment in parallel (multi-threaded, like the main tab's own denoise-all), with a progress bar and a "Reset" button, instead of denoising one frame at a time as you scrub to it.
- **4D signal file type "Auto"**: the ROI Tracker/SAM2 Tracker "Data Type" combo's old "All Files" (matched everything, filtered nothing) is now "Auto" - detects the real file type from what's actually in the selected folder, and pre-selects it automatically when you pick a 4D-signal folder. Logged when used.
- **Mask Transparency**: now a proper slider + spinbox (matching every other control in Display Preferences) instead of a bare spinbox.

## Fixes

- **Fine-Tune Mask edits weren't visible on the main UI**: ROI Tracker's own "ROI with Threshold" live panel never read the saved/fine-tuned mask at all - it always recomputed a fresh, from-scratch threshold every redraw, silently discarding manual paint, D-pad grow/shrink, or Pick Blob edits until "Extract!" was re-run. It now shows the saved mask (with Dilate/Erode/Edge Detection/Manual Edits/Mesh/Blob Selection re-applied fresh) once one exists, falling back to the live threshold preview only for a ROI that hasn't been extracted/edited yet - consistent with what "All Active Objects" mode already did.
- **Blob Selection could silently mask the whole ROI**: fixed two related issues from this feature's initial rollout - (1) a pick made in Fine-Tune Mask no longer turns on the main tab's separate, persistent, every-frame Blob restriction unless it was already active before the dialog opened; (2) "Reset to Tracking" now also clears Blob Selection's own segment history, so a bad pick can't survive a reset.
- **UI renames for clarity**: "Threshold" -> "Mask Threshold" (ROI Tracker ribbon), "Extract" -> "Extract DP" (ROI Tracker/SAM2 Tracker ribbons), "Sum DP" -> "DP Computation" (Navigator ribbon), "Seg Image" -> "Segment Image" (SAM2 Tracker button).
- "Mask Panel"/"Nav. Entries" plotting-mode radios (ROI Tracker) and their SAM2 equivalent now live in a "Plotting Preferences" group box, with shortened "Selected"/"All Active" option labels.

---
*Full changelog since `2.1.20260930.1650`.*
