# EDyssey 2.1.20261001.1026

## Fixes

- **"Extract!" could silently discard a Fine-Tune Mask edit**: ROI Tracker's own "Extract!" always rebuilt every ROI's mask from scratch via a fresh re-threshold, ignoring whatever was already saved in `df_rois['mask']` - including a manual paint, D-pad grow/shrink, or Pick Blob edit made in Fine-Tune Mask (those are baked directly into the mask array, unlike Dilate/Erode/Edge Detection/Mesh/Blob Selection, which are settings re-applied fresh either way). A mask that looked correct right after accepting Fine-Tune Mask could revert to the raw, unrestricted threshold result the moment "Extract!" ran. Extraction now reuses the saved mask once one exists, matching what the main canvas already shows and what Fine-Tune Mask itself already does - a ROI that's never been tracked/extracted/edited still gets its first mask built fresh.
- **A blob picked in the main UI could change after extraction**: a coordinate-space mismatch - every blob centroid the live "ROI with Threshold" panel ever cached was in ROI-local (cropped) pixel coordinates, but `extract_3ded`, "All Active Objects" mode, and the main canvas's own saved-mask display all re-apply that same cache to a FULL navigation-image-sized mask with no translation. The nearest-blob lookup then searched for a tiny local coordinate inside a full-size frame, landing on an essentially different (often spurious) blob. Centroids are now translated into whichever coordinate space the mask being resolved is actually in, and back again before being cached, so a blob picked live survives extraction correctly.
- **Fine-Tune Mask's left panel could overflow past its own width**: restoring the vertical scroll area (see below) only fixed the scroll area's own outer width, not the inner panel - a widget with no explicit width of its own (e.g. the Deviation slider) could demand more space than available and spill into the canvas. The inner panel is now fixed-width again, same as every control inside it was already built for.
- **Blob Selection's "Show Blobs" overlay didn't react to the Method choice**: it was outlining the whole raw mask as one shape regardless of which splitting method was selected, so switching to Watershed/K-Means/GMM never visibly split a connected blob into pieces on screen (picking still worked correctly underneath). It now outlines each detected blob individually.

## Changes

- Fine-Tune Mask's left panel has its scroll area back (removed in an earlier, shorter version of this dialog; brought back now that Blob Selection and the merged Denoise+Threshold box make it taller again).
- "Test Methods" moved onto the same row as "Apply to Segment"/"Reset" in the Denoise box; "Show Blobs" moved onto the same row as "Active" in the Blob Selection box - both to save vertical space.

---
*Full changelog since `2.1.20260930.1920`.*
