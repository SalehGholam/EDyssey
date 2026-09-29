# EDyssey 2.1.20260929.1648

## New Features
- **Fine-Tune Mask: multiple meshes per segment**: a segment can now hold several independent meshes, each with its own angle/cell size/lines-only/fixed/cell selection, selectable and editable from a small list with Add/Delete Mesh - when more than one is enabled, the kept region is their intersection, letting one mesh's selection be narrowed further by another.
- **Fine-Tune Mask: "Fixed Mesh"** (renamed from "Center on Initial Mask") now actually works at real extraction time, not just in the dialog's own preview - it anchors a mesh's grid to one captured point (by default, the object's centroid on the segment's initial mask) instead of recentering every frame. A new **"Center Grid (Click)"** button lets you pick that anchor point yourself: click it, then click the canvas.
- **Fine-Tune Mask: "Center on: Edited Mask / Initial Mask"**: while a mesh isn't Fixed, this picks which mask its per-frame recentering tracks - the object's live/edited mask (the default, same as before) or its initial/pre-edit one, so the grid can keep following the original tracked/segmented position instead of drifting with painted/grown/shrunk edits.
- **Fine-Tune Mask: "Exclude Manual Edits from Effects"** (checked by default): pixels painted, rect-painted, or grown/shrunk by hand now keep exactly that value even when Dilate/Erode or Edge Detection is enabled, instead of those effects reprocessing them along with the rest of the mask.
- **Fine-Tune Mask: mask transparency and color**: the current/tuned mask is now more transparent on the canvas and a different color (orange) from the mask as it was before any edit this session, which stays visible underneath it (fainter, blue), as a constant reference for how much has changed.
- **Fine-Tune Mask: "Find Tilt Axis"/"Show Details..."** now sit side by side instead of stacked.
- **Fine-Tune Mask: Edit Scope now defaults to "Segment"** instead of "Single Frame".

## Fixes
- "Center on Initial Mask" silently had no effect on the extracted/saved mask (only on the dialog's own live preview) - real extraction always recomputed the mesh origin from the live mask centroid, ignoring the flag entirely. Fixed as part of the "Fixed Mesh" rework above.
- The mesh list was unselectable by mouse - a custom row widget filling the whole row swallowed every click before the list could register it as a row selection. Rebuilt on native checkable list items instead.

---
*Full changelog since `2.1.20260917.1414`.*
