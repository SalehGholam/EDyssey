# EDyssey 2.1.20261001.1737

## Fixes

- **Fine-Tune Mask crash on a loaded analysis with an active Mesh**: opening Fine-Tune Mask for an object/ROI whose saved analysis already had an enabled Mesh selection raised `AttributeError: 'MaskEditDialog' object has no attribute 'spinbox_meshCellSize'` - the dialog's Blob Selection box was syncing its own enabled/disabled widgets via a full mask redraw before the Mesh box's own widgets existed yet, which only mattered when there was already a mesh to draw. That redraw is now deferred to the dialog's own final setup step, after every widget is built.
- **"Load Saved Analysis" didn't restore the 4D data folder, save folder, or (SAM2) nav signal fields**: only the navigation signal's source path was ever recorded in `analysis_info.json`; the 4D-STEM data folder and the results/save directory weren't saved at all, and SAM2's own nav-signal field wasn't restored even though it was saved. All three now round-trip; an analysis saved before this fix still loads fully, with a logged warning instead of a silent gap when the 4D folder can't be restored.
- **ROI Tracker's Blob Selection could resolve a different blob after reloading a saved analysis**: which blob got picked on an ambiguous (multi-blob) frame via auto-follow depended on an in-memory, session-only history of already-resolved frames, never saved to disk - "Load Saved Analysis" always started that history empty, so a later re-extraction or resave could silently pick a different blob than the one the analysis was originally saved with. That history now round-trips with the rest of each ROI's settings.

---
*Full changelog since `2.1.20261001.1026`.*
