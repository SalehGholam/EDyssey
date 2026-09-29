# EDyssey 2.1.20260917.1414

## New Features
- **Generic `.hdf5` support**: conventional/third-party HDF5 files are now loaded lazily (h5py + dask), without requiring a HyperSpy-compatible layout. EDyssey walks the file tree to find the single 4D dataset automatically.
- **ROI-on-4D threshold segmentation**: new thresholding workflow for extracting ROIs directly from 4D data, with an Activate checkbox placed next to Method for a more compact ribbon.
- **Extract Frame now updates Extract All results**: fine-tuning a mask on a few frames and re-extracting just those no longer requires re-running the whole series (ROI Tracker and SAM2 Tracker).
- **Fine-Tune Mask "Edit Scope" controls**: a Single Frame / Segment radio toggle above the canvas makes it clear whether an edit applies to just the current frame or the whole tracked range. Per-frame edits persist and reappear when you return to that frame regardless of the toggle position.
  - New **Reset Frame**, **Reset Segment**, and **Reset to Tracking** actions, with a confirmation prompt on Reset to Tracking since it discards all segments and mask edits.
- **Navigator tab** frame/view improvements for consistency with the other tabs.

## Performance
- **Canvas blitting** rolled out across all four main tabs and the on-top dialogs (Mask Edit, Threshold, Blob Segmentation), making repetitive plot updates (sliders, live previews) noticeably smoother.
- **Faster eventem HDF5 reads**, with eventem and h5py's own HDF5 libraries now coexisting without symbol/version conflicts.

## Fixes
- Boolean-dtype 4D datasets are now correctly detected during generic `.hdf5` loading.
- Fixed a crash from a single-pixel ROI click (zero-width/height selection).
- Fixed an `AttributeError` race on `nav_imgs`/`nav_imgs_raw` during signal loading (root cause: an unblocked slider range update firing a settings-changed signal before the data existed).
- Fixed a stale "ghost" artifact appearing in blitted dialogs after a resize.
- Fixed the DP Clipping Threshold not re-anchoring after Extract All (previously required a manual Reset click).
- Fixed tracking/extraction results always jumping to frame 0 on completion instead of staying on the slider's current frame.
- Removed the unnecessary scrollbar from the Fine-Tune Mask dialog's left panel.
- Various small ROI Tracker/SAM2 Tracker ribbon layout fixes.

## Packaging / Docs
- Resolved an HDF5 DLL naming collision between eventem and h5py (eventem's HDF5 library renamed to `h5ev.dll`).
- Updated GPU driver documentation.
- Removed unused CI configuration.

---
*Full changelog since `2.1.20260907.1138`.*
