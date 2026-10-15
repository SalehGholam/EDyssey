# EDyssey 2.1.20261015.0927

## Fixes

- **ROI Tracker: lowering a ROI's End frame after tracking was ignored by "Extract!"**: extraction processed every frame that had a tracked box, so a ROI tracked over the whole signal was still extracted over all of it. It now only extracts frames from the ROI's start frame up to its current End value. The progress total and log line count the frames that will really run. (The SAM2 tab already behaved this way.)

---
*Full changelog since `2.1.20261015.0115`.*
