# EDyssey 2.1.20261014.2054

## Fixes

- **Save Results: clip export failed with `OSError: [WinError 6] The handle is invalid`**: the ffmpeg process was started inheriting the app's stdout/stderr, which can be invalid while another thread has them redirected (e.g. during extraction). ffmpeg is now started with explicit handles, and its own error output is captured and reported if it fails.
- **Save Results: clips and TIFF frames are now exported safely on their background threads**:
  - Clips are drawn on an off-screen figure instead of through pyplot, which built a Qt window on a worker thread.
  - Every stage is logged ("building figure", "encoding with ffmpeg", frame progress), and a failure names the stage, the file and the reason in the log console.
  - A TIFF frame that fails to write is logged with its path and does not stop the remaining frames; a summary error is raised at the end.
  - Re-saving into an existing frames folder only clears old `.tif` files, never other files or folders.
- **Native crashes now leave a trace**: the app writes every thread's Python stack to `%LOCALAPPDATA%\EDyssey\logs\native_crash.log` if the process dies from a hard fault.

---
*Full changelog since `2.1.20261014.1512`.*
