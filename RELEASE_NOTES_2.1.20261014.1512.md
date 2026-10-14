# EDyssey 2.1.20261014.1512

## Fixes

- **Installed app crashed at startup with `AttributeError: 'NoneType' object has no attribute 'origin'`** (in hyperspy's extension scan): the build bundled the metadata of `pyxem` when it was installed in the build environment, without its code. The build now strips it, so the result no longer depends on what else is installed on the build machine.

## New

- **Save Figure ribbon button (all four tabs)**: a floppy-disk icon in the right-hand ribbon saves the tab's whole figure to a path you choose (PNG, PDF, SVG, TIFF or JPEG, at 300 dpi). The save dialog reopens in the last-used folder.
- **Save Data ribbon button (ROI on 4D tab)**: a database icon writes the plotted diffraction pattern plus what produced it - navigation image, ROI rectangle and/or mask (raw and as used), SAM2 points, beam center, and a JSON block of settings - to a single `.npz` file.

---
*Full changelog since `2.1.20261001.1737`.*
