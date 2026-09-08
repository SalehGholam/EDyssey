# EDyssey

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

<p align="center">
  <img src="docs/New_Icon.png" alt="EDyssey Icon" width="180">
  <img src="docs/UI_screenshot.png" alt="EDyssey UI Screenshot" width="520">
</p>

EDyssey is a PyQt5 desktop application for processing
and analyzing 4D-STEM Tomography data (scanning electron diffraction) datasets. It covers a range of necessary steps to reach the final 3D ED dataset.
The normal workflow is:
- Checking the 4D-STEM Files on ROI on 4D tab
- Create a stack of images from navigation images (with virtual detectors) on Navigator tab
- Track particles or regions of interest by either classical opencv trackers or MetaAI's SAM2
- Extract 3D ED frames from the segmented regions

The software currently supports raw Amsterdam Scientific Instruments's tpx3 through [evenTem](https://github.com/EMAT-Jo/evenTem), Quantum Detector's mib, Hyperspy's hspy and
zspy, ASTAR's blo, and both eventem's own and conventional HDF5 data types.

## Install

**Windows installer (recommended):** two variants, both installing the same
app - no Python setup required either way:

- **Online** (small, ~200MB): download `EDyssey_Setup_Online_<version>.exe` from the
  [Releases](https://github.com/SalehGholam/EDyssey/releases) page. The SAM2
  checkpoint and Nano/DaSiamRPN tracker model files download automatically
  the first time you use those specific features, instead of being bundled
  upfront.
- **Offline** (large, ~2.6GB): bundles those same model files, plus
  `torch`/CUDA/`sam2` themselves, so nothing needs to be downloaded or
  installed later - not even on first use of the SAM2 tab. Not published
  on GitHub (too large for a Release asset) - ask whoever maintains this
  repo for a copy, or build it yourself (see Development below).

**From source:** see [INSTALL.md](INSTALL.md).

With the online installer (or running from source), SAM2 support needs one
extra step: **Help > Set Up SAM2...** in the app, which auto-detects your
GPU and installs the matching CUDA build of `torch` for you (a **Check
CUDA** button lets you confirm it actually picked up your GPU afterwards) -
see INSTALL.md's "Enabling SAM2" section. The offline installer already
includes everything, so SAM2 works immediately after install.

## Usage

- [MANUAL.md](MANUAL.md) - full walkthrough of all four tabs (ROI on 4D,
  Navigator, ROI Tracker, SAM2 Tracker), saving/resuming analyses, and
  keyboard shortcuts.
- [CONTROLS.md](CONTROLS.md) - canvas mouse control reference for all four
  tabs.

## Development

- Tests: `pytest tests/`
- Lint: `ruff check .`
- Building the Windows installer:
  - Online: `pyinstaller EDyssey.spec`, then `iscc installer\EDyssey_online.iss`.
  - Offline: `set EDYSSEY_OFFLINE_BUILD=1` first, then
    `pyinstaller EDyssey.spec --distpath dist_offline --workpath build_offline`,
    then `iscc installer\EDyssey_offline.iss` - needs `torch`/`sam2` and the
    model weight files present locally first (see its header comment).

  See [EDyssey.spec](EDyssey.spec)'s header comment for the overall
  packaging rationale.

## Acknowledgements 
Many thanks to Joke Hadermann and Jo Verbeeck for the support and Arno Annys for assistance on setting up the software and [evenTem](https://github.com/EMAT-Jo/evenTem).

## AI Usage
Many parts of the user interface and build of the installers are developed by the help of large language models. The code has been tested to work properly, but not every line of the code has been reviewed by the developer. The logo is made by Nano Banana 2.

## License

GPL-3.0 - see [LICENSE](LICENSE). This follows from two runtime
dependencies (PyQt5, HyperSpy) that are themselves GPLv3, which doesn't
allow a more permissive or usage-restricted license on top.

Some model weights are downloaded on demand rather than bundled - see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for what each one is,
its license, and where it comes from (one of them, the NanoTrack tracker
model, currently has an unclear license - flagged there).
