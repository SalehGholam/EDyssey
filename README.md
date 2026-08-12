# EDyssey

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

EDyssey (repo name `py5DED`) is a PyQt5 desktop application for processing
and analyzing 4D-STEM (scanning electron diffraction) datasets: computing
navigation images, inspecting individual ROIs, tracking objects across a
scan series - either with classical OpenCV trackers or Meta's SAM2 AI
segmentation model - and extracting per-object 3D electron diffraction
(3DED) data.

<!-- screenshot: docs/screenshot.png -->

## Install

**Windows installer (recommended):** two variants, both installing the same
app - no Python setup required either way:

- **Online** (small): download `EDyssey_Setup_Online_<version>.exe` from the
  [Releases](https://github.com/SalehGholam/EDyssey/releases) page. The SAM2
  checkpoint and Nano/DaSiamRPN tracker model files download automatically
  the first time you use those specific features, instead of being bundled
  upfront.
- **Offline** (large, ~2-3GB): bundles those same model files so nothing
  needs to be downloaded later, not even on first use. Not published on
  GitHub (too large for a Release asset) - ask whoever maintains this repo
  for a copy, or build it yourself (see Development below).

**From source:** see [INSTALL.md](INSTALL.md).

Either way, SAM2 support needs one extra manual step (`torch` + the `sam2`
package aren't bundled or auto-installed, since the right build depends on
your GPU/CUDA version) - see INSTALL.md's "Enabling SAM2" section.

## Usage

- [MANUAL.md](MANUAL.md) - full walkthrough of all four tabs (ROI on 4D,
  Navigator, Tracking by CV2, SAM2 Seg.), saving/resuming analyses, and
  keyboard shortcuts.
- [CONTROLS.md](CONTROLS.md) - canvas mouse control reference for the
  tracking tabs.

## Development

- Tests: `pytest tests/`
- Lint: `ruff check .`
- Building the Windows installer: `pyinstaller EDyssey.spec`, then either
  `iscc installer\EDyssey_online.iss` or `iscc installer\EDyssey_offline.iss`
  (the offline one needs the model weight files present locally first - see
  its header comment). See [EDyssey.spec](EDyssey.spec)'s header comment for
  the overall packaging rationale.

## License

GPL-3.0 - see [LICENSE](LICENSE). This follows from two runtime
dependencies (PyQt5, HyperSpy) that are themselves GPLv3, which doesn't
allow a more permissive or usage-restricted license on top.

Some model weights are downloaded on demand rather than bundled - see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for what each one is,
its license, and where it comes from (one of them, the NanoTrack tracker
model, currently has an unclear license - flagged there).
