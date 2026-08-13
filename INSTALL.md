# Installing EDyssey

## Option A: Windows installer (recommended for most users)

Two installer variants exist - same app either way, they only differ in
whether the (large) model weight files come bundled or get downloaded
later:

- **Online** (small, ~200MB): download `EDyssey_Setup_Online_<version>.exe` from the
  [Releases](https://github.com/SalehGholam/EDyssey/releases) page and run
  it - no Python setup needed. The SAM2 checkpoint and the Nano/DaSiamRPN
  tracker model files are downloaded automatically, once, the first time
  you actually use the relevant feature (SAM2 Seg. tab, or the
  "nano"/"dasiamrpn" tracker options in Tracking by CV2) - see
  [EDyssey/tracking_utils/asset_fetch.py](EDyssey/tracking_utils/asset_fetch.py)
  and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exactly what
  gets downloaded and from where. This needs an internet connection the
  first time; nothing is re-downloaded after that.
- **Offline** (~2.6GB): `EDyssey_Setup_Offline_<version>.exe` bundles
  those same model files, **and** `torch`/CUDA/`sam2` themselves, so
  nothing downloads or needs installing later, ever - useful for a machine
  with no/restricted internet access, or just to skip the manual step
  below entirely. This one is too large for a GitHub Release, so it isn't
  published there - ask whoever maintains this repo for a copy directly.

With the online installer (or running from source), `torch`/`sam2` are
**not** installed automatically - see "Enabling SAM2" below. Every other
tab/feature works out of the box either way. The offline installer needs
no such step - SAM2 works immediately after install.

## Option B: Run from source

### 1. Python dependencies

```
pip install -r requirements.txt
```

Then install PyTorch separately, matching your CUDA version (or CPU-only),
following https://pytorch.org/get-started/locally/ - `requirements.txt`
deliberately leaves it out since a blind `pip install torch` often pulls
the wrong build for your GPU.

### 2. SAM2 (Meta's segment-anything-2)

Only needed for the SAM2 tab's AI segmentation. Not on PyPI under a stable
name - install from source:

```
pip install git+https://github.com/facebookresearch/sam2.git
```

(or clone it and `pip install -e .` for a local editable install). The
checkpoint itself (`sam2.1_hiera_large.pt`) no longer needs manual
placement - `worker_sam.py`/`tab_sam2.py` fetch it automatically into
`EDyssey/tracking_utils/SAM2_checkpoints/` on first use, the same as the
packaged installer does. The Nano/DaSiamRPN tracker weights work the same
way (`EDyssey/tracking_utils/opencv_models/`).

### 3. eventem / pacbed (tpx3 loading)

These are compiled binaries (`.pyd` on Windows / `.so` on Linux) that ship
bundled in this repo under `EDyssey/io_utils/` - nothing to install
separately, but they're platform- and Python-version-specific. If you're on
a different platform/Python version than they were built for, `.tpx3` file
loading (the eventem-based fast paths in `EDyssey/io_utils/loaders.py` and
`nav_image.py`) will fail to import; every other file format (.hdf5, .hspy,
.zspy, .mib) works without them.

### 4. ffmpeg (optional, for video export)

The clip-export functions (`EDyssey/io_utils/video.py`) pipe frames to
`ffmpeg` if it's found on `PATH`, for fast `.mp4` encoding. If it's not
found, they fall back to a slower `.gif` via matplotlib - no hard
dependency, just slower/larger output without it.

### 5. Launching

Run `EDyssey_MainWindow.py`. See [MANUAL.md](MANUAL.md) for usage.

## Enabling SAM2 (running from source, or the online installer)

Skip this section if you're using the offline installer - it already
includes `torch`/CUDA/`sam2`.

`torch` and the `sam2` package are deliberately not installed by
`requirements.txt` or the online installer - `torch` alone is a multi-GB,
CUDA-version-specific download (see https://pytorch.org/get-started/locally/),
so there's no single build that would be right for every machine.

- **Running from source**: `pip install torch` (matching your CUDA version)
  into whatever environment you installed `requirements.txt` into, then
  the SAM2-from-source step above.
- **Windows installer**: install both into the app's own bundled
  environment rather than a system Python, since the installed app doesn't
  use one:
  ```
  pip install --target "<install_dir>\_internal" torch --index-url https://download.pytorch.org/whl/cu121
  pip install --target "<install_dir>\_internal" git+https://github.com/facebookresearch/sam2.git
  ```
  Swap the `--index-url` per pytorch.org for your GPU (or omit it for
  CPU-only), and replace `<install_dir>` with wherever EDyssey was
  installed (shown in the app's "About"/install location, typically
  `%LocalAppData%\Programs\EDyssey` for a per-user install or
  `C:\Program Files\EDyssey` for an all-users one). This needs *some*
  separate Python+pip available on the machine as a tool - not the
  installed app itself, which doesn't ship one. If you skip this step,
  every tab except SAM2 Seg. still works; that tab shows a message with
  this same command when you try to use it without `torch`/`sam2` present.
