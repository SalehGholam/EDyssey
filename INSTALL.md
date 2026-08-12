# Installing EDyssey

## 1. Python dependencies

```
pip install -r requirements.txt
```

Then install PyTorch separately, matching your CUDA version (or CPU-only),
following https://pytorch.org/get-started/locally/ - `requirements.txt`
deliberately leaves it out since a blind `pip install torch` often pulls
the wrong build for your GPU.

## 2. SAM2 (Meta's segment-anything-2)

Only needed for the SAM2 tab's AI segmentation. Not on PyPI under a stable
name - install from source:

```
pip install git+https://github.com/facebookresearch/sam2.git
```

(or clone it and `pip install -e .` for a local editable install). You'll
also need a SAM2 checkpoint (`.pt`) and its matching config, pointed to by
`worker_sam.py`.

## 3. eventem / pacbed (tpx3 loading)

These are compiled binaries (`.pyd` on Windows / `.so` on Linux) that ship
bundled in this repo under `EDyssey/io_utils/` - nothing to install
separately, but they're platform- and Python-version-specific. If you're on
a different platform/Python version than they were built for, `.tpx3` file
loading (the eventem-based fast paths in `EDyssey/io_utils/loaders.py` and
`nav_image.py`) will fail to import; every other file format (.hdf5, .hspy,
.zspy, .mib) works without them.

## 4. ffmpeg (optional, for video export)

The clip-export functions (`EDyssey/io_utils/video.py`) pipe frames to
`ffmpeg` if it's found on `PATH`, for fast `.mp4` encoding. If it's not
found, they fall back to a slower `.gif` via matplotlib - no hard
dependency, just slower/larger output without it.

## 5. Launching

Run `EDyssey_MainWindow.py`. See [MANUAL.md](MANUAL.md) for usage.
