# Installing EDyssey

## Option A: Windows installer (recommended for most users)

Two installer variants exist - same app either way, they only differ in
whether the (large) model weight files come bundled or get downloaded
later:

- **Online** (small, ~200MB): download `EDyssey_Setup_Online_<version>.exe` from the
  [Releases](https://github.com/SalehGholam/EDyssey/releases) page and run
  it - no Python setup needed. The SAM2 checkpoint and the Nano/DaSiamRPN
  tracker model files are downloaded automatically, once, the first time
  you actually use the relevant feature (SAM2 Tracker tab, or the
  "nano"/"dasiamrpn" tracker options in the ROI Tracker tab) - see
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
pip install https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip
```

(or clone it and `pip install -e .` for a local editable install - needs
`git`, unlike the plain URL above, which doesn't). The
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
`nav_image.py`) will fail to import; every other file format
(`.hdf5 (eventem)`, `.hdf5`, `.hspy`, `.zspy`, `.mib`, `.blo`) works without
them - `.hdf5 (eventem)` (eventem's own export layout, internally the
'.hdf5_eventem' dtype) is read directly via h5py/dask, not the compiled
eventem extension itself, despite the name.

### 4. ffmpeg (optional, for video export)

The clip-export functions (`EDyssey/io_utils/video.py`) pipe frames to
`ffmpeg` if it's found on `PATH`, for fast `.mp4` encoding. If it's not
found, they fall back to a slower `.gif` via matplotlib - no hard
dependency, just slower/larger output without it.

### 5. Launching

Run `EDyssey_MainWindow.py`. See [MANUAL.md](MANUAL.md) for usage.

## Enabling SAM2 (running from source, or the online installer)

**You need an NVIDIA GPU driver already installed for GPU support to work
at all** - this applies to every install method, including the offline
installer. `torch`'s CUDA builds only bundle the CUDA *runtime* libraries,
not the GPU driver itself - that's a separate, OS-level component only
NVIDIA provides (via [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx)
or your GPU/laptop vendor's own updater), not something `pip install torch`
or this app can install for you. Without one, SAM2 still works, just on
CPU (much slower) - Set Up SAM2's GPU dropdown quietly falls back to "CPU
only" if it can't detect a driver via `nvidia-smi`, and Check CUDA
reporting `torch.cuda.is_available(): False` after a GPU install usually
means this, not a broken install.

Skip the rest of this section if you're using the offline installer - it
already includes `torch`/CUDA/`sam2`.

`torch` and the `sam2` package are deliberately not installed by
`requirements.txt` or the online installer - `torch` alone is a multi-GB,
CUDA-version-specific download (see https://pytorch.org/get-started/locally/),
so there's no single build that would be right for every machine.

**Easiest**: open EDyssey and use **Help > Set Up SAM2...** in the menu
bar. It picks the install location for you, auto-detects your GPU (reading
your NVIDIA driver via `nvidia-smi`) and pre-selects a matching CUDA
option in the dropdown - every CUDA line PyTorch currently publishes is
offered, plus CPU-only - and runs the install commands with the output
shown live in the dialog, no manual pip commands or path-hunting needed.
Use the **Check CUDA** button afterwards to confirm `torch.cuda.is_available()`
actually reports your GPU, rather than assuming the install worked.

**Windows installer, run EDyssey as Administrator for this step**: the
installer's default location is `C:\Program Files\EDyssey`, which needs
elevated rights to write into - Set Up SAM2 installs directly into that
folder (`_internal\sam2_packages`), so it fails there unless EDyssey itself
is running elevated (right-click `EDyssey.exe` or its Start Menu shortcut >
**Run as administrator**). Not needed if you installed to a per-user
location instead (the installer's directory-selection page lets you choose
one under `%LocalAppData%\Programs`), or if you're running from source.

It still needs *some* separate Python+pip already on the machine as a tool
(not the installed app itself, which doesn't ship one) - the dialog says
so and links to python.org if none is found. This isn't just an install-time
requirement: the packaged app always runs the SAM2 tab through this same
real Python interpreter rather than its own frozen executable (torch needs
things a frozen build can't provide), so don't remove that Python install
afterwards.

Manual alternative, if you'd rather run the commands yourself:

- **Running from source**: `pip install torch` (matching your CUDA version)
  into whatever environment you installed `requirements.txt` into, then
  the SAM2-from-source step above.
- **Windows installer**: install into a dedicated `sam2_packages` subfolder
  of the app's own install (not a system Python's site-packages, and not
  `_internal` directly - it stays isolated from what the app itself already
  bundles):
  ```
  pip install --target "<install_dir>\_internal\sam2_packages" --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install --target "<install_dir>\_internal\sam2_packages" --upgrade "https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip" --no-deps
  pip install --target "<install_dir>\_internal\sam2_packages" --upgrade hydra-core iopath pillow tqdm
  ```
  The `--no-deps` on the second command matters: sam2's own `setup.py`
  lists an unpinned `torch`/`torchvision` requirement, and without
  `--no-deps` this command would silently resolve that against the default
  PyPI index (CPU-only builds only) and downgrade the GPU torch the first
  command just installed - the third command installs sam2's actual other
  dependencies instead.

  Swap the `--index-url` per pytorch.org for your GPU (or omit it for
  CPU-only), and replace `<install_dir>` with wherever EDyssey was
  installed (shown in the app's "About"/install location, typically
  `%LocalAppData%\Programs\EDyssey` for a per-user install or
  `C:\Program Files\EDyssey` for an all-users one).

If you skip this entirely, every tab except SAM2 Tracker still works; that
tab shows a message pointing at Help > Set Up SAM2... when you try to use
it without `torch`/`sam2` present.
