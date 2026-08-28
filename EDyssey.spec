# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for EDyssey (py5DED), a PyQt5 desktop app.

Build with: pyinstaller EDyssey.spec
Output: dist/EDyssey/ (onedir - see rationale below), wrapped by either
installer/EDyssey_online.iss or installer/EDyssey_offline.iss (Inno Setup)
for actual distribution.

Onedir, not onefile: the dependency stack (hyperspy + dask + PyQt5 + scipy +
scikit-image + pandas + matplotlib + opencv-contrib + hdf5 DLLs) is too
large for onefile's extract-to-%TEMP%-on-every-launch model (minutes-long
startup, and a stale/AV-quarantined extraction directory breaks the next
launch). Onedir - a plain folder installed normally - is also the far more
common shape for antivirus heuristics to trust.

Two build modes, toggled by the EDYSSEY_OFFLINE_BUILD env var:

- Default (unset/"0"): the "online" build. Heavy model weights (SAM2
  checkpoint, Nano/DaSiamRPN tracker ONNX files, ~1.06GB) and torch/sam2
  (multi-GB, CUDA-version-specific) are all deliberately NOT bundled -
  EDyssey/tracking_utils/asset_fetch.py downloads the model weights on
  first use instead, and INSTALL.md documents installing torch/sam2
  manually. Most installs never touch SAM2 or the Nano/DaSiamRPN trackers
  at all, and bundling everything would put the installer uncomfortably
  close to GitHub Releases' 2GB-per-asset cap.
- EDYSSEY_OFFLINE_BUILD=1: the "offline" build. Bundles torch+CUDA+
  torchvision+sam2 (and their own dependencies, hydra-core/iopath/
  omegaconf) directly, so the SAM2 tab works with zero setup after
  install. Model *weight* files (the SAM2 checkpoint, tracker ONNX files)
  are still not bundled here - those are staged separately by
  installer/EDyssey_offline.iss's [Files] section, copied straight from
  wherever they already sit on the build machine, since they're plain
  data files asset_fetch.py's own "already present at the right size,
  skip downloading" check picks up with zero extra wiring. This build is
  large (multi-GB) and built locally, not by CI - see
  installer/EDyssey_offline.iss's header comment.

  set EDYSSEY_OFFLINE_BUILD=1  (cmd)
  $env:EDYSSEY_OFFLINE_BUILD=1  (PowerShell)
"""
import os
from PyInstaller.utils.hooks import collect_all

OFFLINE_BUILD = os.environ.get('EDYSSEY_OFFLINE_BUILD', '0') == '1'

hs_datas, hs_binaries, hs_hidden = collect_all('hyperspy')
rs_datas, rs_binaries, rs_hidden = collect_all('rsciio')
dask_datas, dask_binaries, dask_hidden = collect_all('dask')
# pyxem is an optional hyperspy extension - tracking_utils_ui.py already
# guards `import pyxem` in a try/except to make it optional at the app
# level. But hyperspy/extensions.py unconditionally looks up every
# extension package's importlib metadata/spec at import time (not just the
# ones this app actually imports), and crashes with AttributeError if
# pyxem's spec can't be resolved (tried excluding it outright first - that
# left stale entry-point metadata behind without the actual package,
# which is worse). Bundling it for real, matching what's installed in the
# build environment, is what actually reproduces a working non-frozen run.
px_datas, px_binaries, px_hidden = collect_all('pyxem')
# orix (a pyxem dependency, pulled in via pyxem -> ... -> orix.crystal_map)
# uses the `lazy_loader` package, which reads a sibling `.pyi` stub file at
# runtime to know what to lazily expose - a real file `lazy_loader` opens
# by path, not something import tracing or PYZ-embedded bytecode
# satisfies. collect_all('pyxem') above only collects the pyxem package
# itself, not this separate third-party dependency, so orix's .py files
# got traced/bundled normally but its .pyi stub didn't, crashing with
# "Cannot load imports from non-existent stub '...\\orix\\crystal_map\\
# __init__.pyi'" the first time a hyperspy loader path reaches pyxem
# (e.g. loading a .mib file). Collecting orix explicitly, the same way as
# pyxem, fixes it.
orix_datas, orix_binaries, orix_hidden = collect_all('orix')

torch_datas, torch_binaries, torch_hidden = [], [], []
torch_excludes = ['torch', 'sam2', 'torchvision']
# Every worker_*.py script is a loose `datas` file, NOT a scripts= entry -
# runpy.run_path() (worker_dispatch.run_worker(), the receiving end of the
# --worker dispatch) needs a real file on disk to run; one only present as
# compiled bytecode inside the PYZ archive isn't (confirmed the hard way:
# worker_extract_frame.py/worker_nav_img.py used to be scripts= entries
# instead, which crashed `--worker extract_frame`/`--worker nav_img` with
# "can't find '__main__' module" - runpy falls back to treating a
# non-existent path as a module-search target). Their imports don't need
# separate tracing either: they're already a strict subset of
# EDyssey_MainWindow.py's own (fully traced as the real scripts= entry),
# except worker_sam.py's torch/sam2 imports, which must NOT be traced for
# the online build - see the torch/sam2/torchvision exclude below.
# Bundling torch/sam2 for the offline build doesn't need scripts= tracing
# either: collect_all()'s hiddenimports already force-includes a package's
# submodules independent of whether anything traced imports them - that's
# the whole point of hiddenimports, and it's what actually wires torch/
# sam2 into the build.
scripts = ['EDyssey_MainWindow.py']
extra_datas = [
    ('worker_sam.py', '.'),
    ('worker_extract_frame.py', '.'),
    ('worker_nav_img.py', '.'),
    ('worker_nav_img_batch.py', '.'),
    # Imported (not runpy'd) by every *_batch.py worker and by ui_tabs/
    # tab_*.py directly - a loose file regardless, same as the rest of this
    # list, since nothing currently traced as a scripts= entry imports it,
    # so PyInstaller's static analysis wouldn't otherwise bundle it.
    ('worker_pool_utils.py', '.'),
]
if OFFLINE_BUILD:
    torch_excludes = []
    for pkg in ('torch', 'torchvision', 'sam2', 'hydra', 'iopath', 'omegaconf'):
        d, b, h = collect_all(pkg)
        torch_datas += d
        torch_binaries += b
        torch_hidden += h

block_cipher = None

a = Analysis(
    # No worker_*.py script is listed here in either build mode - see the
    # scripts=/extra_datas comment above. Keeping worker_sam.py
    # specifically out of Analysis' own tracing also matters for a second
    # reason: if PyInstaller's static analyzer ever traced a script that
    # `import torch`s, it would bundle torch (and any local CUDA DLLs) into
    # the online build too, which is exactly what that build doesn't want.
    scripts,
    pathex=[],
    binaries=[
        ('EDyssey/io_utils/eventem.cp312-win_amd64.pyd', 'EDyssey/io_utils'),
        ('EDyssey/io_utils/hdf5.dll', 'EDyssey/io_utils'),
        ('EDyssey/io_utils/hdf5_cpp.dll', 'EDyssey/io_utils'),
        ('EDyssey/io_utils/hdf5_hl.dll', 'EDyssey/io_utils'),
    ] + hs_binaries + rs_binaries + dask_binaries + px_binaries + orix_binaries + torch_binaries,
    datas=[
        # io_utils_ui.py is imported two ways elsewhere in this codebase:
        # package-relative (EDyssey/io_utils/__init__.py) AND as a bare
        # `import io_utils_ui` via sys.path (worker_nav_img.py,
        # EDyssey/tracking_utils/tracking_utils_ui.py). PyInstaller compiles
        # pure-Python modules into the embedded PYZ archive, which satisfies
        # the first form but not the second - it needs an actual loose file
        # on disk too.
        ('EDyssey/io_utils/io_utils_ui.py', 'EDyssey/io_utils'),
        ('ui_tabs/logo', 'ui_tabs/logo'),
    ] + extra_datas + hs_datas + rs_datas + dask_datas + px_datas + orix_datas + torch_datas,
    hiddenimports=(['matplotlib.backends.backend_qt5agg']
                    + hs_hidden + rs_hidden + dask_hidden + px_hidden + orix_hidden + torch_hidden),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        *torch_excludes,  # backstop in the online build - scripts= above already keeps these unreachable there
        # The app only uses PyQt5 (see e.g. EDyssey_MainWindow.py's
        # `import PyQt5.QtWidgets`, and the explicit backend_qt5agg hidden
        # import above) - PySide6/PySide2/PyQt6 showing up here is just
        # whatever else happens to be installed in the build environment
        # (e.g. an IDE's own Qt bindings). PyInstaller refuses to bundle
        # more than one Qt binding in the same frozen app, so these must be
        # excluded explicitly rather than left for its hook auto-detection
        # to trip over.
        'PySide6', 'PySide2', 'PyQt6',
        # cupy is an optional GPU-array backend dask.array.chunk_types.py
        # probes for (`try: import cupy ... except ImportError: pass`) -
        # this app never uses GPU dask arrays. Installed in this build
        # machine's environment for unrelated reasons, cupy IS importable
        # there, but its native DLL-directory setup
        # (cupy/_environment.py's _setup_win32_dll_directory) looks for a
        # 'bin' folder relative to its own package location that doesn't
        # exist in a frozen layout, raising FileNotFoundError - a type the
        # `except ImportError` above doesn't catch, crashing the frozen
        # app at import time. Excluding it outright makes `import cupy`
        # fail with the plain ImportError dask already handles, exactly as
        # it does on any machine without cupy installed at all.
        'cupy', 'cupyx', 'cupy_backends',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EDyssey',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # console=True: gives the app a real console window (raw prints, torch/
    # CUDA's own stderr chatter, tracebacks that don't make it into the Qt
    # log console all land somewhere visible). EDyssey_MainWindow.py pushes
    # it behind the main window on startup so it doesn't steal focus - it's
    # still right there in the taskbar to check. Worker subprocesses
    # (QProcess, no special creation flags) inherit this same console
    # rather than popping up their own new windows - standard Windows
    # child-process console inheritance when CREATE_NEW_CONSOLE isn't set.
    console=True,

    icon='ui_tabs/logo/EDyssey_logo.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='EDyssey',
)
