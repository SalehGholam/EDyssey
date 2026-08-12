# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for EDyssey (py5DED), a PyQt5 desktop app.

Build with: pyinstaller EDyssey.spec
Output: dist/EDyssey/ (onedir - see rationale below), wrapped by
installer/EDyssey.iss (Inno Setup) for actual distribution.

Onedir, not onefile: the dependency stack (hyperspy + dask + PyQt5 + scipy +
scikit-image + pandas + matplotlib + opencv-contrib + hdf5 DLLs) is too
large for onefile's extract-to-%TEMP%-on-every-launch model (minutes-long
startup, and a stale/AV-quarantined extraction directory breaks the next
launch). Onedir - a plain folder installed normally - is also the far more
common shape for antivirus heuristics to trust.

Heavy model weights (SAM2 checkpoint, Nano/DaSiamRPN tracker ONNX files,
~1.06GB total) are deliberately NOT bundled here - EDyssey/tracking_utils/
asset_fetch.py downloads them on first use of the relevant feature instead
(most installs never touch SAM2 or the Nano/DaSiamRPN trackers at all, and
bundling them would put the installer uncomfortably close to GitHub
Releases' 2GB-per-asset cap). Same reasoning for torch/sam2 - not bundled,
see INSTALL.md.
"""
from PyInstaller.utils.hooks import collect_all

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

block_cipher = None

a = Analysis(
    # worker_sam.py is deliberately NOT listed here (see excludes= below) -
    # if PyInstaller's static analyzer ever traces a script that
    # `import torch`s, it bundles torch (and any local CUDA DLLs) into the
    # build automatically, which is exactly what we don't want. It still
    # reaches the frozen app as a loose `datas` file below, executed later
    # via worker_dispatch.run_worker()'s runpy.run_path().
    ['EDyssey_MainWindow.py', 'worker_extract_frame.py', 'worker_nav_img.py'],
    pathex=[],
    binaries=[
        ('EDyssey/io_utils/eventem.cp312-win_amd64.pyd', 'EDyssey/io_utils'),
        ('EDyssey/io_utils/hdf5.dll', 'EDyssey/io_utils'),
        ('EDyssey/io_utils/hdf5_cpp.dll', 'EDyssey/io_utils'),
        ('EDyssey/io_utils/hdf5_hl.dll', 'EDyssey/io_utils'),
    ] + hs_binaries + rs_binaries + dask_binaries + px_binaries,
    datas=[
        ('worker_sam.py', '.'),
        # io_utils_ui.py is imported two ways elsewhere in this codebase:
        # package-relative (EDyssey/io_utils/__init__.py) AND as a bare
        # `import io_utils_ui` via sys.path (worker_nav_img.py,
        # EDyssey/tracking_utils/tracking_utils_ui.py). PyInstaller compiles
        # pure-Python modules into the embedded PYZ archive, which satisfies
        # the first form but not the second - it needs an actual loose file
        # on disk too.
        ('EDyssey/io_utils/io_utils_ui.py', 'EDyssey/io_utils'),
        ('ui_tabs/logo', 'ui_tabs/logo'),
    ] + hs_datas + rs_datas + dask_datas + px_datas,
    hiddenimports=['matplotlib.backends.backend_qt5agg'] + hs_hidden + rs_hidden + dask_hidden + px_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'sam2', 'torchvision',  # backstop - scripts= above already keeps these unreachable
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
    console=False,  # windowed app; worker subprocess I/O goes through QProcess pipes, not a console
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
