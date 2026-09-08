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
import re
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.win32 import versioninfo as _winversioninfo

OFFLINE_BUILD = os.environ.get('EDYSSEY_OFFLINE_BUILD', '0') == '1'

# EDyssey.exe's own Explorer Properties > Details tab (CompanyName,
# FileVersion, ...) - built programmatically here (never a separate file to
# keep in sync) from EDyssey_MainWindow.py's own APP_VERSION, so it can't
# silently drift out of date the way a hand-edited version_info.txt would.
# A Win32 version resource caps each of the 4 numeric FILEVERSION fields at
# 65535, which APP_VERSION's YYYYMMDD segment doesn't fit - re-encoded as
# MMDD/HHMM instead (drops the year; this tuple is purely decorative
# Explorer-Properties metadata, not used anywhere the app itself checks its
# own version - see installer/EDyssey_common.iss's identical VersionInfo*
# re-encoding for setup.exe's own copy of this same problem).
with open('EDyssey_MainWindow.py', encoding='utf-8') as _f:
    _app_version_str = re.search(r"APP_VERSION = '([\d.]+)'", _f.read()).group(1)
_ver_major, _ver_minor, _ver_yyyymmdd, _ver_hhmm = _app_version_str.split('.')
_filevers = (int(_ver_major), int(_ver_minor), int(_ver_yyyymmdd[4:8]), int(_ver_hhmm))

_version_info = _winversioninfo.VSVersionInfo(
    ffi=_winversioninfo.FixedFileInfo(filevers=_filevers, prodvers=_filevers),
    kids=[
        _winversioninfo.StringFileInfo([
            _winversioninfo.StringTable('040904B0', [
                _winversioninfo.StringStruct('CompanyName', 'Saleh Gholam'),
                _winversioninfo.StringStruct(
                    'FileDescription', 'EDyssey - 4D-STEM Tomography Analysis'),
                _winversioninfo.StringStruct('FileVersion', _app_version_str),
                _winversioninfo.StringStruct('InternalName', 'EDyssey'),
                _winversioninfo.StringStruct('LegalCopyright', 'Copyright (c) Saleh Gholam'),
                _winversioninfo.StringStruct('OriginalFilename', 'EDyssey.exe'),
                _winversioninfo.StringStruct('ProductName', 'EDyssey'),
                _winversioninfo.StringStruct('ProductVersion', _app_version_str),
            ]),
        ]),
        _winversioninfo.VarFileInfo([_winversioninfo.VarStruct('Translation', [1033, 1200])]),
    ],
)

hs_datas, hs_binaries, hs_hidden = collect_all('hyperspy')
rs_datas, rs_binaries, rs_hidden = collect_all('rsciio')
dask_datas, dask_binaries, dask_hidden = collect_all('dask')
# pyxem/orix are NOT collected (and never traced/imported anywhere in this
# codebase - tracking_utils_ui.py used to optionally use pyxem, but no
# longer does). Not calling collect_all('pyxem') at all - rather than
# excluding it - matters: hyperspy/extensions.py unconditionally iterates
# importlib.metadata.entry_points(group="hyperspy.extensions") at import
# time and crashes if any *registered* extension's spec can't be resolved.
# collect_all() is what bundles a package's entry-point metadata in the
# first place (confirmed by reproducing the exact crash and reading
# hyperspy's own extensions.py); never bundling it at all means hyperspy
# finds nothing registered under that group, instead of finding a stale
# registration with no package behind it.
# scikit-learn (Blob Selection's K-Means/GMM segmentation methods - see
# EDyssey/io_utils/blob_segmentation.py) ships several compiled Cython
# submodules PyInstaller's static import-tracing commonly misses (e.g.
# sklearn.utils._cython_blas, sklearn.neighbors._partition_nodes) - same
# class of problem as hyperspy/rsciio/dask above, so it gets
# the same collect_all() treatment rather than relying on default tracing.
sk_datas, sk_binaries, sk_hidden = collect_all('sklearn')

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
# worker_*.py live under EDyssey/workers/ in the install (matches
# worker_dispatch._base_dir()) - grouped under EDyssey/ along with
# EDyssey/io_utils and ui_tabs/logo below, so every EDyssey-authored file
# staged as a loose file sits in one named folder rather than scattered
# across several top-level ones.
extra_datas = [
    ('workers/worker_sam.py', 'EDyssey/workers'),
    ('workers/worker_extract_frame.py', 'EDyssey/workers'),
    ('workers/worker_extract_frame_batch.py', 'EDyssey/workers'),
    ('workers/worker_nav_img.py', 'EDyssey/workers'),
    ('workers/worker_nav_img_batch.py', 'EDyssey/workers'),
    ('workers/worker_pool_utils.py', 'EDyssey/workers'),
    # worker_sam.py's _load_asset_fetch() loads these two directly by file
    # path (bypassing EDyssey.io_utils/EDyssey.tracking_utils's own
    # __init__.py, which pulls in hyperspy/dask/PyQt5) - staged under
    # EDyssey/workers rather than EDyssey/io_utils or EDyssey/tracking_utils
    # because either of those destinations collides with PyInstaller
    # auto-compiling them into the PYZ via tab_sam2.py's own normal import of
    # them (which silently wins over an explicit datas= entry at the same
    # path) - EDyssey/workers has no such collision, same as the entries above.
    ('EDyssey/io_utils/app_dirs.py', 'EDyssey/workers'),
    ('EDyssey/tracking_utils/asset_fetch.py', 'EDyssey/workers'),
]
if OFFLINE_BUILD:
    torch_excludes = []
    for pkg in ('torch', 'torchvision', 'sam2', 'hydra', 'iopath', 'omegaconf'):
        d, b, h = collect_all(pkg)
        torch_datas += d
        torch_binaries += b
        torch_hidden += h

# Forces a loose .py source copy (not just the default compiled-into-PYZ
# form) for torch/sam2's own full runtime dependency closure, offline build
# only - worker_sam.py runs this closure via a real, separate portable
# Python interpreter (see python_finder.find_bundled_python()), which can
# only import a package from a plain file/folder on disk, not PyInstaller's
# own embedded PYZ archive (readable only by the frozen exe's own bootstrap
# importer). Confirmed the hard way: an early offline build attempt hit
# "No module named 'typing_extensions'" - present in the frozen app's own
# _internal (as a .dist-info folder) but with no matching loose .py file,
# because nothing had ever needed one until a *separate* interpreter tried
# to import it. This list is the actual unconditional (no environment
# marker - i.e. not an optional extra) dependency closure of torch,
# torchvision, sam2, hydra-core, iopath and omegaconf together, walked via
# importlib.metadata against this exact build environment and hand-
# reviewed - not derived at build time, so it stays reviewable/stable
# across rebuilds rather than silently changing if the build machine's own
# installed package set ever does. Does NOT need to cover the *stdlib*
# side of the same inspect.getsource() problem (e.g. `enum`) - the bundled
# portable Python interpreter carries its own real stdlib source
# independently (see EDyssey/portable_python/Lib/), unrelated to anything
# PyInstaller collects here.
_TORCH_DEPENDENCY_CLOSURE_IMPORT_NAMES = [
    'PIL', '_distutils_hack', '_yaml', 'antlr4', 'filelock', 'fsspec',
    'functorch', 'hydra', 'iopath', 'isympy', 'jinja2', 'markupsafe',
    'mpmath', 'networkx', 'numpy', 'omegaconf', 'packaging', 'pkg_resources',
    'portalocker', 'setuptools', 'sympy', 'torch', 'torchgen', 'torchvision',
    'tqdm', 'typing_extensions', 'yaml',
]
module_collection_mode = ({name: 'pyz+py' for name in _TORCH_DEPENDENCY_CLOSURE_IMPORT_NAMES}
                           if OFFLINE_BUILD else None)

# PyInstaller's own default manifest (PyInstaller/utils/win32/winmanifest.py)
# declares no DPI awareness at all, which leaves it up to Qt5's own runtime
# negotiation (SetProcessDpiAwarenessContext, called from qwindows.dll before
# any window exists) to make the process per-monitor DPI aware - normally
# fine, but unreliable enough in practice (this Qt5 build, this exact
# Windows version/monitor combo) that a user hit oversized/blurry UI on a
# second monitor with a different scale factor than their primary, only
# fixed by manually forcing Windows' own compatibility override (Properties
# > Compatibility > Change high DPI settings > "System (Enhanced)" +
# "Fix scaling problems"). Declaring PerMonitorV2 awareness directly in the
# exe's own manifest - same mechanism that Windows Settings toggle itself
# writes - makes Windows tell the real per-monitor DPI (and skip bitmap-
# stretching) from process start, without depending on Qt's own runtime
# negotiation succeeding; EDyssey_MainWindow.py's own
# AA_EnableHighDpiScaling/PassThrough rounding policy (set before
# QApplication()) then does the actual logical-to-device-pixel scaling once
# it receives that real per-monitor DPI. `dpiAwareness` (2016 namespace,
# Windows 10 1703+, needs the Win10 supportedOS GUID below - already
# present) is the modern per-monitor-v2 declaration; `dpiAware` (2005
# namespace, `true/pm`) is the legacy per-monitor-v1 fallback for older
# Windows - both included together per Microsoft's own documented pattern.
# Everything else here is copied verbatim from PyInstaller's own default
# manifest (winmanifest.py's _DEFAULT_MANIFEST_XML) since supplying a custom
# manifest replaces rather than merges with it.
_MANIFEST_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <supportedOS Id="{e2011457-1546-43c5-a5fe-008deee3d3f0}"></supportedOS>
      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"></supportedOS>
      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"></supportedOS>
      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"></supportedOS>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"></supportedOS>
    </application>
  </compatibility>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2, PerMonitor, System</dpiAwareness>
    </windowsSettings>
  </application>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"></assemblyIdentity>
    </dependentAssembly>
  </dependency>
</assembly>
"""

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
    ] + hs_binaries + rs_binaries + dask_binaries + sk_binaries + torch_binaries,
    datas=[
        # io_utils_ui.py is imported two ways elsewhere in this codebase:
        # package-relative (EDyssey/io_utils/__init__.py) AND as a bare
        # `import io_utils_ui` via sys.path (worker_nav_img.py,
        # EDyssey/tracking_utils/tracking_utils_ui.py). PyInstaller compiles
        # pure-Python modules into the embedded PYZ archive, which satisfies
        # the first form but not the second - it needs an actual loose file
        # on disk too.
        ('EDyssey/io_utils/io_utils_ui.py', 'EDyssey/io_utils'),
        ('ui_tabs/logo', 'EDyssey/ui_tabs/logo'),
    ] + extra_datas + hs_datas + rs_datas + dask_datas + sk_datas + torch_datas,
    hiddenimports=(['matplotlib.backends.backend_qt5agg',
                     # stdlib module PyInstaller's static tracing misses -
                     # torch/sam2's own deps (hydra/omegaconf/iopath) import
                     # it transitively at runtime, not traceably.
                     'modulefinder']
                    + hs_hidden + rs_hidden + dask_hidden + sk_hidden + torch_hidden),
    hookspath=[],
    hooksconfig={},
    # Must run before PyInstaller's own pyi_rth_pyqt5.py - see the hook's
    # own docstring for why.
    runtime_hooks=['pyinstaller_hooks/rthook_preload_system_crt.py'],
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
    module_collection_mode=module_collection_mode,
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
    manifest=_MANIFEST_XML,
    version=_version_info,
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
