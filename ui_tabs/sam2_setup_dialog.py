# -*- coding: utf-8 -*-
"""Help > Set Up SAM2... - automates the "Enabling SAM2" steps INSTALL.md
documents for running from source or the online installer (the offline
installer already bundles torch/sam2, and doesn't need this).

Runs `pip install [--target <install_dir>\\_internal\\sam2_packages] torch
[--index-url ...]` and the same for the `sam2` package, streaming their
output into a
log box instead of making the user copy/paste commands from a message box
themselves (see tab_sam2.py's _show_missing_dependency_dialog, which still
shows the manual command as a fallback if this dialog can't run - e.g. no
system Python found).
"""
import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys

import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QTextCursor

from ui_tabs.python_finder import (
    find_bundled_python, find_system_python, read_interpreter_marker, write_interpreter_marker)

# (label, cuda_version, index_url), highest CUDA first, CPU only last -
# every index PyTorch currently publishes wheels under (confirmed against
# download.pytorch.org/whl/<tag>/torch/ directly, 2026-09) - cu132/cu130/
# cu126 are the actively-updated lines (latest torch on each as of writing),
# cu128/cu124/cu121/cu118 are older lines PyTorch stopped publishing new
# torch releases for but are still installable for older drivers.
# _pick_default_device below picks the newest one a detected driver can
# still run (backward-compatible), so this order matters.
_CUDA_OPTIONS = [
    ('NVIDIA GPU (CUDA 13.2)', (13, 2), 'https://download.pytorch.org/whl/cu132'),
    ('NVIDIA GPU (CUDA 13.0)', (13, 0), 'https://download.pytorch.org/whl/cu130'),
    ('NVIDIA GPU (CUDA 12.8)', (12, 8), 'https://download.pytorch.org/whl/cu128'),
    ('NVIDIA GPU (CUDA 12.6)', (12, 6), 'https://download.pytorch.org/whl/cu126'),
    ('NVIDIA GPU (CUDA 12.4)', (12, 4), 'https://download.pytorch.org/whl/cu124'),
    ('NVIDIA GPU (CUDA 12.1)', (12, 1), 'https://download.pytorch.org/whl/cu121'),
    ('NVIDIA GPU (CUDA 11.8)', (11, 8), 'https://download.pytorch.org/whl/cu118'),
    ('CPU only (no GPU)', None, None),
]
_INDEX_URL_BY_LABEL = {label: url for label, _v, url in _CUDA_OPTIONS}


def _detect_cuda_version():
    """(major, minor) CUDA version the installed NVIDIA driver supports,
    via `nvidia-smi`'s own header - None if no NVIDIA GPU/driver found.
    Older drivers print "CUDA Version: X.Y"; newer ones (confirmed on
    driver 616.56) print "CUDA UMD Version: X.Y" instead - matches either."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            m = re.search(r'CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)', result.stdout)
            if m:
                return (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass
    return None


def _pick_default_device(detected_version):
    """The newest _CUDA_OPTIONS label the driver can run, or CPU only."""
    if detected_version is not None:
        for label, version, _url in _CUDA_OPTIONS:
            if version is not None and version <= detected_version:
                return label
    return _CUDA_OPTIONS[-1][0]

# A plain HTTP(S) archive URL, not `git+https://...` - pip can install
# straight from this with no `git` executable on PATH at all (confirmed:
# `pip install <this-url> --no-deps` installs the same "SAM-2" package with
# git absent from PATH), unlike the `git+` form, which shells out to a real
# git clone and fails outright on a machine without git installed - a real
# gap, since this app doesn't require/bundle git anywhere else. Tracks the
# same ref git+https://github.com/facebookresearch/sam2.git would (no `@ref`
# on either form - both just follow the repo's default branch).
SAM2_GIT_URL = 'https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip'

# Frozen builds install into this dedicated subfolder of _internal, not
# _internal itself - torch's own dependencies (numpy, markupsafe, ...)
# overlap with ones PyInstaller already bundled AND the running app already
# has loaded (e.g. markupsafe via jinja2, pulled in by hyperspy/dask) -
# pip --upgrade can't delete a loaded .pyd (WinError 5), so this avoids
# ever touching those files at all. worker_sam.py puts this folder on its
# own sys.path before importing torch.
_TARGET_SUBDIR = 'sam2_packages'


def _torch_sam2_available(target_dir=None):
    """Whether both packages are usable - checked as plain directories
    under target_dir, or its parent (frozen builds, see _TARGET_SUBDIR)
    rather than importing, so this never loads a native extension into the
    long-running main GUI process; via normal import machinery otherwise
    (dev mode). The parent check is for an *offline* build, which bundles
    torch/sam2 directly into _internal (target_dir's own parent) at
    PyInstaller build time - see worker_sam.py's identical online-vs-
    offline distinction for why target_dir itself (sam2_packages) is
    checked first: an online install's pip-installed copy there should
    always be treated as the answer over an offline build's bundled one, if
    a Reinstall ever created both."""
    if target_dir:
        def _has_both(d):
            return os.path.isdir(os.path.join(d, 'torch')) and os.path.isdir(os.path.join(d, 'sam2'))
        return _has_both(target_dir) or _has_both(os.path.dirname(target_dir))
    importlib.invalidate_caches()
    return (importlib.util.find_spec('torch') is not None
            and importlib.util.find_spec('sam2') is not None)


_find_system_python = find_system_python


class SAM2SetupDialog(qtw.QDialog):
    """Self-contained "install torch/sam2" helper, launched from Help > Set
    Up SAM2... (see EDyssey_MainWindow.py)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Set Up SAM2')
        self.resize(640, 480)
        self._process = None
        self._steps = []
        self._target_dir = None
        self._python_prefix = None
        self._build_ui()
        self._refresh_status()

    def _build_ui(self):
        layout = qtw.QVBoxLayout(self)

        self.label_status = qtw.QLabel()
        self.label_status.setWordWrap(True)
        self.label_status.setTextFormat(Qt.RichText)
        self.label_status.setOpenExternalLinks(True)
        layout.addWidget(self.label_status)

        form = qtw.QFormLayout()
        self.combo_device = qtw.QComboBox()
        self.combo_device.addItems([label for label, _v, _u in _CUDA_OPTIONS])
        self.combo_device.setCurrentText(_pick_default_device(_detect_cuda_version()))
        form.addRow('GPU:', self.combo_device)
        layout.addLayout(form)

        note = qtw.QLabel(
            'GPU auto-detected from your NVIDIA driver above (via nvidia-smi) - change it '
            'if that looks wrong, or check '
            '<a href="https://pytorch.org/get-started/locally/" style="color:#6db3ff;">'
            'pytorch.org/get-started/locally</a> yourself. "CPU only" if you don\'t have '
            'an NVIDIA GPU.')
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        layout.addWidget(note)

        self.log = qtw.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        layout.addWidget(self.log, 1)

        button_row = qtw.QHBoxLayout()
        self.button_check_cuda = qtw.QPushButton('Check CUDA')
        self.button_check_cuda.setToolTip(
            "Actually import torch and print torch.cuda.is_available() - the auto-detected "
            "GPU option above only reflects what nvidia-smi reports, not whether the "
            "installed torch build can actually see the GPU.")
        self.button_check_cuda.clicked.connect(self._check_cuda)
        button_row.addWidget(self.button_check_cuda)
        button_row.addStretch(1)
        self.button_install = qtw.QPushButton('Install')
        self.button_install.clicked.connect(self._start_install)
        button_row.addWidget(self.button_install)
        self.button_close = qtw.QPushButton('Close')
        self.button_close.clicked.connect(self.close)
        button_row.addWidget(self.button_close)
        layout.addLayout(button_row)

    def _refresh_status(self):
        """Update the status label and the availability of the Install
        button/GPU picker for the current torch/sam2 + system-Python state."""
        frozen = getattr(sys, 'frozen', False)
        if frozen:
            install_dir = os.path.dirname(sys.executable)
            self._target_dir = os.path.join(install_dir, '_internal', _TARGET_SUBDIR)
            where = f'into this install ({install_dir})'
        else:
            self._target_dir = None
            where = f'into the current Python environment ({sys.executable})'

        already_installed = _torch_sam2_available(self._target_dir)
        self.button_install.setText('Reinstall' if already_installed else 'Install')

        if already_installed:
            status = (
                'torch and sam2 are already installed - the SAM2 tab is ready to use. '
                'Use Check CUDA below to verify GPU support, or pick a different GPU '
                'option and Reinstall if you need to change it (e.g. it picked CPU by '
                'mistake, or you want a different CUDA version).')
        else:
            status = (
                'SAM2 needs torch and the sam2 package, which EDyssey does not '
                f'bundle by default (see INSTALL.md) - not installed yet. This '
                f'installs both {where}. Pick your GPU below, then Install - this '
                'downloads several GB and can take a while.')

        can_install = True

        # Only shown when the install actually looks unwritable (e.g. the
        # default C:\Program Files\EDyssey location) - installed to a
        # per-user location (%LocalAppData%\Programs\...) instead, this
        # never fires, so the note doesn't show up for people it doesn't
        # apply to.
        if frozen and not already_installed and not os.access(
                os.path.join(install_dir, '_internal'), os.W_OK):
            status += (
                '<br><br><b>This install location needs administrator rights to write '
                'to.</b> Close this, then re-open EDyssey as administrator (right-click '
                'its shortcut > Run as administrator) before installing.')
            can_install = False

        # A bundled portable Python (offline installer only, see
        # find_bundled_python()) makes SAM2 itself runnable with no system
        # Python at all - only pip operations (Install/Reinstall, to
        # change torch/CUDA version) still need one, so this warning no
        # longer implies SAM2 itself won't work when one's bundled.
        has_bundled_python = frozen and self._target_dir and find_bundled_python(
            os.path.dirname(self._target_dir)) is not None
        if frozen and _find_system_python() is None:
            if has_bundled_python:
                status += (
                    '<br><br>No separate system Python found on this machine - not '
                    'needed to run SAM2 (this offline install already bundles one), '
                    'but Install/Reinstall above (e.g. to switch CUDA versions) needs '
                    'one. Install a recent version from '
                    '<a href="https://www.python.org/downloads/" style="color:#6db3ff;">'
                    'python.org</a> first if you need that.')
            else:
                status += (
                    '<br><br><b>No Python installation found on this machine</b> - one is '
                    "needed as a separate tool to install and run SAM2 (EDyssey itself "
                    "doesn't ship one - see INSTALL.md). Any reasonably recent version "
                    'works, install one from '
                    '<a href="https://www.python.org/downloads/" style="color:#6db3ff;">'
                    'python.org</a> first, then reopen this dialog.')
            # Disables Install/Reinstall specifically (pip needs a real
            # system Python either way) - not a statement that SAM2 itself
            # won't work, see has_bundled_python above.
            can_install = False

        self.button_install.setEnabled(can_install)
        self.combo_device.setEnabled(can_install)

        self.label_status.setText(status)

    def _python_prefix_for_run(self):
        """[python, *args] to run pip in (Install/Reinstall only - Check
        CUDA uses _python_prefix_for_check() instead, see its own docstring
        for why) - None if frozen and no usable system Python is available.
        Reuses whatever interpreter a prior install actually used (via the
        marker file in target_dir) if one is recorded, instead of
        re-resolving fresh each time - keeps Reinstall consistent with the
        original install even if find_system_python()'s own default pick
        would differ on a later call (e.g. a new Python got installed/
        uninstalled meanwhile). Never prefers a bundled portable Python
        (see find_bundled_python()) the way _python_prefix_for_check() does
        - the embeddable package it's built from has no pip of its own."""
        if getattr(sys, 'frozen', False):
            python_prefix = None
            if self._target_dir:
                python_prefix = read_interpreter_marker(self._target_dir)
            if python_prefix is None:
                python_prefix = _find_system_python()
            if python_prefix is None:
                return None
            self._python_prefix = python_prefix
            return list(python_prefix)
        return [sys.executable]

    def _python_prefix_for_check(self):
        """[python, *args] for Check CUDA - mirrors worker_launch.py's own
        _sam_command_frozen() resolution order exactly (bundled portable
        Python first, if this is an *offline* build - see
        find_bundled_python() - then the marker/system-Python fallback
        _python_prefix_for_run() also uses), so Check CUDA verifies the
        SAME interpreter SAM2 itself would actually run under, rather than
        one that only makes sense for pip (which a bundled portable Python
        doesn't have - see _python_prefix_for_run())."""
        if getattr(sys, 'frozen', False) and self._target_dir:
            python_prefix = find_bundled_python(os.path.dirname(self._target_dir))
            if python_prefix is not None:
                self._python_prefix = python_prefix
                return list(python_prefix)
        return self._python_prefix_for_run()

    def _set_process_working_dir(self, process, working_dir=None):
        """Point `process` at working_dir (target_dir/sam2_packages by
        default) instead of leaving QProcess's default of inheriting this
        app's own working directory - which EDyssey_MainWindow.py's own
        `os.chdir(fld_path)` sets to sys._MEIPASS (`_internal`) in every
        frozen build, for the entire lifetime of the app. `_internal` is
        packed with this app's own bundled dependencies (numpy, etc.),
        built against whatever Python built the app itself - a system
        Python/pip subprocess implicitly launched with that as its CWD can
        end up resolving an import (or a native .pyd's own DLL search)
        against `_internal`'s copy instead of the one actually pip-
        installed for THIS interpreter, producing the exact same "Module
        use of pythonXYZ.dll conflicts" crash worker_sam.py's own sys.path
        bug did - see worker_sam.py's own comment on the analogous fix
        there. Check CUDA passes _torch_import_dir() explicitly instead of
        relying on this default, since that may correctly resolve to
        _internal itself for an *offline* build (bundled torch, no
        sam2_packages) - _internal only needs avoiding when it'd be a
        foreign shadow, not when it's genuinely where torch lives."""
        working_dir = working_dir or self._target_dir
        if working_dir:
            # Must actually exist first, or QProcess.start() fails outright
            # (CreateProcess itself rejects a nonexistent working
            # directory) - target_dir may not exist yet this early (e.g.
            # _start_install() just wiped it, see its own docstring; pip
            # would normally create it itself, but only after its own
            # process has already started).
            os.makedirs(working_dir, exist_ok=True)
            process.setWorkingDirectory(working_dir)

    def _torch_import_dir(self):
        """Directory to sys.path.insert(0, ...) so `import torch` in a
        Check CUDA subprocess finds it - target_dir (sam2_packages) if an
        online install actually put torch there, else its parent
        (_internal) if an *offline* build bundled torch directly instead -
        see worker_sam.py's identical online-vs-offline resolution and
        _torch_sam2_available's matching "check both locations" logic.
        Falls back to target_dir itself (even though it doesn't have
        torch) if neither does, so a genuinely not-yet-installed state
        still surfaces a plain, sensible ImportError instead of an empty
        sys.path.insert."""
        if os.path.isdir(os.path.join(self._target_dir, 'torch')):
            return self._target_dir
        parent = os.path.dirname(self._target_dir)
        if os.path.isdir(os.path.join(parent, 'torch')):
            return parent
        return self._target_dir

    def _pip_base_cmd(self):
        """[python, *args, -m, pip, install, (--target <dir>)]. None if
        frozen and no matching-version system Python is available."""
        cmd = self._python_prefix_for_run()
        if cmd is None:
            return None
        cmd += ['-m', 'pip', 'install']
        if self._target_dir:
            # --upgrade: otherwise pip silently skips any dependency
            # PyInstaller already bundled a copy of (numpy, setuptools,
            # ...), leaving torch running against a stale/mismatched one.
            cmd += ['--target', self._target_dir, '--upgrade']
        return cmd

    def _check_cuda(self):
        """Actually import torch (in the same interpreter/target dir the
        Install button uses) and report torch.cuda.is_available() - the
        auto-detected GPU picker only reflects nvidia-smi's own report, not
        whether the installed torch build can actually see the GPU."""
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return  # an install is already running - don't clobber it
        cmd = self._python_prefix_for_check()
        if cmd is None:
            self._refresh_status()
            return
        script = (
            f"import sys\nsys.path.insert(0, {self._torch_import_dir()!r})\n"
            if self._target_dir else "import sys\n"
        )
        script += (
            "try:\n"
            "    import torch\n"
            "except ImportError as exc:\n"
            "    print('torch is not importable:', exc)\n"
            "else:\n"
            "    print('torch version:', torch.__version__)\n"
            "    print('CUDA build:', torch.version.cuda)\n"
            "    available = torch.cuda.is_available()\n"
            "    print('torch.cuda.is_available():', available)\n"
            "    if available:\n"
            "        print('GPU:', torch.cuda.get_device_name(0))\n"
        )
        cmd += ['-c', script]
        self._append_log('\n$ Check CUDA\n')
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._set_process_working_dir(
            self._process, self._torch_import_dir() if self._target_dir else None)
        self._process.start(cmd[0], cmd[1:])

    def _start_install(self):
        # Wipe target_dir first (Install and Reinstall both land here) rather
        # than relying on `pip install --upgrade --target` alone: pip's own
        # "is this already satisfied" check just reads the recorded
        # dist-info metadata, oblivious to which interpreter wrote it - so a
        # Reinstall picking a *different* system Python than a previous
        # attempt (installed torch as e.g. cp311, this run resolves to
        # cp312) can leave the old interpreter's compiled .pyd files sitting
        # there unreplaced if pip decides the recorded version already
        # satisfies the requirement, producing exactly the "Module use of
        # pythonXYZ.dll conflicts" mismatch this dialog exists to prevent.
        # Starting from an empty directory every time makes every install
        # fully self-consistent regardless of what an earlier attempt left
        # behind.
        if self._target_dir and os.path.isdir(self._target_dir):
            shutil.rmtree(self._target_dir, ignore_errors=True)

        base_cmd = self._pip_base_cmd()
        if base_cmd is None:
            self._refresh_status()
            return
        index_url = _INDEX_URL_BY_LABEL[self.combo_device.currentText()]
        torch_cmd = base_cmd + ['torch', 'torchvision']
        if index_url:
            torch_cmd += ['--index-url', index_url]
        # --no-deps: sam2's own setup.py (REQUIRED_PACKAGES) lists a loose,
        # unpinned torch/torchvision requirement. Without --no-deps, this
        # step (which - unlike torch_cmd above - has no reason to pass a
        # CUDA --index-url, since sam2 itself isn't CUDA-specific) would
        # resolve that requirement against the *default* PyPI index (the
        # only place CUDA builds don't exist), and --upgrade (already on
        # base_cmd) would silently swap the correctly GPU-installed torch
        # above back to a CPU-only build to "satisfy" it. sam2's actual
        # non-torch runtime deps (its setup.py's REQUIRED_PACKAGES, minus
        # torch/torchvision/numpy already covered above) are installed
        # explicitly right after instead - omegaconf/antlr4-python3-runtime/
        # PyYAML/portalocker/colorama etc. still come along transitively via
        # these, just not torch itself.
        sam2_cmd = base_cmd + [SAM2_GIT_URL, '--no-deps']
        deps_cmd = base_cmd + ['hydra-core', 'iopath', 'pillow', 'tqdm']
        self._steps = [torch_cmd, sam2_cmd, deps_cmd]

        self.button_install.setEnabled(False)
        self.combo_device.setEnabled(False)
        self.button_close.setEnabled(False)
        self.log.clear()
        self._run_next_step()

    def _run_next_step(self):
        """Pop and run the next queued command, or - once the queue is
        empty - re-enable the dialog and report the final outcome."""
        if not self._steps:
            self.button_close.setEnabled(True)
            self._refresh_status()
            if _torch_sam2_available(self._target_dir):
                if self._target_dir and self._python_prefix:
                    write_interpreter_marker(self._target_dir, self._python_prefix)
                self._append_log('\nDone - torch and sam2 are installed. '
                                  'You can close this dialog and use the SAM2 tab.')
            else:
                self._append_log('\nInstall finished, but torch/sam2 are still '
                                  'not importable - see the log above for errors.')
            return
        cmd = self._steps.pop(0)
        self._append_log(f'$ {" ".join(cmd)}\n')
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._on_step_finished)
        self._set_process_working_dir(self._process)
        self._process.start(cmd[0], cmd[1:])

    def _on_output(self):
        text = bytes(self._process.readAllStandardOutput()).decode('utf-8', errors='replace')
        self._append_log(text, newline=False)

    def _append_log(self, text, newline=True):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text + ('\n' if newline else ''))
        self.log.moveCursor(QTextCursor.End)

    def _on_step_finished(self, exit_code, exit_status):
        if exit_code != 0 or exit_status == QProcess.CrashExit:
            self._append_log(f'\nCommand failed (exit code {exit_code}) - stopping.')
            self._steps = []
        self._run_next_step()

    def closeEvent(self, event):
        """Confirm before closing mid-install - killing the QProcess here
        would leave a half-installed torch/sam2 behind."""
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            reply = qtw.QMessageBox.question(
                self, 'Install in Progress',
                'An install is still running - close anyway? '
                'This may leave torch/sam2 partially installed.',
                qtw.QMessageBox.Yes | qtw.QMessageBox.No, qtw.QMessageBox.No)
            if reply != qtw.QMessageBox.Yes:
                event.ignore()
                return
            self._process.kill()
        super().closeEvent(event)
