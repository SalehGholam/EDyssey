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
import subprocess
import sys

import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QTextCursor

from ui_tabs.python_finder import APP_PYTHON_VERSION, find_system_python, write_interpreter_marker

_APP_PYTHON_VERSION = APP_PYTHON_VERSION

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

SAM2_GIT_URL = 'git+https://github.com/facebookresearch/sam2.git'

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
    under target_dir (frozen builds, see _TARGET_SUBDIR) rather than
    importing, so this never loads a native extension into the long-running
    main GUI process; via normal import machinery otherwise (dev mode)."""
    if target_dir:
        return (os.path.isdir(os.path.join(target_dir, 'torch'))
                and os.path.isdir(os.path.join(target_dir, 'sam2')))
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

        if _torch_sam2_available(self._target_dir):
            self.label_status.setText(
                'torch and sam2 are already installed - the SAM2 tab is ready to use.')
            self.button_install.setEnabled(False)
            self.combo_device.setEnabled(False)
            return

        status = (
            'SAM2 needs torch and the sam2 package, which EDyssey does not '
            f'bundle by default (see INSTALL.md) - not installed yet. This '
            f'installs both {where}. Pick your GPU below, then Install - this '
            'downloads several GB and can take a while.')

        if frozen and _find_system_python() is None:
            status += (
                f'<br><br><b>No Python {_APP_PYTHON_VERSION} installation found on this '
                'machine</b> - one is needed as a separate tool to run pip (EDyssey itself '
                f'does not ship one), and it must be Python {_APP_PYTHON_VERSION} specifically '
                '(matching this build) - installing via a different version silently produces '
                "files this app's own Python can't load. Install Python "
                f'{_APP_PYTHON_VERSION} from '
                '<a href="https://www.python.org/downloads/" style="color:#6db3ff;">'
                'python.org</a> first, then reopen this dialog.')
            self.button_install.setEnabled(False)
            self.combo_device.setEnabled(False)
        else:
            self.button_install.setEnabled(True)
            self.combo_device.setEnabled(True)

        self.label_status.setText(status)

    def _python_prefix_for_run(self):
        """[python, *args] to run torch/pip in - None if frozen and no
        matching-version system Python is available."""
        if getattr(sys, 'frozen', False):
            python_prefix = _find_system_python()
            if python_prefix is None:
                return None
            self._python_prefix = python_prefix
            return list(python_prefix)
        return [sys.executable]

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
        cmd = self._python_prefix_for_run()
        if cmd is None:
            self._refresh_status()
            return
        script = (
            "import sys\n"
            f"sys.path.insert(0, {self._target_dir!r})\n" if self._target_dir else "import sys\n"
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
        self._process.start(cmd[0], cmd[1:])

    def _start_install(self):
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
