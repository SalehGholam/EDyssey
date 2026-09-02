# -*- coding: utf-8 -*-
"""Help > Set Up SAM2... - automates the "Enabling SAM2" steps INSTALL.md
documents for running from source or the online installer (the offline
installer already bundles torch/sam2, and doesn't need this).

Runs `pip install [--target <install_dir>\\_internal] torch [--index-url
...]` and the same for the `sam2` package, streaming their output into a
log box instead of making the user copy/paste commands from a message box
themselves (see tab_sam2.py's _show_missing_dependency_dialog, which still
shows the manual command as a fallback if this dialog can't run - e.g. no
system Python found).
"""
import importlib
import importlib.util
import os
import shutil
import sys

import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QTextCursor

# Matches INSTALL.md's "Enabling SAM2" section - kept to a handful of
# common builds rather than scraping pytorch.org, so this needs no network
# access just to populate the dropdown. "CPU only" omits --index-url
# entirely, which gets pip's default (CPU) wheels from PyPI.
_CUDA_INDEX_URLS = {
    'NVIDIA GPU (CUDA 12.6)': 'https://download.pytorch.org/whl/cu126',
    'NVIDIA GPU (CUDA 12.4)': 'https://download.pytorch.org/whl/cu124',
    'NVIDIA GPU (CUDA 12.1)': 'https://download.pytorch.org/whl/cu121',
    'CPU only (no GPU)': None,
}

SAM2_GIT_URL = 'git+https://github.com/facebookresearch/sam2.git'


def _torch_sam2_available():
    """Whether both packages are importable right now - re-checked (not
    cached) since this is called again right after an install completes,
    from the same process, and importlib.invalidate_caches() is needed for
    that to see files a just-finished pip subprocess wrote."""
    importlib.invalidate_caches()
    return (importlib.util.find_spec('torch') is not None
            and importlib.util.find_spec('sam2') is not None)


def _find_system_python():
    """A Python interpreter usable to run pip, other than this app's own -
    a frozen build's sys.executable is EDyssey.exe, which can't run
    `-m pip` (see EDyssey.spec's torch_excludes comment for why torch/pip
    aren't bundled). Prefers the Windows 'py' launcher (most reliable at
    finding a real install even when 'python' isn't directly on PATH),
    falls back to 'python'/'python3'."""
    for candidate in ('py', 'python', 'python3'):
        found = shutil.which(candidate)
        if found:
            return found
    return None


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
        self.combo_device.addItems(list(_CUDA_INDEX_URLS.keys()))
        form.addRow('GPU:', self.combo_device)
        layout.addLayout(form)

        note = qtw.QLabel(
            'Not sure which CUDA version to pick? Check '
            '<a href="https://pytorch.org/get-started/locally/" style="color:#6db3ff;">'
            'pytorch.org/get-started/locally</a> against your GPU driver, '
            'or pick "CPU only" if you don\'t have an NVIDIA GPU.')
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        layout.addWidget(note)

        self.log = qtw.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        layout.addWidget(self.log, 1)

        button_row = qtw.QHBoxLayout()
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
        if _torch_sam2_available():
            self.label_status.setText(
                'torch and sam2 are already installed - the SAM2 tab is ready to use.')
            self.button_install.setEnabled(False)
            self.combo_device.setEnabled(False)
            return

        frozen = getattr(sys, 'frozen', False)
        if frozen:
            install_dir = os.path.dirname(sys.executable)
            self._target_dir = os.path.join(install_dir, '_internal')
            where = f'into this install ({install_dir})'
        else:
            self._target_dir = None
            where = f'into the current Python environment ({sys.executable})'

        status = (
            'SAM2 needs torch and the sam2 package, which EDyssey does not '
            f'bundle by default (see INSTALL.md) - not installed yet. This '
            f'installs both {where}. Pick your GPU below, then Install - this '
            'downloads several GB and can take a while.')

        if frozen and _find_system_python() is None:
            status += (
                '<br><br><b>No Python installation found on this machine</b> - '
                'one is needed as a separate tool to run pip (EDyssey itself '
                'does not ship one). Install Python from '
                '<a href="https://www.python.org/downloads/" style="color:#6db3ff;">'
                'python.org</a> first, then reopen this dialog.')
            self.button_install.setEnabled(False)
            self.combo_device.setEnabled(False)
        else:
            self.button_install.setEnabled(True)
            self.combo_device.setEnabled(True)

        self.label_status.setText(status)

    def _pip_base_cmd(self):
        """[python, -m, pip, install, (--target <dir>)] - the fixed prefix
        every install command below is built from."""
        if getattr(sys, 'frozen', False):
            python = _find_system_python()
            cmd = [python]
            if os.path.basename(python).lower() in ('py', 'py.exe'):
                # The 'py' launcher needs telling which Python to use;
                # 'python'/'python3' executables are already that Python.
                cmd.append('-3')
        else:
            cmd = [sys.executable]
        cmd += ['-m', 'pip', 'install']
        if self._target_dir:
            cmd += ['--target', self._target_dir]
        return cmd

    def _start_install(self):
        index_url = _CUDA_INDEX_URLS[self.combo_device.currentText()]
        torch_cmd = self._pip_base_cmd() + ['torch']
        if index_url:
            torch_cmd += ['--index-url', index_url]
        sam2_cmd = self._pip_base_cmd() + [SAM2_GIT_URL]
        self._steps = [torch_cmd, sam2_cmd]

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
            if _torch_sam2_available():
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
