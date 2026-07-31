# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:11:56 2024

@author: SGholam
"""
import sys
from PyQt5.QtCore import (QThread, pyqtSignal, QMutex, QMutexLocker, QObject,
                          QRunnable, QThreadPool)
from functools import partial
#%% general worker thread
class WorkerThread_General(QRunnable):
    # exec_signal = pyqtSignal(object)
    # exec_signal = pyqtSignal()
    stopped = pyqtSignal()
    
    def __init__(self, func, index=0, *args, **kwargs):
        super(WorkerThread_General, self).__init__() #TODO check
        self.func = func
        self.index = index
        self.args = args
        # print(self.args)
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.is_running = True
    
    def run(self):
        if self.is_running:
            try:
                self.result = self.func(*self.args, **self.kwargs)
            except Exception:
                # Without this, an exception here would silently vanish (Qt
                # swallows it at the QRunnable/thread boundary) and
                # `results` would simply never fire - any caller that
                # disabled a button "until results arrive" would then stay
                # disabled forever, with no way to retry or even see that
                # anything went wrong.
                import traceback
                self.signals.error.emit(traceback.format_exc(), self.index)
                self.signals.finished.emit()
                return
            self.signals.results.emit(self.result, self.index)
        else:
            self.signals.stopped.emit()
        self.signals.finished.emit()

    def stop(self):
        self.is_running = False
#%%
# Step 1: Create a Worker class that inherits from QRunnable
class WorkerSignals(QObject):
    # Define custom signals (for communicating between thread and GUI)
    finished = pyqtSignal()  # Task is done
    stopped = pyqtSignal()
    # results = pyqtSignal(object, int)  # Task returns a result
    results = pyqtSignal(object, object)  # Task returns a result
    error = pyqtSignal(object, object)  # (formatted traceback string, index)

# =============================================================================
# class Worker(QRunnable):
#     def __init__(self):
#         super().__init__()
#         self.signals = WorkerSignals()
#
#     def run(self):
#         pass
# =============================================================================
#%% QProcess stderr handling shared across every tab's worker subprocesses
class ProcessStderrBuffer:
    """Handles a QProcess's stderr the way it actually needs to be handled
    everywhere in this app, instead of each tab re-deriving (and
    occasionally forgetting) the same fix: compiled extensions used by our
    worker subprocesses - eventem for tpx3 loading (worker_extract_frame.py,
    worker_nav_img.py), SAM2/PyTorch (worker_sam.py) - write their progress
    bars and routine diagnostics straight to stderr on every run, success or
    failure. Treating a non-empty stderr chunk as an error (as a couple of
    call sites used to) means a completely successful tpx3 load pops an
    error dialog for every progress-bar update.

    Two usage patterns, matching the two ways call sites actually need this:
      - log_info(): no separate success/failure check exists downstream, so
        just log immediately at info level and move on (SAM2 segmentation).
      - append() + pop_text(): buffer quietly and only retrieve (once, then
        cleared) as context if the process is independently found to have
        failed - e.g. its stdout couldn't be decoded (nav-image/3DED-frame
        extraction, where "no valid output" is the actual failure signal,
        not "wrote something to stderr").
    Either way, the real OS-level stderr stream is still echoed byte-for-
    byte to this process's own stderr, so a console/terminal keeps showing
    it live exactly as if this class weren't involved at all.
    """
    def __init__(self, echo_to_console=True):
        self._buffers = {}
        self._echo = echo_to_console

    def _read(self, process):
        chunk = bytes(process.readAllStandardError())
        if chunk and self._echo:
            try:
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()
            except Exception:
                pass
        return chunk

    def log_info(self, process, logger, label=''):
        """Read, echo, and immediately log any new stderr at info level."""
        chunk = self._read(process)
        text = chunk.decode('utf-8', errors='replace').strip()
        if text:
            logger.info('%s%s', f'{label}: ' if label else '', text)

    def append(self, process):
        """Read, echo, and buffer any new stderr for later retrieval."""
        chunk = self._read(process)
        if chunk:
            self._buffers.setdefault(process, bytearray()).extend(chunk)

    def pop_text(self, process):
        """Retrieve and clear the buffered stderr for `process` as text -
        callers slice it to taste (e.g. `[:500]`) for a popup vs. a log line."""
        raw = bytes(self._buffers.pop(process, b''))
        return raw.decode('utf-8', errors='replace').strip()

    def discard(self, process):
        """Clear the buffer for `process` without retrieving it (call once
        it's known the process succeeded, so stale buffers don't accumulate)."""
        self._buffers.pop(process, None)
