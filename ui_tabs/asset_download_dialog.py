# -*- coding: utf-8 -*-
"""Shared "confirm, then show live progress" wrapper around an
asset_fetch.py ensure_*() download - used for the SAM2 checkpoint, CV2
tracker model weights, and ffmpeg, so all three ask before downloading and
show real byte progress instead of a generic spinner or nothing at all.
"""
import threading

import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt

from ui_tabs.worker_thread import WorkerThread_General
from EDyssey.tracking_utils.asset_fetch import AssetDownloadCancelled


def confirm_and_download(parent, threadpool, title, confirm_message, download_fn,
                          on_ready, on_failed, download_kwargs=None):
    """Ask the user to confirm via a Yes/No dialog, then run `download_fn`
    (one of asset_fetch.py's ensure_*() functions) in a background worker
    with a cancellable progress dialog.

    Args:
        parent: Parent widget for the dialogs.
        threadpool: QThreadPool to run the download worker on.
        title: Dialog window title.
        confirm_message: Text for the initial Yes/No confirmation.
        download_fn: Callable accepting progress_callback/cancel_event
            kwargs and returning the downloaded asset's path.
        on_ready: `on_ready(result)` - called with download_fn's return
            value once it succeeds.
        on_failed: `on_failed(error_msg)` - called if the user declines,
            cancels, or the download genuinely fails. error_msg is empty
            for a decline/cancel (not a real error to report).
        download_kwargs: Extra kwargs to pass to `download_fn` (e.g.
            tracking_method for ensure_tracker_models).
    """
    reply = qtw.QMessageBox.question(
        parent, title, confirm_message,
        qtw.QMessageBox.Yes | qtw.QMessageBox.No, qtw.QMessageBox.No)
    if reply != qtw.QMessageBox.Yes:
        on_failed('')
        return

    progress_dlg = qtw.QProgressDialog('Starting download...', 'Cancel', 0, 0, parent)
    progress_dlg.setWindowTitle(title)
    progress_dlg.setWindowModality(Qt.WindowModal)
    progress_dlg.setMinimumDuration(0)
    progress_dlg.setValue(0)

    cancel_event = threading.Event()
    kwargs = dict(download_kwargs or {})
    kwargs['cancel_event'] = cancel_event

    worker = WorkerThread_General(download_fn, 0, **kwargs)

    # Runs on the worker thread - only ever touches Qt objects via the
    # (thread-safe, queued) signal below, never progress_dlg directly.
    def _progress_cb(bytes_done, total_bytes):
        worker.signals.progress.emit(bytes_done, total_bytes)
    worker.kwargs['progress_callback'] = _progress_cb

    def _on_progress(bytes_done, total_bytes):
        if total_bytes:
            progress_dlg.setMaximum(total_bytes)
            progress_dlg.setValue(bytes_done)
            mb_done, mb_total = bytes_done / 1e6, total_bytes / 1e6
            progress_dlg.setLabelText(f'Downloading... {mb_done:.0f} / {mb_total:.0f} MB')
        else:
            progress_dlg.setMaximum(0)  # indeterminate - server sent no Content-Length

    def _ready(result, _idx):
        progress_dlg.close()
        on_ready(result)

    def _failed(error_msg, _idx):
        progress_dlg.close()
        # AssetDownloadCancelled's own traceback text still contains its
        # class name - a plain substring check is enough to tell it apart
        # from a genuine failure without re-running the download to catch
        # the exception object itself (it crossed a thread boundary as text).
        on_failed('' if 'AssetDownloadCancelled' in error_msg else error_msg)

    worker.signals.progress.connect(_on_progress)
    worker.signals.results.connect(_ready)
    worker.signals.error.connect(_failed)
    progress_dlg.canceled.connect(cancel_event.set)

    threadpool.start(worker)
    progress_dlg.show()
