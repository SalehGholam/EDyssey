# -*- coding: utf-8 -*-
"""
Shared logging setup for EDyssey.

Each tab gets its own logger writing to its own rotating file under
logs/<tab_name>.log (so a session's history stays readable per-tab
instead of interleaved), and every logger also feeds a single Qt signal
that the main window uses to show live output in its console box.
"""

import os
import sys
import html
import logging
import traceback
from logging.handlers import RotatingFileHandler
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QTextCursor
import PyQt5.QtWidgets as qtw
from EDyssey.io_utils.app_dirs import writable_data_dir

LOG_DIR = os.path.join(writable_data_dir(), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

_FORMATTER = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER_PREFIX = 'EDyssey.'


class QtLogHandler(logging.Handler, QObject):
    """Re-emits every log record as a Qt signal (tab_name, formatted
    message, level, progress_key) so a GUI console box can display it live.

    `progress_key` is empty for ordinary log calls (always appended as a new
    line); a caller can pass `extra={'progress_key': some_stable_id}` to
    `logger.info(...)` to mark a record as a progress update that should
    replace its own previous line instead of appending a new one - see
    LogConsole._append_log.

    Instantiation is deferred to get_qt_log_handler() rather than done at
    import time, because a QObject with signals needs a QApplication to
    already exist for cross-thread (QueuedConnection) delivery to work,
    and tab/module imports happen before QApplication is constructed.
    """
    log_emitted = pyqtSignal(str, str, int, str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(_FORMATTER)

    def emit(self, record):
        """logging.Handler override: format `record` and re-emit it via
        `log_emitted`, stripping the `_LOGGER_PREFIX` from the logger name."""
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        name = record.name
        tab_name = name[len(_LOGGER_PREFIX):] if name.startswith(_LOGGER_PREFIX) else name
        progress_key = getattr(record, 'progress_key', '')
        self.log_emitted.emit(tab_name, msg, record.levelno, progress_key)


_qt_log_handler = None


def get_qt_log_handler():
    """Return the process-wide QtLogHandler singleton, creating it on first call."""
    global _qt_log_handler
    if _qt_log_handler is None:
        _qt_log_handler = QtLogHandler()
    return _qt_log_handler


def shutdown_qt_log_handler():
    """Detach the QtLogHandler from every logger, and from logging's own
    internal atexit shutdown registry, while its underlying Qt/C++ object
    is still alive - call this once, from MainWindow.closeEvent, before the
    QApplication itself starts tearing down.

    Without this: `logging` registers its own `logging.shutdown()` via
    atexit at *import* time (very early in the process). atexit callbacks
    run in reverse (LIFO) order, so anything Qt/sip registers for its own
    C++ object teardown - which happens later, once the app is actually
    closing - runs *before* `logging.shutdown()` does. By the time
    `logging.shutdown()` finally runs, this handler's underlying C++
    QObject has often already been deleted (even though the Python object
    itself is still alive) - any attribute access on it then raises
    "RuntimeError: wrapped C/C++ object of type QtLogHandler has been
    deleted" from inside `logging.shutdown()` itself, printed as an ugly
    (harmless, but alarming) "Exception ignored in atexit callback" message
    right as the app closes.

    `logger.removeHandler()`/`handler.close()` alone don't prevent this -
    `logging.shutdown()` walks its own separate registry
    (`logging._handlerList`, a weakref list every Handler subclass instance
    is added to on construction) independently of which loggers reference
    it, so that registry needs to be pruned directly too.
    """
    global _qt_log_handler
    if _qt_log_handler is None:
        return
    handler = _qt_log_handler
    for name in list(logging.Logger.manager.loggerDict):
        logging.getLogger(name).removeHandler(handler)
    handler.close()
    logging._handlerList[:] = [wr for wr in logging._handlerList if wr() is not handler]
    _qt_log_handler = None


def get_tab_logger(tab_name):
    """Return the logger for `tab_name`, creating it (rotating file
    handler + shared GUI handler) on first call. Safe to call repeatedly
    for the same tab_name (e.g. a tab re-instantiated during the session)."""
    logger = logging.getLogger(_LOGGER_PREFIX + tab_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, f'{tab_name}.log'),
        maxBytes=2_000_000, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)
    logger.addHandler(get_qt_log_handler())

    return logger


class LogConsole(qtw.QPlainTextEdit):
    """Read-only console showing the whole app's log output (every tab's
    messages, not just whichever tab embeds this widget). Each tab embeds
    its own instance below its own plot area (so the log strip only ever
    eats space from that tab's plot column, never from its left parameter
    panel) - all instances are driven by the same shared Qt log signal, so
    background activity in another tab still stays visible regardless of
    which tab is currently in front.
    """
    _LEVEL_COLORS = {
        logging.ERROR: '#ff6b6b',
        logging.WARNING: '#e0c341',
    }

    def __init__(self, parent=None, height=140):
        """Build the console widget (dark theme, 2000-line scrollback) and
        start listening for log signals."""
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        self.setFixedHeight(height)
        self.setStyleSheet(
            "background-color: #1e1e1e; color: #d0d0d0;"
            "font-family: Consolas, monospace; font-size: 9pt;"
            "border-top: 1px solid #555;")
        # Tracks the line last written for a given progress_key, so repeated
        # updates (e.g. a tpx3 load's progress ticks) can replace that one
        # line in place instead of appending a new line every time.
        self._progress_cursors = {}
        self._progress_text = {}
        get_qt_log_handler().log_emitted.connect(self._append_log)

    def _append_log(self, tab_name, msg, levelno, progress_key=''):
        """Append `msg` as a new console line, colored by level; if
        `progress_key` matches the line last written for that key, replace
        that line in place instead of appending a new one."""
        if levelno >= logging.ERROR:
            color = self._LEVEL_COLORS[logging.ERROR]
        elif levelno >= logging.WARNING:
            color = self._LEVEL_COLORS[logging.WARNING]
        else:
            color = None
        line = f'[{tab_name}] {html.escape(msg)}'

        if progress_key:
            cursor = self._progress_cursors.get(progress_key)
            # The text check (not just block validity) guards against
            # setMaximumBlockCount() trimming/renumbering blocks from the
            # top of the document - a stale cursor can silently end up
            # pointing at an unrelated line rather than becoming invalid.
            if (cursor is not None and not cursor.isNull()
                    and cursor.block().isValid()
                    and cursor.block().text() == self._progress_text.get(progress_key)):
                cursor.movePosition(QTextCursor.StartOfBlock)
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                if color:
                    cursor.insertHtml(f'<span style="color:{color}">{line}</span>')
                else:
                    cursor.insertText(line)
                self._progress_text[progress_key] = line
                return

        if color:
            self.appendHtml(f'<span style="color:{color}">{line}</span>')
        else:
            self.appendPlainText(line)

        if progress_key:
            new_cursor = QTextCursor(self.document())
            new_cursor.movePosition(QTextCursor.End)
            new_cursor.movePosition(QTextCursor.StartOfBlock)
            self._progress_cursors[progress_key] = new_cursor
            self._progress_text[progress_key] = line

    def disconnect_log(self):
        """Stop receiving log signals - call from the owning tab's
        cleanup() so a closed-but-not-destroyed window (see MainWindow's
        reuse-across-reruns comment) doesn't keep pushing log lines into a
        hidden widget."""
        try:
            get_qt_log_handler().log_emitted.disconnect(self._append_log)
        except TypeError:
            pass  # already disconnected


def install_excepthook():
    """Route uncaught exceptions (including those raised inside Qt slots,
    which PyQt5 routes through sys.excepthook) to their own log file and
    the GUI console box, instead of them only reaching a console window
    that may not exist for a windowed app."""
    app_logger = get_tab_logger('app')

    def _handle(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        text = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        app_logger.error('Uncaught exception:\n%s', text)

    sys.excepthook = _handle
