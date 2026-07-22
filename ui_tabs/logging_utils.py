# -*- coding: utf-8 -*-
"""
Shared logging setup for py5DED.

Each tab gets its own logger writing to its own rotating file under
logs/<tab_name>.log (so a session's history stays readable per-tab
instead of interleaved), and every logger also feeds a single Qt signal
that the main window uses to show live output in its console box.
"""

import os
import sys
import logging
import traceback
from logging.handlers import RotatingFileHandler
from PyQt5.QtCore import QObject, pyqtSignal

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.dirname(_HERE), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

_FORMATTER = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER_PREFIX = 'py5DED.'


class QtLogHandler(logging.Handler, QObject):
    """Re-emits every log record as a Qt signal (tab_name, formatted
    message, level) so a GUI console box can display it live.

    Instantiation is deferred to get_qt_log_handler() rather than done at
    import time, because a QObject with signals needs a QApplication to
    already exist for cross-thread (QueuedConnection) delivery to work,
    and tab/module imports happen before QApplication is constructed.
    """
    log_emitted = pyqtSignal(str, str, int)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(_FORMATTER)

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        name = record.name
        tab_name = name[len(_LOGGER_PREFIX):] if name.startswith(_LOGGER_PREFIX) else name
        self.log_emitted.emit(tab_name, msg, record.levelno)


_qt_log_handler = None


def get_qt_log_handler():
    global _qt_log_handler
    if _qt_log_handler is None:
        _qt_log_handler = QtLogHandler()
    return _qt_log_handler


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
