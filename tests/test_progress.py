# -*- coding: utf-8 -*-
"""Tests for EDyssey.io_utils.progress - specifically redirect_console_to_
logger/pipe_process_output_to_logger's throttling (a real, suspected-but-
unconfirmed fix for a reported "second progress bar never updates, looks
hung" symptom - see redirect_console_to_logger's own docstring). Uses a
tiny min_interval so these run fast and deterministically instead of
waiting on the real ~0.3s default.
"""
import os
import subprocess
import sys
import time

from EDyssey.io_utils import progress as prog


class _FakeLogger:
    def __init__(self):
        self.calls = []

    def info(self, fmt, *args, **kwargs):
        self.calls.append(fmt % args if args else fmt)


def _write_line(text):
    """Writes straight to the raw OS fd 1, bypassing sys.stdout/print()
    entirely - what redirect_console_to_logger/_capture_fd actually exists
    to intercept (a compiled extension's own printf/std::cout, which never
    goes through Python's io layer at all). Plain print() would instead go
    through whatever object sys.stdout currently is - under pytest's own
    default capture, that's pytest's own in-memory buffer, not the real fd,
    so it would never reach _capture_fd's dup2'd pipe at all."""
    os.write(1, (text + '\n').encode('utf-8'))


def test_redirect_console_to_logger_throttles_rapid_updates():
    logger = _FakeLogger()
    with prog.redirect_console_to_logger(logger, label='Test', min_interval=1.0):
        for i in range(50):
            _write_line(f'tick {i}')
    # 50 lines written essentially instantaneously, with a 1s throttle -
    # nowhere near 50 log calls should have gone through.
    assert 1 <= len(logger.calls) < 10


def test_redirect_console_to_logger_always_emits_final_line():
    logger = _FakeLogger()
    with prog.redirect_console_to_logger(logger, label='Test', min_interval=1.0):
        for i in range(20):
            _write_line(f'tick {i}')
        _write_line('final result')
    # Whatever got throttled along the way, the very last line must always
    # show up once - otherwise a real result/completion message could be
    # silently dropped.
    assert any('final result' in c for c in logger.calls)


def test_redirect_console_to_logger_forwards_widely_spaced_updates():
    logger = _FakeLogger()
    with prog.redirect_console_to_logger(logger, label='Test', min_interval=0.05):
        for i in range(3):
            _write_line(f'tick {i}')
            time.sleep(0.1)
    # Each write is spaced well beyond min_interval - none should be
    # throttled away (the throttle must not eat genuinely spaced-out
    # updates, only a flood).
    assert sum(1 for c in logger.calls if 'tick' in c) == 3


def test_redirect_console_to_logger_none_logger_is_noop():
    with prog.redirect_console_to_logger(None, label='Test'):
        _write_line('should just go to the real console, not raise')


def test_pipe_process_output_to_logger_throttles_and_returns_full_tail():
    logger = _FakeLogger()
    script = (
        "import sys\n"
        "for i in range(50):\n"
        "    print(f'tick {i}', flush=True)\n"
        "print('DONE', flush=True)\n"
    )
    proc = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    returncode, tail = prog.pipe_process_output_to_logger(proc, logger, label='Test', min_interval=1.0)
    assert returncode == 0
    assert 1 <= len(logger.calls) < 10
    assert any('DONE' in c for c in logger.calls)  # final line always gets through
    assert tail[-1] == 'DONE'  # the unthrottled tail sees every line regardless


def test_pipe_process_output_to_logger_no_logger_prints(capfd):
    script = "print('hello world', flush=True)\n"
    proc = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    returncode, tail = prog.pipe_process_output_to_logger(proc, None, label='Test')
    assert returncode == 0
    assert tail == ['hello world']
    captured = capfd.readouterr()
    assert 'hello world' in captured.out


def test_pipe_process_output_to_logger_reports_nonzero_exit():
    script = "import sys; print('boom', flush=True); sys.exit(1)\n"
    proc = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    returncode, tail = prog.pipe_process_output_to_logger(proc, None, label='Test')
    assert returncode == 1
    assert tail == ['boom']
