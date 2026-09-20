# -*- coding: utf-8 -*-
"""hang_watchdog: the diagnostic added after extensive, targeted attempts
to reproduce a reported ROI-computation hang (declustering toggled
on/off/on again) could not trigger it on a different machine. Wraps
Worker_CalculateDP/Worker_CalculateDP_Mask's actual decode call so that if
it genuinely never returns in time, every thread's real call stack gets
dumped to a file instead of leaving nothing to go on beyond "it never came
back".

These tests use a short timeout (real production use is 25s) so the suite
doesn't have to actually wait that long to exercise both outcomes.
"""
import logging
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt5")

from ui_tabs.logging_utils import LOG_DIR, hang_watchdog  # noqa: E402


class _CollectingLogger:
    def __init__(self):
        self.errors = []

    def error(self, *a, **k):
        self.errors.append(a[0] % a[1:] if a[1:] else a[0])


def test_a_fast_call_leaves_no_dump_file_and_logs_nothing():
    logger = _CollectingLogger()
    before = set(os.listdir(LOG_DIR))
    with hang_watchdog(logger, "test_fast", timeout_s=1.0):
        pass  # returns immediately, well under the timeout
    after = set(os.listdir(LOG_DIR))
    assert after == before, "a call that finished in time must not leave a dump file behind"
    assert logger.errors == []


def test_a_hung_call_produces_a_real_thread_dump_and_logs_its_path():
    logger = _CollectingLogger()
    label = "test_hang_unique_marker"
    before = set(f for f in os.listdir(LOG_DIR) if label in f)

    with hang_watchdog(logger, label, timeout_s=0.3):
        time.sleep(1.0)  # longer than the watchdog's timeout

    after = set(f for f in os.listdir(LOG_DIR) if label in f)
    new_files = after - before
    assert len(new_files) == 1, f"expected exactly one new dump file, got {new_files}"
    dump_path = os.path.join(LOG_DIR, new_files.pop())
    try:
        content = Path(dump_path).read_text(encoding="utf-8")
        # A real faulthandler dump names the thread and shows a frame from
        # this test file - not just an empty or generic placeholder.
        assert "Thread" in content
        assert "test_hang_watchdog.py" in content

        assert len(logger.errors) == 1
        assert dump_path in logger.errors[0] or os.path.basename(dump_path) in logger.errors[0]
    finally:
        os.remove(dump_path)


def test_an_exception_inside_the_block_still_cancels_the_watchdog_cleanly():
    """The watchdog must not swallow or alter an exception raised inside
    the wrapped block, and must still clean up (no dump file, since the
    call didn't hang - it just failed quickly)."""
    logger = _CollectingLogger()
    label = "test_exception_marker"
    before = set(f for f in os.listdir(LOG_DIR) if label in f)

    with pytest.raises(ValueError, match="boom"):
        with hang_watchdog(logger, label, timeout_s=1.0):
            raise ValueError("boom")

    after = set(f for f in os.listdir(LOG_DIR) if label in f)
    assert after == before
    assert logger.errors == []
