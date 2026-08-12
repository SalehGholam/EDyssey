# -*- coding: utf-8 -*-
"""Progress/console-output plumbing shared by the rest of EDyssey.io_utils:
routing dask's own progress bar, and raw OS-level stdout/stderr writes from
compiled extensions (eventem), through a Python logger instead of the
terminal - so a Qt log console can show them live.
"""
import os
import sys
import threading
import contextlib
import uuid
import datetime
from dask.diagnostics import ProgressBar

#%% elapsed-time formatting

def format_duration_hms(seconds):
    """Format an elapsed/duration time (in seconds) as H:MM:SS - the
    standard shape every "completed/failed after ..." log message in this
    app uses, instead of each call site picking its own '%.1f s'/'%.1f min'.
    """
    return str(datetime.timedelta(seconds=round(seconds)))

#%% dask progress reporting
class _NullWriter:
    """Discards ProgressBar's own raw stdout writes - LoggingProgressBar
    reports through a logger instead (see below), so the base class's
    carriage-return text would otherwise still land on stdout."""
    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass


class LoggingProgressBar(ProgressBar):
    """dask ProgressBar that reports through a Python logger instead of
    writing raw carriage-return text to stdout - meaningless once mixed
    into a Qt log widget/rotating log file. Only logs when the integer
    percentage actually changes, so this doesn't spam a line every `dt`.
    A no-op (does nothing) when `logger` is None, so call sites can always
    wrap a `.compute()` in this without checking first."""
    def __init__(self, logger=None, label='', minimum=1.0, dt=0.5):
        super().__init__(minimum=minimum, dt=dt, out=_NullWriter())
        self.logger = logger
        self.label = label
        self._last_percent = -1

    def _draw_bar(self, frac, elapsed):
        """dask ProgressBar hook invoked periodically with the current
        fraction/elapsed time; logs only when the rounded percentage changes."""
        if self.logger is None:
            return
        percent = int(100 * frac)
        if percent == self._last_percent:
            return
        self._last_percent = percent
        from dask.utils import format_time
        prefix = f'{self.label}: ' if self.label else ''
        self.logger.info('%s%d%% complete (%s elapsed)', prefix, percent, format_time(elapsed))

def _log_or_print(logger, msg):
    """Report a status message through `logger` if given, otherwise print it
    - lets shared helpers (e.g. the create_clip_* functions) report through
    a tab's Qt log console when called from the GUI, while still working
    standalone (no logger) from a plain script/console session."""
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

#%% console progress -> Qt logger bridge
# RLock (not Lock): redirect_console_to_logger nests two _capture_fd calls
# (fd 1 and fd 2) from the same thread - a plain Lock would self-deadlock.
_fd_redirect_lock = threading.RLock()

@contextlib.contextmanager
def _capture_fd(fd, on_line):
    """Tee OS file descriptor `fd` (1=stdout, 2=stderr) through a pipe for
    the duration of the block: everything written to it still reaches the
    real fd exactly as before (so a console/terminal keeps showing it live,
    same as before this existed), while also calling `on_line(text)` for
    each printed line - or each carriage-return-updated segment of an
    in-place progress bar - so it can additionally be forwarded elsewhere
    (e.g. a Qt log console). Needed for progress from compiled extensions
    (e.g. eventem's tpx3 loading/ROI-extraction progress) that write
    straight to the OS-level fd and are invisible to `sys.stdout`/
    `sys.stderr` overrides.

    A given fd is process-wide, not per-thread, so a lock serializes
    overlapping uses of *any* fd captured this way - two concurrent
    redirects would otherwise corrupt each other's captured output (or, if
    they target different fds, still race on the shared dup()'d-fd
    bookkeeping below).
    """
    with _fd_redirect_lock:
        sys.stdout.flush()
        sys.stderr.flush()
        original_fd = os.dup(fd)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)

        last_line = [None]

        def _pump():
            """Read chunks from `read_fd`, tee them to `original_fd`, and call
            `on_line` once per completed \\n/\\r-terminated line (deduping
            consecutive repeats)."""
            buf = b''
            try:
                while True:
                    chunk = os.read(read_fd, 65536)
                    if not chunk:
                        break
                    # Tee: the real console/terminal still sees this exactly
                    # as if it hadn't been intercepted at all.
                    os.write(original_fd, chunk)
                    buf += chunk
                    while True:
                        idx_n = buf.find(b'\n')
                        idx_r = buf.find(b'\r')
                        candidates = [i for i in (idx_n, idx_r) if i != -1]
                        if not candidates:
                            break
                        idx = min(candidates)
                        line = buf[:idx].decode('utf-8', errors='replace').strip()
                        buf = buf[idx + 1:]
                        if line and line != last_line[0]:
                            last_line[0] = line
                            on_line(line)
                tail = buf.decode('utf-8', errors='replace').strip()
                if tail and tail != last_line[0]:
                    on_line(tail)
            finally:
                os.close(read_fd)

        reader_thread = threading.Thread(target=_pump, daemon=True)
        reader_thread.start()
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(original_fd, fd)
            # Unbounded: the pipe's write end is deterministically closed by
            # the dup2 above (the only other reference, write_fd, was closed
            # right after the initial dup2), so EOF - and thus _pump's exit
            # - is guaranteed, not a hang risk. A bounded join here used to
            # leave the reader thread running concurrently with whatever the
            # caller did right after this context manager exited; do not
            # reintroduce a timeout without re-checking every call site for
            # that race.
            reader_thread.join()
            # Only close original_fd once the reader thread is fully done -
            # it tees every chunk through to original_fd (see _pump above),
            # so closing it earlier risks a write racing the close, or (once
            # the OS recycles the fd number) landing in a wrong,
            # unrelated file entirely.
            os.close(original_fd)


@contextlib.contextmanager
def redirect_console_to_logger(logger, label=''):
    """Capture both real stdout and stderr (OS fds 1 and 2) and forward
    everything written to either, line by line, to a Python logger instead
    of the terminal - eventem's native progress output has been observed on
    both, depending on version/platform, so both are captured to be safe. A
    no-op when `logger` is None, so call sites can always wrap a blocking
    call in this without checking first.

    All lines from one call (i.e. one load/compute operation) share a single
    `progress_key`, so consecutive progress updates collapse onto one
    updating line in the Qt log console (see LogConsole._append_log) instead
    of accumulating as one appended line per update.
    """
    if logger is None:
        yield
        return

    progress_key = f'{label or "console"}-{uuid.uuid4().hex[:8]}'

    def _on_line(line):
        logger.info('%s%s', f'{label}: ' if label else '', line,
                    extra={'progress_key': progress_key})

    with _capture_fd(1, _on_line), _capture_fd(2, _on_line):
        yield
