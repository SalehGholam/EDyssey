# -*- coding: utf-8 -*-
"""Progress/console-output plumbing shared by the rest of EDyssey.io_utils:
routing dask's own progress bar, and raw OS-level stdout/stderr writes from
compiled extensions (eventem), through a Python logger instead of the
terminal - so a Qt log console can show them live.
"""
import os
import sys
import threading
import time
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

def _safe_flush(stream):
    """Best-effort flush - `stream` is None in a PyInstaller windowed build
    (console=False) with no console attached: sys.stdout/sys.stderr are
    None there, not just closed or redirected, so a bare .flush() call
    raises AttributeError. This is only making sure already-buffered
    Python output is out before the raw fd gets redirected below, not
    essential to what follows, so any failure here is safe to ignore."""
    try:
        stream.flush()
    except (AttributeError, OSError, ValueError):
        pass


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
        _safe_flush(sys.stdout)
        _safe_flush(sys.stderr)
        try:
            original_fd = os.dup(fd)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, fd)
            os.close(write_fd)
        except OSError:
            # No real OS-level fd to capture from - e.g. a PyInstaller
            # windowed build with no console attached, where fd 1/2 may
            # not be valid/duplicable at all. Degrade to a plain no-op
            # rather than crashing the operation this wraps: losing live
            # progress-in-log-console for this one call is much better
            # than losing the whole load/compute.
            yield
            return

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
            _safe_flush(sys.stdout)
            _safe_flush(sys.stderr)
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
def redirect_console_to_logger(logger, label='', min_interval=0.3):
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

    `min_interval`: forwarding is throttled to at most one logger.info() per
    `min_interval` seconds (plus one guaranteed final call once the wrapped
    block exits - see below), not one per line. A live progress bar (tqdm's
    default ~0.1s tick, or eventem's own) updates far more often than a
    cross-thread, GUI-queued Qt signal should be emitted - LogConsole's own
    _append_log does real work per call (HTML formatting, a QTextCursor
    block replace) on the GUI thread, and _pump here runs on a background
    thread with no backpressure of its own, so an unthrottled multi-minute
    run can queue far more of these than the GUI thread drains in the same
    time. Suspected (not confirmed with a debugger, but consistent with
    everything observed) explanation for a real report: after a first
    declustered run, a *second* run's own progress bar stayed stuck on its
    very first tick - indistinguishable from a hung computation - while the
    same sequence outside the GUI (no Qt log console attached) never
    reproduced any hang at all. Exposed as a parameter mainly so tests can
    use a tiny value instead of waiting on the real one.
    """
    if logger is None:
        yield
        return

    progress_key = f'{label or "console"}-{uuid.uuid4().hex[:8]}'
    _last_emit_time = [0.0]
    _last_line_seen = [None]
    _last_line_emitted = [None]

    def _emit(line):
        logger.info('%s%s', f'{label}: ' if label else '', line, extra={'progress_key': progress_key})
        _last_line_emitted[0] = line

    def _on_line(line):
        _last_line_seen[0] = line
        now = time.monotonic()
        if now - _last_emit_time[0] < min_interval:
            return
        _last_emit_time[0] = now
        _emit(line)

    with _capture_fd(1, _on_line), _capture_fd(2, _on_line):
        yield

    # A throttled-away final line (the actual "N events processed"/"done"
    # summary, not just an intermediate tick) would otherwise never reach
    # the log console at all - always emit whatever was last seen, once,
    # after the real computation has fully finished.
    if _last_line_seen[0] is not None and _last_line_seen[0] != _last_line_emitted[0]:
        _emit(_last_line_seen[0])


def pipe_process_output_to_logger(proc, logger, label='', tail_lines=20, min_interval=0.3):
    """Block until `proc` (a subprocess.Popen with stdout=PIPE,
    stderr=STDOUT) exits, forwarding its combined output to `logger` the
    same way redirect_console_to_logger does for in-process console writes
    - same \\r/\\n splitting and dedup, same progress_key collapsing
    (see that function's own docstring) - used for a computation that has
    to run in a genuinely separate process (eventem_backend's New-eventem
    subprocess delegation) rather than one whose own stdout/stderr fds this
    process can capture directly.

    A plain `for line in proc.stdout` would not work here: it only splits
    on '\\n', but a live progress bar (eventem's own, or tqdm's) updates via
    bare '\\r' with no '\\n' between ticks - every intermediate update would
    silently vanish, arriving only once as one giant blob at the final
    '\\n'. Reads raw bytes instead and splits on both, exactly like
    _capture_fd's own pump.

    Prints to the real stdout instead of logging when `logger` is None -
    unlike redirect_console_to_logger's own no-op in that case (which
    leaves an in-process write to reach the *already-visible* real
    console/terminal on its own), this process's output would otherwise be
    silently swallowed by the very act of piping it for capture.

    Returns `(returncode, last_lines)` - `last_lines` is up to the final
    `tail_lines` distinct lines seen (most useful on failure: eventem_new's
    own progress ticks dominate the full output, so the caller doesn't have
    to sift through it to find the actual Python traceback at the end).

    Forwarding to `logger` is throttled the same way redirect_console_to_
    logger's own is (see that function's docstring for why) - the `seen`
    tail used for error reporting on a nonzero exit is tracked separately
    and unthrottled, so a failure's actual traceback is never among the
    lines silently dropped by the throttle.
    """
    progress_key = f'{label or "console"}-{uuid.uuid4().hex[:8]}' if logger else None
    last_line = [None]
    seen = []
    _last_emit_time = [0.0]
    _last_line_emitted = [None]

    def _record(line):
        seen.append(line)
        if len(seen) > tail_lines:
            del seen[0]
        if logger is None:
            print(line)
            return
        now = time.monotonic()
        if now - _last_emit_time[0] < min_interval:
            return
        _last_emit_time[0] = now
        _last_line_emitted[0] = line
        logger.info('%s%s', f'{label}: ' if label else '', line, extra={'progress_key': progress_key})

    buf = b''
    fd = proc.stdout.fileno()
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
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
                _record(line)
    tail = buf.decode('utf-8', errors='replace').strip()
    if tail and tail != last_line[0]:
        _record(tail)
    returncode = proc.wait()
    # Same as redirect_console_to_logger's own final flush: a throttled-away
    # last line must still reach the log console once, now that the
    # subprocess has actually finished.
    if logger is not None and last_line[0] is not None and last_line[0] != _last_line_emitted[0]:
        logger.info('%s%s', f'{label}: ' if label else '', last_line[0], extra={'progress_key': progress_key})
    return returncode, seen
