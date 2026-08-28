# -*- coding: utf-8 -*-
"""Builds the (program, arguments) pair QProcess needs to launch one of the
worker_*.py scripts (worker_extract_frame.py, worker_nav_img.py,
worker_sam.py) as a subprocess.

Used instead of every tab hand-rolling `process.setProgram(sys.executable);
process.setArguments(["worker_X.py", *args])` - that pattern only works when
`sys.executable` is a real Python interpreter that can be handed a bare .py
path to run. In a PyInstaller-frozen build, `sys.executable` is the frozen
app itself (no bundled python.exe), so the worker script has to be run a
different way: the frozen exe re-invokes itself with `--worker <name> ...`,
which EDyssey_MainWindow.py's entry point recognizes and dispatches to
worker_dispatch.py (repo root) instead of starting the GUI. This module is
the one place that decides which of those two invocation shapes to build,
so no call site needs to know or care whether it's running frozen or not.
"""
import sys
import os

# Repo root - two levels up from this file (ui_tabs/worker_launch.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_WORKER_NAMES = {'extract_frame', 'nav_img', 'nav_img_batch', 'sam'}


def worker_command(worker_name, args):
    """Return (program, arguments) for QProcess.setProgram/.setArguments to
    launch worker_<worker_name>.py with `args` (each converted to str).

    Args:
        worker_name: One of 'extract_frame', 'nav_img', 'nav_img_batch',
            'sam' - see worker_dispatch.WORKER_SCRIPTS for the name ->
            script mapping.
        args: Positional arguments to pass to the worker script, in the same
            order its own `if __name__ == '__main__':` block expects them.
    """
    if worker_name not in _WORKER_NAMES:
        raise ValueError(f'Unknown worker name: {worker_name!r}')
    str_args = [str(a) for a in args]
    if getattr(sys, 'frozen', False):
        # sys.executable is this app's own frozen exe here - it already is
        # the entry point, so no script path is needed, just the dispatch flag.
        return sys.executable, ['--worker', worker_name] + str_args
    # sys.executable is a plain python.exe here, which needs an explicit
    # script to run - unlike the frozen case above.
    entry = os.path.join(_REPO_ROOT, 'EDyssey_MainWindow.py')
    return sys.executable, [entry, '--worker', worker_name] + str_args
