# -*- coding: utf-8 -*-
"""Receiving side of ui_tabs/worker_launch.py's `--worker <name> <args>`
invocation - runs the named worker_*.py script in-process via `runpy`, as if
it had been launched directly as `python worker_X.py <args>`.

Using `runpy.run_path` (rather than importing each worker as a module and
calling a specific function) means none of the worker scripts' own
`if __name__ == '__main__':` entry-point code has to change - each one
still parses `sys.argv` and calls `sys.exit(...)` exactly as it always has,
whether invoked directly (`python worker_extract_frame.py ...`, still works
unchanged) or dispatched through here (dev-mode subprocess re-invoking
5DED_MainWidow.py, or a PyInstaller-frozen build re-invoking itself).
"""
import os
import sys
import runpy

WORKER_SCRIPTS = {
    'extract_frame': 'worker_extract_frame.py',
    'nav_img': 'worker_nav_img.py',
    'sam': 'worker_sam.py',
}


def _base_dir():
    """Directory the worker_*.py scripts live in - the PyInstaller-extracted
    temp dir (`sys._MEIPASS`) when frozen and bundled as data files, this
    file's own directory (the repo root) otherwise."""
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.dirname(os.path.abspath(__file__))


def run_worker(worker_name, args):
    """Run worker_<worker_name>.py with `args` as its `sys.argv[1:]`, then
    exit - the worker script's own `if __name__ == '__main__':` block
    determines the process exit code via its own `sys.exit(...)` calls, so
    this only need to fall through to a clean 0 if the script doesn't call
    sys.exit itself."""
    script = WORKER_SCRIPTS.get(worker_name)
    if script is None:
        print(f'Unknown worker: {worker_name!r} (expected one of {sorted(WORKER_SCRIPTS)})',
              file=sys.stderr)
        sys.exit(1)
    fn = os.path.join(_base_dir(), script)
    sys.argv = [fn] + list(args)
    runpy.run_path(fn, run_name='__main__')
    sys.exit(0)
