# -*- coding: utf-8 -*-
"""Receiving side of ui_tabs/worker_launch.py's `--worker <name> <args>`
invocation - runs the named worker_*.py script (workers/, see _base_dir())
via `runpy`, so its own `if __name__ == '__main__':` code needs no changes.
worker_dispatch.py itself stays at the repo root; only the scripts it
dispatches to live in workers/ (EDyssey/workers/ in a frozen install - see
EDyssey.spec's extra_datas).
"""
import os
import sys
import runpy

WORKER_SCRIPTS = {
    'extract_frame': 'worker_extract_frame.py',
    'extract_frame_batch': 'worker_extract_frame_batch.py',
    'nav_img': 'worker_nav_img.py',
    'nav_img_batch': 'worker_nav_img_batch.py',
    'sam': 'worker_sam.py',
}


def _base_dir():
    """EDyssey/workers/ under sys._MEIPASS when frozen (see EDyssey.spec's
    extra_datas), or plain workers/ next to this file's own directory (the
    repo root) otherwise."""
    if getattr(sys, 'frozen', False):
        root = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, 'EDyssey', 'workers')
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, 'workers')


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
    base_dir = _base_dir()
    # workers/ on sys.path, so a worker script can bare-import its own
    # siblings (worker_extract_frame_batch.py's `import worker_pool_utils`
    # / `from worker_extract_frame import ...`, and the same in
    # worker_nav_img_batch.py). `python worker_x.py` would put the script's
    # own directory on sys.path automatically, but runpy.run_path() does
    # NOT - and ui_tabs/__init__.py, which is what puts workers/ on
    # sys.path for the GUI process, never runs here: EDyssey_MainWindow.py
    # dispatches --worker before any ui_tabs import. Without this, every
    # batch worker died at import with "No module named
    # 'worker_pool_utils'" (both frozen and from source), taking
    # Navigator's Calculate All and ROI Tracker/SAM2's Extract! with it.
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    fn = os.path.join(base_dir, script)
    sys.argv = [fn] + list(args)
    runpy.run_path(fn, run_name='__main__')
    sys.exit(0)
