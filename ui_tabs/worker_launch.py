# -*- coding: utf-8 -*-
"""Builds the (program, arguments) pair QProcess needs to launch one of the
worker_*.py scripts (worker_extract_frame.py, worker_nav_img.py,
worker_sam.py) as a subprocess.

Used instead of every tab hand-rolling `process.setProgram(sys.executable);
process.setArguments(["worker_X.py", *args])` - that pattern only works when
`sys.executable` is a real Python interpreter that can be handed a bare .py
path to run. In a PyInstaller-frozen build, `sys.executable` is the frozen
app itself (no bundled python.exe), so most workers are run a different
way: the frozen exe re-invokes itself with `--worker <name> ...`, which
EDyssey_MainWindow.py's entry point recognizes and dispatches to
worker_dispatch.py (repo root) instead of starting the GUI.

The 'sam' worker is a deliberate exception to that, frozen or not: torch
(specifically SAM2's own `torch.jit.script(...)` call) needs `inspect.
getsource()` to read real .py source for stdlib modules like `enum`, which
a PyInstaller-frozen build never has (it ships compiled bytecode only, no
source) - and separately, PyInstaller's own mandatory PyQt5 runtime hook
loads a bundled Qt5\\bin\\msvcp140.dll at startup that torch's own c10.dll
then gets forced to reuse instead of the system one, crashing outright. Both
are structural mismatches between "running inside the frozen GUI exe" and
"running torch", not fixable by patching the frozen build - so 'sam' is
instead run via the same real, version-matched system Python that
sam2_setup_dialog.py used to pip-install it (see python_finder.py), pointed
directly at the staged workers/worker_sam.py, exactly like dev-mode already
does successfully today. Every other worker has no such dependency and
keeps using the frozen self-relaunch path.
"""
import sys
import os

from ui_tabs.python_finder import find_bundled_python, find_system_python, read_interpreter_marker

# Repo root - two levels up from this file (ui_tabs/worker_launch.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_WORKER_NAMES = {'extract_frame', 'extract_frame_batch', 'nav_img', 'nav_img_batch', 'sam'}


def _sam_command_frozen(str_args):
    """(program, arguments) for the 'sam' worker specifically, run via a
    real system Python instead of self-relaunching the frozen exe - see this
    module's own docstring for why. Prefers a bundled portable Python (an
    *offline* build only - see find_bundled_python()) over the marker/
    system-Python resolution an *online* build depends on, since it's
    guaranteed to match whatever torch/sam2 were bundled for. Falls back to
    the old self-relaunch (which will surface a clean "missing_dependency"
    message if torch truly isn't installed) if no matching Python can be
    found here - shouldn't normally happen, since installing needed one too."""
    install_dir = os.path.dirname(sys.executable)
    internal_dir = os.path.join(install_dir, '_internal')
    sam2_packages_dir = os.path.join(internal_dir, 'sam2_packages')
    worker_script = os.path.join(internal_dir, 'EDyssey', 'workers', 'worker_sam.py')

    python_prefix = (find_bundled_python(internal_dir)
                      or read_interpreter_marker(sam2_packages_dir)
                      or find_system_python())
    if python_prefix is not None and os.path.isfile(worker_script):
        return python_prefix[0], python_prefix[1:] + [worker_script] + str_args
    return sys.executable, ['--worker', 'sam'] + str_args


def worker_command(worker_name, args):
    """Return (program, arguments) for QProcess.setProgram/.setArguments to
    launch worker_<worker_name>.py with `args` (each converted to str).

    Args:
        worker_name: One of 'extract_frame', 'extract_frame_batch',
            'nav_img', 'nav_img_batch', 'sam' - see
            worker_dispatch.WORKER_SCRIPTS for the name -> script mapping.
        args: Positional arguments to pass to the worker script, in the same
            order its own `if __name__ == '__main__':` block expects them.
    """
    if worker_name not in _WORKER_NAMES:
        raise ValueError(f'Unknown worker name: {worker_name!r}')
    str_args = [str(a) for a in args]
    if getattr(sys, 'frozen', False):
        if worker_name == 'sam':
            return _sam_command_frozen(str_args)
        # sys.executable is this app's own frozen exe here - it already is
        # the entry point, so no script path is needed, just the dispatch flag.
        return sys.executable, ['--worker', worker_name] + str_args
    # sys.executable is a plain python.exe here, which needs an explicit
    # script to run - unlike the frozen case above.
    entry = os.path.join(_REPO_ROOT, 'EDyssey_MainWindow.py')
    return sys.executable, [entry, '--worker', worker_name] + str_args
