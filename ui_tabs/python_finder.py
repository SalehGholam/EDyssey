# -*- coding: utf-8 -*-
"""Finds a real system Python to run torch/sam2 in.

Shared by sam2_setup_dialog.py (to run pip) and worker_launch.py (to run
worker_sam.py directly, instead of through the frozen EDyssey.exe - see
worker_launch.py's own docstring for why SAM2 specifically needs this).

No particular version is required: SAM2 always runs via this same real
interpreter (never the frozen exe itself, see worker_launch.py), so there's
no cross-version .pyd-loading mismatch to avoid the way there would be if
torch/sam2 had to be loaded by the app's own embedded Python - any Python
recent enough for torch/sam2 themselves to support is fine. This app's own
embedded version (APP_PYTHON_VERSION) is tracked separately below only for
reference/logging, not as a requirement.
"""
import json
import os
import shutil
import subprocess
import sys

# e.g. '3.12' - this app's own embedded Python version. Not a requirement
# for the system Python used to run SAM2 - see this module's docstring.
APP_PYTHON_VERSION = '%d.%d' % sys.version_info[:2]

# torch/sam2's own realistic floor - just a sanity check against a stray,
# genuinely-too-old `python` on PATH, not a pin to any specific version.
MIN_PYTHON_VERSION = (3, 9)

# Records which [python, *args] prefix sam2_setup_dialog.py actually used to
# pip-install into sam2_packages, so worker_launch.py can reuse that exact
# interpreter later instead of re-detecting (which could pick a different
# one if e.g. multiple Pythons are installed).
INTERPRETER_MARKER_FILENAME = '.python_interpreter.json'

# Where the *offline* installer stages a portable Python runtime (see
# EDyssey_offline.iss's [Files] section and EDyssey/portable_python/'s own
# README-less origin: a plain python.org Windows embeddable package, plus
# the real stdlib .py source copied in alongside it - the embeddable
# package's own bundled stdlib is compiled-only, which would hit the exact
# same inspect.getsource() failure worker_launch.py's own docstring
# describes for running torch inside the frozen exe). Relative to
# _internal (sys._MEIPASS), matching where worker_sam.py/worker_launch.py
# already resolve _internal_dir/install_dir from.
_BUNDLED_PYTHON_SUBDIR = 'portable_python'


def find_bundled_python(internal_dir):
    """[python.exe] if this install has a bundled portable Python (see
    _BUNDLED_PYTHON_SUBDIR above - only the *offline* installer stages
    one), else None. Checked before read_interpreter_marker()/
    find_system_python() by both worker_launch.py (to actually run
    worker_sam.py) and sam2_setup_dialog.py's Check CUDA: an offline
    build's torch/sam2 are bundled for exactly this app's own
    APP_PYTHON_VERSION with no pip step to ever adapt them to a
    differently-versioned system Python, so the bundled interpreter -
    guaranteed to match - is strictly preferred over depending on
    whatever's (or isn't) installed on the target machine. Not used for
    pip operations (_pip_base_cmd) - the embeddable package this is built
    from has no pip of its own, deliberately (SAM2 never needs to install
    anything into it, torch/sam2 are already right there)."""
    candidate = os.path.join(internal_dir, _BUNDLED_PYTHON_SUBDIR, 'python.exe')
    if os.path.isfile(candidate) and python_is_usable(candidate):
        return [candidate]
    return None


def write_interpreter_marker(sam2_packages_dir, python_prefix):
    try:
        with open(os.path.join(sam2_packages_dir, INTERPRETER_MARKER_FILENAME), 'w') as f:
            json.dump({'python_prefix': list(python_prefix)}, f)
    except OSError:
        pass


def read_interpreter_marker(sam2_packages_dir):
    """The [python, *args] prefix recorded by write_interpreter_marker, if
    the file exists and that interpreter is still usable - None otherwise
    (caller should fall back to find_system_python())."""
    try:
        with open(os.path.join(sam2_packages_dir, INTERPRETER_MARKER_FILENAME)) as f:
            prefix = json.load(f)['python_prefix']
    except (OSError, ValueError, KeyError):
        return None
    if python_is_usable(prefix[0], prefix[1:]):
        return prefix
    return None


def python_is_usable(python_exe, extra_args=()):
    """Whether `python_exe extra_args` actually runs and reports at least
    MIN_PYTHON_VERSION."""
    try:
        result = subprocess.run(
            [python_exe, *extra_args, '-c',
             'import sys; print("%d.%d" % sys.version_info[:2])'],
            capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return False
        major, minor = (int(v) for v in result.stdout.strip().split('.'))
        return (major, minor) >= MIN_PYTHON_VERSION
    except Exception:
        return False


def find_system_python():
    """A [python, *args] invocation prefix for a usable system Python - None
    if none is found. Always returns a version-pinned command (e.g.
    ['py.exe', '-3.12'] or ['C:\\...\\python.exe']), never a bare `py` with
    no version flag - bare `py`'s own default resolution is NOT stable
    across invocations from different environments: it prefers an active
    virtualenv's own interpreter when one happens to be active (as in a dev
    shell), and otherwise falls back to whatever the highest-registered
    version is, which can differ machine to machine and drift over time as
    new Pythons get installed. A caller resolving `py` this way at install
    time and again later at run time (a real app process, no venv active
    either time) can silently get two DIFFERENT interpreters, producing
    "Module use of pythonXYZ.dll conflicts with this version of Python" the
    moment a compiled .pyd built for one gets loaded by the other - this
    happened for real, see git history around this comment. Prefers
    APP_PYTHON_VERSION (this app's own embedded version, the best-tested
    combination) if available, then whatever `py`'s default actually - and
    verifiably - resolves to, then `python`/`python3` on PATH.
    """
    py_launcher = shutil.which('py')
    if py_launcher:
        versioned_flag = f'-{APP_PYTHON_VERSION}'
        if python_is_usable(py_launcher, [versioned_flag]):
            return [py_launcher, versioned_flag]
        if python_is_usable(py_launcher):
            # Bare `py` is usable but not APP_PYTHON_VERSION - pin the
            # command to whichever version it actually resolved to, so the
            # returned prefix stays reproducible later even if the bare
            # default itself would resolve differently by then.
            try:
                result = subprocess.run(
                    [py_launcher, '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'],
                    capture_output=True, text=True, timeout=15)
                resolved_version = result.stdout.strip()
                if result.returncode == 0 and resolved_version:
                    return [py_launcher, f'-{resolved_version}']
            except Exception:
                pass
    for candidate in ('python', 'python3'):
        found = shutil.which(candidate)
        if found and python_is_usable(found):
            return [found]
    return None
