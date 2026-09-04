# -*- coding: utf-8 -*-
"""Finds a real system Python matching this app's own embedded version.

Shared by sam2_setup_dialog.py (to run pip) and worker_launch.py (to run
worker_sam.py directly, instead of through the frozen EDyssey.exe - see
worker_launch.py's own docstring for why SAM2 specifically needs this).
"""
import json
import os
import shutil
import subprocess
import sys

# e.g. '3.12' - this app's own embedded Python version, even when frozen.
APP_PYTHON_VERSION = '%d.%d' % sys.version_info[:2]

# Records which [python, *args] prefix sam2_setup_dialog.py actually used to
# pip-install into sam2_packages, so worker_launch.py can reuse that exact
# interpreter later instead of re-detecting (which could pick a different
# one if e.g. both `py -3.12` and a PATH python.exe match).
INTERPRETER_MARKER_FILENAME = '.python_interpreter.json'


def write_interpreter_marker(sam2_packages_dir, python_prefix):
    try:
        with open(os.path.join(sam2_packages_dir, INTERPRETER_MARKER_FILENAME), 'w') as f:
            json.dump({'python_prefix': list(python_prefix)}, f)
    except OSError:
        pass


def read_interpreter_marker(sam2_packages_dir):
    """The [python, *args] prefix recorded by write_interpreter_marker, if
    the file exists and that interpreter still resolves to a real,
    version-matching Python - None otherwise (caller should fall back to
    find_system_python())."""
    try:
        with open(os.path.join(sam2_packages_dir, INTERPRETER_MARKER_FILENAME)) as f:
            prefix = json.load(f)['python_prefix']
    except (OSError, ValueError, KeyError):
        return None
    if python_version_matches(prefix[0], prefix[1:]):
        return prefix
    return None


def python_version_matches(python_exe, extra_args=()):
    """Whether `python_exe extra_args` actually runs as APP_PYTHON_VERSION -
    a --target install via a different version produces .pyd files this
    app's own Python can't load."""
    try:
        result = subprocess.run(
            [python_exe, *extra_args, '-c',
             'import sys; print("%d.%d" % sys.version_info[:2])'],
            capture_output=True, text=True, timeout=15)
        return result.returncode == 0 and result.stdout.strip() == APP_PYTHON_VERSION
    except Exception:
        return False


def find_system_python():
    """A [python, *args] invocation prefix for a system Python matching
    APP_PYTHON_VERSION - None if no matching one exists (never falls back
    to a mismatched version). Tries 'py -3.12' (exact pin) first, then
    'python'/'python3' on PATH if either actually matches."""
    py_launcher = shutil.which('py')
    if py_launcher:
        versioned_flag = f'-{APP_PYTHON_VERSION}'
        if python_version_matches(py_launcher, [versioned_flag]):
            return [py_launcher, versioned_flag]
    for candidate in ('python', 'python3'):
        found = shutil.which(candidate)
        if found and python_version_matches(found):
            return [found]
    return None
