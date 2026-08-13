# -*- coding: utf-8 -*-
"""Where EDyssey is allowed to write its own runtime state (logs, caches,
on-demand-downloaded models) - as opposed to where it's installed.

Those are the same directory in dev mode (next to the source tree, which is
convenient and already git-ignored), but must NOT be the same directory in
a frozen build: the default Windows installer location (`Program Files`)
needs admin rights to write into, so anything that assumed "next to the
app" == "writable" crashed for any non-admin install there - see
logging_utils.py's LOG_DIR (crashed at import time, before the window even
opened) and pets.py's _CACHE_DIR (crashed later, when actually saving PETS2
params) for the two bugs this was written to fix.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def writable_data_dir():
    """Per-user, always-writable app data directory - do not assume the
    install directory itself is writable."""
    if getattr(sys, 'frozen', False):
        return os.path.join(os.environ['LOCALAPPDATA'], 'EDyssey')
    return _REPO_ROOT
