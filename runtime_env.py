# -*- coding: utf-8 -*-
"""Process-wide environment setup that must run before numba (pyeventem) is imported.

A bare module at the repo root, like tqdm_safety.py: importing it must not pull in the heavy
EDyssey.io_utils package first.
"""
import os
import sys


def configure_numba_cache(frozen=None, local_app_data=None):
    """Point numba's on-disk kernel cache at a per-user folder in a frozen (installed) build.

    By default numba caches next to the package files, which are read-only under Program Files.
    On Windows, probing such a folder makes tempfile retry ~10000 times per cached function, so the
    first `import pyeventem` effectively hangs for minutes. An explicit NUMBA_CACHE_DIR skips the
    probe entirely. An existing NUMBA_CACHE_DIR is left alone. Returns the directory in use or None.
    """
    frozen = getattr(sys, 'frozen', False) if frozen is None else frozen
    if not frozen:
        return None
    if os.environ.get('NUMBA_CACHE_DIR'):
        return os.environ['NUMBA_CACHE_DIR']
    base = local_app_data or os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    cache_dir = os.path.join(base, 'EDyssey', 'numba_cache')
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        return None
    os.environ['NUMBA_CACHE_DIR'] = cache_dir
    return cache_dir
