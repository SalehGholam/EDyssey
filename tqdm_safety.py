# -*- coding: utf-8 -*-
"""Disables tqdm's own background "monitor" thread, process-wide, as early
in this app's startup as possible.

**The bug this prevents, confirmed with a real thread-stack dump from a
hung session:** tqdm's monitor thread wakes periodically to force-refresh
any bar it considers stale, writing to `file` (stderr by default) *while
still holding tqdm's own global lock*. This app's own
``redirect_console_to_logger`` (see ``progress.py``) ``os.dup2()``s
stdout/stderr to a plain pipe with its own reader thread, to route
eventem/pyeventem's console progress into the Qt log console. If the
monitor thread's write lands while that pipe isn't being drained at that
exact instant, it blocks mid-write without releasing the lock - and any
*later* ``tqdm(...)`` construction anywhere in the process (a second
declustered ROI computation, say) then deadlocks forever waiting for a
lock a thread stuck on an unrelated I/O wait will never release.

**Why this lives here, called at the very top of EDyssey_MainWindow.py,
rather than only inside pyeventem itself** (which has its own, narrower
``_tqdm_safe.import_tqdm()``): ``tqdm.monitor``/``monitor_interval`` are
class attributes shared by *every* importer of ``tqdm.tqdm`` in this
process, and setting ``monitor_interval = 0`` only stops a *future*
``tqdm.__new__`` call from starting the monitor thread - it does nothing
to one that already exists. This app has other, unrelated bare
``from tqdm import tqdm`` call sites (``io_utils/video.py``'s clip
export, ``tracking_utils/tracking_utils_ui.py``'s per-ROI extraction
loop) that know nothing about pyeventem's own fix. If either of those
runs first and creates the monitor thread with tqdm's real default, that
thread keeps running with the interval it was built with regardless of
what pyeventem sets afterward - confirmed directly: a real report of
this exact deadlock recurring *after* pyeventem's own fix landed traced
back to exactly this gap. Calling this once, here, before any of those
modules get a chance to construct a bar, is what actually closes it -
this needs to be a host-level fix, not something one library dependency
can secure entirely on its own.

Deliberately depends on nothing beyond ``tqdm`` itself (no dask, no
PyQt5, no hyperspy) - EDyssey_MainWindow.py calls this before its own
``--worker``/``--multiprocessing-fork`` subprocess dispatch specifically
to avoid paying for those heavier imports in every short-lived worker
subprocess, and this fix has to stay just as cheap to keep that property.
``tqdm`` itself is a required (non-optional) dependency of this app
(requirements.txt), so no try/except around the import is needed.

**Lives at the repo root, as a bare module - not inside the ``EDyssey``
package**, matching ``worker_dispatch.py``'s own placement and for the
same reason: ``EDyssey/io_utils/__init__.py`` does ``from .io_utils_ui
import *``, which transitively imports dask/hyperspy/matplotlib/PyQt5 -
*any* package-qualified import reaching into ``EDyssey.io_utils``, even
of an otherwise-empty module, pays that whole cost first. Verified
directly (a first draft placed this at
``EDyssey/io_utils/tqdm_safety.py`` and a subprocess-isolated import test
caught dask/hyperspy/matplotlib all getting pulled in) - see
``tests/test_tqdm_safety.py``.
"""

from __future__ import annotations


def disable_monitor_thread() -> None:
    """Set ``tqdm.tqdm.monitor_interval = 0`` - see this module's own
    docstring for why. Idempotent and safe to call more than once (each
    call is a plain class-attribute assignment); harmless to call even if
    some other code already created the monitor thread before this ran -
    it just means that particular thread's own risk isn't closed, which
    is exactly why this is called as early as possible."""
    import tqdm as tqdm_module

    tqdm_module.tqdm.monitor_interval = 0
