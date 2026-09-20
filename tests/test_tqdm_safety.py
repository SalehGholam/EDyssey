# -*- coding: utf-8 -*-
"""tqdm_safety (repo root, a bare module - see its own docstring for why
it can't live inside EDyssey.io_utils): the host-level half of a real,
confirmed deadlock fix.

**Why this exists in EDyssey itself, not only in pyeventem.** pyeventem's
own ``_tqdm_safe.import_tqdm()`` sets ``tqdm.monitor_interval = 0`` too,
but only stops *pyeventem's* calls from creating tqdm's background
monitor thread. That thread is one shared, process-wide object - if
anything else in this app that does a bare, unaware ``from tqdm import
tqdm`` (``io_utils/video.py``'s clip export, ``tracking_utils_ui.py``'s
per-ROI extraction loop, both real call sites in this app - see the grep
in tqdm_safety.py's own docstring) constructs a bar *first*, with tqdm's
real default, that thread keeps running with the interval it captured at
construction time, completely unaffected by pyeventem setting
``monitor_interval = 0`` afterward. A real report of the underlying
deadlock recurring *after* pyeventem's own fix landed is exactly this
gap - confirmed directly below, not assumed.

These tests exercise the mechanism directly rather than importing
EDyssey_MainWindow.py itself, which has heavy, process-global side
effects at module level (os.chdir(), an AppTheme QObject) unsuitable for
a unit test to trigger.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tqdm")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _restore_tqdm_class_state():
    """monitor_interval/monitor are class attributes on tqdm.tqdm, shared
    by every test in the process - save and restore around each test
    here, and stop any monitor thread a test leaves running so it can't
    leak into unrelated tests (see pyeventem's own test_tqdm_safe.py for
    the identical concern and the same fix)."""
    from tqdm import tqdm as tqdm_cls

    saved_interval = tqdm_cls.monitor_interval
    saved_monitor = tqdm_cls.monitor
    yield
    if tqdm_cls.monitor is not None and tqdm_cls.monitor is not saved_monitor:
        tqdm_cls.monitor.exit()
    tqdm_cls.monitor_interval = saved_interval
    tqdm_cls.monitor = saved_monitor


def test_disable_monitor_thread_sets_the_class_attribute():
    from tqdm_safety import disable_monitor_thread

    disable_monitor_thread()
    from tqdm import tqdm as tqdm_cls

    assert tqdm_cls.monitor_interval == 0


def test_called_early_it_protects_code_that_knows_nothing_about_it():
    """The actual scenario this exists for: video.py/tracking_utils_ui.py
    do a bare `from tqdm import tqdm` with no awareness of this module at
    all. Calling disable_monitor_thread() first (as EDyssey_MainWindow.py
    does, before either of those could ever run) must still prevent the
    monitor thread even from that unaware caller."""
    from tqdm_safety import disable_monitor_thread

    disable_monitor_thread()

    from tqdm import tqdm as tqdm_cls  # exactly what video.py itself does

    bar = tqdm_cls(total=5, disable=True)
    bar.update(1)
    bar.close()

    assert tqdm_cls.monitor is None, \
        "even a caller unaware of tqdm_safety must not get a monitor thread once this ran first"


def test_the_gap_this_closes_is_real_not_hypothetical():
    """Reproduces the exact failure that prompted this fix: pyeventem's
    own, narrower fix runs, but only *after* something else already
    created the monitor thread with the real default. Confirms that
    ordering genuinely defeats pyeventem's fix on its own - i.e. that
    this host-level fix is not redundant with it."""
    pytest.importorskip("pyeventem")
    from tqdm import tqdm as tqdm_cls

    # Something oblivious to any fix runs first (video.py's own pattern).
    bar = tqdm_cls(total=5, disable=True)
    bar.update(1)
    bar.close()
    pre_existing_monitor = tqdm_cls.monitor
    assert pre_existing_monitor is not None
    assert pre_existing_monitor.is_alive()

    # pyeventem's own fix runs after the fact, as it would the first time
    # a declustered ROI computation happens in the same process.
    from pyeventem._tqdm_safe import import_tqdm
    import_tqdm()

    assert tqdm_cls.monitor_interval == 0, "pyeventem's own fix still applies going forward"
    assert tqdm_cls.monitor is pre_existing_monitor, \
        "but it cannot retroactively stop a monitor thread that already exists"
    assert tqdm_cls.monitor.is_alive(), \
        "the pre-existing thread keeps running with its own captured interval regardless"


def test_importing_it_does_not_pull_in_the_heavy_dependencies():
    """EDyssey_MainWindow.py calls this before its own --worker /
    --multiprocessing-fork subprocess dispatch specifically to avoid
    paying for PyQt5/dask/hyperspy in every short-lived worker
    subprocess (see its own comment on that dispatch). A fix placed that
    early has to actually honor that property, not quietly reintroduce a
    heavy import - checked in a fresh subprocess, since this process may
    already have those imported from other tests."""
    code = (
        "import sys; sys.modules_before = set(sys.modules)\n"
        "from tqdm_safety import disable_monitor_thread\n"
        "disable_monitor_thread()\n"
        "new_modules = set(sys.modules) - sys.modules_before\n"
        "heavy = {m for m in new_modules if m.split('.')[0] in "
        "('PyQt5', 'dask', 'hyperspy', 'matplotlib', 'torch')}\n"
        "print(sorted(heavy))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"unexpectedly heavy imports: {result.stdout}"


def test_a_worker_subprocess_actually_gets_the_fix_applied():
    """The real, behavioral version of "the call happens before the
    --worker dispatch": actually launch EDyssey_MainWindow.py in
    --worker mode (an unknown worker name, so it exits almost
    immediately via worker_dispatch's own error path) and check
    tqdm.monitor_interval inside that exact process, instead of scanning
    source text for the call. A text-based check (e.g. "does
    'disable_monitor_thread()' appear before the --worker check") passed
    even with the call commented out - the string was still *present*,
    just inert - which is exactly the class of mutation a real behavioral
    check has to catch and a textual one won't.
    """
    # worker_dispatch.run_worker prints an error and sys.exit(1)s for an
    # unrecognized name - never reaches PyQt5/the GUI - but not before
    # this file's own top-of-module code (including disable_monitor_thread)
    # has already run, which is the only part being checked here.
    code = (
        "import faulthandler; faulthandler.dump_traceback_later(10, exit=True)\n"
        "import sys; sys.argv = ['EDyssey_MainWindow.py', '--worker', '__test_probe__']\n"
        "import tqdm\n"
        "_orig_exit = sys.exit\n"
        "def _probe_and_exit(*a, **k):\n"
        "    print('MONITOR_INTERVAL=' + repr(tqdm.tqdm.monitor_interval))\n"
        "    _orig_exit(0)\n"
        "sys.exit = _probe_and_exit\n"
        "import runpy\n"
        "runpy.run_path('EDyssey_MainWindow.py', run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=30,
    )
    assert "MONITOR_INTERVAL=0" in result.stdout, (
        f"a --worker subprocess did not have the fix applied by the time it exits\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )
