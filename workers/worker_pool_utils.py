# -*- coding: utf-8 -*-
"""Shared machinery for the "one subprocess, one internal process pool"
batch workers (worker_nav_img_batch.py, worker_extract_frame_batch.py) and
their GUI-side launchers (tab_create_navSignal.py, tab_tracking_cv2.py,
tab_sam2.py).

Two independent halves live here:

- GUI-process side: write_tasks_json() (builds the one task file a batch
  driver reads) and create_job_object()/assign_process_to_job()/kill_job()/
  kill_pid() - thin ctypes wrappers around the Windows Job Object API (plus
  a PID-based backstop, see kill_pid's docstring) used to kill a batch
  driver AND every pool worker it ever spawned in one shot.
- Worker/driver-process side: load_tasks() (reads that file back),
  run_pool_batch() (drives a ProcessPoolExecutor over the task list,
  reporting per-task progress on stdout and recovering from a crashed pool
  worker), and wait_for_job_assignment() (see the Job Object note below).

Both halves are kept in one module (rather than split worker-side/GUI-side)
because they're two ends of the same protocol and because every worker_*.py
here is already imported directly by ui_tabs/*.py today (e.g.
tab_tracking_cv2.py imports worker_extract_frame.load_dp) - same precedent,
same "loose file next to the frozen exe" packaging (see EDyssey.spec).
"""
import ctypes
from ctypes import wintypes
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np


#%% GUI-process side - task-list I/O

def _json_default(obj):
    """json.dumps(..., default=...) backstop for numpy scalars (np.int64,
    np.bool_, ...) - not JSON-serializable by default, and with hundreds/
    thousands of tasks in one file, a single missed field anywhere aborts
    serialization for the *whole* batch, not just that task. Callers should
    still coerce explicitly where practical (e.g. list(scanSize) with
    already-plain-int elements) - this is defense in depth, not a
    replacement for that."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def write_tasks_json(path, tasks):
    """Write `tasks` (a list of plain-Python-typed dicts, one per file/
    frame/patch) to `path` as JSON - the one file a batch driver reads
    instead of each task's args being stringified onto a QProcess argv
    list (which risks Windows' command-line length limit for a large
    batch, and is exactly the per-task subprocess overhead this refactor
    removes). Every task dict must include an 'i_index' key (the same
    index the driver's "DONE <i_index>"/"FAIL <i_index>" stdout lines and
    "<i_index>.npy" result files use).

    Round-trip note: JSON has no tuple type, so anything that needs to be
    a tuple on the read side (e.g. scanSize, det_shape) comes back as a
    list from load_tasks() - callers must re-tuple() explicitly wherever
    the rest of the code expects one."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, default=_json_default)


#%% Worker/driver-process side

def load_tasks(tasks_path):
    """Read the GUI-written task list back into a list of dicts, with the
    coercions lost in the JSON round trip restored (see write_tasks_json's
    docstring for what those are)."""
    with open(tasks_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_result_npy(temp_dir, i_index, result):
    """Write one task's result array to `<temp_dir>/<i_index>.npy`, atomically
    (write-then-rename) so a process killed mid-write - e.g. by Cancel -
    never leaves a torn file the GUI could mistake for a finished one, and
    so the GUI never needs to poll/glob the directory: it only trusts a
    result once it has seen this task's own "DONE" stdout line.

    Named "<i_index>.tmp.npy", not "<i_index>.npy.tmp" - np.save() appends
    ".npy" to any filename that doesn't already end with it, which would
    otherwise silently turn a ".npy.tmp" target into ".npy.tmp.npy"."""
    tmp_path = os.path.join(temp_dir, f'{i_index}.tmp.npy')
    final_path = os.path.join(temp_dir, f'{i_index}.npy')
    np.save(tmp_path, result)
    os.replace(tmp_path, final_path)


def _report_new_worker_pids(executor, reported_pids):
    """Print `WORKERPID <pid>` for any ProcessPoolExecutor worker process
    not already in `reported_pids` (updated in place).

    Why this exists: a Windows Job Object (see below) is supposed to make
    Cancel kill a driver process and everything it ever spawns in one shot,
    but that was verified empirically (real OS processes, not mocked) to
    sometimes still leave a grandchild pool worker alive - reproducible even
    with the driver correctly assigned to the job *before* it builds its
    pool, i.e. not just the documented "assignment isn't retroactive"
    startup race. Cause not fully isolated (plausibly specific to nested
    process sandboxing in some environments), so this is a deliberate,
    independent second kill path: the GUI also force-kills every PID
    reported here directly, which doesn't depend on job-nesting semantics
    at all. Uses ProcessPoolExecutor's private `_processes` dict (stable
    across CPython 3.7+ in this form, but undocumented) - if a future
    Python version removes/renames it, this degrades to "no PID list
    reported", not a crash (see the try/except below); Cancel then falls
    back to the Job Object kill alone.
    """
    try:
        current_pids = set(executor._processes.keys())
    except AttributeError:
        return
    new_pids = current_pids - reported_pids
    for pid in new_pids:
        print(f'WORKERPID {pid}')
    if new_pids:
        reported_pids.update(new_pids)
        sys.stdout.flush()


def run_pool_batch(compute_fn, tasks, temp_dir, n_workers):
    """Run `compute_fn(task)` for every task in `tasks` across a
    ProcessPoolExecutor of `n_workers` workers, writing each result to
    `temp_dir` and printing progress to stdout (flushed immediately) for
    the GUI's incremental reader (see the line-buffering reader added to
    each tab's readyReadStandardOutput handler):

        DONE <i_index>
        FAIL <i_index> <short error message>
        WORKERPID <pid>    (see _report_new_worker_pids - a Cancel kill-path
                             backstop, not progress; one line per pool
                             worker process as it's first seen)

    A single task's exception is caught per-task (a "FAIL" line, everything
    else keeps going) - only a worker *process* dying outright (a native
    crash in eventem/hdf5/opencv, not a normal Python exception) takes the
    whole pool down as BrokenProcessPool, in which case a fresh pool is
    built and only the not-yet-completed tasks are resubmitted, rather than
    losing the rest of a long batch to one bad file.

    `compute_fn` must be a real module-level function reached via a genuine
    `import` (e.g. worker_nav_img.calculate_nav_img_core) - never something
    defined inside the batch-driver script itself. That script is executed
    via runpy.run_path(..., run_name='__main__') (see worker_dispatch.py),
    which aliases sys.modules['__main__'] to a synthetic module for the
    duration of the call; a function defined there would pickle as
    ('__main__', name), and reconstructing that in a spawned child is not
    reliable for a frozen Windows executable.
    """
    pending = {task['i_index']: task for task in tasks}
    reported_pids = set()
    while pending:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(compute_fn, task): i_index
                          for i_index, task in pending.items()}
                _report_new_worker_pids(executor, reported_pids)
                for future in as_completed(futures):
                    _report_new_worker_pids(executor, reported_pids)
                    i_index = futures[future]
                    try:
                        result = future.result()
                        _save_result_npy(temp_dir, i_index, result)
                        print(f'DONE {i_index}')
                    except Exception as exc:
                        print(f'FAIL {i_index} {exc}')
                    finally:
                        pending.pop(i_index, None)
                        sys.stdout.flush()
        except BrokenProcessPool:
            # A worker process died outright rather than raising - the
            # executor (and every future still pending on it) is unusable
            # now. `pending` already only holds tasks not yet reported
            # DONE/FAIL above, so looping just retries those in a fresh pool.
            continue


def wait_for_job_assignment():
    """Block until the parent GUI process confirms (over this process's own
    stdin) that it has assigned this process to a Windows Job Object with
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, before returning - so the driver
    only starts spawning its own ProcessPoolExecutor workers once every
    process it's about to create is guaranteed to be covered by a later
    Cancel. Job membership isn't retroactive: a pool worker spawned before
    this handshake completes would escape a kill of the job.

    Always returns (never raises) - if the parent's job-object creation
    itself failed (e.g. a locked-down EDR environment), it still writes a
    line here so the batch isn't stuck waiting forever; Cancel then falls
    back to killing just this driver process."""
    try:
        sys.stdin.readline()
    except Exception:
        pass


#%% GUI-process side - Windows Job Object ctypes wrappers

_kernel32 = ctypes.windll.kernel32 if os.name == 'nt' else None

JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('ReadOperationCount', ctypes.c_uint64),
        ('WriteOperationCount', ctypes.c_uint64),
        ('OtherOperationCount', ctypes.c_uint64),
        ('ReadTransferCount', ctypes.c_uint64),
        ('WriteTransferCount', ctypes.c_uint64),
        ('OtherTransferCount', ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ('PerProcessUserTimeLimit', ctypes.c_int64),
        ('PerJobUserTimeLimit', ctypes.c_int64),
        ('LimitFlags', wintypes.DWORD),
        ('MinimumWorkingSetSize', ctypes.c_size_t),
        ('MaximumWorkingSetSize', ctypes.c_size_t),
        ('ActiveProcessLimit', wintypes.DWORD),
        ('Affinity', ctypes.c_size_t),
        ('PriorityClass', wintypes.DWORD),
        ('SchedulingClass', wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ('BasicLimitInformation', _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ('IoInfo', _IO_COUNTERS),
        ('ProcessMemoryLimit', ctypes.c_size_t),
        ('JobMemoryLimit', ctypes.c_size_t),
        ('PeakProcessMemoryUsed', ctypes.c_size_t),
        ('PeakJobMemoryUsed', ctypes.c_size_t),
    ]


if _kernel32 is not None:
    # Explicit argtypes/restype throughout - HANDLE is pointer-sized, and
    # ctypes silently assumes a 32-bit c_int return value for any function
    # left undeclared, which truncates/corrupts a 64-bit handle on 64-bit
    # Python instead of raising. Getting this wrong here would corrupt the
    # very handle Cancel later relies on to kill a whole process tree.
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateProcess.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL


def create_job_object():
    """Create a Windows Job Object configured so that closing/terminating it
    kills every process ever assigned to it, plus everything those
    processes go on to spawn themselves (a ProcessPoolExecutor's workers) -
    even ones spawned after the originally-assigned process is already
    dead/hung.

    Returns the job handle, or None on any failure (missing kernel32 on a
    non-Windows platform, or the call itself failing - e.g. a locked-down
    EDR environment refusing job creation). Callers must treat None as
    "fall back to killing just the one process you have a handle for",
    never as fatal - Cancel must still work, just less thoroughly."""
    if _kernel32 is None:
        return None
    handle = _kernel32.CreateJobObjectW(None, None)
    if not handle:
        return None
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(info))
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = _kernel32.SetInformationJobObject(
        handle, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
    if not ok:
        _kernel32.CloseHandle(handle)
        return None
    return handle


def assign_process_to_job(job_handle, pid):
    """Assign the process `pid` (and, from then on, anything it spawns) to
    `job_handle`. Returns True on success. `job_handle` may be None (a
    create_job_object() failure already handled upstream) - returns False
    without raising, same "not fatal" contract as create_job_object()."""
    if job_handle is None:
        return False
    proc_handle = _kernel32.OpenProcess(PROCESS_TERMINATE | PROCESS_SET_QUOTA, False, pid)
    if not proc_handle:
        return False
    try:
        return bool(_kernel32.AssignProcessToJobObject(job_handle, proc_handle))
    finally:
        _kernel32.CloseHandle(proc_handle)


def kill_job(job_handle):
    """Kill every process currently assigned to `job_handle` and close it.
    No-op if `job_handle` is None (nothing to do - the caller already fell
    back to plain process.kill() on just the driver PID in that case)."""
    if job_handle is None:
        return
    _kernel32.TerminateJobObject(job_handle, 1)
    _kernel32.CloseHandle(job_handle)


def kill_pid(pid):
    """Force-kill one process by PID directly (OpenProcess+TerminateProcess),
    independent of any job-object membership. Used as a second, redundant
    Cancel kill-path for each PID reported via a driver's "WORKERPID" stdout
    lines (see _report_new_worker_pids) - kill_job() alone was empirically
    observed to sometimes still leave a pool-worker grandchild alive even
    with job assignment done correctly before the pool was built, so Cancel
    calls both, not either/or. Silently does nothing if the process is
    already gone (already exited, or already taken down by kill_job())."""
    if _kernel32 is None:
        return
    proc_handle = _kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
    if not proc_handle:
        return
    try:
        _kernel32.TerminateProcess(proc_handle, 1)
    finally:
        _kernel32.CloseHandle(proc_handle)
