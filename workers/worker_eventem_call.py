# -*- coding: utf-8 -*-
"""Qt-free subprocess entry point for a single EDyssey.io_utils.eventem_backend
run_*() call - the actual fix for 'New eventem' segfaulting whenever it's
imported in a process that already has PyQt5 loaded (see
eventem_backend._new_eventem's own comment for the full story: 100%
reproducible, root cause is inside eventem_new.pyd's own compiled code, not
diagnosable further without a debugger). Rather than just failing cleanly
in that case, eventem_backend.py now runs the *entire* New-eventem call
through this script instead - a real, separate, Qt-free process, exactly
like "Extract!"/"Calculate All"/"Compute Virtual Image" already use for
their own (batch) work, just synchronous and generic over which run_*
function is being called.

Invoked as `--worker eventem_call <request.json path> <result.npz path>`
via worker_launch.worker_command()/worker_dispatch.py, the same indirection
every other worker script here uses.

request.json: {"func": "run_pacbed"|"run_vstem"|"run_var"|"run_roi"|"run_roi_masked",
               "kwargs": {...plain JSON-serializable eventem_backend.run_*
               keyword arguments - the caller has already stripped out
               anything non-serializable (a real logger object, a mask
               ndarray is instead written to its own .npy and referenced
               by path - see eventem_backend._run_new_eventem_via_subprocess)}}

Writes result.npz (np.savez) with whichever arrays that function actually
returns, or prints a traceback to stderr and exits 1 on failure - the
parent reads this file back and reconstructs the same return shape
(a plain ndarray for run_pacbed/run_vstem/run_var, or an eventem_backend.
RoiResult for run_roi/run_roi_masked) the in-process call would have
returned, so no caller of the facade needs to know this subprocess
indirection is happening at all.
"""
import sys
import os
import json

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_THIS_DIR), 'EDyssey', 'io_utils'))
# io_utils_ui, not a bare `import eventem_backend` - eventem_backend.py uses
# relative imports (`from .progress import ...`), which fail with "attempted
# relative import with no known parent package" when the module is imported
# bare like this, outside its EDyssey.io_utils package context. io_utils_ui
# re-exports it as `io.eb` via absolute imports instead (see its own
# docstring) - the same workaround worker_extract_frame.py/worker_nav_img.py
# already use for exactly this reason.
import io_utils_ui as io
eb = io.eb

_ARRAY_RESULT_FUNCS = {'run_pacbed', 'run_vstem', 'run_var'}
_ROI_RESULT_FUNCS = {'run_roi', 'run_roi_masked'}


def _load_mask_if_referenced(kwargs):
    """`mask` (run_roi_masked only) travels as a path to a .npy file, not
    inline in the JSON request - a real boolean array is not JSON-
    serializable, and inlining it as nested lists would bloat the request
    file for no reason when a temp .npy is just as easy for the caller to
    write."""
    mask_path = kwargs.pop('mask_path', None)
    if mask_path is not None:
        kwargs['mask'] = np.load(mask_path)
    return kwargs


def run(request_path, result_path):
    with open(request_path, 'r', encoding='utf-8') as f:
        request = json.load(f)
    func_name = request['func']
    kwargs = _load_mask_if_referenced(request['kwargs'])
    if 'scan_size' in kwargs:
        kwargs['scan_size'] = tuple(kwargs['scan_size'])
    if 'det_shape' in kwargs:
        kwargs['det_shape'] = tuple(kwargs['det_shape'])
    if kwargs.get('roi_rect') is not None:
        kwargs['roi_rect'] = tuple(kwargs['roi_rect'])

    func = getattr(eb, func_name)
    result = func(**kwargs)

    if func_name in _ARRAY_RESULT_FUNCS:
        np.savez(result_path, array=np.asarray(result))
    elif func_name in _ROI_RESULT_FUNCS:
        arrays = {
            'scan_image': np.asarray(result.Roi_scan_image),
            'diffraction_pattern': np.asarray(result.Roi_diffraction_pattern),
        }
        try:
            arrays['roi_4d'] = np.asarray(result.get_4D())
        except RuntimeError:
            pass  # not computed with get_4d=True - fine, the parent handles this the same way
        np.savez(result_path, **arrays)
    else:
        raise ValueError(f'Unknown eventem_backend function: {func_name!r}')


if __name__ == '__main__':
    try:
        run(sys.argv[1], sys.argv[2])
        sys.exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
