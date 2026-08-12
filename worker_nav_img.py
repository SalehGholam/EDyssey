# -*- coding: utf-8 -*-
import sys
import os
import json
import numpy as np

file_path = os.path.abspath(__file__)
main_path = os.path.dirname(file_path)
eventem_path = os.path.join(main_path, 'EDyssey', 'io_utils')
sys.path.append(eventem_path)
os.chdir(main_path)

import io_utils_ui as io


def _read_scansize_hdf5(fn):
    """Read scan dimensions (nx, ny) directly from an .hdf5 file's metadata.

    Handles both 4-D `f['4D']` arrays and flat 1-D storage where shape is in `f['shape']`.

    Args:
        fn: Path to the .hdf5 file.

    Returns:
        Tuple (nx, ny) as integers.
    """
    import h5py
    with h5py.File(fn, 'r') as f:
        if len(f['4D'].shape) == 1:
            return tuple(int(x) for x in f['shape'][:][:2])
        else:
            return tuple(int(x) for x in f['4D'].shape[:2])


def calculate_nav_img_worker(fn, dtype, scanSize, dwellTime, i_index, temp_dir,
                             detectors_json=None, fn_pattern=None, det_shape=None):
    """Compute one navigation image, save it to `temp_dir` as a .npy file, and
    print the saved path to stdout - instead of the array itself, base64+
    pickle-encoded. Transferring a multi-MB encoded array through the
    stdout pipe for every file in a batch (with many workers doing this
    concurrently) was making "Calculate All" progressively slower as the
    batch went on; a short path string avoids that IPC cost entirely. The
    parent (tab_create_navSignal.py) loads the array back with `np.load`
    and removes the file once it has.

    Called as a subprocess by `tab_create_navSignal.py`. Reads arguments from the
    command line (all as strings), computes the navigation image via `io.calculate_nav_img`
    (or `io.calculate_nav_img_masked` when a virtual detector mask is given), and exits 0 on
    success or 1 on error.

    Args:
        fn: Absolute path to the 4D-STEM file.
        dtype: File extension (e.g. `.hdf5`, `.tpx3`, `.hspy`).
        scanSize: Scan dimensions as string `'(nx, ny)'` or `'None'`.
        dwellTime: Dwell time in microseconds as a string.
        i_index: Position of this file in the overall file list, as a string.
        temp_dir: Directory (already created by the parent) to save the
            result .npy file into.
        detectors_json: Optional JSON-encoded list of {'center': [x, y],
            'r_in', 'r_out'} dicts describing the virtual detector mask(s)
            to use, as a string (or 'None'/None for no mask) - see
            Tab_Create_NavSignal.get_active_detectors().
        fn_pattern: Optional path to a smart-scan pattern file for `fn` (empty
            string/'None' for a normal dense file) - see `loaders.load_tpx3`/
            `loaders._load_mib_smart_scan`.
    """
    try:
        fn_pattern = None if fn_pattern in (None, '', 'None') else fn_pattern
        det_shape = (tuple(map(int, det_shape.strip("()").split(",")))
                    if det_shape not in (None, '', 'None') else (512, 512))
        if scanSize == 'None':
            if dtype == '.hdf5':
                scanSize = _read_scansize_hdf5(fn)
            else:
                scanSize = None
        else:
            scanSize = tuple(map(int, scanSize.strip("()").split(",")))
        dwellTime = int(dwellTime)
        i_index = int(i_index)

        # Same functions "Test File" calls in-thread - eventem's native
        # progress output (tpx3 loading) lands on stderr, which QProcess
        # already keeps separate from stdout (this script's IPC channel for
        # the result below), so nothing special needs to happen here; the
        # parent process reads/handles stderr on its own side.
        #
        # n_threads=1: eventem auto-sizes its own internal thread pool to
        # the whole machine per instance unless told otherwise - fine for a
        # single call, but this script runs as one of up to spinbox_cpuCores
        # *concurrent* processes, each of which would otherwise also try to
        # claim the whole machine's threads for itself. That oversubscription
        # (N processes x full-core-count threads each) is what was causing
        # per-file throughput to collapse as more workers piled up
        # concurrently - pinning each process to 1 internal thread makes
        # total concurrency match what the user actually configured.
        if detectors_json not in (None, 'None'):
            detectors = json.loads(detectors_json)
            for d in detectors:
                d['center'] = tuple(d['center'])
            result = io.calculate_nav_img_masked(fn, dtype=dtype, scanSize=scanSize,
                                                 dwellTime=dwellTime, detectors=detectors,
                                                 n_threads=1, fn_pattern=fn_pattern,
                                                 det_shape=det_shape)
        else:
            result = io.calculate_nav_img(fn, dtype=dtype, scanSize=scanSize,
                                          dwellTime=dwellTime, n_threads=1,
                                          fn_pattern=fn_pattern, det_shape=det_shape)

        fn_out = os.path.join(temp_dir, f'{i_index}.npy')
        np.save(fn_out, result)
        print(fn_out)
        sys.stdout.flush()
        sys.exit(0)

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    calculate_nav_img_worker(*sys.argv[1:])
