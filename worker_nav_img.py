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


def calculate_nav_img_core(task):
    """Compute one navigation image from an already-typed task dict - the
    pure-computation half of calculate_nav_img_worker (below), factored out
    so worker_nav_img_batch.py's pooled batch driver can call it directly
    per task instead of via a fresh CLI-parsing subprocess each time.
    calculate_nav_img_worker (the single-file CLI entry point, still used
    directly by ROI on 4D/SAM2's one-off "Compute Virtual Image") is now a
    thin wrapper around this.

    Args:
        task: dict with keys 'fn' (path), 'dtype' (file extension),
            'scanSize' ((nx, ny)/[nx, ny] or None - resolved from the
            .hdf5 file's own metadata if still None here and dtype is
            '.hdf5'), 'dwellTime' (int, microseconds), 'detectors'
            (optional list of {'center': [x, y] or (x, y), 'r_in',
            'r_out'} dicts - a virtual detector mask, or None/absent for
            none), 'fn_pattern' (optional smart-scan pattern-file path),
            'det_shape' (optional (det_x, det_y)/[det_x, det_y], defaults
            to (512, 512)), 'mode' ('sum' or 'variance', defaults to
            'sum').

    Returns:
        numpy.ndarray - the computed navigation image.
    """
    fn = task['fn']
    dtype = task['dtype']
    scanSize = task.get('scanSize')
    scanSize = tuple(scanSize) if scanSize is not None else None
    if scanSize is None and dtype == '.hdf5':
        scanSize = _read_scansize_hdf5(fn)
    dwellTime = int(task['dwellTime'])
    det_shape = task.get('det_shape')
    det_shape = tuple(det_shape) if det_shape is not None else (512, 512)
    fn_pattern = task.get('fn_pattern') or None
    mode = task.get('mode') or 'sum'
    detectors = task.get('detectors')

    # n_threads=1: eventem auto-sizes its own internal thread pool to the
    # whole machine per instance unless told otherwise - fine for a single
    # call, but this runs as one of several concurrent ProcessPoolExecutor
    # workers within the same batch driver (or, for calculate_nav_img_worker
    # below, one of several concurrent subprocesses), each of which would
    # otherwise also try to claim the whole machine's threads for itself.
    # Pinning each worker to 1 internal thread makes total concurrency
    # match what the user actually configured (spinbox_cpuCores).
    if detectors:
        detectors = [dict(d, center=tuple(d['center'])) for d in detectors]
        result = io.calculate_nav_img_masked(fn, dtype=dtype, scanSize=scanSize,
                                             dwellTime=dwellTime, detectors=detectors,
                                             n_threads=1, fn_pattern=fn_pattern,
                                             det_shape=det_shape, mode=mode)
    else:
        result = io.calculate_nav_img(fn, dtype=dtype, scanSize=scanSize,
                                      dwellTime=dwellTime, n_threads=1,
                                      fn_pattern=fn_pattern, det_shape=det_shape, mode=mode)
    return result


def calculate_nav_img_worker(fn, dtype, scanSize, dwellTime, i_index, temp_dir,
                             detectors_json=None, fn_pattern=None, det_shape=None, mode='sum'):
    """Compute one navigation image, save it to `temp_dir` as a .npy file, and
    print the saved path to stdout - instead of the array itself, base64+
    pickle-encoded. Transferring a multi-MB encoded array through the
    stdout pipe for every file in a batch (with many workers doing this
    concurrently) was making "Calculate All" progressively slower as the
    batch went on; a short path string avoids that IPC cost entirely. The
    parent (tab_create_navSignal.py) loads the array back with `np.load`
    and removes the file once it has.

    Single-file CLI entry point - still used directly (not via the pooled
    batch driver, worker_nav_img_batch.py) by ROI on 4D/SAM2's one-off
    "Compute Virtual Image", which has no benefit from pooling a single
    task. "Calculate All" (tab_create_navSignal.py) uses the batch driver
    instead, which calls calculate_nav_img_core directly per task.

    Called as a subprocess by `tab_create_navSignal.py`/`tab_roi_4d.py`/
    `tab_sam2.py`. Reads arguments from the command line (all as strings),
    parses them into the same task-dict shape calculate_nav_img_core
    expects, computes the navigation image, and exits 0 on success or 1 on
    error.

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
        mode: 'sum' (default) or 'variance' - see
            EDyssey.io_utils.nav_image.calculate_nav_img_hdf5's docstring.
    """
    try:
        fn_pattern = None if fn_pattern in (None, '', 'None') else fn_pattern
        det_shape = (tuple(map(int, det_shape.strip("()").split(",")))
                    if det_shape not in (None, '', 'None') else (512, 512))
        scanSize = None if scanSize == 'None' else tuple(map(int, scanSize.strip("()").split(",")))
        dwellTime = int(dwellTime)
        i_index = int(i_index)
        mode = 'sum' if mode in (None, '', 'None') else mode
        detectors = None
        if detectors_json not in (None, 'None'):
            detectors = json.loads(detectors_json)
            for d in detectors:
                d['center'] = tuple(d['center'])

        task = {'fn': fn, 'dtype': dtype, 'scanSize': scanSize, 'dwellTime': dwellTime,
               'detectors': detectors, 'fn_pattern': fn_pattern, 'det_shape': det_shape,
               'mode': mode}
        result = calculate_nav_img_core(task)

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
