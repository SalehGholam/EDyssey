# -*- coding: utf-8 -*-
import sys
import os
import base64
import pickle

file_path = os.path.abspath(__file__)
main_path = os.path.dirname(file_path)
eventem_path = os.path.join(main_path, 'py4DTomo', 'io_utils')
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


def calculate_nav_img_worker(fn, dtype, scanSize, dwellTime, i_index,
                             r_in=None, r_out=None, center=None):
    """Compute one navigation image and write the result to stdout as base64+pickle.

    Called as a subprocess by `tab_create_navSignal.py`. Reads arguments from the
    command line (all as strings), computes the navigation image via `io.calculate_nav_img`
    (or `io.calculate_nav_img_masked` when a virtual detector mask is given), serialises
    `(result, i_index)` with pickle, encodes with base64, prints to stdout, and exits 0 on
    success or 1 on error.

    Args:
        fn: Absolute path to the 4D-STEM file.
        dtype: File extension (e.g. `.hdf5`, `.tpx3`, `.hspy`).
        scanSize: Scan dimensions as string `'(nx, ny)'` or `'None'`.
        dwellTime: Dwell time in microseconds as a string.
        i_index: Position of this file in the overall file list, as a string.
        r_in: Optional inner radius (pixels) of the virtual detector mask, as a string.
        r_out: Optional outer radius (pixels) of the virtual detector mask, as a string.
        center: Optional `'(x, y)'` center of the virtual detector mask, as a string.
    """
    try:
        if scanSize == 'None':
            if dtype == '.hdf5':
                scanSize = _read_scansize_hdf5(fn)
            else:
                scanSize = None
        else:
            scanSize = tuple(map(int, scanSize.strip("()").split(",")))
        dwellTime = int(dwellTime)
        i_index = int(i_index)

        if r_in is not None:
            r_in = float(r_in)
            r_out = float(r_out)
            center = tuple(map(float, center.strip("()").split(",")))
            result = io.calculate_nav_img_masked(fn, dtype=dtype, scanSize=scanSize,
                                                 dwellTime=dwellTime, r_in=r_in,
                                                 r_out=r_out, center=center)
        else:
            result = io.calculate_nav_img(fn, dtype=dtype, scanSize=scanSize, dwellTime=dwellTime)

        serialized = base64.b64encode(pickle.dumps((result, i_index))).decode('utf-8')
        print(serialized)
        sys.stdout.flush()
        sys.exit(0)

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    calculate_nav_img_worker(*sys.argv[1:])
