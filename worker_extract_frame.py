# -*- coding: utf-8 -*-
"""
Created on Tue May  6 22:47:49 2025

@author: sgholam
"""

import sys
import dask.array as da
import h5py
from dask.distributed import Client, LocalCluster
import os
from dask import config
import numpy as np
from dask.diagnostics import ProgressBar
import base64
import pickle
file_path = os.path.abspath(__file__)
main_path = os.path.dirname(file_path)
eventem_path = os.path.join(main_path, 'py4DTomo', 'io_utils')
sys.path.append(eventem_path)
# os.chdir()
import eventem
from hyperspy.api import signals, load
# from py4DTomo.io_utils import load_signal
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="distributed")
#%%
#TODO add cupy if possible
def extract_3ded_mask_single_frame(fn, roi, mask_path, dtype, scanSize, i_c):
    """Extract a single 3DED diffraction pattern from one 4D-STEM frame using a binary mask.

    Subprocess entry point called by `tab_sam2.py`. Deserialises CLI arguments, loads the
    per-frame mask from `mask_path`, sums diffraction patterns at mask-True pixels, and
    prints the result as a base64+pickle-encoded `(dp, i_c)` tuple.

    Args:
        fn: Path to the 4D-STEM file.
        roi: Scan-space crop as a string representation of a numpy array, e.g. `'[x y w h]'`.
        mask_path: Path to a `.npy` file containing the binary 2-D mask.
        dtype: File extension string (e.g. `.tpx3`, `.hdf5`).
        scanSize: Scan dimensions as `'(nx, ny)'` string.
        i_c: Frame index within the extraction batch, as a string.
    """
    try:
        # Parse input args
        scanSize = tuple(map(int, scanSize.strip("()").split(",")))
        # roi = tuple(map(int, roi.strip("()").split(",")))
        roi = [int(a) for a in str(roi)[1:-1].split()] # read as str of numpy array
        # roi = eval(roi) # read as str of list
        mask = np.load(mask_path)
        # mask = mask_path
        
        dp = load_dp(fn, roi=roi, mask=mask, scanSize=scanSize)
        # Serialize result to base64 string and print it to stdout
        serialized = base64.b64encode(pickle.dumps((dp, i_c))).decode('utf-8')
        print(serialized)  # <- this goes to QProcess output
        
        sys.stdout.flush()  # Ensure it's pushed
        sys.exit(0)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)  # fail code
        
def load_dp(fn, **kwargs):
    """Dispatch diffraction pattern loading to the format-specific function.

    Args:
        fn: Path to the 4D-STEM file.
        **kwargs: Passed through to the format-specific loader (roi, mask, scanSize, etc.).

    Returns:
        numpy.ndarray of shape (det_y, det_x) with the summed diffraction pattern.
    """
    try:
        dtype = kwargs.get('dtype')
    except:
        dtype = None
    if dtype is None:
        dtype = os.path.splitext(fn)[1]
    
    if dtype == '.tpx3':
        result = load_tpx3(fn, **kwargs)        
    elif dtype == '.hdf5':
        result = load_hdf5(fn, **kwargs)        
    elif dtype in ['.zspy', '.hspy']:
        result = load_hs(fn, **kwargs)
    elif dtype == '.mib':
        result = load_mib(fn, **kwargs)
    return result

def load_tpx3(fn, mask, scanSize, roi, dwellTime=1, **kwargs):
    """Load a .tpx3 file, apply an ROI crop and mask, and return the summed diffraction pattern.

    Args:
        fn: Path to the .tpx3 file.
        mask: 2-D boolean array matching the scan dimensions.
        scanSize: (nx, ny) scan dimensions.
        roi: (x, y, w, h) scan-space crop or None for the full scan.
        dwellTime: Dwell time in microseconds.

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    repetitions = 1
    bitDepth=16
    if roi is None:
        x=0
        y=0 
        w=scanSize[0]
        h=scanSize[1]
    else:
        # y, x, h, w = roi
        x, y, w, h = roi
        
    s_roi = eventem.Roi(repetitions=repetitions, extract_4D=True)
    s_roi.set_bitdepth(bitDepth)
    s_roi.nx = scanSize[0]
    s_roi.ny = scanSize[1]
    s_roi.set_file(fn)
    s_roi.set_roi(x=x, y=y, width=w, height=h)
    s_roi.set_dwell_time(dwellTime*1000)
    s_roi.run()
    # ROI_scan_image = np.asarray(roi.Roi_scan_image)
    # ROI_diffp = np.asarray(roi.Roi_diffraction_pattern).reshape(512, 512)
    s = np.asarray(s_roi.get_4D())

    mask_crop = mask[y:y+h, x:x+w]
    s = s.reshape(-1, *s.shape[-2:])
    dp = s[np.where(mask_crop.flatten() == 1)[0]].sum(axis=0)
    # dp = dp.compute()
    return dp

def load_hdf5(fn, roi, mask, scanSize=None, chunks=(8, 512, 512, 512), **kwargs):
    """Load an .hdf5 file and sum diffraction patterns at mask-True scan pixels.

    Args:
        fn: Path to the .hdf5 file.
        roi: (x, y, w, h) scan-space crop (currently unused in this loader).
        mask: 2-D boolean array matching the full scan dimensions.
        scanSize: (nx, ny) scan dimensions for reshaping flat storage.
        chunks: Dask chunk shape.

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    with h5py.File(fn, 'r') as f:
        shape = tuple(f['shape'][:])
        if len(f['4D'].shape) == 4:
            s = da.from_array(f['4D'], chunks=chunks)
        elif len(f['4D'].shape) == 1:
            s = da.from_array(f['4D'], chunks=np.prod(chunks))
            s = s.reshape(shape)
        s = s.reshape(-1, *s.shape[2:])
        mask_idx = np.where(mask.flatten() == 1)[0]
        dp = s[mask_idx].sum(axis=0).compute()
    return dp
    
def load_hs(fn, roi, mask, chunks=(16, 16, 64, 64), **kwargs):
    """Load a .hspy/.zspy file and sum diffraction patterns at mask-True scan pixels.

    Args:
        fn: Path to the HyperSpy signal file.
        roi: (y, x, h, w) scan-space crop (applied to the mask transpose).
        mask: 2-D boolean array.
        chunks: Dask rechunk shape.

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    s = load(fn, lazy=True)
    try:
        s.rechunk(chunks)
    except:
        pass
    y,x,h,w = roi # TODO check
    # s = s.inav[x:x+w, y:y+h].data
    # mask = mask[]
    s = s.data
    s = s.reshape(-1, *s.shape[2:])
    dp = s[np.where(mask.T.flatten() == 1)[0]].sum(axis=0)
    dp = dp.compute()
    return dp   

def load_mib(fn, roi, mask, scanSize=None, chunks=(16, 16, 64, 64), **kwargs):
    """Load a .mib file and sum diffraction patterns at mask-True scan pixels.

    Args:
        fn: Path to the .mib file.
        roi: (x, y, w, h) scan-space crop applied via HyperSpy `inav`.
        mask: 2-D boolean array.
        scanSize: (nx, ny) scan dimensions. Reads from `default.hdr` if None.
        chunks: Dask rechunk shape.

    Returns:
        numpy.ndarray of shape (det_y, det_x).
    """
    s = load(fn, lazy=True)
    if len(s.data.shape) == 3:
        if scanSize is None:
            det_shape = s.data[-1]
            fld = os.path.split(fn)[0]
            fn_hdr = os.path.join(fld, 'default.hdr')
            scanSize = get_scan_size_mib_hdr(fn_hdr)
        s.reshape(scanSize[0], scanSize[1], det_shape, det_shape)
    s.rechunk(chunks)
    x,y,w,h = roi
    s = s.inav[x:x+w, y:y+h].data
    s = s.reshape(-1, *s.shape[2:])
    dp = s[np.where(mask.flatten() == 1)[0]].sum(axis=0)
    dp = dp.compute() 
    return dp

def get_scan_size_mib_hdr(fn_hdr):
    """Parse scan dimensions from a Merlin .hdr header file.

    Args:
        fn_hdr: Path to the `.hdr` file (typically `default.hdr` in the .mib folder).

    Returns:
        Tuple (nx, ny) where nx = frames per trigger and ny = total frames / nx.
    """
    with open(fn_hdr, 'r') as file:
        hdr = file.readlines()
    fpt = [line for line in hdr if 'Frames per Trigger' in line][0]
    fpt = fpt.split(':')[1]
    fpt = int(fpt)
    
    framesAcq = [line for line in hdr if 'Frames in Acquisition' in line][0]
    framesAcq = framesAcq.split(':')[1]
    framesAcq = int(framesAcq)
    scanSize = (fpt, int(framesAcq/fpt))
    return scanSize
#%%
if __name__ == "__main__":
    extract_3ded_mask_single_frame(*sys.argv[1:])
    
# =============================================================================
#     # for debugging
#     path_mask = r'D:\0_5ded test\Ayush\5DED Analysis\2026-03-26__17-11-25\roi No 1\output_mask.npy'
#     mask = np.load(path_mask)[-1]
#     args = ['D:/0_5ded test/Ayush/4d signals\\raw_0652_-37.75_000000.tpx3', 
#              '[314 117  23  23]', 
#              mask, 
#              '.tpx3', 
#              '(512, 512)', 
#              '(1, 9)']
#     extract_3ded_mask_single_frame(*args)
# =============================================================================
