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

#TODO add cupy if possible
def extract_3ded_mask_single_frame(fn, roi, mask_path, dtype, scanSize, i_c):
    try:
        # Parse input args
        scanSize = tuple(map(int, scanSize.strip("()").split(",")))
        # roi = tuple(map(int, roi.strip("()").split(",")))
        roi = [int(a) for a in str(roi)[1:-1].split()] # read as str of numpy array
        # roi = eval(roi) # read as str of list
        mask = np.load(mask_path)
        
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

def load_tpx3(fn, mask, scanSize, roi, dwellTime=50, **kwargs):
    repetitions = 1
    bitDepth=16
    if roi is None:
        x=0
        y=0 
        w=scanSize[0]
        y=scanSize[1]
    else:
        x, y, w, h = roi
        
    roi = eventem.Roi(repetitions=repetitions, extract_4D=True)
    roi.set_bitdepth(bitDepth)
    roi.nx = scanSize[0]
    roi.ny = scanSize[1]
    roi.set_file(fn)
    roi.set_roi(x=x, y=y, width=w, height=h)
    roi.set_dwell_time(dwellTime*1000)
    roi.run()
    # ROI_scan_image = np.asarray(roi.Roi_scan_image)
    # ROI_diffp = np.asarray(roi.Roi_diffraction_pattern).reshape(512, 512)
    s = np.asarray(roi.get_4D())
    s = s.reshape(-1, *s[-2:])
    dp = s[np.where(mask.flatten() == 1)[0]].sum(axis=0)
    dp = dp.compute()
    return dp

def load_hdf5(fn, roi, mask, scanSize=None, chunks=(8,512,512,512), **kwargs):
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
    
def load_hs(fn, roi, mask, chunks=(16,16,64,64), **kwargs):
    s = load(fn, lazy=True)
    try:
        s.rechunk(chunks)
    except:
        pass
    x,y,w,h = roi # TODO check
    # s = s.inav[x:x+w, y:y+h].data
    # mask = mask[]
    s = s.data
    s = s.reshape(-1, *s.shape[2:])
    dp = s[np.where(mask.flatten() == 1)[0]].sum(axis=0)
    dp = dp.compute()
    return dp   

def load_mib(fn, roi, mask, scanSize=None, chunks=(16,16,64,64), **kwargs):
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