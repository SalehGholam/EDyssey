# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:12:07 2025

@author: sgholam
"""

import os
from glob import glob
import h5py
import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
import py4DTomo.io_utils as io
import dask.config as daco
import matplotlib.pyplot as plt
import hyperspy.api as hs

path = r'C:\My Files\Microscope Data\Tecnai\24-09-11_TiO2_200 kV\S1\4D signals'
fns = glob(os.path.join(path, '*.hdf5'))

# fns_4th = fns[::4]
#%%
def load_dataset(f, scanSize=(512,512)):
    scanSize = tuple(f['shape'])[:2]
    lines = 4096 / scanSize[0]
    chunks=(lines, scanSize[0], 512, 512)
    chunks1D = np.prod(chunks)
    s = da.from_array(f['4D'], chunks=chunks1D) # only 1D hdf5 files
# =============================================================================
#         if len(s.shape) == 1: # for 1D array
#             with daco.set(**{'array.slicing.split_large_chunks': False}):
#                 s = s.reshape(scanSize[0], scanSize[1], 512,512)
# =============================================================================
    return s

fn = fns[0]
scanSize = (512,512)

with h5py.File(fn, 'r') as f:
    s1 = load_dataset(f)
    s1 = s1.reshape(scanSize[0], scanSize[1], 512,512)
    navImg = s1.sum(axis=(2,3))
    with ProgressBar():
        navImg = navImg.compute()



fig, ax = plt.subplots()
ax.imshow(navImg)
#%%
step = 0.05
integ_angle = 0.50 # deg/file
fnNo = len(fns)
chunk_size = int(np.floor(integ_angle / step))
rem_files = int(np.ceil(fnNo % chunk_size))
fns_cut = fns[:-1*rem_files]
counter = np.arange(len(fns_cut)).reshape(len(fns_cut)//chunk_size, chunk_size)
#%%
# test
# counter = counter[:4]

path_save = r'C:\My Files\Microscope Data\Tecnai\24-09-11_TiO2_200 kV\S1\combined_4D'
for i_c, item in enumerate(counter):
    arr_new = da.zeros((512**4), chunks=(512**3), dtype='uint8')
    files = []
    for i_f in item:
        fn = fns[i_f]
        f = h5py.File(fn, 'r')
        files.append(f)
        arr_new += da.from_array(f['4D'], chunks=512**3)
    s = hs.signals.Signal2D(arr_new.reshape(512,512,512,512))
    s = s.as_lazy()
    fn_save = f'{i_c:04d}.zspy'
    s.save(os.path.join(path_save, fn_save), chunks=(16,16,512,512), overwrite=True)
    _ = [f.close() for f in files]
