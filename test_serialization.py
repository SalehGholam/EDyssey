# -*- coding: utf-8 -*-
"""
Created on Fri May  9 17:55:36 2025

@author: sgholam
"""

import numpy as np
import dask.array as da
import h5py
import os
from glob import glob
from tqdm import tqdm
from dask.distributed import Client, LocalCluster, as_completed
from tqdm.dask import TqdmCallback
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import cupy as cp
import time
#%%
# @delayed
# def extract_dp(fn, mask, i_c=0):
def extract_dp(args):
    fn, mask, i_c = args
    try:
        shape = (512, 512, 512, 512)
        with h5py.File(fn, 'r') as f:
            if len(f['4D'].shape) == 4:
                s = da.from_array(f['4D'], chunks=(8, 512, 512, 512))
            elif len(f['4D'].shape) == 1:
                s = da.from_array(f['4D'], chunks=(8 * 512**3,))
                s = s.reshape(shape)
            s = s.reshape(-1, *s.shape[2:])
            mask_idx = np.where(mask.flatten() == 1)[0]
            # dp = s[mask_idx].sum(axis=0).compute()
            dp = s[mask_idx].sum(axis=0).compute()
        return dp, i_c
    except Exception as e:
        print(f"Error: {e} in file {fn}")
        return None, i_c
#%% GPU
def extract_dp_cupy(fn, mask_np):
    def to_cupy_block(x):
        return cp.asarray(x)

    # Open with h5py to get shape info (without reading all data)
    # with h5py.File(fn, 'r') as f:
    #     original_shape = f['4D'].shape
    #     dtype = f['4D'].dtype
    
    # # Suppose you know the target 4D shape (for example): (t, z, y, x)
    target_shape = (512,512,512,512)  # adjust this to your expected shape
    # assert cp.prod(cp.array(target_shape)) == original_shape[0], "Mismatch in reshaping dimensions"
    
    # Create the dask array with CuPy backend
    # darr = da.from_array(h5py.File(fn, 'r')['4D'], chunks=(8*512**3), asarray=cp.asarray)
    # darr = da.from_array(h5py.File(fn, 'r')['4D'], chunks=(16*16*64*64))
    # darr = darr.reshape(target_shape)
    
    darr = da.from_zarr(fn, chunks=(16,16,64,64))
    darr = darr.map_blocks(to_cupy_block)
    
    mask = cp.asarray(mask_np)
    # Broadcast the mask to the full shape
    expanded_mask = mask[:, :, None, None]
    
    # Apply the mask
    masked_array = da.where(expanded_mask, darr, 0)
    
    # Step 4: Sum over the first two axes
    with ProgressBar():
        result = masked_array.sum(axis=(0, 1)).compute()
    
    # `result` is a CuPy array on the GPU with shape (256, 256)
    result_cpu = cp.asnumpy(result)
    return result_cpu



#%% Main script
if __name__ == '__main__':
    path_4d = r'C:\My Files\Microscope Data\Tecnai\24-09-11_TiO2_200 kV\S2\4D Signals'
    path_save = r'C:\My Files\Microscope Data\Tecnai\24-09-11_TiO2_200 kV\S2\5DED Analysis\2025-05-03__15-49-37\roi No 3\extract 2\frames'
    path_mask = r'C:\My Files\Microscope Data\Tecnai\24-09-11_TiO2_200 kV\S2\5DED Analysis\2025-05-03__15-49-37\roi No 3'
    
# =============================================================================
#     end = 10
#     
#     fn_mask = [fn for fn in os.listdir(path_mask) if 'segmentation masks' in fn][0]
#     masks = np.load(os.path.join(path_mask, fn_mask))[:end]
#     masks_dilated = masks
#     fns_4d = glob(os.path.join(path_4d, '*.hdf5'))[:end]
# =============================================================================
    
    fn_mask = [fn for fn in os.listdir(path_mask) if 'segmentation masks' in fn][0]
    masks = np.load(os.path.join(path_mask, fn_mask))
    masks_dilated = masks
    fns_4d = glob(os.path.join(path_4d, '*.hdf5'))

    from multiprocessing import Pool
    if __name__ == '__main__':
        count = np.arange(0, len(masks_dilated), dtype=int)
        with Pool(processes=5) as pool:
            # dps = pool.map(extract_dp, zip(fns_4d, masks_dilated))
            dps = list(tqdm(pool.imap(extract_dp, zip(fns_4d, masks_dilated, count)), total=len(fns_4d)))
    #%%
    import hyperspy.api as hs
    # dps = np.array(results)
    # s = hs.signals.Signal2D(dps)
    
    imgs = np.zeros((len(dps),512,512), dtype='uint32')
    for img, i_c in dps:
        imgs[i_c] = img
    imgs = np.array(imgs)
    
    # s = hs.signals.Signal2D(imgs)
    # s.plot(norm='symlog', cmap='viridis', vmin=1)
    #%%
    import tifffile
    for i, img in enumerate(imgs):
        tifffile.imwrite(os.path.join(path_save, f'{i:04d}.tif'), img)
