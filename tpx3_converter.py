# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:35:44 2026

@author: Saleh Gholam & Arno Annys
"""

import argparse
import sys
import os
file_path = os.path.abspath(__file__)
fld_path = os.path.dirname(file_path)
sys.path.append(os.path.join(fld_path, 'py4DTomo', 'io_utils')) # path to the evetem
import eventem
from multiprocessing import Pool
from glob import glob
import os
from time import perf_counter, sleep
import numpy as np

def process_file(in_file,out_file, fn_pattern=None):
    sleep(0.1)
    # tic = perf_counter()

    #### SET PARAMETERS #####
    det_size = (512, 512)
    det_bin = 1  # noqa: F841 - not wired up to fourD yet, kept as a documented parameter slot
    scan_bin = 1  # noqa: F841 - not wired up to fourD yet, kept as a documented parameter slot
    scan_size = (30872,1)
    dwellTime = 200 # usec
    chunksize = 8
    compression_factor =  4 # 1 is least compression, 9 is most compression
    bitdepth = 8
    ##########################

    dwellTime *= 1000
    # print(f"RAM requirement of 4D chunk: {(scan_size//scan_bin)*(chunksize//scan_bin) * (det_size//det_bin)**2 * (bitdepth/8) * 2 /1000000000} GB per process") #also ram used by raw data that is processed

    if bitdepth == 8:
        FourD = eventem.FourD8
    elif bitdepth == 16:
        FourD = eventem.FourD16
    elif bitdepth == 32:
        FourD = eventem.FourD32
    fourD = FourD(output_filename=out_file, repetitions=1, bitdepth=bitdepth,
                  compression_factor=compression_factor) # create a new instance of the FourD class with the output file name
    fourD.set_file(in_file) # set the input file
    if fn_pattern:
        fourD.set_pattern_file(fn_pattern)
    fourD.detector_size_x = det_size[0]
    fourD.detector_size_y = det_size[1]
    fourD.chunksize = chunksize
    fourD.nx = scan_size[0]
    fourD.ny = scan_size[1]
    fourD.allocate_chunk() # allocate the memory for the 4D array
    fourD.init_4D_file() # initialize the hdf5 file writing
    fourD.set_dwell_time(dwellTime)
    fourD.run() # run the processing
    # fourD.save_dose_image() # save the dose image
    dose_image = np.array(fourD.Dose_image).reshape(scan_size)
    fn_dose = os.path.join(out_file + '_nav')
    np.save(fn_dose, dose_image)

    # toc = perf_counter()
    # print(f'Duration: {(toc-tic)/60:0.2f} min')

def delete_existing(fns_tpx3, path_hdf5):
    fns_tpx3_new = []
    # fns_hdf5_new = []
    fns_tpx3_2 = [os.path.splitext(os.path.split(fn)[1])[0] for fn in fns_tpx3]
    fns_hdf5 = [fn[:-5] for fn in os.listdir(path_hdf5)]
    for i, fn in enumerate(fns_tpx3):
        if fns_tpx3_2[i] not in fns_hdf5:
            fns_tpx3_new.append(fn)
            # fns_hdf5_new.append(os.path.join(path_hdf5, fns_tpx3_2[i]))
    # return fns_tpx3_new, fns_hdf5_new
    return fns_tpx3_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert .tpx3 files to .hdf5 (single-process, smart-scan capable).')
    parser.add_argument('path_in', help='Directory containing the .tpx3 files (and, '
                         'for a smart scan, their matching pattern .txt files).')
    parser.add_argument('--path-out', default=None,
                         help='Directory to write converted files to (default: same as path_in).')
    parser.add_argument('--smart-scanned', dest='smart_scanned', action='store_true', default=True,
                         help='Treat the acquisition as smart-scanned (default).')
    parser.add_argument('--full-scan', dest='smart_scanned', action='store_false',
                         help='Treat the acquisition as a normal dense raster scan.')
    args = parser.parse_args()

    path_in = args.path_in
    path_out = args.path_out if args.path_out is not None else path_in
    smartScanned = args.smart_scanned

    in_files = glob(os.path.join(path_in, '*.tpx3'))
    if smartScanned:
        fns_pat = glob(os.path.join(path_in, '*.txt'))
        fns_pat = [fn for fn in fns_pat if 'comment.txt' not in fn]
        fns_pat.sort()

    #### delete existing files
    # in_files = delete_existing(in_files, path_out)
    tic = perf_counter()
    out_files = [os.path.split(fn)[1] for fn in in_files]
    out_files = [os.path.join(path_out, os.path.splitext(fn)[0]) for fn in out_files]
    if smartScanned:
        for fn_in, fn_out, fn_pat in zip(in_files,out_files, fns_pat):
            process_file(fn_in, fn_out, fn_pat)

    else:
        for fn_in, fn_out in zip(in_files,out_files):
            process_file(fn_in, fn_out)
    toc = perf_counter()
    print(f'Duration: {(toc-tic)/60:0.2f} min')
