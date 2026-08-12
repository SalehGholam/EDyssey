'''
@author: Saleh Gholam & Arno Annys

Help:
    1. Set the parameters for the detector and 4D-STEM data inside "process_file" function
    2. Pass the directory where the .tpx3 files are as the "path_in" argument
    3. Optionally pass --path-out for the directory converted files will be written to
       (defaults to path_in)
    4. Optionally pass --n-processes (depends on your PC specs - better to first
       test with a small dataset)
'''

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

def process_file(in_file,out_file):
    sleep(0.1)
    # tic = perf_counter()

    #### SET PARAMETERS #####
    det_size = 512
    det_bin = 1
    scan_bin = 1  # noqa: F841 - not wired up to fourD yet, kept as a documented parameter slot
    scan_size = (512,512)
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
    fourD.detector_size = det_size
    fourD.det_bin = det_bin
    fourD.chunksize = chunksize
    fourD.nx = scan_size[0]
    fourD.ny = scan_size[1]
    fourD.allocate_chunk() # allocate the memory for the 4D array
    fourD.init_4D_file() # initialize the hdf5 file writing
    fourD.set_dwell_time(dwellTime)
    fourD.run() # run the processing
    fourD.save_dose_image() # save the dose image
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
        description='Convert .tpx3 files to .hdf5 across a multiprocessing pool.')
    parser.add_argument('path_in', help='Directory containing the .tpx3 files.')
    parser.add_argument('--path-out', default=None,
                         help='Directory to write converted files to (default: same as path_in).')
    parser.add_argument('--n-processes', type=int, default=1,
                         help='Number of worker processes (default: 1).')
    args = parser.parse_args()

    path_in = args.path_in
    path_out = args.path_out if args.path_out is not None else path_in
    in_files = glob(os.path.join(path_in, '*.tpx3'))

    #### delete existing files
    # in_files = delete_existing(in_files, path_out)
    tic = perf_counter()
    out_files = [os.path.split(fn)[1] for fn in in_files]
    out_files = [os.path.join(path_out, os.path.splitext(fn)[0]) for fn in out_files]
    with Pool(args.n_processes) as p:
        p.starmap(process_file, zip(in_files,out_files))
    toc = perf_counter()
    print(f'Duration: {(toc-tic)/60:0.2f} min')
