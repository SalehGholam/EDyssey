import sys
sys.path.append(r'C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\py5DED\py4DTomo\io_utils') # path to the pyLiveProcessing.pyd
# import pyLiveProcessing as pyLP
import eventem as pyLP
from multiprocessing import Pool
from glob import glob
import os
from time import perf_counter, sleep

def process_file(in_file,out_file):
    sleep(0.1)
    tic = perf_counter()
    det_size = 128
    det_bin = 1
    scan_bin = 1
    scan_size = 512
    dwellTime = 10 # usec
    dwellTime *= 1000
    chunksize = 8
    compression_factor =  4 #1 is least compression, 9 is most compression
    bitdepth = 8

    print(f"RAM requirement of 4D chunk: {(scan_size//scan_bin)*(chunksize//scan_bin) * (det_size//det_bin)**2 * (bitdepth/8) * 2 /1000000000} GB per process") #also ram used by raw data that is processed

    if bitdepth == 8:
        FourD = pyLP.FourD8
    elif bitdepth == 16:
        FourD = pyLP.FourD16
    elif bitdepth == 32:
        FourD = pyLP.FourD32
    fourD = FourD(output_filename = out_file,repetitions = 1, bitdepth=bitdepth, compression_factor=compression_factor) # create a new instance of the FourD class with the output file name
    fourD.set_file(in_file) # set the input file
    fourD.detector_size = det_size
    fourD.det_bin = det_bin 
    fourD.chunksize = chunksize 
    fourD.nx = scan_size
    fourD.ny = scan_size
    fourD.allocate_chunk() # allocate the memory for the 4D array
    fourD.init_4D_file() # initialize the hdf5 file writing
    fourD.set_dwell_time(dwellTime)
    fourD.run() # run the processing
    fourD.save_dose_image() # save the dose image
    toc = perf_counter()
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
#%%
if __name__ == '__main__':
    path_in = r'Z:\emattecnai\Saleh_Tecnai\250305_GoldCalib\cl_300_c250\2025-03-05_15-22-02'
    # path_out = path_in
    path_out = r'Z:\emattecnai\Saleh_Tecnai\250305_GoldCalib\cl_300_c250\2025-03-05_15-22-02'
    in_files = glob(os.path.join(path_in, '*.tpx3'))
    
    #### cutting files
    # in_files = in_files[2:3]
    
    #### delete existing files
    # in_files = delete_existing(in_files, path_out)
    tic = perf_counter()
    out_files = [os.path.split(fn)[1] for fn in in_files]
    out_files = [os.path.join(path_out, os.path.splitext(fn)[0]) for fn in out_files]
    N_processes = 1
    with Pool(N_processes) as p:
        p.starmap(process_file, zip(in_files,out_files))
    toc = perf_counter()
    print(f'Duration: {(toc-tic)/60:0.2f} min')