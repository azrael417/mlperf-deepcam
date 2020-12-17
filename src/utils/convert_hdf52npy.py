import os
import glob
import h5py as h5
import numpy as np
import math
import argparse as ap
from mpi4py import MPI

def filter_func(item, lst):
    item = os.path.basename(item).replace(".h5", ".npy")
    return item not in lst


def main(args):

    # get rank
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # get input files
    inputfiles_all = glob.glob(os.path.join(args.input_directory, "*.h5"))
    
    # create output directory
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory, exist_ok=True)

    # check what has been done
    filesdone = [os.path.basename(x) for x in glob.glob(os.path.join(args.output_directory, '*.npy'))]
    filesdone = set([x for x in filesdone if x.startswith("data-")])
    inputfiles = list(filter(lambda x: filter_func(x, filesdone), inputfiles_all))

    # wait for everybody to be done
    comm.barrier()
    
    if comm_rank == 0:
        print(f"{len(inputfiles_all)} files found, {len(filesdone)} done, {len(inputfiles)} to do.")
    
    # split across ranks: round robin
    inputfiles_local = []
    for idx, ifname in enumerate(inputfiles):
        if idx % comm_size == comm_rank:
            inputfiles_local.append(ifname)
    
    # convert files
    for ifname in inputfiles_local:
        ofname_data = os.path.join(args.output_directory, os.path.basename(ifname).replace(".h5", ".npy"))
        ofname_label = os.path.join(args.output_directory, os.path.basename(ifname).replace("data-", "label-").replace(".h5", ".npy"))
        
        with h5.File(ifname, 'r') as f:
            data = f["climate/data"][...]
            label = f["climate/labels_0"][...]
        
        # save data and label
        np.save(ofname_label, label)
        np.save(ofname_data, data)

    # wait for the others
    comm.barrier()


if __name__ == "__main__":
    
    AP = ap.ArgumentParser()
    AP.add_argument("--input_directory", type=str, help="Directory with input files", required = True)
    AP.add_argument("--output_directory", type=str, help="Directory with output files.", required = True)
    pargs = AP.parse_args()
    
    main(pargs)
