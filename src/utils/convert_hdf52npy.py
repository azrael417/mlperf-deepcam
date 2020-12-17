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


def convert(ifname, ofname_data, ofname_label):
    with h5.File(ifname, 'r') as f:
        data = f["climate/data"][...]
        label = f["climate/labels_0"][...]
        
        # save data and label
    np.save(ofname_label, label)
    np.save(ofname_data, data)

    
def fix(ifname, ofname_data, ofname_label):
    # assume it is good
    good = True

    try:
        np.load(ofname_label)
        np.load(ofname_data)
    except:
        good = False

    if not good:
        print(f"Files {ofname_label}, {ofname_data} need to be repaired.")
        convert(ifname, ofname_data, ofname_label)
        

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
    if pargs.fix:
        inputfiles = inputfiles_all
    else:
        filesdone = [os.path.basename(x) for x in glob.glob(os.path.join(args.output_directory, '*.npy'))]
        filesdone = set([x for x in filesdone if x.startswith("data-")])
        inputfiles = list(filter(lambda x: filter_func(x, filesdone), inputfiles_all))

    # wait for everybody to be done
    comm.barrier()
    
    if comm_rank == 0 :
        if not pargs.fix:
            print(f"{len(inputfiles_all)} files found, {len(filesdone)} done, {len(inputfiles)} to do.")
        else:
            print(f"Checking {len(inputfiles)} files.")
    
    # split across ranks: round robin
    inputfiles_local = []
    for idx, ifname in enumerate(inputfiles):
        if idx % comm_size == comm_rank:
            inputfiles_local.append(ifname)
    
    # convert files
    for ifname in inputfiles_local:
        ofname_data = os.path.join(args.output_directory, os.path.basename(ifname).replace(".h5", ".npy"))
        ofname_label = os.path.join(args.output_directory, os.path.basename(ifname).replace("data-", "label-").replace(".h5", ".npy"))

        if pargs.fix:
            # fix
            fix(ifname, ofname_data, ofname_label)
        else:
            # convert
            convert(ifname, ofname_data, ofname_label)

    # wait for the others
    comm.barrier()


if __name__ == "__main__":
    
    AP = ap.ArgumentParser()
    AP.add_argument("--input_directory", type=str, help="Directory with input files", required = True)
    AP.add_argument("--output_directory", type=str, help="Directory with output files.", required = True)
    AP.add_argument("--fix", action="store_true", help="Checks if all files got converted propery and repairs if requested")
    pargs = AP.parse_args()
    
    main(pargs)
