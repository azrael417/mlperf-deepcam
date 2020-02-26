import os
import sys
import numpy as np
import h5py as h5
from mpi4py import MPI

#merge function helper
def merge_token(token1, token2):
    #extract data
    #first
    n1 = token1[0]
    dmean1 = token1[1]
    dsqmean1 = token1[2]
    dmin1 = token1[3]
    dmax1 = token1[4]
    #second
    n2 = token2[0]
    dmean2 = token2[1]
    dsqmean2 = token2[2]
    dmin2 = token2[3]
    dmax2 = token2[4]

    #create new token
    nres = n1 + n2
    dmeanres = float(n1)/float(nres)*dmean1 + float(n2)/float(nres)*dmean2
    dsqmeanres = float(n1)/float(nres)*dsqmean1 + float(n2)/float(nres)*dsqmean2
    dminres = np.minimum(dmin1, dmin2)
    dmaxres = np.maximum(dmax1, dmax2)

    return (nres, dmeanres, dsqmeanres, dminres, dmaxres)


#create data token
def create_token(filename, data_format="nchw", rank = 0):

    try:
        with h5.File(filename, "r") as f:
            arr = f["climate/data"][...]
    except:
        raise IOError("Cannot open file {} on rank {}".format(filename, rank))
        
    #prep axis for ops
    axis = (1,2) if data_format == "nchw" else (0,1)

    #how many samples do we have: just 1 here
    n = 1
    #compute stats
    mean = np.mean(arr, axis=axis)
    meansq = np.mean(np.square(arr), axis=axis)
    minimum = np.amin(arr, axis=axis)
    maximum = np.amax(arr, axis=axis)

    #result
    result = (n, mean, meansq, minimum, maximum)
    
    return result
        

#global parameters
overwrite = False
data_format = "nhwc"
data_path_prefix = "/data"

#MPI
comm = MPI.COMM_WORLD.Dup()
comm_rank = comm.rank
comm_size = comm.size


#root path
root = os.path.join( data_path_prefix, "train" )

#get files
allfiles = [ os.path.join(root, x)  for x in os.listdir(root) \
              if x.endswith('.h5') and x.startswith('data-') ]

#split list
numfiles = len(allfiles)
chunksize = int(np.ceil(numfiles / comm_size))
start = chunksize * comm_rank
end = min([start + chunksize, numfiles])
files = allfiles[start:end]

#get first token and then merge recursively
token = create_token(files[0], data_format)
for filename in files[1:]:
    token = merge_token(create_token(filename, data_format, comm_rank), token)

#communicate results
alltoken = comm.gather(token, 0)

if comm_rank == 0:
    #merge the remaining token
    token = alltoken[0]
    for token2 in alltoken[1:]:
        token = merge_token(token, token2)

    #save the stuff
    with h5.File(os.path.join(data_path_prefix, "stats.h5"), "w") as f:
        f["climate/count"]=token[0]
        f["climate/mean"]=token[1]
        f["climate/sqmean"]=token[2]
        f["climate/minval"]=token[3]
        f["climate/maxval"]=token[4]
        
#wait for rank 0 to complete
comm.barrier()
