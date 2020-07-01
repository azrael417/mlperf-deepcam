import os
import h5py as h5
import numpy as np
import math
from time import sleep

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


#dataset class
class CamDataset(Dataset):
  
    def init_reader(self):
        #shuffle
        if self.shuffle:
            self.rng.shuffle(self.all_files)
            
        #shard the dataset
        self.global_size = len(self.all_files)
        if self.allow_uneven_distribution:
            # this setting covers the data set completely, but the
            # last worker might get more samples than the rest
            num_files_local = self.global_size // self.comm_size
            start_idx = self.comm_rank * num_files_local
            if self.comm_rank != (self.comm_size - 1):
                end_idx = start_idx + num_files_local
            else:
                end_idx = self.global_size
            self.files = self.all_files[start_idx:end_idx]
        else:
            # here, every worker gets the same number of samples, 
            # potentially under-sampling the data
            num_files_local = self.global_size // self.comm_size
            start_idx = self.comm_rank * num_files_local
            end_idx = start_idx + num_files_local
            self.files = self.all_files[start_idx:end_idx]
            self.global_size = self.comm_size * len(self.files)
            
        #my own files
        self.local_size = len(self.files)

        #print sizes
        #print("Rank {} local size {} (global {})".format(self.comm_rank, self.local_size, self.global_size))

  
    def __init__(self, source, statsfile, channels, allow_uneven_distribution = False, shuffle = False, preprocess = True, comm_size = 1, comm_rank = 0, seed = 12345):
        self.source = source
        self.statsfile = statsfile
        self.channels = channels
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.all_files = sorted( [ os.path.join(self.source,x) for x in os.listdir(self.source) ] )
        self.comm_size = comm_size
        self.comm_rank = comm_rank
        self.allow_uneven_distribution = allow_uneven_distribution
        
        #split list of files
        self.rng = np.random.RandomState(seed)
        
        #init reader
        self.init_reader()

        #get shapes
        filename = os.path.join(self.source, self.files[0])
        with h5.File(filename, "r") as fin:
            self.data_shape = fin['climate']['data'].shape
            self.label_shape = fin['climate']['labels_0'].shape
        
        #get statsfile for normalization
        #open statsfile
        with h5.File(self.statsfile, "r") as f:
            data_shift = f["climate"]["minval"][self.channels]
            data_scale = 1. / ( f["climate"]["maxval"][self.channels] - data_shift )

        #reshape into broadcastable shape
        self.data_shift = np.reshape( data_shift, (data_shift.shape[0], 1, 1) ).astype(np.float32)
        self.data_scale = np.reshape( data_scale, (data_scale.shape[0], 1, 1) ).astype(np.float32)

        if comm_rank == 0:
            print("Initialized dataset with ", self.global_size, " samples.")


    def __len__(self):
        return self.local_size


    @property
    def shapes(self):
        return self.data_shape, self.label_shape


    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.files[idx])

        #load data and project
        with h5.File(filename, "r") as f:
            data = f["climate/data"][..., self.channels]
            label = f["climate/labels_0"][...]
        
        #transpose to NCHW
        data = np.transpose(data, (2,0,1))
        
        #preprocess
        data = self.data_scale * (data - self.data_shift)
        
        return data, label, filename
