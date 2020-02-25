import os
import h5py as h5
import numpy as np
from time import sleep

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


#dataset class
class CamDataset(Dataset):
  
    def init_reader(self):
        #shuffle
        if self.shuffle:
            np.random.shuffle(self.all_files)
            
        #split
        num_files_local = len(self.all_files) // self.comm_size
        start_idx = comm_rank * num_files_local
        end_idx = start_idx + num_files_local
        self.files = self.all_files[start_idx:end_idx]
        
        #my own files
        self.length = len(self.files)

  
    def __init__(self, source, statsfile, channels, shuffle = False, preprocess = True, comm_size = 1, comm_rank = 0, seed = 12345):
        self.source = source
        self.statsfile = statsfile
        self.channels = channels
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.all_files = sorted( [ os.path.join(self.source,x) for x in os.listdir(self.source) ] )
        self.comm_size = comm_size
        self.comm_rank = comm_rank
        
        #split list of files
        np.random.seed(seed)
        
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
            data_shift = f["climate/minval"][self.channels]
            data_scale = 1. / ( f["climate/maxval"][self.channels] - data_shift )

        #reshape into broadcastable shape
        data_shift = np.reshape( data_shift, (1, data_shift.shape[0], 1, 1) ).astype(np.float32)
        data_scale = np.reshape( data_scale, (1, data_scale.shape[0], 1, 1) ).astype(np.float32)
        #store into tensors
        self.data_shift = torch.tensor(data_shift, requires_grad=False)
        self.data_scale	= torch.tensor(data_scale, requires_grad=False)
        
        print("Initialized dataset with ", self.length, " samples.")


    def __len__(self):
        return self.length


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
        
        #unsqueeze
        data = data.unsqueeze(0)
        label = label.unsqueeze(0)
        
        #preprocess
        data = self.data_scale * (data - self.data_shift)
        label = self.data_scale * (label - self.data_shift)
        
        return data, label, filename
