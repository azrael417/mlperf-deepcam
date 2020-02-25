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
  
    def __init__(self, source, statsfile, channels, shuffle = False, preprocess = True):
        self.source = source
        self.statsfile = statsfile
        self.channels = channels
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.files = sorted(os.listdir(self.source))

        if self.shuffle:
            np.random.shuffle(self.files)
        
        self.length = len(self.files)

        #get shapes
        filename = os.path.join(self.source, self.files[0])
        with h5.File(filename, "r") as fin:
            self.data_shape = fin['climate']['data'].shape
            self.label_shape = fin['climate']['labels_0'].shape
        
        #get statsfile for normalization
        
        
        print("Initialized dataset with ", self.length, " samples.")

    def __len__(self):
        return self.length

    @property
    def shapes(self):
        return self.data_shape, self.label_shape

    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.files[idx])

        while(True):
            try:
                fin = h5.File(filename, "r")
                break
            except OSError:
                print("Could not open file " + filename + ", trying again in 5 seconds.")
                sleep(5)

        X = fin['climate']['data'][()]
        if self.preprocess:
            X = X[...,self.channels]
            X = np.moveaxis(X, -1, 0)
        Y = fin['climate']['labels_0'][()]
        fin.close()
        
        return X, Y, filename
