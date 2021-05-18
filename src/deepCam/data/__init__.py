import os
from glob import glob
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

# helper function for determining the data shapes
def get_datashapes(pargs, root_dir):
    
    if not pargs.enable_dali:
        return peek_shapes_hdf5(os.path.join(root_dir, "train"))
    else:
        return peek_shapes_numpy(os.path.join(root_dir, "train"))
    

# helper function to de-clutter the main training script
def get_dataloaders(pargs, root_dir, device, seed, comm_size, comm_rank):
    
    if not pargs.enable_dali:

        # import only what we need
        from .cam_hdf5_dataset import CamDataset, peek_shapes_hdf5
        
        train_dir = os.path.join(root_dir, "train")
        train_set = CamDataset(train_dir, 
                               statsfile = os.path.join(root_dir, 'stats.h5'),
                               channels = pargs.channels,
                               allow_uneven_distribution = False,
                               shuffle = True, 
                               preprocess = True,
                               transpose = not pargs.enable_nhwc,
                               augmentations = pargs.data_augmentations,
                               comm_size = 1,
                               comm_rank = 0)

        distributed_train_sampler = DistributedSampler(train_set,
                                                       num_replicas = comm_size,
                                                       rank = comm_rank,
                                                       shuffle = True,
                                                       drop_last = True)
    
        train_loader = DataLoader(train_set,
                                  pargs.local_batch_size,
                                  num_workers = min([pargs.max_inter_threads, pargs.local_batch_size]),
                                  sampler = distributed_train_sampler,
                                  pin_memory = True,
                                  drop_last = True)

        train_size = train_set.global_size

    else:
        from .cam_numpy_dali_dataset import CamDaliDataloader, peek_shapes_numpy
        
        train_dir = os.path.join(root_dir, "train")
        train_loader = CamDaliDataloader(train_dir,
                                         'data-*.npy',
                                         'label-*.npy',
                                         os.path.join(root_dir, 'stats.h5'),
                                         pargs.local_batch_size,
                                         file_list_data = "train_files_data.lst",
                                         file_list_label = "train_files_label.lst",
                                         num_threads = min([pargs.max_inter_threads, pargs.local_batch_size]),
                                         device = device,
                                         num_shards = comm_size,
                                         shard_id = comm_rank,
                                         stick_to_shard = False,
                                         shuffle = True,
                                         is_validation = False,
                                         lazy_init = True,
                                         transpose = not pargs.enable_nhwc,
                                         augmentations = pargs.data_augmentations,
                                         read_gpu = False,
                                         use_mmap = False,
                                         seed = seed)
        train_size = train_loader.global_size
    
    # validation: we only want to shuffle the set if we are cutting off validation after a certain number of steps
    if not pargs.enable_dali:
        validation_dir = os.path.join(root_dir, "validation")
        validation_set = CamDataset(validation_dir, 
                                    statsfile = os.path.join(root_dir, 'stats.h5'),
                                    channels = pargs.channels,
                                    allow_uneven_distribution = True,
                                    shuffle = (pargs.max_validation_steps is not None),
                                    preprocess = True,
                                    transpose = not pargs.enable_nhwc,
                                    augmentations = [],
                                    comm_size = comm_size,
                                    comm_rank = comm_rank)
    
        # use batch size = 1 here to make sure that we do not drop a sample
        validation_loader = DataLoader(validation_set,
                                       1,
                                       num_workers = min([pargs.max_inter_threads, pargs.local_batch_size]),
                                       pin_memory = True,
                                       drop_last = False)

        validation_size = validation_set.global_size
        
    else:
        validation_dir = os.path.join(root_dir, "validation")
        validation_loader = CamDaliDataloader(validation_dir,
                                              'data-*.npy',
                                              'label-*.npy',
                                              os.path.join(root_dir, 'stats.h5'),
                                              1,
                                              file_list_data = "validation_files_data.lst",
                                              file_list_label = "validation_files_label.lst",
                                              num_threads = min([pargs.max_inter_threads, pargs.local_batch_size]),
                                              device = device,
                                              num_shards = comm_size,
                                              shard_id = comm_rank,
                                              stick_to_shard = True,
                                              shuffle = True,
                                              is_validation = True,
                                              lazy_init = True,
                                              transpose = not pargs.enable_nhwc,
                                              augmentations = [],
                                              read_gpu = False,
                                              use_mmap = False,
                                              seed = seed)
        validation_size = validation_loader.global_size
    
        
    return train_loader, train_size, validation_loader, validation_size
