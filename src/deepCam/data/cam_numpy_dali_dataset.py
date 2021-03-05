import os
import sys
import glob
import h5py as h5
import numpy as np
import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

class NumpyReadPipeline(Pipeline):
    def __init__(self, file_root, data_files, label_files, batch_size, mean, stddev, num_threads,
                 device, io_device, num_shards=1, shard_id=0, shuffle=False, stick_to_shard=False,
                 is_validation=False, lazy_init = False, transpose = True, augmentations = [], use_mmap=True, seed=333):
        super(NumpyReadPipeline, self).__init__(batch_size, num_threads, device.index, seed)
        
        self.data = ops.NumpyReader(device = io_device,
                                    file_root = file_root,
                                    files = data_files,
                                    num_shards = num_shards,
                                    shard_id = shard_id,
                                    stick_to_shard = stick_to_shard,
                                    shuffle_after_epoch = shuffle,
                                    prefetch_queue_depth = 2,
                                    cache_header_information = False,
                                    register_buffers = True,
                                    pad_last_batch = True,
                                    dont_use_mmap = not use_mmap,
                                    lazy_init = lazy_init,
                                    seed = seed)
        
        self.label = ops.NumpyReader(device = io_device,
                                     file_root = file_root,
                                     files = label_files,
                                     num_shards = num_shards,
                                     shard_id = shard_id,
                                     stick_to_shard = stick_to_shard,
                                     shuffle_after_epoch = shuffle,
                                     prefetch_queue_depth = 2,
                                     cache_header_information = False,
                                     register_buffers = True,
                                     pad_last_batch = True,
                                     dont_use_mmap = not use_mmap,
                                     lazy_init = lazy_init,
                                     seed = seed)

        self.normalize = ops.Normalize(device = "gpu", mean = mean, stddev = stddev, scale = 1.)

        self.do_transpose = transpose
        if self.do_transpose:
            self.transpose = ops.Transpose(device = "gpu", perm = [2, 0, 1])

        self.augmentations = augmentations
        if self.augmentations:
            # casts
            self.fcast = ops.Cast(dtype=types.DALIDataType.FLOAT32)
            self.icast = ops.Cast(dtype=types.DALIDataType.INT32)
            self.bcast = ops.Cast(dtype=types.DALIDataType.BOOL)

            # random stuff
            self.rng = ops.CoinFlip()

            # shape stuff
            self.reshape_in = ops.Reshape(device = "gpu", shape = [768, 1152, 1], layout="HWC")
            self.reshape_out = ops.Reshape(device = "gpu", shape = [768, 1152], layout="HW")

            # special ops
            if "flip" in self.augmentations:
                self.flip = ops.Flip(device = "gpu", depthwise = 0, horizontal = 0)

            if "jitter" in self.augmentations:
                self.jitter_data = ops.Jitter(device = "gpu",
                                              nDegree = 4,
                                              seed = seed)
                self.jitter_label = ops.Jitter(device = "gpu",
                                               nDegree = 4,
                                               seed = seed)

            if "gauss" in self.augmentations:
                self.blur = ops.GaussianBlur(device = "gpu",
                                             sigma = 1.,
                                             window_size = 3)


    def define_graph(self):
        data = self.data(name = "data")
        label = self.label(name = "label")

        # copy to GPU
        data = data.gpu()
        label = label.gpu()

        # augment if requested
        if self.augmentations:

            if "flip" in self.augmentations:
                # enable or disable
                condition = self.icast(self.rng())
                
                # flip
                data = self.flip(data, vertical = condition)
                label = self.reshape_out(self.flip(self.reshape_in(label), vertical = condition))

            if "jitter" in self.augmentations:
                # enable or disable
                condition = self.icast(self.rng())

                data = self.jitter_data(data, mask = condition)
                label = self.reshape_out(self.jitter_label(self.reshape_in(label), mask = condition))

            if "gauss" in self.augmentations:
                condition = self.bcast(self.rng())
                not_condition = condition ^ True

                data_blurred = self.blur(data)
                label_blurred = self.icast(self.blur(self.fcast(label)))
                
                data = data * not_condition + data_blurred * condition
                label = self.icast(label * not_condition + label_blurred * condition)

        # normalize now:
        data = self.normalize(data)

        # transpose data to NCHW if requested
        if self.do_transpose:
            data = self.transpose(data)
                
        return data, label


class CamDaliDataloader(object):

    def init_files(self, root_dir, prefix_data, prefix_label, statsfile, file_list_data = None, file_list_label = None, transpose = True):
        self.root_dir = root_dir
        self.prefix_data = prefix_data
        self.prefix_label = prefix_label

        # get files
        # data
        if file_list_data is not None and os.path.isfile(os.path.join(root_dir, file_list_data)):
            with open(os.path.join(root_dir, file_list_data), "r") as f:
                token = f.readlines()
            self.data_files = [os.path.join(root_dir, x.strip()) for x in token]
        else:
            self.data_files = glob.glob(os.path.join(self.root_dir, self.prefix_data))
        # label
        if file_list_label is not None and os.path.isfile(os.path.join(root_dir, file_list_label)):
            with open(os.path.join(root_dir, file_list_label), "r") as f:
                token = f.readlines()
            self.label_files = [os.path.join(root_dir, x.strip()) for x in token]
        else:
            self.label_files = glob.glob(os.path.join(self.root_dir, self.prefix_label))

        # get shapes
        self.data_shape = np.load(self.data_files[0]).shape
        self.label_shape = np.load(self.label_files[0]).shape

        # open statsfile
        with h5.File(statsfile, "r") as f:
            data_mean = f["climate"]["minval"][...]
            data_stddev = (f["climate"]["maxval"][...] - data_mean)
            
        #reshape into broadcastable shape: channels first
        self.data_mean = np.reshape( data_mean, (1, 1, data_mean.shape[0]) ).astype(np.float32)
        self.data_stddev = np.reshape( data_stddev, (1, 1, data_stddev.shape[0]) ).astype(np.float32)

        # clean up old iterator
        if self.iterator is not None:
            del(self.iterator)
            self.iterator = None
        
        # clean up old pipeline
        if self.pipeline is not None:
            del(self.pipeline)
            self.pipeline = None

        # io devices
        self.io_device = "gpu" if self.read_gpu else "cpu"
            
        # define pipes
        self.pipeline = NumpyReadPipeline(file_root = self.root_dir,
                                          data_files = [os.path.basename(x) for x in self.data_files],
                                          label_files = [os.path.basename(x) for x in self.label_files],
                                          batch_size = self.batchsize,
                                          mean = self.data_mean,
                                          stddev = self.data_stddev,
                                          num_threads = self.num_threads,
                                          device = self.device,
                                          io_device = self.io_device,
                                          num_shards = self.num_shards,
                                          shard_id = self.shard_id,
                                          shuffle = self.shuffle,
                                          is_validation = self.is_validation,
                                          lazy_init = self.lazy_init,
                                          transpose = self.transpose,
                                          augmentations = self.augmentations,
                                          use_mmap = self.use_mmap,
                                          seed = self.seed)
        
        # build pipes
        self.global_size = len(self.data_files)
        self.pipeline.build()

        # init iterator
        if not self.lazy_init:
            self.init_iterator()
        

    def init_iterator(self):
        self.iterator = DALIGenericIterator([self.pipeline], ['data', 'label'], auto_reset = True,
                                            reader_name = "data",
                                            last_batch_policy = LastBatchPolicy.PARTIAL if self.is_validation else LastBatchPolicy.DROP)
        self.epoch_size = self.pipeline.epoch_size()
        
        
    def __init__(self, root_dir, prefix_data, prefix_label, statsfile,
                 batchsize, file_list_data = None, file_list_label = None,
                 num_threads = 1, device = torch.device("cpu"),
                 num_shards = 1, shard_id = 0, stick_to_shard = False,
                 shuffle = False, is_validation = False,
                 lazy_init = False, transpose = True, augmentations = [],
                 use_mmap = True, read_gpu = False, seed = 333):
    
        # read filenames first
        self.batchsize = batchsize
        self.num_threads = num_threads
        self.device = device
        self.io_device = "gpu" if read_gpu else "cpu"
        self.use_mmap = use_mmap
        self.shuffle = shuffle
        self.read_gpu = read_gpu
        self.pipeline = None
        self.iterator = None
        self.lazy_init = lazy_init
        self.transpose = transpose
        self.augmentations = augmentations
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.stick_to_shard = stick_to_shard
        self.is_validation = is_validation
        self.seed = seed
        self.epoch_size = 0

        # init files
        self.init_files(root_dir, prefix_data, prefix_label,
                        statsfile, file_list_data, file_list_label)
        

    @property
    def shapes(self):
        return self.data_shape, self.label_shape

    
    def __iter__(self):
        #self.iterator.reset()
        for token in self.iterator:
            data = token[0]['data']
            label = token[0]['label']
            
            yield data, label, ""
