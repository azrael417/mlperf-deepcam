# The MIT License (MIT)
#
# Copyright (c) 2018 Pyjcsx
# Modifications Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Basics
import os
import numpy as np
import datetime as dt
import subprocess as sp

# logging
# wandb
have_wandb = False
try:
    import wandb
    have_wandb = True
except ImportError:
    pass

# mlperf logger
import utils.mlperf_log_utils as mll

# Torch
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

# utils
from torchsummary import summary

# Custom
from driver import train_step, validate
from utils import parser
from utils import losses
from utils import parsing_helpers as ph
from data import cam_hdf5_dataset as cam
from data import cam_numpy_dali_dataset as cam_dali
from architecture import deeplab_xception

#warmup scheduler
have_warmup_scheduler = False
try:
    from warmup_scheduler import GradualWarmupScheduler
    have_warmup_scheduler = True
except ImportError:
    pass

# vis stuff
from PIL import Image
from utils import visualizer as vizc

# DDP
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

# APEX
try:
    import apex.optimizers as aoptim
    have_apex = True
except ImportError:
    from utils import optimizer as uoptim
    have_apex = False

# amp
import torch.cuda.amp as amp

#comm wrapper
from utils import comm

#main function
def main(pargs):

    # this should be global
    global have_wandb

    #init distributed training
    comm.init(pargs.wireup_method)
    comm_rank = comm.get_rank()
    comm_local_rank = comm.get_local_rank()
    comm_size = comm.get_size()

    # set up logging
    pargs.logging_frequency = max([pargs.logging_frequency, 1])
    log_file = os.path.normpath(os.path.join(pargs.output_dir, "logs", pargs.run_tag + ".log"))
    logger = mll.mlperf_logger(log_file, "deepcam", "Umbrella Corp.")
    logger.log_start(key = "init_start", sync = True)        
    logger.log_event(key = "cache_clear")
    
    #set seed
    seed = pargs.seed
    logger.log_event(key = "seed", value = seed)
    
    # Some setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda", comm_local_rank)
        torch.cuda.manual_seed(seed)
        #necessary for AMP to work
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    #visualize?
    visualize_train = (pargs.training_visualization_frequency > 0)
    visualize_validation = (pargs.validation_visualization_frequency > 0)
        
    #set up directories
    root_dir = os.path.join(pargs.data_dir_prefix)
    output_dir = pargs.output_dir
    plot_dir = os.path.join(output_dir, "plots")
    if comm_rank == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if (visualize_train or visualize_validation) and not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
    
    # Setup WandB
    if not pargs.enable_wandb:
        have_wandb = False
    if have_wandb and (comm_rank == 0):
        # get wandb api token
        certfile = os.path.join(pargs.wandb_certdir, ".wandbirc")
        try:
            with open(certfile) as f:
                token = f.readlines()[0].replace("\n","").split()
                wblogin = token[0]
                wbtoken = token[1]
        except IOError:
            print("Error, cannot open WandB certificate {}.".format(certfile))
            have_wandb = False

        if have_wandb:
            # log in: that call can be blocking, it should be quick
            sp.call(["wandb", "login", wbtoken])
        
            #init db and get config
            resume_flag = pargs.run_tag if pargs.resume_logging else False
            wandb.init(entity = wblogin, project = 'deepcam', 
                       dir = output_dir,
                       name = pargs.run_tag, id = pargs.run_tag, 
                       resume = resume_flag)
            config = wandb.config
        
            #set general parameters
            config.root_dir = root_dir
            config.output_dir = pargs.output_dir
            config.max_epochs = pargs.max_epochs
            config.local_batch_size = pargs.local_batch_size
            config.num_workers = comm_size
            config.channels = pargs.channels
            config.optimizer = pargs.optimizer
            config.start_lr = pargs.start_lr
            config.adam_eps = pargs.adam_eps
            config.weight_decay = pargs.weight_decay
            config.model_prefix = pargs.model_prefix
            config.enable_amp = pargs.enable_amp
            config.loss_weight_pow = pargs.loss_weight_pow
            config.lr_warmup_steps = pargs.lr_warmup_steps
            config.lr_warmup_factor = pargs.lr_warmup_factor
            
            # lr schedule if applicable
            if pargs.lr_schedule is not None:
                for key in pargs.lr_schedule:
                    config.update({"lr_schedule_"+key: pargs.lr_schedule[key]}, allow_val_change = True)


    # Logging hyperparameters
    logger.log_event(key = "global_batch_size", value = (pargs.local_batch_size * comm_size))
    logger.log_event(key = "opt_name", value = pargs.optimizer)
    logger.log_event(key = "opt_base_learning_rate", value = pargs.start_lr * pargs.lr_warmup_factor)
    logger.log_event(key = "opt_learning_rate_warmup_steps", value = pargs.lr_warmup_steps)
    logger.log_event(key = "opt_learning_rate_warmup_factor", value = pargs.lr_warmup_factor)
    logger.log_event(key = "opt_epsilon", value = pargs.adam_eps)

    # Define architecture
    n_input_channels = len(pargs.channels)
    n_output_channels = 3
    net = deeplab_xception.DeepLabv3_plus(n_input = n_input_channels, 
                                          n_classes = n_output_channels, 
                                          os=16, pretrained=False, 
                                          rank = comm_rank)
    net.to(device)

    if pargs.enable_nhwc:
        net = net.to(memory_format = torch.channels_last)

    #select loss
    loss_pow = pargs.loss_weight_pow
    #some magic numbers
    class_weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
    fpw_1 = 2.61461122397522257612
    fpw_2 = 1.71641974795896018744
    #criterion = losses.FPLoss(class_weights, fpw_1, fpw_2).to(device)
    criterion = losses.CELoss(class_weights).to(device)
    if pargs.enable_jit:
        criterion = torch.jit.script(criterion)

    #select optimizer
    optimizer = None
    if pargs.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    elif pargs.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    elif pargs.optimizer == "LAMB":
        if have_apex:
            optimizer = aoptim.FusedLAMB(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
        else:
            optimizer = uoptim.Lamb(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay, clamp_value = torch.iinfo(torch.int32).max)
    else:
        raise NotImplementedError("Error, optimizer {} not supported".format(pargs.optimizer))
    
    # gradient scaler
    gscaler = amp.GradScaler(enabled = pargs.enable_amp)
    
    #make model distributed
    ddp_net = DDP(net, device_ids=[device.index],
                  output_device=[device.index],
                  find_unused_parameters=True)

    #restart from checkpoint if desired
    #if (comm_rank == 0) and (pargs.checkpoint):
    #load it on all ranks for now
    if pargs.checkpoint:
        checkpoint = torch.load(pargs.checkpoint, map_location = device)
        start_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        ddp_net.load_state_dict(checkpoint['model'])
    else:
        start_step = 0
        start_epoch = 0
        
    #select scheduler
    if pargs.lr_schedule:
        scheduler_after = ph.get_lr_schedule(pargs.start_lr, pargs.lr_schedule, optimizer, last_step = start_step)

        # LR warmup
        if pargs.lr_warmup_steps > 0:
            if have_warmup_scheduler:
                scheduler = GradualWarmupScheduler(optimizer, multiplier=pargs.lr_warmup_factor,
                                                   total_epoch=pargs.lr_warmup_steps,
                                                   after_scheduler=scheduler_after)
            # Throw an error if the package is not found
            else:
                raise Exception(f'Requested {pargs.lr_warmup_steps} LR warmup steps '
                                'but warmup scheduler not found. Install it from '
                                'https://github.com/ildoonet/pytorch-gradual-warmup-lr')
        else:
            scheduler = scheduler_after

    #broadcast model and optimizer state
    steptens = torch.tensor(np.array([start_step, start_epoch]), requires_grad=False).to(device)
    dist.broadcast(steptens, src = 0)

    #unpack the bcasted tensor
    start_step = int(steptens.cpu().numpy()[0])
    start_epoch = int(steptens.cpu().numpy()[1])

    # Set up the data feeder
    if not pargs.enable_dali:
        train_dir = os.path.join(root_dir, "train")
        train_set = cam.CamDataset(train_dir, 
                                   statsfile = os.path.join(root_dir, 'stats.h5'),
                                   channels = pargs.channels,
                                   allow_uneven_distribution = False,
                                   shuffle = True, 
                                   preprocess = True,
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
        train_dir = os.path.join(root_dir, "train")
        train_loader = cam_dali.CamDaliDataloader(train_dir,
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
                                                  augmentations = pargs.data_augmentations,
                                                  read_gpu = False,
                                                  use_mmap = False,
                                                  seed = seed)
        train_size = train_loader.global_size
    
    # validation: we only want to shuffle the set if we are cutting off validation after a certain number of steps
    if not pargs.enable_dali:
        validation_dir = os.path.join(root_dir, "validation")
        validation_set = cam.CamDataset(validation_dir, 
                                        statsfile = os.path.join(root_dir, 'stats.h5'),
                                        channels = pargs.channels,
                                        allow_uneven_distribution = True,
                                        shuffle = (pargs.max_validation_steps is not None),
                                        preprocess = True,
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
        validation_loader = cam_dali.CamDaliDataloader(validation_dir,
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
                                                       augmentations = [],
                                                       read_gpu = False,
                                                       use_mmap = False,
                                                       seed = seed)
        validation_size = validation_loader.global_size

    ## print model summary :
    #if comm_rank == 0:
    #    tmp_shape = train_loader.shapes[0]
    #    input_shape = (tmp_shape[2], tmp_shape[0], tmp_shape[1])
    #    summary(net.module, input_size = input_shape)

    # jit stuff
    if pargs.enable_jit:
        # input_shape:
        if pargs.enable_dali:
            tshapes = train_loader.shapes[0]
        else:
            tshapes = train_loader.dataset.shapes[0]
        input_shape = [tshapes[2], tshapes[0], tshapes[1]]
        # example input
        train_example = torch.randn( (pargs.local_batch_size, *input_shape) ).to(device)
        validation_example = torch.randn( (1, *input_shape) ).to(device)

        # backup old network
        net_validate = net
        
        # NHWC if requested
        if pargs.enable_nhwc:
            train_example = train_example.contiguous(memory_format = torch.channels_last)
            validation_example = validation_example.contiguous(memory_format = torch.channels_last)

        if pargs.enable_amp:
            with amp.autocast(enabled = pargs.enable_amp):
                ddp_net.module = torch.jit.trace(ddp_net.module, train_example, check_trace = False)
        else:
            ddp_net.module = torch.jit.script(ddp_net.module)
        net_train = ddp_net
            
    else:
        net_validate = ddp_net
        net_train = ddp_net
        
    # log size of datasets
    logger.log_event(key = "train_samples", value = train_size)
    if pargs.max_validation_steps is not None:
        val_size = min([validation_size, pargs.max_validation_steps * pargs.local_batch_size * comm_size])
    else:
        val_size = validation_size
    logger.log_event(key = "eval_samples", value = val_size)

    # do sanity check
    if pargs.max_validation_steps is not None:
        logger.log_event(key = "invalid_submission")
    
    #for visualization
    viz = None
    if (visualize_train or visualize_validation):
        viz = vizc.CamVisualizer(plot_dir)   
    
    # Train network
    if have_wandb and not pargs.enable_jit and (comm_rank == 0):
        wandb.watch(net_train)
    
    step = start_step
    epoch = start_epoch
    current_lr = pargs.start_lr if not pargs.lr_schedule else scheduler.get_last_lr()[0]
    stop_training = False
    net_train.train()

    # start trining
    logger.log_end(key = "init_stop", sync = True)
    logger.log_start(key = "run_start", sync = True)

    # start prefetching
    if pargs.enable_dali:
        train_loader.init_iterator()
        validation_loader.init_iterator()

    # training loop
    while True:

        # start epoch
        logger.log_start(key = "epoch_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync=True)

        if not pargs.enable_dali:
            distributed_train_sampler.set_epoch(epoch)

        # epoch loop
        with torch.autograd.profiler.emit_nvtx(enabled = True):
            for inputs, label, filename in train_loader:
                
                step = train_step(pargs, comm_rank, comm_size,
                                  device, step, epoch, 
                                  net_train, criterion, 
                                  optimizer, gscaler, scheduler,
                                  inputs, label, filename, 
                                  logger, have_wandb, viz)
            
                # validation step if desired
                if (step % pargs.validation_frequency == 0):
                    
                    stop_training = validate(pargs, comm_rank, comm_size,
                                             device, step, epoch, 
                                             net_validate, criterion, validation_loader, 
                                             logger, have_wandb, viz)
            
                #save model if desired
                if (pargs.save_frequency > 0) and (step % pargs.save_frequency == 0):
                    logger.log_start(key = "save_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
                    if comm_rank == 0:
                        checkpoint = {
                            'step': step,
                            'epoch': epoch,
                            'model': net_train.state_dict(),
                            'optimizer': optimizer.state_dict()
		                }
                        torch.save(checkpoint, os.path.join(output_dir, pargs.model_prefix + "_step_" + str(step) + ".cpt") )
                    logger.log_end(key = "save_stop", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)

                # Stop training?
                if stop_training:
                    break
            
        # log the epoch
        logger.log_end(key = "epoch_stop", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
        epoch += 1
        
        # are we done?
        if epoch >= pargs.max_epochs or stop_training:
            break

    # run done
    logger.log_end(key = "run_stop", sync = True, metadata = {'status' : 'success'})


if __name__ == "__main__":

    #arguments
    pargs = parser.parse_arguments()
    
    #run the stuff
    main(pargs)
