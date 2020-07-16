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
import argparse as ap
import datetime as dt
import subprocess as sp

# Torch
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom
from utils import utils
from utils import losses
from utils import parsing_helpers as ph
from data import cam_hdf5_dataset as cam
from architecture import deeplab_xception

#DDP
import torch.distributed as dist
try:
    from apex import amp
    import apex.optimizers as aoptim
    from apex.parallel import DistributedDataParallel as DDP
    have_apex = True
except:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
    have_apex = False

#comm wrapper
from utils import comm

#we need pycuda
import pycuda.autoinit
import pycuda as pyc

#dict helper for argparse
class StoreDictKeyPair(ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def printr(msg, rank=0):
    if comm.get_rank() == rank:
        print(msg)


class Profile():
    
    def __init__(self, params, flag, steps):
        self.flag = flag
        self.tflag = params.profile
        self.num_warmup_steps = params.num_warmup_steps
        self.steps = steps
        self.profiler_started = False

    def __enter__(self):
        if (self.flag == self.tflag) and (self.num_warmup_steps >= self.steps):
            pyc.driver.start_profiler()
            self.profiler_started = True

    def __exit__(self, *args):
        if self.profiler_started:
            pyc.driver.stop_profiler()
            self.profiler_started = False


#main function
def main(pargs):

    #init distributed training
    comm.init(pargs.wireup_method)
    comm_rank = comm.get_rank()
    comm_local_rank = comm.get_local_rank()
    comm_size = comm.get_size()
    
    #set seed
    seed = 333
    
    # Some setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        printr("Using GPUs",0)
        device = torch.device("cuda", comm_local_rank)
        torch.cuda.manual_seed(seed)
        #necessary for AMP to work
        torch.cuda.set_device(device)
    else:
        printr("Using CPUs",0)
        device = torch.device("cpu")

    #set up directories
    root_dir = os.path.join(pargs.data_dir_prefix)
    output_dir = pargs.output_dir
    plot_dir = os.path.join(output_dir, "plots")
    if comm_rank == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    # Define architecture
    n_input_channels = len(pargs.channels)
    n_output_channels = 3
    net = deeplab_xception.DeepLabv3_plus(n_input = n_input_channels, 
                                          n_classes = n_output_channels, 
                                          os=16, pretrained=False, 
                                          rank = comm_rank)
    net.to(device)

    #select loss
    loss_pow = pargs.loss_weight_pow
    #some magic numbers
    class_weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
    fpw_1 = 2.61461122397522257612
    fpw_2 = 1.71641974795896018744
    criterion = losses.fp_loss

    #select optimizer
    optimizer = None
    if pargs.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    elif pargs.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    elif have_apex and (pargs.optimizer == "LAMB"):
        optimizer = aoptim.FusedLAMB(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    else:
        raise NotImplementedError("Error, optimizer {} not supported".format(pargs.optimizer))

    if have_apex:
        #wrap model and opt into amp
        net, optimizer = amp.initialize(net, optimizer, opt_level = pargs.amp_opt_level)
    
    #make model distributed
    net = DDP(net)

    #select scheduler
    if pargs.lr_schedule:
        scheduler = ph.get_lr_schedule(pargs.start_lr, pargs.lr_schedule, optimizer, last_step = 0)

    # Set up the data feeder
    # train
    train_dir = os.path.join(root_dir, "train")
    train_set = cam.CamDataset(train_dir, 
                               statsfile = os.path.join(root_dir, 'stats.h5'),
                               channels = pargs.channels,
                               shuffle = True, 
                               preprocess = True,
                               comm_size = comm_size,
                               comm_rank = comm_rank)
    train_loader = DataLoader(train_set, pargs.local_batch_size, num_workers=min([pargs.max_inter_threads, pargs.local_batch_size]), drop_last=True)
    
        
    printr('{:14.4f} REPORT: starting warmup'.format(dt.datetime.now().timestamp()), 0)
    step = 0
    current_lr = pargs.start_lr if not pargs.lr_schedule else scheduler.get_last_lr()[0]
    current_lr = pargs.start_lr
    net.train()
    while True:
        
        #for inputs_raw, labels, source in train_loader:
        for inputs, label, filename in train_loader:
            
            # Print status
            if step == pargs.num_warmup_steps:
                printr('{:14.4f} REPORT: starting profiling'.format(dt.datetime.now().timestamp()), 0)
            
            # Forward pass
            with Profile(pargs, "Forward", step):
                
                #send data to device
                inputs = inputs.to(device)
                label = label.to(device)
                
                # Compute output
                outputs = net.forward(inputs)
            
                # Compute loss
                loss = criterion(outputs, label, weight=class_weights, fpw_1=fpw_1, fpw_2=fpw_2)

                
            # allreduce for loss
            loss_avg = loss.detach()
            dist.reduce(loss_avg, dst=0, op=dist.ReduceOp.SUM)
            
            # Compute score
            predictions = torch.max(outputs, 1)[1]
            iou = utils.compute_score(predictions, label, device_id=device, num_classes=3)
            iou_avg = iou.detach()
            dist.reduce(iou_avg, dst=0, op=dist.ReduceOp.SUM)
            
            # Backprop
            with Profile(pargs, "Backward", step):

                # reset grads
                optimizer.zero_grad()
                
                # compute grads
                if have_apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    
                
            # weight update
            with Profile(pargs, "Optimizer", step):
                # update weights
                optimizer.step()

            # advance the scheduler
            if pargs.lr_schedule:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                
            #step counter
            step += 1

            #are we done?
            if step >= (pargs.num_warmup_steps + pargs.num_profile_steps):
                break

        #need to check here too
        if step >= (pargs.num_warmup_steps + pargs.num_profile_steps):
            break

    printr('{:14.4f} REPORT: finishing profiling'.format(dt.datetime.now().timestamp()), 0)
    

if __name__ == "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--wireup_method", type=str, default="nccl-openmpi", choices=["nccl-openmpi", "nccl-slurm", "nccl-slurm-pmi", "mpi"],
                    help="Specify what is used for wiring up the ranks")
    AP.add_argument("--run_tag", type=str, help="Unique run tag, to allow for better identification")
    AP.add_argument("--output_dir", type=str, help="Directory used for storing output. Needs to read/writeable from rank 0")
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    AP.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs to train")
    AP.add_argument("--local_batch_size", type=int, default=1, help="Number of samples per local minibatch")
    AP.add_argument("--num_warmup_steps", type=int, default=5, help="Number of warmup steps")
    AP.add_argument("--num_profile_steps", type=int, default=1, help="Number of profiling steps")
    AP.add_argument("--profile", type=str, default="Forward", choices=["Forward", "Backward", "Optimizer"], help="Flag which parts to profile")
    AP.add_argument("--channels", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], help="Channels used in input")
    AP.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "LAMB"], help="Optimizer to use (LAMB requires APEX support).")
    AP.add_argument("--start_lr", type=float, default=1e-3, help="Start LR")
    AP.add_argument("--adam_eps", type=float, default=1e-8, help="Adam Epsilon")
    AP.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    AP.add_argument("--loss_weight_pow", type=float, default=-0.125, help="Decay factor to adjust the weights")
    AP.add_argument("--lr_schedule", action=StoreDictKeyPair)
    AP.add_argument("--amp_opt_level", type=str, default="O0", help="AMP optimization level")
    pargs = AP.parse_args()

    #run the stuff
    main(pargs)
