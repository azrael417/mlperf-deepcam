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

# Custom
from utils import utils
from utils import losses
from utils import parsing_helpers as ph
from data import cam_hdf5_dataset as cam
from architecture import deeplab_xception

#warmup scheduler
have_warmup_scheduler = False
try:
    from warmup_scheduler import GradualWarmupScheduler
    have_warmup_scheduler = True
except ImportError:
    pass

#vis stuff
from PIL import Image
from utils import visualizer as vizc

#DDP
import torch.distributed as dist
try:
    from apex import amp
    import apex.optimizers as aoptim
    from apex.parallel import DistributedDataParallel as DDP
    have_apex = True
except ImportError:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
    have_apex = False

#comm wrapper
from utils import comm


#dict helper for argparse
class StoreDictKeyPair(ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


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
    seed = 333
    logger.log_event(key = "seed", value = seed)
    
    # Some setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda", comm_local_rank)
        torch.cuda.manual_seed(seed)
        #necessary for AMP to work
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    #visualize?
    visualize = (pargs.training_visualization_frequency > 0) or (pargs.validation_visualization_frequency > 0)
        
    #set up directories
    root_dir = os.path.join(pargs.data_dir_prefix)
    output_dir = pargs.output_dir
    plot_dir = os.path.join(output_dir, "plots")
    if comm_rank == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if visualize and not os.path.isdir(plot_dir):
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
            wandb.init(entity = wblogin, project = 'deepcam', name = pargs.run_tag, id = pargs.run_tag, resume = resume_flag)
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
            config.amp_opt_level = pargs.amp_opt_level
            config.loss_weight_pow = pargs.loss_weight_pow
            config.lr_warmup_steps = pargs.lr_warmup_steps
            config.lr_warmup_factor = pargs.lr_warmup_factor
            
            # lr schedule if applicable
            if pargs.lr_schedule:
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

    #restart from checkpoint if desired
    #if (comm_rank == 0) and (pargs.checkpoint):
    #load it on all ranks for now
    if pargs.checkpoint:
        checkpoint = torch.load(pargs.checkpoint, map_location = device)
        start_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['model'])
        if have_apex:
            amp.load_state_dict(checkpoint['amp'])
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
    
    ##broadcast model and optimizer state
    #hvd.broadcast_parameters(net.state_dict(), root_rank = 0)
    #hvd.broadcast_optimizer_state(optimizer, root_rank = 0)

    #unpack the bcasted tensor
    start_step = steptens.cpu().numpy()[0]
    start_epoch = steptens.cpu().numpy()[1]

    # Set up the data feeder
    # train
    train_dir = os.path.join(root_dir, "train")
    train_set = cam.CamDataset(train_dir, 
                               statsfile = os.path.join(root_dir, 'stats.h5'),
                               channels = pargs.channels,
                               allow_uneven_distribution = False,
                               shuffle = True, 
                               preprocess = True,
                               comm_size = comm_size,
                               comm_rank = comm_rank)
    train_loader = DataLoader(train_set,
                              pargs.local_batch_size,
                              num_workers = min([pargs.max_inter_threads, pargs.local_batch_size]),
                              pin_memory = True,
                              drop_last = True)
    
    # validation: we only want to shuffle the set if we are cutting off validation after a certain number of steps
    validation_dir = os.path.join(root_dir, "validation")
    validation_set = cam.CamDataset(validation_dir, 
                               statsfile = os.path.join(root_dir, 'stats.h5'),
                               channels = pargs.channels,
                               allow_uneven_distribution = True,
                               shuffle = (pargs.max_validation_steps is not None),
                               preprocess = True,
                               comm_size = comm_size,
                               comm_rank = comm_rank)
    # use batch size = 1 here to make sure that we do not drop a sample
    validation_loader = DataLoader(validation_set,
                                   1,
                                   num_workers = min([pargs.max_inter_threads, pargs.local_batch_size]),
                                   pin_memory = True,
                                   drop_last = True)

    # log size of datasets
    logger.log_event(key = "train_samples", value = train_set.global_size)
    if pargs.max_validation_steps is not None:
        val_size = min([validation_set.global_size, pargs.max_validation_steps * pargs.local_batch_size * comm_size])
    else:
        val_size = validation_set.global_size
    logger.log_event(key = "eval_samples", value = val_size)

    # do sanity check
    if pargs.max_validation_steps is not None:
        logger.log_event(key = "invalid_submission")
    
    #for visualization
    if visualize:
        viz = vizc.CamVisualizer()   
    
    # Train network
    if have_wandb and (comm_rank == 0):
        wandb.watch(net)
    
    step = start_step
    epoch = start_epoch
    current_lr = pargs.start_lr if not pargs.lr_schedule else scheduler.get_last_lr()[0]
    stop_training = False
    net.train()

    # start trining
    logger.log_end(key = "init_stop", sync = True)
    logger.log_start(key = "run_start", sync = True)

    # training loop
    while True:

        # start epoch
        logger.log_start(key = "epoch_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync=True)

        # epoch loop
        for inputs, label, filename in train_loader:
            
            # send to device
            inputs = inputs.to(device)
            label = label.to(device)
            
            # forward pass
            outputs = net.forward(inputs)
            
            # Compute loss and average across nodes
            loss = criterion(outputs, label, weight=class_weights, fpw_1=fpw_1, fpw_2=fpw_2)
            
            # Backprop
            optimizer.zero_grad()
            if have_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # step counter
            step += 1
            
            if pargs.lr_schedule:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

            #visualize if requested
            if (step % pargs.training_visualization_frequency == 0) and (comm_rank == 0):
                # Compute predictions
                predictions = torch.max(outputs, 1)[1]
                
                # extract sample id and data tensors
                sample_idx = np.random.randint(low=0, high=label.shape[0])
                plot_input = inputs.detach()[sample_idx, 0,...].cpu().numpy()
                plot_prediction = predictions.detach()[sample_idx,...].cpu().numpy()
                plot_label = label.detach()[sample_idx,...].cpu().numpy()
                
                # create filenames
                outputfile = os.path.basename(filename[sample_idx]).replace("data-", "training-").replace(".h5", ".png")
                outputfile = os.path.join(plot_dir, outputfile)
                
                # plot
                viz.plot(filename[sample_idx], outputfile, plot_input, plot_prediction, plot_label)
                
                #log if requested
                if have_wandb:
                    img = Image.open(outputfile)
                    wandb.log({"train_examples": [wandb.Image(img, caption="Prediction vs. Ground Truth")]}, step = step)
            
            
            #log if requested
            if (step % pargs.logging_frequency == 0):

                # allreduce for loss
                loss_avg = loss.detach()
                dist.reduce(loss_avg, dst=0, op=dist.ReduceOp.SUM)
                loss_avg_train = loss_avg.item() / float(comm_size)

                # Compute score
                predictions = torch.max(outputs, 1)[1]
                iou = utils.compute_score(predictions, label, device_id=device, num_classes=3)
                iou_avg = iou.detach()
                dist.reduce(iou_avg, dst=0, op=dist.ReduceOp.SUM)
                iou_avg_train = iou_avg.item() / float(comm_size)
                
                logger.log_event(key = "learning_rate", value = current_lr, metadata = {'epoch_num': epoch+1, 'step_num': step})
                logger.log_event(key = "train_accuracy", value = iou_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step})
                logger.log_event(key = "train_loss", value = loss_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step})
                
                if have_wandb and (comm_rank == 0):
                    wandb.log({"train_loss": loss_avg.item() / float(comm_size)}, step = step)
                    wandb.log({"train_accuracy": iou_avg.item() / float(comm_size)}, step = step)
                    wandb.log({"learning_rate": current_lr}, step = step)

            
            # validation step if desired
            if (step % pargs.validation_frequency == 0):
                
                logger.log_start(key = "eval_start", metadata = {'epoch_num': epoch+1})

                #eval
                net.eval()
                
                count_sum_val = torch.Tensor([0.]).to(device)
                loss_sum_val = torch.Tensor([0.]).to(device)
                iou_sum_val = torch.Tensor([0.]).to(device)
                
                # disable gradients
                with torch.no_grad():
                
                    # iterate over validation sample
                    step_val = 0
                    # only print once per eval at most
                    visualized = False
                    for inputs_val, label_val, filename_val in validation_loader:
                        
                        #send to device
                        inputs_val = inputs_val.to(device)
                        label_val = label_val.to(device)
                        
                        # forward pass
                        outputs_val = net.forward(inputs_val)

                        # Compute loss and average across nodes
                        loss_val = criterion(outputs_val, label_val, weight=class_weights, fpw_1=fpw_1, fpw_2=fpw_2)
                        loss_sum_val += loss_val
                        
                        #increase counter
                        count_sum_val += 1.
                        
                        # Compute score
                        predictions_val = torch.max(outputs_val, 1)[1]
                        iou_val = utils.compute_score(predictions_val, label_val, device_id=device, num_classes=3)
                        iou_sum_val += iou_val

                        # Visualize
                        if (step_val % pargs.validation_visualization_frequency == 0) and (not visualized) and (comm_rank == 0):
                            #extract sample id and data tensors
                            sample_idx = np.random.randint(low=0, high=label_val.shape[0])
                            plot_input = inputs_val.detach()[sample_idx, 0,...].cpu().numpy()
                            plot_prediction = predictions_val.detach()[sample_idx,...].cpu().numpy()
                            plot_label = label_val.detach()[sample_idx,...].cpu().numpy()
                            
                            #create filenames
                            outputfile = os.path.basename(filename[sample_idx]).replace("data-", "validation-").replace(".h5", ".png")
                            outputfile = os.path.join(plot_dir, outputfile)
                            
                            #plot
                            viz.plot(filename[sample_idx], outputfile, plot_input, plot_prediction, plot_label)
                            visualized = True
                            
                            #log if requested
                            if have_wandb:
                                img = Image.open(outputfile)
                                wandb.log({"eval_examples": [wandb.Image(img, caption="Prediction vs. Ground Truth")]}, step = step)
                        
                        #increase eval step counter
                        step_val += 1
                        
                        if (pargs.max_validation_steps is not None) and step_val > pargs.max_validation_steps:
                            break
                        
                # average the validation loss
                dist.all_reduce(count_sum_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_sum_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(iou_sum_val, op=dist.ReduceOp.SUM)
                loss_avg_val = loss_sum_val.item() / count_sum_val.item()
                iou_avg_val = iou_sum_val.item() / count_sum_val.item()
                
                # print results
                logger.log_event(key = "eval_accuracy", value = iou_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})
                logger.log_event(key = "eval_loss", value = loss_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})

                # log in wandb
                if have_wandb and (comm_rank == 0):
                    wandb.log({"eval_loss": loss_avg_val}, step=step)
                    wandb.log({"eval_accuracy": iou_avg_val}, step=step)

                if (iou_avg_val >= pargs.target_iou):
                    logger.log_event(key = "target_accuracy_reached", value = pargs.target_iou, metadata = {'epoch_num': epoch+1, 'step_num': step})
                    stop_training = True

                # set to train
                net.train()

                logger.log_end(key = "eval_stop", metadata = {'epoch_num': epoch+1})
            
            #save model if desired
            if (pargs.save_frequency > 0) and (step % pargs.save_frequency == 0):
                logger.log_start(key = "save_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
                if comm_rank == 0:
                    checkpoint = {
                        'step': step,
                        'epoch': epoch,
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict()
		    }
                    if have_apex:
                        checkpoint['amp'] = amp.state_dict()
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
    AP = ap.ArgumentParser()
    AP.add_argument("--wireup_method", type=str, default="nccl-openmpi", choices=["nccl-openmpi", "nccl-slurm", "nccl-slurm-pmi", "mpi"], help="Specify what is used for wiring up the ranks")
    AP.add_argument("--wandb_certdir", type=str, default="/opt/certs", help="Directory in which to find the certificate for wandb logging.")
    AP.add_argument("--run_tag", type=str, help="Unique run tag, to allow for better identification")
    AP.add_argument("--output_dir", type=str, help="Directory used for storing output. Needs to read/writeable from rank 0")
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    AP.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs to train")
    AP.add_argument("--save_frequency", type=int, default=100, help="Frequency with which the model is saved in number of steps")
    AP.add_argument("--validation_frequency", type=int, default=100, help="Frequency with which the model is validated")
    AP.add_argument("--max_validation_steps", type=int, default=None, help="Number of validation steps to perform. Helps when validation takes a long time. WARNING: setting this argument invalidates submission. It should only be used for exploration, the final submission needs to have it disabled.")
    AP.add_argument("--logging_frequency", type=int, default=100, help="Frequency with which the training progress is logged. If not positive, logging will be disabled")
    AP.add_argument("--training_visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during training")
    AP.add_argument("--validation_visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during validation")
    AP.add_argument("--local_batch_size", type=int, default=1, help="Number of samples per local minibatch")
    AP.add_argument("--channels", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], help="Channels used in input")
    AP.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "LAMB"], help="Optimizer to use (LAMB requires APEX support).")
    AP.add_argument("--start_lr", type=float, default=1e-3, help="Start LR")
    AP.add_argument("--adam_eps", type=float, default=1e-8, help="Adam Epsilon")
    AP.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    AP.add_argument("--loss_weight_pow", type=float, default=-0.125, help="Decay factor to adjust the weights")
    AP.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for linear LR warmup")
    AP.add_argument("--lr_warmup_factor", type=float, default=1., help="Multiplier for linear LR warmup")
    AP.add_argument("--lr_schedule", action=StoreDictKeyPair)
    AP.add_argument("--target_iou", type=float, default=0.82, help="Target IoU score.")
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--amp_opt_level", type=str, default="O0", help="AMP optimization level")
    AP.add_argument("--enable_wandb", action='store_true')
    AP.add_argument("--resume_logging", action='store_true')
    pargs = AP.parse_args()
    
    #run the stuff
    main(pargs)
