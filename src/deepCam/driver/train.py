# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

# base stuff
import os

# torch
import torch
import torch.cuda.amp as amp
import torch.distributed as dist

# custom stuff
from utils import utils

def train_step(pargs, comm_rank, comm_size, 
               step, epoch, 
               net, criterion, 
               optimizer, gscaler, scheduler,
               inputs, label, filename, 
               logger, have_wandb, viz):

    if not pargs.enable_dali:
        # send to device
        inputs = inputs.to(net.device)
        label = label.to(net.device)
    
    # to NHWC
    if pargs.enable_nhwc:
        inputs = inputs.contiguous(memory_format = torch.channels_last)
    
    # forward pass
    if pargs.enable_jit:
        # JIT
        outputs = net.forward(inputs)
        with amp.autocast(enabled = pargs.enable_amp):
            # to NCHW
            if pargs.enable_nhwc:
                outputs = outputs.contiguous(memory_format = torch.contiguous_format)
            loss = criterion(outputs, label)
    else:
        # NO-JIT
        with amp.autocast(enabled = pargs.enable_amp):
            outputs = net.forward(inputs)
            # to NCHW
            if pargs.enable_nhwc:
                outputs = outputs.contiguous(memory_format = torch.contiguous_format)
            loss = criterion(outputs, label)
    
    # Backprop
    #optimizer.zero_grad(set_to_none = True)
    optimizer.zero_grad()
    gscaler.scale(loss).backward()
    gscaler.step(optimizer)
    gscaler.update()
    
    # step counter
    step += 1
    
    if pargs.lr_schedule:
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
    
    #visualize if requested
    if (viz is not None) and (step % pargs.training_visualization_frequency == 0) and (comm_rank == 0):
        # Compute predictions
        predictions = torch.max(outputs, 1)[1]
        
        # extract sample id and data tensors
        sample_idx = np.random.randint(low=0, high=label.shape[0])
        plot_input = inputs.detach()[sample_idx, 0,...].cpu().numpy()
        plot_prediction = predictions.detach()[sample_idx,...].cpu().numpy()
        plot_label = label.detach()[sample_idx,...].cpu().numpy()
    
        # create filenames
        outputfile = os.path.basename(filename[sample_idx]).replace("data-", "training-").replace(".h5", ".png")
    
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
        iou = utils.compute_score(predictions, label, num_classes=3)
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
            
    return step