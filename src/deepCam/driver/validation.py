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
import torch.distributed as dist

# custom stuff
from utils import metric

# import wandb
try:
    import wandb
except ImportError:
    pass


def validate(pargs, comm_rank, comm_size,
             device, step, epoch, 
             net, criterion, validation_loader, 
             logger, have_wandb, viz):
    
    logger.log_start(key = "eval_start", metadata = {'epoch_num': epoch+1})

    # do we visualize? 
    visualize_validation = (pargs.validation_visualization_frequency > 0)

    #eval
    net.eval()

    count_sum_val = torch.zeros((1), dtype=torch.float32, device=device)
    loss_sum_val = torch.zeros((1), dtype=torch.float32, device=device)
    iou_sum_val = torch.zeros((1), dtype=torch.float32, device=device)

    # disable gradients
    with torch.no_grad():

        # iterate over validation sample
        step_val = 0
        # only print once per eval at most
        visualized = False
        for inputs_val, label_val, filename_val in validation_loader:

            if not pargs.enable_dali:
                #send to device
                inputs_val = inputs_val.to(device)
                label_val = label_val.to(device)
            else:
                if inputs_val.numel() == 0:
                    # we are done
                    continue

            # to NHWC
            if pargs.enable_nhwc:
                inputs_val = inputs_val.contiguous(memory_format = torch.channels_last)
            
            # forward pass
            #with amp.autocast(enabled = pargs.enable_amp):
            outputs_val = net.forward(inputs_val)

            # NCHW
            if pargs.enable_nhwc:
                outputs_val = outputs_val.contiguous(memory_format = torch.contiguous_format)
        
            # Compute loss
            loss_val = criterion(outputs_val, label_val)

            # accumulate loss
            loss_sum_val += loss_val
        
            #increase counter
            count_sum_val += 1.
        
            # Compute score
            predictions_val = torch.max(outputs_val, 1)[1]
            iou_val = metric.compute_score_new(predictions_val, label_val, num_classes=3)
            iou_sum_val += iou_val

            # Visualize
            if visualize_validation and (step_val % pargs.validation_visualization_frequency == 0) and (not visualized) and (comm_rank == 0):
                #extract sample id and data tensors
                sample_idx = np.random.randint(low=0, high=label_val.shape[0])
                plot_input = inputs_val.detach()[sample_idx, 0,...].cpu().numpy()
                plot_prediction = predictions_val.detach()[sample_idx,...].cpu().numpy()
                plot_label = label_val.detach()[sample_idx,...].cpu().numpy()
                
                #create filenames
                outputfile = os.path.basename(filename[sample_idx]).replace("data-", "validation-").replace(".h5", ".png")
                
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
        dist.all_reduce(count_sum_val, op=dist.ReduceOp.SUM, async_op=False)
        dist.reduce(loss_sum_val, dst=0, op=dist.ReduceOp.SUM)
        dist.all_reduce(iou_sum_val, op=dist.ReduceOp.SUM, async_op=False)
        loss_avg_val = loss_sum_val.item() / count_sum_val.item()
        iou_avg_val = iou_sum_val.item() / count_sum_val.item()

    # print results
    logger.log_event(key = "eval_accuracy", value = iou_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})
    logger.log_event(key = "eval_loss", value = loss_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})

    # log in wandb
    if have_wandb and (comm_rank == 0):
        wandb.log({"eval_loss": loss_avg_val}, step=step)
        wandb.log({"eval_accuracy": iou_avg_val}, step=step)

    stop_training = False
    if (iou_avg_val >= pargs.target_iou):
        logger.log_event(key = "target_accuracy_reached", value = pargs.target_iou, metadata = {'epoch_num': epoch+1, 'step_num': step})
        stop_training = True

    # set to train
    net.train()

    logger.log_end(key = "eval_stop", metadata = {'epoch_num': epoch+1})
    
    return stop_training
