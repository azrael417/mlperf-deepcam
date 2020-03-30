# Basics
import os
import wandb
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

#warmup scheduler
from warmup_scheduler import GradualWarmupScheduler

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
except:
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


def printr(msg, rank=0):
    if comm.get_rank() == rank:
        print(msg)


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
    if (pargs.logging_frequency > 0) and (comm_rank == 0):
        # get wandb api token
        with open(os.path.join(pargs.wandb_certdir, ".wandbirc")) as f:
            token = f.readlines()[0].replace("\n","").split()
            wblogin = token[0]
            wbtoken = token[1]
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

        if pargs.lr_warmup_steps > 0:
            scheduler = GradualWarmupScheduler(optimizer, multiplier=pargs.lr_warmup_factor, total_epoch=pargs.lr_warmup_steps, after_scheduler=scheduler_after)
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
                               shuffle = True, 
                               preprocess = True,
                               comm_size = comm_size,
                               comm_rank = comm_rank)
    train_loader = DataLoader(train_set, pargs.local_batch_size, num_workers=min([pargs.max_inter_threads, pargs.local_batch_size]), drop_last=True)
    
    # validation: we only want to shuffle the set if we are cutting off validation after a certain number of steps
    validation_dir = os.path.join(root_dir, "validation")
    validation_set = cam.CamDataset(validation_dir, 
                               statsfile = os.path.join(root_dir, 'stats.h5'),
                               channels = pargs.channels,
                               shuffle = (pargs.max_validation_steps is not None),
                               preprocess = True,
                               comm_size = comm_size,
                               comm_rank = comm_rank)
    validation_loader = DataLoader(validation_set, pargs.local_batch_size, num_workers=min([pargs.max_inter_threads, pargs.local_batch_size]), drop_last=True)

    
    #for visualization
    if visualize:
        viz = vizc.CamVisualizer()   
    
    # Train network
    if (pargs.logging_frequency > 0) and (comm_rank == 0):
        wandb.watch(net)
    
    printr('{:14.4f} REPORT: starting training'.format(dt.datetime.now().timestamp()), 0)
    step = start_step
    epoch = start_epoch
    current_lr = pargs.start_lr if not pargs.lr_schedule else scheduler.get_last_lr()[0]
    net.train()
    while True:
        
        printr('{:14.4f} REPORT: starting epoch {}'.format(dt.datetime.now().timestamp(), epoch), 0)
        
        #for inputs_raw, labels, source in train_loader:
        for inputs, label, filename in train_loader:
            
            #send to device
            inputs = inputs.to(device)
            label = label.to(device)
            
            # forward pass
            outputs = net.forward(inputs)
            
            # Compute loss and average across nodes
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
            optimizer.zero_grad()
            if have_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            #step counter
            step += 1
            
            if pargs.lr_schedule:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

            #print some metrics
            printr('{:14.4f} REPORT training: step {} loss {} iou {} LR {}'.format(dt.datetime.now().timestamp(), step,
                                                                                   loss_avg.item() / float(comm_size),
                                                                                   iou_avg.item() / float(comm_size),
                                                                                   current_lr), 0)

            #visualize if requested
            if (step % pargs.training_visualization_frequency == 0) and (comm_rank == 0):
                #extract sample id and data tensors
                sample_idx = np.random.randint(low=0, high=label.shape[0])
                plot_input = inputs.detach()[sample_idx, 0,...].cpu().numpy()
                plot_prediction = predictions.detach()[sample_idx,...].cpu().numpy()
                plot_label = label.detach()[sample_idx,...].cpu().numpy()
                
                #create filenames
                outputfile = os.path.basename(filename[sample_idx]).replace("data-", "training-").replace(".h5", ".png")
                outputfile = os.path.join(plot_dir, outputfile)
                
                #plot
                viz.plot(filename[sample_idx], outputfile, plot_input, plot_prediction, plot_label)
                
                #log if requested
                if pargs.logging_frequency > 0:
                    img = Image.open(outputfile)
                    wandb.log({"Training Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth")]}, step = step)
            
            
            #log if requested
            if (pargs.logging_frequency > 0) and (step % pargs.logging_frequency == 0) and (comm_rank == 0):
                wandb.log({"Training Loss": loss_avg.item() / float(comm_size)}, step = step)
                wandb.log({"Training IoU": iou_avg.item() / float(comm_size)}, step = step)
                wandb.log({"Current Learning Rate": current_lr}, step = step)
                
            # validation step if desired
            if (step % pargs.validation_frequency == 0):
                
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
                        loss_val = criterion(outputs_val, label_val, weight=class_weights)
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
                            if pargs.logging_frequency > 0:
                                img = Image.open(outputfile)
                                wandb.log({"Validation Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth")]}, step = step)
                        
                        #increase eval step counter
                        step_val += 1
                        
                        if (pargs.max_validation_steps is not None) and step_val > pargs.max_validation_steps:
                            break
                        
                # average the validation loss
                dist.reduce(count_sum_val, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(loss_sum_val, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(iou_sum_val, dst=0, op=dist.ReduceOp.SUM)
                loss_avg_val = loss_sum_val.item() / count_sum_val.item()
                iou_avg_val = iou_sum_val.item() / count_sum_val.item()
                
                # print results
                printr('{:14.4f} REPORT validation: step {} loss {} iou {}'.format(dt.datetime.now().timestamp(), step, loss_avg_val, iou_avg_val), 0)

                # log in wandb
                if (pargs.logging_frequency > 0) and (comm_rank == 0):
                    wandb.log({"Validation Loss": loss_avg_val}, step=step)
                    wandb.log({"Validation IoU": iou_avg_val}, step=step)

                # set to train
                net.train()
            
            #save model if desired
            if (step % pargs.save_frequency == 0) and (comm_rank == 0):
                checkpoint = {
                    'step': step,
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict()
		}
                if have_apex:
                    checkpoint['amp'] = amp.state_dict()
                torch.save(checkpoint, os.path.join(output_dir, pargs.model_prefix + "_step_" + str(step) + ".cpt") )

        #do some after-epoch prep, just for the books
        epoch += 1
        if comm_rank==0:
          
            # Save the model
            checkpoint = {
                'step': step,
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if have_apex:
                checkpoint['amp'] = amp.state_dict()
            torch.save(checkpoint, os.path.join(output_dir, pargs.model_prefix + "_epoch_" + str(epoch) + ".cpt") )

        #are we done?
        if epoch >= pargs.max_epochs:
            break

    printr('{:14.4f} REPORT: finishing training'.format(dt.datetime.now().timestamp()), 0)
    

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
    AP.add_argument("--max_validation_steps", type=int, default=None, help="Number of validation steps to perform. Helps when validation takes a long time")
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
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--amp_opt_level", type=str, default="O0", help="AMP optimization level")
    AP.add_argument("--resume_logging", action='store_true')
    pargs = AP.parse_args()

    #run the stuff
    main(pargs)
