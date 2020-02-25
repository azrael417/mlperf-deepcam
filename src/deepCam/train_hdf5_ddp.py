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

#vis stuff
from PIL import Image

#DDP
from torch.nn.parallel import DistributedDataParallel as DDP
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

    #init DDP
    torch.distributed.init_process_group(backend="nccl")
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
    else:
        printr("Using CPUs",0)
        device = torch.device("cpu")

    #set up directories
    fac = 2 if pargs.num_raid == 4 else 1
    mod = 4 if pargs.num_raid == 4 else 2
    root_dir = os.path.join(pargs.data_dir_prefix, "data{}".format( fac * (comm_local_rank // mod) + 1 ), \
                            "ecmwf_data", "gpu{}".format( comm_local_rank ))
    output_dir = pargs.output_dir
    if comm_rank == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    
    ## Setup WandB
    #if (pargs.logging_frequency > 0) and (comm_rank == 0):
    #    # get wandb api token
    #    with open("/root/.wandbirc") as f:
    #        wbtoken = f.readlines()[0].replace("\n","")
    #    # log in: that call can be blocking, it should be quick
    #    sp.call(["wandb", "login", wbtoken])
    #    
    #    #init db and get config
    #    resume_flag = pargs.run_tag if pargs.resume_logging else False
    #    wandb.init(project = 'ERA5 prediction', name = pargs.run_tag, id = pargs.run_tag, resume = resume_flag)
    #    config = wandb.config
    #
    #    #set general parameters
    #    config.root_dir = root_dir
    #    config.output_dir = pargs.output_dir
    #    config.max_steps = pargs.max_steps
    #    config.local_batch_size = pargs.local_batch_size
    #    config.num_workers = hvd.size()
    #    config.channels = pargs.channels
    #    config.noise_dimensions = pargs.noise_dimensions
    #    config.noise_type = pargs.noise_type
    #    config.optimizer = pargs.optimizer
    #    config.start_lr = pargs.start_lr
    #    config.adam_eps = pargs.adam_eps
    #    config.weight_decay = pargs.weight_decay
    #    config.loss_type = pargs.loss_type
    #    config.model_prefix = pargs.model_prefix
    #    config.precision = "fp16" if  pargs.enable_fp16 else "fp32"
    #
    #    # lr schedule if applicable
    #    for key in pargs.lr_schedule:
    #        config.update({"lr_schedule_"+key: pargs.lr_schedule[key]}, allow_val_change = True)
            

    # Define architecture
    n_input_channels = len(pargs.channels)
    n_output_channels = 3
    net = deeplab_xception.DeepLabv3_plus(nInputChannels = n_input_channels, n_output = n_output_channels, os=16, pretrained=False)

    #select loss
    loss_pow = pargs.loss_pow
    #some magix numbers
    class_weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
    fpw_1 = 2.61461122397522257612
    fpw_2 = 1.71641974795896018744
    criterion = losses.fp_loss

    #select optimizer
    optimizer = None
    if pargs.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=pargs.start_lr, eps=pargs.adam_eps, weight_decay=pargs.weight_decay)
    else:
        raise ValueError("Error, optimizer {} not supported".format(pargs.optimizer))

    #restart from checkpoint if desired
    if (hvd.rank() == 0) and (pargs.checkpoint):
        checkpoint = torch.load(pargs.checkpoint, map_location = device)
        start_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['model'])
    else:
        start_step = 0
        start_epoch = 0
        
    #select scheduler
    if pargs.lr_schedule:
        scheduler = ph.get_lr_schedule(pargs.start_lr, pargs.lr_schedule, optimizer, last_step = start_step)
        
    #broadcast model and optimizer state
    steptens = torch.tensor(np.array([start_step, start_epoch]), requires_grad=False)
    hvd.broadcast(steptens, root_rank = 0)
    hvd.broadcast_parameters(net.state_dict(), root_rank = 0)
    hvd.broadcast_optimizer_state(optimizer, root_rank = 0)
    if pargs.enable_fp16:
        net.half()
    net.to(device)

    #unpack the bcasted tensor
    start_step = steptens.numpy()[0]
    start_epoch = steptens.numpy()[1]
    
    #wrap the optimizer
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=net.named_parameters(),
                                         compression = hvd.Compression.none,
                                         op = hvd.Average)

    # Set up the data feeder
    # train
    train_dir = os.path.join(root_dir, "train")
    train_set = cam.CamDataset(train_dir, 
                               statsfile = os.path.join(train_dir, 'stats.h5'),
                               channels = pargs.channels,
                               shuffle = True, 
                               preprocess = True)
    train_loader = DataLoader(train_set, local_batch_size, num_workers=min([max_inter_threads, local_batch_size]), drop_last=True)
    
    # validation
    validation_dir = os.path.join(root_dir, "validation")
    validation_set = cam.CamDataset(validation_dir, 
                               statsfile = os.path.join(train_dir, 'stats.h5'),
                               channels = pargs.channels,
                               shuffle = False, 
                               preprocess = True)
    validation_loader = DataLoader(validation_set, local_batch_size, num_workers=min([max_inter_threads, local_batch_size]), drop_last=True)
        
    # Train network
    if (pargs.logging_frequency > 0) and (hvd.rank() == 0):
        wandb.watch(net)
    printr('{:14.4f} REPORT: starting training'.format(dt.datetime.now().timestamp()), 0)
    step = start_step
    epoch = start_epoch
    current_lr = pargs.start_lr if not pargs.lr_schedule else scheduler.get_last_lr()[0]
    net.train()
    while True:
        
        printr('{:14.4f} REPORT: starting epoch {}'.format(dt.datetime.now().timestamp(), epoch), 0)
        mse_list = []
        
        #for inputs_raw, labels, source in train_loader:
        for inputs, label, filename in train_loader:

            # forward pass
            outputs = net.forward(inputs)
            
            # Compute loss and average across nodes
            loss = criterion(outputs, label, weight=class_weights, fpw_1=fpw_1, fpw_2=fpw_2)
            
            # Compute score
            predictions = torch.max(outputs, 1)[1]
            local_iou = utils.get_iou(predictions, labels, n_classes=3) #/ local_batch_size
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #step counter
            step += 1

            if pargs.lr_schedule:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

            #print some metrics
            printr('{:14.4f} REPORT training: step {} loss {} LR {}'.format(dt.datetime.now().timestamp(), step, loss_avg, current_lr), 0)

            ##visualize if requested
            #if (step % pargs.visualization_frequency == 0) and (hvd.rank() == 0):
            #    sample_idx = np.random.randint(low=0, high=label.shape[0])
            #    filename = os.path.join(output_dir, "plot_step{}_sampleid{}.png".format(step,sample_idx))
            #    prediction = outputs.detach()[sample_idx,...].cpu().numpy()
            #    groundtruth = label.detach()[sample_idx,...].cpu().numpy()
            #    ev.visualize_prediction(filename, prediction, groundtruth)
            #    
            #    #log if requested
            #    if pargs.logging_frequency > 0:
            #        img = Image.open(filename)
            #        wandb.log({"Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)
            #
            ##log if requested
            #if (pargs.logging_frequency > 0) and (step % pargs.logging_frequency == 0) and (hvd.rank() == 0):
            #    wandb.log({"Training Loss": loss_avg}, step = step)
            #    wandb.log({"Current Learning Rate": current_lr}, step = step)
                
            # validation step if desired
            if (step % pargs.validation_frequency == 0):
                
                #eval
                net.eval()
                
                # vali loss
                mse_list_val = []

                # disable gradients
                with torch.no_grad():
                
                    # iterate over validation sample
                    for inputs_raw_val, label_val, inputs_info_val, label_info_val in validation_loader:

                        # generate random sample: format of input data is NHWC 
                        if pargs.noise_dimensions > 0:
                            ishape_val = inputs_raw_val.shape
                            inputs_noise_val = udist.rsample( (ishape_val[0], pargs.noise_dimensions, ishape_val[2], ishape_val[3]) ).to(device)

                            # concat tensors
                            inputs_val = torch.cat([inputs_raw_val, inputs_noise_val], dim=1)
                        else:
                            inputs_val = inputs_raw_val
                    
                        # forward pass
                        outputs_val = net.forward(inputs_val)

                        # Compute loss and average across nodes
                        loss_val = criterion(outputs_val, label_val)

                        # append to list
                        mse_list_val.append(loss_val)

                # average the validation loss
                count_val = float(len(mse_list_val))
                count_val_global = metric_average(count_val, "val_count", op=hvd.Sum)
                loss_val = sum(mse_list_val)
                loss_val_global = metric_average(loss_val, "val_loss", op=hvd.Sum)
                loss_val_avg = loss_val_global / count_val_global
                
                # print results
                printr('{:14.4f} REPORT validation: step {} loss {}'.format(dt.datetime.now().timestamp(), step, loss_val_avg), 0)

                ## log in wandb
                #if (pargs.logging_frequency > 0) and (hvd.rank() == 0):
                #    wandb.log({"Validation Loss": loss_val_avg}, step=step)

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
            torch.save(checkpoint, os.path.join(output_dir, pargs.model_prefix + "_epoch_" + str(epoch) + ".cpt") )

        #are we done?
        if epoch >= pargs.max_epochs:
            break

    printr('{:14.4f} REPORT: finishing training'.format(dt.datetime.now().timestamp()), 0)
    

if __name__ == "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--run_tag", type=str, help="Unique run tag, to allow for better identification")
    AP.add_argument("--output_dir", type=str, help="Directory used for storing output. Needs to read/writeable from rank 0")
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--num_raid", type=int, default=4, choices=[4, 8], help="Number of available raid drives")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    #AP.add_argument("--max_intra_threads", type=int, default=8, help="Maximum degree of parallelism within reader")
    AP.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs to train")
    AP.add_argument("--save_frequency", type=int, default=100, help="Frequency with which the model is saved in number of steps")
    AP.add_argument("--validation_frequency", type=int, default=100, help="Frequency with which the model is validated")
    AP.add_argument("--logging_frequency", type=int, default=100, help="Frequency with which the training progress is logged. If not positive, logging will be disabled")
    AP.add_argument("--visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during training")
    AP.add_argument("--local_batch_size", type=int, default=1, help="Number of samples per local minibatch")
    AP.add_argument("--channels", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], help="Channels used in input")
    AP.add_argument("--optimizer", type=str, default="Adam", choices=["Adam"], help="Optimizer to use")
    AP.add_argument("--start_lr", type=float, default=1e-3, help="Start LR")
    AP.add_argument("--adam_eps", type=float, default=1e-8, help="Adam Epsilon")
    AP.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    AP.add_argument("--loss_pow", type=float, default=-0.125, help="Decay factor to adjust the weights")
    AP.add_argument("--lr_schedule", action=StoreDictKeyPair)
    AP.add_argument("--lr_decay_patience", type=int, default=3, help="Minimum number of steps used to wait before decreasing LR")
    AP.add_argument("--lr_decay_rate", type=float, default=0.25, help="LR decay factor")
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--disable_gds", action='store_true')
    AP.add_argument("--enable_fp16", action='store_true')
    AP.add_argument("--resume_logging", action='store_true')
    pargs = AP.parse_args()

    #run the stuff
    main(pargs)
