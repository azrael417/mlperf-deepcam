# Basics
import os
import numpy as np
import time
import argparse as ap

# Torch
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#horovod
import horovod.torch as hvd

#torch helper
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def printr(msg, rank=0):
    if hvd.rank() == rank:
        print(msg)

# Custom
from utils import utils
from utils import losses
from data import cam_hdf5_dataset as cam
from architecture import deeplab_xception
#from utils import visualizer as vizc

from tqdm import tqdm



def main(pargs):

    #init horovod
    hvd.init()

    # parameters fro prediction
    visualize = pargs.visualize
    use_fp16 = pargs.enable_fp16
    do_inference = pargs.inference
    preprocess = pargs.preprocess
    model_path = "./share/model.pth"
    channels = [0,1,2,10]
    batch_size = pargs.batch_size
    local_batch_size = batch_size // hvd.size()
    max_workers = pargs.max_workers

    #be careful here
    if do_inference:
        assert(do_inference == preprocess)

    # parameters for visualization
    output_dir = "/data1/cam5_data/output/hdf5"
    predict_dir = os.path.join(output_dir, "predict")
    os.makedirs(predict_dir, exist_ok=True)
    truth_dir = os.path.join(output_dir, "true")
    os.makedirs(truth_dir, exist_ok=True)


    # Initialize run
    torch.manual_seed(333)

    # Define architecture
    if torch.cuda.is_available():
        printr("Using GPUs",0)
        device = torch.device("cuda", hvd.local_rank())
    else:
        printr("Using CPUs",0)
        device = torch.device("cpu")

    #init data parallel model
    net = torch.nn.DataParallel(deeplab_xception.DeepLabv3_plus(nInputChannels=4, n_classes=3, os=16, pretrained=False), device_ids = [device])
    if hvd.rank() == 0:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path))
        else:
            net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        
    #broadcast
    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    net.eval()
    if use_fp16:
        net.half()
    net.to(device)


    # Get data
    fac = 2 if pargs.num_raid == 4 else 1
    mod = 4 if pargs.num_raid == 4 else 2
    data_dir = "/data{}/cam5_data/viz/hdf5_full/gpu{}".format( fac * (hvd.local_rank() // mod) + 1, hvd.local_rank())
    data = cam.subset(data_dir, channels, 0, -1, 1, hvd, shuffle = pargs.shuffle, preprocess = preprocess)
    data_loader = DataLoader(data, local_batch_size, num_workers=min([max_workers, local_batch_size]), drop_last=True)
    # train, val, test = cam.split(data_dir, channels, subset_length, train_frac, val_frac, test_frac)
    # data_loader = DataLoader(test, batch_size, num_workers=8, drop_last=True)

    #create vizc instance
    if visualize:
        viz = vizc.CamVisualizer()

        
    printr("starting inference", 0)

    #do multiple experiments if requested
    for nr in range(-pargs.num_warmup_runs, pargs.num_runs):

        tstart = time.time()
        it = 0
        
        for inputs, labels, source in data_loader:

            #increase iteration count
            it += 1
            
            # Push data on GPU and pass forward
            inputs, labels = inputs.to(device), labels.to(device)
            if use_fp16:
                inputs, labels = inputs.half(), labels.half()        

            if do_inference:
                #compute outputs
                with torch.no_grad():
                    outputs = net(inputs)

                ## Calculate test IoU
                predictions = torch.max(outputs, 1)[1]
                iou = utils.get_iou(predictions, labels, n_classes=3) / local_batch_size
                iou_avg = metric_average(iou, "IoU")

                printr("batch IoU: " + str(iou_avg), 0)

                #do we want to plot?
                if visualize:
        
                    #extract tensors as numpy arrays
                    datatens = inputs.cpu().detach().numpy()
                    predtens = predictions.cpu().detach().numpy()
                    labeltens = labels.cpu().detach().numpy()

                    #plot
                    for i in range(0,len(source)):
                        print("visualizing " + source[i])
                        h5path = source[i]
                        h5base = os.path.basename(h5path)
                        year = h5base[5:9]
                        month = h5base[10:12]
                        day = h5base[13:15]
                        hour = h5base[16:18]
                        
                        viz.plot(os.path.join(predict_dir, os.path.splitext(os.path.basename(h5base))[0]),
                                 "Predicted",
                                 np.squeeze(datatens[i,0,...]),
                                 np.squeeze(predtens[i,...]),
                                 year=year,
                                 month=month,
                                 day=day,
                                 hour=hour)
                        
                        viz.plot(os.path.join(truth_dir, os.path.splitext(os.path.basename(h5base))[0]),
                                 "Ground truth",
                                 np.squeeze(datatens[i,0,...]),
                                 np.squeeze(labeltens[i,...]),
                                 year=year,
                                 month=month,
                                 day=day,
                                 hour=hour)
            
        #print time:
        tend = time.time()
        printr("inference complete\n", 0)
        printr("total time: {} seconds for {} samples".format(tend - tstart, it * batch_size), 0)
        printr("iteration time: {} seconds/sample".format((tend - tstart)/float(it * batch_size)), 0)
        tst_data = cam.CamDataset(data_dir, channels)
        data_size = np.prod(tst_data.shapes[0]) * 4
        label_size = np.prod(tst_data.shapes[1]) * 4
        sample_size = (data_size + label_size) / 1024 / 1024 / 1024
        printr("bandwidth: {} GB/s".format(float(it * batch_size * sample_size) / (tend - tstart)), 0)

        #write results to file
        if (nr >= 0) and (hvd.rank() == 0):
            mode = ('a' if nr > 0 else 'w+')
            with open(pargs.outputfile, mode) as f:
                f.write("run {}:\n".format(nr + 1))
                f.write("total time: {} seconds for {} samples\n".format(tend - tstart, it * batch_size))
                f.write("iteration time: {} seconds/sample\n".format((tend - tstart)/float(it * batch_size)))
                f.write("bandwidth: {} GB/s\n".format(float(it * batch_size * sample_size) / (tend - tstart)))
                f.write("\n")


if __name__== "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--outputfile", type=str, help="Full path to output file.")
    AP.add_argument("--num_raid", type=int, default=4, choices=[4, 8], help="Number of raid drives.")
    AP.add_argument("--num_warmup_runs", type=int, default=1, help="Number of warmup experiments to run.")
    AP.add_argument("--num_runs", type=int, default=1, help="Number of experiments to run.")
    AP.add_argument("--batch_size", type=int, default=16, help="Global batch size. Make sure it is bigger than the number of ranks.")
    AP.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent workers")
    AP.add_argument("--shuffle", action='store_true')
    AP.add_argument("--visualize", action='store_true')
    AP.add_argument("--preprocess", action='store_true')
    AP.add_argument("--inference", action='store_true')
    AP.add_argument("--enable_fp16", action='store_true')
    parsed = AP.parse_args()
    
    main(parsed)

