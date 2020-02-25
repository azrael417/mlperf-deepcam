import timeit
from datetime import datetime
import os
import numpy as np

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.utils import make_grid
from torchvision.utils import save_image
# Custom includes
from dataloaders import utils
from networks import deeplab_xception, deeplab_resnet

from torchvision import transforms

from cam_dataset import get_split_random
import losses
         
torch.manual_seed(333)
  
device = torch.device("cuda")  
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

# Setting parameters
nEpochs = 20  # Number of epochs for training
batch_size = 16  # Training batch size

# Network definition
net = torch.nn.DataParallel(deeplab_xception.DeepLabv3_plus(nInputChannels=4, n_classes=3, os=16, pretrained=False))
#net.load_state_dict(torch.load("model.pth"))
loss_pow = -1/8
class_weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
fpw_1 = 2.61461122397522257612
fpw_2 = 1.71641974795896018744
criterion = losses.fp_loss
net.to(device)

# Use the following optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3, eps=1e-8, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=3, verbose=1)

train, validation, test = get_split_random('/global/cscratch1/sd/amahesh/gb_data/All-Hist/', [0,1,2,10], train_size=30000, validation_size=1000, test_size=1000)

train_loader = DataLoader(train, batch_size=batch_size, num_workers=8, drop_last=True)
val_loader = DataLoader(validation, batch_size=batch_size, num_workers=8, drop_last=True)

num_img_tr = len(train_loader)
running_loss_tr = 0.0
global_step = 0
print("Training Network")
start_time = timeit.default_timer()

# Main Training and Testing Loop
for epoch in range(0, nEpochs):
    print('epoch ', epoch)
    ious = []

    net.train()
    for batch_index, (inputs, labels) in enumerate(train_loader):
        
        inputs = inputs.permute(0,3,1,2)
        
        # Forward-Backward of the mini-batch
        inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
        global_step += inputs.data.shape[0]
        
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net.forward(inputs)
        
        loss = criterion(outputs, labels, weight=class_weights, fpw_1=fpw_1, fpw_2=fpw_2)
        running_loss_tr += loss.item()
        
        predictions = torch.max(outputs, 1)[1]
        iou = utils.get_iou(predictions, labels, n_classes=3) / batch_size
        ious.append(iou)
        
        # Print stuff
        '''
        if batch_index % 2 == 0:
            running_loss_tr = running_loss_tr / num_img_tr
            stop_time = timeit.default_timer()
            print("Epoch: ", epoch, "numImages: ", ((batch_index+1) * batch_size), " at time ", str(stop_time - start_time))
            print('Current loss: %f' % loss)
            print('Current iou: %f' % iou)
        

            running_loss_tr = 0
            
        '''
            
        # Backward the averaged gradient
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        
    
    stop_time = timeit.default_timer()
    print("Finished epoch at time ", stop_time)
    print("train iou mean: ", np.mean(ious))
    print("train iou std: ", np.std(ious))
    print("train iou min: ", np.min(ious))
    print("train iou max: ", np.max(ious))

    net.eval()
    ious = []
    val_losses = []
    for ii, sample_batched in enumerate(val_loader):
        inputs, labels = sample_batched[0], sample_batched[1]
        
        inputs = inputs.permute(0,3,1,2)
          
        # Forward pass of the mini-batch
        inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = net.forward(inputs)

            predictions = torch.max(outputs, 1)[1]
            val_loss = criterion(outputs, labels, weight=class_weights)

        predictions = torch.max(outputs, 1)[1]
        iou = utils.get_iou(predictions, labels, n_classes=3) / batch_size
        ious.append(iou)
        val_losses.append(val_loss)
                
        if ii == 0:
            grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, padding=30, pad_value=100, normalize=False,
                                       range=(0, 255))
            save_image(grid_image, 'Predictions.jpg')
            grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, padding=30, pad_value=100, normalize=False, range=(0, 255))
            save_image(grid_image, 'Labels.jpg')
                
    print("val iou mean: ", np.mean(ious))
    print("val iou std: ", np.std(ious))
    print("val iou min: ", np.min(ious))
    print("val iou max: ", np.max(ious))
    
    scheduler.step(np.mean(ious))
          
    # Save the model
    torch.save(net.state_dict(), "model_allhist.pth")
    print("Saved model.")