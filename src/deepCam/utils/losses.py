import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

def fp_loss(logit, target, weight, fpw_1=0, fpw_2=0):
    
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    
    #later should use cuda
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().to(target.device), reduction='none')
    losses = criterion(logit, target.long())
    
    preds = torch.max(logit, 1)[1]
    
    #is fp 1
    is_fp_one = (torch.eq(preds, 1) & torch.ne(preds, 1)).float()
    fp_matrix_one = (is_fp_one * fpw_1) + 1
    losses = torch.mul(fp_matrix_one, losses)
        
    #is fp 1
    is_fp_two = (torch.eq(preds, 2) & torch.ne(preds, 2)).float()
    fp_matrix_two = (is_fp_two * fpw_2) + 1
    losses = torch.mul(fp_matrix_two, losses)
    
    loss = torch.mean(losses)

    return loss
