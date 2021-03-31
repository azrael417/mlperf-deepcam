import torch
import torch.nn as nn
import torch.distributed as dist

class BatchNormStatsAverage(nn.Module):

    def __init__(self, model):
        super(BatchNormStatsAverage, self).__init__()
        
        # create tensor list
        for m in model.modules():
            if hasattr(m, "running_mean"):
                print(m)
            

