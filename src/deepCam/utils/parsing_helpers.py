import re
import numpy as np
import torch
import torch.optim as optim

def get_lr_schedule(start_lr, scheduler_arg, optimizer, last_step = -1):
    #add the initial_lr to the optimizer
    optimizer.param_groups[0]["initial_lr"] = start_lr

    #now check
    if scheduler_arg["type"] == "multistep":
        milestones = [ int(x) for x in scheduler_arg["milestones"].split() ]
        gamma = float(scheduler_arg["decay_rate"])
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch = last_step)
    else:
        raise ValueError("Error, scheduler type {} not supported.".format(scheduler_arg["type"]))
