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

import re
import numpy as np
import torch
import torch.optim as optim

try:
    import apex.optimizers as aoptim
    import apex.contrib.optimizers as acoptim
    have_apex = True
except ImportError:
    from utils import optimizer as uoptim
    print("NVIDIA APEX not found")
    have_apex = False

def get_lr_schedule(start_lr, scheduler_arg, optimizer, last_step = -1):
    #add the initial_lr to the optimizer
    optimizer.param_groups[0]["initial_lr"] = start_lr

    #now check
    if scheduler_arg["type"] == "multistep":
        milestones = [ int(x) for x in scheduler_arg["milestones"].split() ]
        gamma = float(scheduler_arg["decay_rate"])
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch = last_step)
    elif scheduler_arg["type"] == "cosine_annealing":
        t_max = int(scheduler_arg["t_max"])
        eta_min = 0. if "eta_min" not in scheduler_arg else float(scheduler_arg["eta_min"])
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = t_max, eta_min = eta_min)
    else:
        raise ValueError("Error, scheduler type {} not supported.".format(scheduler_arg["type"]))


def get_optimizer(net, pargs):
    optimizer = None
    if pargs.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    elif pargs.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    elif pargs.optimizer == "LAMB":
        if have_apex:
            optimizer = aoptim.FusedLAMB(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
            #from apex.contrib.optimizers.distributed_fused_lamb import DistributedFusedLAMB
            #optimizer = DistributedFusedLAMB(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
        else:
            optimizer = uoptim.Lamb(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay, clamp_value = torch.iinfo(torch.int32).max)
    else:
        raise NotImplementedError("Error, optimizer {} not supported".format(pargs.optimizer))

    return optimizer
