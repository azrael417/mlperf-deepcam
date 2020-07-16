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

import os

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def compute_score(prediction, gt, num_classes, device_id, type="iou", weights=None):
    #flatten input
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    iou = [0.] * num_classes

    #cast type for GT tensor. not needed for new
    #pytorch but much cleaner
    gt = gt.type(torch.long)
    
    equal = (prediction == gt)
    not_equal = (prediction != gt)
    for j in range(0, num_classes):
        #true positve: prediction and gt agree and gt is of class j
        tp[j] += torch.sum(equal[gt == j])
        #false positive: prediction is of class j and gt not of class j
        fp[j] += torch.sum(not_equal[prediction == j])
        #false negative: prediction is not of class j and gt is of class j
        fn[j] += torch.sum(not_equal[gt == j])

    for j in range(0, num_classes):
        union = tp[j] + fp[j] + fn[j]
        if union.item() == 0:
            iou[j] = torch.tensor(1.)
        else:
            iou[j] = tp[j].float() / union.float()
    
    return sum(iou)/float(num_classes)
