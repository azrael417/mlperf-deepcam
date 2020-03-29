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
