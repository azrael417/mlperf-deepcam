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
        iou[j] = tp[j].float() / (tp[j].float() + fp[j].float() + fn[j].float())
    
    return sum(iou)/float(num_classes)
