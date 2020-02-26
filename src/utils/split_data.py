import os
import numpy as np

filedir="/gpfs/fs1/tkurth/cam5_dataset/All-Hist/all"
train_fraction=0.8
validation_fraction=0.1
test_fraction=0.1

files = sorted([ os.path.join(filedir, x) for x in os.listdir(filedir) ])

print(files)