#!/bin/bash

srun -A hpc --mpi=pmix -N 1 -n 4 --pty \
     --container-workdir=/opt/utils \
     --container-image=gitlab-master.nvidia.com/tkurth/mlperf-deepcam:debug \
     --container-mounts=/gpfs/fs1/tkurth/cam5_dataset/All-Hist:/data bash
