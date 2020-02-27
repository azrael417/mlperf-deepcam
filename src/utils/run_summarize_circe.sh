#!/bin/bash
#SBATCH -A hpc
#SBATCH -J summarize_cam5
#SBATCH -t 01:00:00

rankspernode=48
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))

srun --mpi=pmix -N ${SLURM_NNODES} -n ${totalranks} -c $(( 96 / ${rankspernode} )) \
     --container-workdir=/opt/utils \
     --container-mounts=/gpfs/fs1/tkurth/cam5_dataset/All-Hist:/data \
     --container-image=gitlab-master.nvidia.com/tkurth/mlperf-deepcam:debug \
     python summarize_data.py
