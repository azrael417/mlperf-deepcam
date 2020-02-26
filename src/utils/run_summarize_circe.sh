#!/bin/bash
#SBATCH -A hpc
#SBATCH -J summarize_cam5
#SBATCH -t 01:00:00
#SBATCH --mpi=pmix
#SBATCH --container-workdir=/opt/utils
#SBATCH --container-mounts=/gpfs/fs1/tkurth/cam5_dataset/All-Hist:/data
#SBATCH --container-image=gitlab-master.nvidia.com/tkurth/mlperf-deepcam:debug

rankspernode=32
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))

srun -N ${SLURM_NNODES} -n ${totalranks} python summarize_data.py
