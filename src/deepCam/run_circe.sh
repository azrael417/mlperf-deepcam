#!/bin/bash
#SBATCH -A hpc
#SBATCH --mpi=pmix
#SBATCH --container-mounts=/gpfs/fs1/tkurth/cam5_dataset/All-Hist:/data:ro
#SBATCH --container-image=gitlab-master.nvidia.com/tkurth/mlperf-deepcam:debug
#SBATCH --container-workdir "/opt/deepCam"

rankspernode=16
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))

srun -N ${SLURM_NNODES} -n ${totalranks} python train_hdf5_ddp.py \
       --wireup_method "slurm" \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --model_prefix "classifier" \
       --start_lr 1e-3 \
       --validation_frequency 200 \
       --logging_frequency 0 \
       --save_frequency 400 \
       --max_epochs 30 \
       --amp_opt_level O1 \
       --local_batch_size 2 |& tee ${output_dir}/train.out
