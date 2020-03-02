#!/bin/bash
#SBATCH -A hpc
#SBATCH -J train_cam5
#SBATCH -t 08:00:00

#ranks per node
rankspernode=16
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))

#parameters
run_tag="deepcam_prediction_run3"
data_dir_prefix="/data"
output_dir="/runs/${run_tag}"

#run training
srun --wait=30 --mpi=pmix -N ${SLURM_NNODES} -n ${totalranks} -c $(( 96 / ${rankspernode} )) --cpu_bind=cores \
     --container-workdir /opt/deepCam \
     --container-mounts=/gpfs/fs1/tkurth/cam5_dataset/All-Hist:/data:ro,/gpfs/fs1/tkurth/cam5_runs:/runs:rw \
     --container-image=gitlab-master.nvidia.com/tkurth/mlperf-deepcam:debug \
     python train_hdf5_ddp.py \
       --wireup_method "nccl-slurm" \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --model_prefix "classifier" \
       --start_lr 1e-3 \
       --lr_schedule type="multistep",milestones="20000 40000",decay_rate="0.1" \
       --validation_frequency 200 \
       --max_validation_steps 50 \
       --logging_frequency 50 \
       --save_frequency 400 \
       --max_epochs 30 \
       --amp_opt_level O1 \
       --local_batch_size 2 |& tee -a ${output_dir}/train.out
