#!/bin/bash
#SBATCH -A dasrepo
#SBATCH -J profile_cam5
#SBATCH -t 08:00:00
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --image=registry.services.nersc.gov/tkurth/mlperf-deepcam:profile

#ranks per node
rankspernode=1
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))

#parameters
run_tag="deepcam_prediction_run1-cori"
data_dir_prefix="/global/cscratch1/sd/tkurth/data/cam5_data/All-Hist"
output_dir="/global/cscratch1/sd/tkurth/data/cam5_runs/${run_tag}"

#create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

#run training
srun -u -N ${SLURM_NNODES} -n ${totalranks} -c $(( 40 / ${rankspernode} )) --cpu_bind=cores \
     $(which python) profile_hdf5_ddp.py \
     --wireup_method "mpi" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 0 \
     --model_prefix "classifier" \
     --optimizer "AdamW" \
     --start_lr 1e-3 \
     --num_warmup_steps 5 \
     --num_profile_steps 1 \
     --profile "Forward" \
     --lr_schedule type="multistep",milestones="15000 25000",decay_rate="0.1" \
     --lr_warmup_steps 0 \
     --lr_warmup_factor $(( ${SLURM_NNODES} / 8 )) \
     --weight_decay 1e-2 \
     --amp_opt_level O1 \
     --local_batch_size 2 |& tee -a ${output_dir}/profile.out
