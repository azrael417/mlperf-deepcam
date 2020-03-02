#!/bin/bash
#SBATCH -A dasrepo
#SBATCH -J train_cam5
#SBATCH -t 08:00:00
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -S 2

#load stuff
conda activate mlperf_deepcam
module load pytorch/v1.4.0

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
srun -N ${SLURM_NNODES} -n ${totalranks} -c $(( 256 / ${rankspernode} )) --cpu_bind=cores \
     python train_hdf5_ddp.py \
     --wireup_method "mpi" \
     --wandb_certdir ${output_dir}/.. \
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
