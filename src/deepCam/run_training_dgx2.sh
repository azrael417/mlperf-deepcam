#!/bin/bash

#check number of arguments
if [ "$#" -ne 1 ]; then
    totalranks=1
else
    totalranks=$1
fi

#env
export OMPI_MCA_btl=^openib

#directories
run_tag="deepcam_prediction_run1"
data_dir_prefix="/data"
output_dir="${data_dir_prefix}/runs/${run_tag}"

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#profilestring
#profile="nsys profile --stats=true -f true -o numpy_rank_%q{PMIX_RANK}_metric_time.qdstrm -t osrt,cuda -s cpu -c cudaProfilerApi"
profile=""

#prepare dir:
mkdir -p ${output_dir}

#run the stuff
mpirun -np ${totalranks} ${mpioptions} python train_hdf5_ddp.py \
       --wireup_method "nccl-openmpi" \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --model_prefix "classifier" \
       --optimizer "LAMB" \
       --start_lr 1e-3 \
       --lr_schedule type="multistep",milestones="15000 25000",decay_rate="0.1" \
       --lr_warmup_steps 0 \
       --lr_warmup_factor 1. \
       --weight_decay 1e-2 \
       --validation_frequency 200 \
       --training_visualization_frequency 200 \
       --validation_visualization_frequency 40 \
       --max_validation_steps 50 \
       --logging_frequency 0 \
       --save_frequency 400 \
       --max_epochs 200 \
       --amp_opt_level O1 \
       --local_batch_size 2 |& tee -a ${output_dir}/train.out
