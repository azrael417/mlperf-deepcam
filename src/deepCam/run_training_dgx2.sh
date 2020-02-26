#!/bin/bash

#check number of arguments
if [ "$#" -ne 1 ]; then
    echo "Please pass a number of ranks to this bash script"
    exit
fi

#env
export OMPI_MCA_btl=^openib

#total number of ranks
totalranks=$1
run_tag="deepcam_prediction_run1"
#data_dir_prefix="/global/cfs/cdirs/dasrepo/tkurth/DataScience/ECMWF/data"
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
mpirun -np ${totalranks} ${mpioptions} ${profile} python train_hdf5_ddp.py \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --model_prefix "classifier" \
       --start_lr 1e-3 \
       --validation_frequency 5 \
       --logging_frequency 0 \
       --save_frequency 400 \
       --max_epochs 30 \
       --amp_opt_level O1 \
       --local_batch_size 2 |& tee ${output_dir}/train.out
