#!/bin/bash

#loaf pytorch
#source activate pytorch-gds

#env
export OMPI_MCA_btl=^openib

#total number of ranks
totalranks=16

#mpi options
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#global parameters
num_warmup_runs=2
num_runs=10
max_threads=8

for batch_size in 16 32 64 128; do
    for max_workers in 1 2 4 8; do

        #check if we have too many threads
        if [[ ${max_workers} -gt ${max_threads} ]]; then
            break
        fi

	#check if output already exists
	outputpath="/data1/cam5_data/performance"
	outputfilename="iostats_hdf5_featurespreprocess-shuffle_bs${batch_size}_ninterthreads${max_workers}_nintrathreads1.out"

	if [ ! -d "${outputpath}" ]; then
	    mkdir -p ${outputpath}
	fi

	if [ -f "${outputpath}/${outputfilename}" ]; then
	    echo "${outputpath}/${outputfilename} already exists, skipping"
	    continue
	fi

	#run the stuff
	srun -n ${totalranks} ${slurmoptions} python train_hdf5_ddp.py \
	       --outputfile "${outputpath}/${outputfilename}" \

    done
done

