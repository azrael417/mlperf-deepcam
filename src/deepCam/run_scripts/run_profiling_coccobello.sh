# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/bin/bash

#ranks per node
rankspernode=1
totalranks=${rankspernode}

#env
export OMPI_MCA_btl=^openib

#profile base
profilebase="$(which nv-nsight-cu-cli) --profile-from-start off -f"

#mpi stuff
mpioptions="--allow-run-as-root --map-by ppr:4:socket:PE=3"

#parameters
run_tag="deepcam_prediction_run2-cori"
data_dir_prefix="/data"
output_dir="${data_dir_prefix}/profiles/${run_tag}"

#create files
mkdir -p ${output_dir}
touch ${output_dir}/profile.out


#metrics
### Tensor Core utilization
metrics="smsp__inst_executed_pipe_tensor_op_hmma.sum \
sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active "

### FLOP
# SP
metrics+="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum \
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum \
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum "
# HP
metrics+="smsp__sass_thread_inst_executed_op_hadd_pred_on.sum \
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum \
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum "

### Time
# CUDA Core time
metrics+="smsp__cycles_elapsed.sum \
smsp__cycles_elapsed.sum.per_second "
# Tensor Core time
metrics+="smsp__pipe_tensor_op_hmma_cycles_active.sum \
smsp__pipe_tensor_op_hmma_cycles_active.sum.per_second "

### L1 transactions
# local
metrics+="l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum \
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum "
# shared
metrics+="l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum \
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum "
# global
metrics+="l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum "
# atomic
metrics+="l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum \
l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum \
l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom.sum \
l1tex__t_set_accesses_pipe_tex_mem_surface_op_red.sum "

### L2 transactions
# read + write
metrics+="lts__t_sectors_op_read.sum \
lts__t_sectors_op_write.sum "
#atomic
metrics+="lts__t_sectors_op_atom.sum \
lts__t_sectors_op_red.sum "

### DRAM transactions
metrics+="dram__sectors_read.sum \
dram__sectors_write.sum "

### PCI/NVLINK transactions
metrics+="lts__t_sectors_aperture_sysmem_op_read.sum \
lts__t_sectors_aperture_sysmem_op_write.sum "


#run training
count=0
for metric in ${metrics}; do

    #assemble profile string
    output_prefix="${output_dir}/profile.pass_backward.batchsize_2.metric_${metric}"
    profilecmd="${profilebase} --metrics ${metric} -o ${output_prefix}"
    
    #since profiling takes ages, better not overwrite
    if [ -f "${output_prefix}.ncu-rep" ]; then
	echo "${output_prefix}.ncu-rep already exists, skipping"
	continue
    fi
    
    #run the profiling
    mpirun -np ${totalranks} ${mpioptions} \
    ${profilecmd} \
    $(which python) ../profile_hdf5_ddp.py \
	   --wireup_method "nccl-openmpi" \
	   --run_tag ${run_tag} \
	   --data_dir_prefix ${data_dir_prefix} \
	   --output_dir ${output_dir} \
	   --max_inter_threads 0 \
	   --optimizer "AdamW" \
	   --start_lr 1e-3 \
	   --num_warmup_steps 5 \
	   --num_profile_steps 1 \
	   --profile "Backward" \
	   --lr_schedule type="multistep",milestones="15000 25000",decay_rate="0.1" \
	   --weight_decay 1e-2 \
	   --amp_opt_level O1 \
	   --local_batch_size 2 |& tee -a ${output_dir}/profile.out

    count=$(( ${count} + 1 ))
done
