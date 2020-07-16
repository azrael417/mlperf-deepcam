#!/bin/bash
#SBATCH -A m1759
#SBATCH -J profile_cam5
#SBATCH -t 00:05:00
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --image=registry.services.nersc.gov/tkurth/mlperf-deepcam:profile
#SBATCH --volume="/global/cscratch1/sd/tkurth/data/cam5_data/All-Hist:/data"

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

#ranks per node
rankspernode=1
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))

#parameters
run_tag="deepcam_prediction_run1-cori"
data_dir_prefix="/data"
output_dir="./runs/profiles/${run_tag}"

#profile base
profilebase="/usr/local/cuda/bin/nv-nsight-cu-cli --profile-from-start off -f"

#create files
mkdir -p ${output_dir}
touch ${output_dir}/profile.out

#metrics
### Tensor Core utilization
metrics="smsp__inst_executed_pipe_tensor_op_hmma.sum"
#"sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active "

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
lts__t_sectors_aperture_sysmem_op_read.sum"

#do the stuff
for metric in ${metrics}; do

    #assemble profile string
    profilecmd="${profilebase} --metrics ${metric} -o ${output_dir}/profile_${metric}"

    #run training
    srun -u -N ${SLURM_NNODES} -n ${totalranks} -c $(( 40 / ${rankspernode} )) --cpu_bind=cores \
	 shifter \
	 ${profilecmd} \
	 /opt/conda/bin/python ../profile_hdf5_ddp.py \
	 --wireup_method "nccl-slurm-pmi" \
	 --run_tag ${run_tag} \
	 --data_dir_prefix ${data_dir_prefix} \
	 --output_dir ${output_dir} \
	 --max_inter_threads 0 \
	 --optimizer "Adam" \
	 --start_lr 1e-3 \
	 --num_warmup_steps 5 \
	 --num_profile_steps 1 \
	 --profile "Forward" \
	 --weight_decay 1e-2 \
	 --amp_opt_level O1 \
	 --local_batch_size 2 |& tee -a ${output_dir}/profile.out

done
