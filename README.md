# Deep Learning Climate Segmentation Benchmark

PyTorch implementation for the climate segmentation benchmark, based on the
Exascale Deep Learning for Climate Analytics codebase here:
https://github.com/azrael417/ClimDeepLearn, and the paper:
https://arxiv.org/abs/1810.01993

## Dataset

The dataset for this benchmark comes from CAM5 [1] simulations and is hosted at
NERSC. The samples are stored in HDF5 files with input images of shape
(768, 1152, 16) and pixel-level labels of shape (768, 1152). The labels have
three target classes (background, atmospheric river, tropical cycline) and were
produced with TECA [2].

The current recommended way to get the data is to use GLOBUS and the following
globus endpoint:

https://app.globus.org/file-manager?origin_id=0b226e2c-4de0-11ea-971a-021304b0cca7&origin_path=%2F

The dataset folder contains a README with some technical description of the
dataset and an All-Hist folder containing all of the data files.

### Preprocessing
The dataset is split into train/val/test and ships with the `stats.h5` file containing summary statistics.

## Before you run

Make sure you have a working python environment with `pytorch` and `h5py` setup. 
If you want to use learning rate warmup, you must also install the warmup-scheduler package
available at https://github.com/ildoonet/pytorch-gradual-warmup-lr.

## How to run the benchmark

Submission example scripts are in `src/deepCam/run_scripts`.

## Hyperparameters

The table below contains the modifiable hyperparameters. Unless otherwise stated, parameters not
listed in the table below are fixed and changing those could lead to an invalid submission.

|Parameter Name |Default | Allowed Range  | Description|
--- | --- | --- | ---
`--optimizer` | `"Adam"` | Optimizer of ADAM or LAMB* type. This benchmark implements "Adam", "AdamW" from PyTorch and "FusedLAMB" from NVIDIA APEX. Algorithmic equivalent implementations to those listed before are allowed. | The optimizer to choose
`--start_lr` | 1e-3 | >= 0. | Start learning rate (or base learning rate if warmup is used)
`--optimizer_betas` | `[0.9, 0.999]` | N/A | Momentum terms for Adam-type optimizers
`--weight_decay` | 1e-6 | >= 0. | L2 weight regularization term
`--lr_warmup_steps` | 0 | >= 0 | Number of steps for learning rate warmup
`--lr_warmup_factor` | 1. | >= 1. | When warmup is used, the target learning_rate will be lr_warmup_factor * start_lr
`--lr_schedule` | - | `type="multistep",milestones="<milestone_list>",decay_rate="<value>"` | Specifies the learning rate schedule. Multistep decays the current learning rate by `decay_rate` at every milestone in the list. Note that the milestones are in unit of steps, not epochs. Number and value of milestones and the `decay_rate` can be chosen arbitrarily. For a milestone list, please specify it as whitespace separated values, for example `milestones="5000 10000"`
`--batchnorm_group_size` | 1 | >= 1 | Determines how many ranks participate in the batchnorm. Specifying a value > 1 will replace nn.BatchNorm2d with nn.SyncBatchNorm everywhere in the model. Currently, nn.SyncBatchNorm only supports node-local batch normalization, but using an Implementation of that same functionality which span arbitrary number of workers is allowed
`--seed` | 333 | Arbitrary but varying | Random number generator seed. Multiple submissions which employ the same seed are discouraged. Please specify a seed depending on system clock or similar.

*LAMB optimizer has additional hyperparameters such as the global grad clipping norm value. For the purpose of this benchmark, consider all those parameters which are LAMB specific and fixed. The defaults are specified in the [NVIDIA APEX documentation for FusedLAMB](https://nvidia.github.io/apex/_modules/apex/optimizers/fused_lamb.html).

### Using Docker

The implementation comes with a Dockerfile optimized for NVIDIA workstations but usable on 
other NVIDIA multi-gpu systems. Use the Dockerfile 
`docker/Dockerfile.train` to build the container and the script `src/deepCam/run_scripts/run_training.sh`
for training. The data_dir variable should point to the full path of the `All-Hist` directory containing the downloaded dataset.

## References

1. Wehner, M. F., Reed, K. A., Li, F., Bacmeister, J., Chen, C.-T., Paciorek, C., Gleckler, P. J., Sperber, K. R., Collins, W. D., Gettelman, A., et al.: The effect of horizontal resolution on simulation quality in the Community Atmospheric Model, CAM5. 1, Journal of Advances in Modeling Earth Systems, 6, 980-997, 2014.
2. Prabhat, Byna, S., Vishwanath, V., Dart, E., Wehner, M., Collins, W. D., et al.: TECA: Petascale pattern recognition for climate science, in: International Conference on Computer Analysis of Images and Patterns, pp. 426-436, Springer, 2015b.
