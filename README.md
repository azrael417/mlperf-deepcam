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
Unfortunately we don't yet have the dataset split into train/val/test, but we have a selection of scripts which achieves this. The splitting scripts are under `src/utils`. 

For *splitting* the dataset, please change the lines 5 and 6 (`inputdir` and `outputdir`) in `split_data.py` accordingly. The first variable should specify the absolute path to the full dataset, the second variable specifies the parent directory of where the train/validation/test splits end up. Instead of copying the files, symbolic links will be created. Therefore, if you plan to run the code from a container or system with different mount points than those used for the splitting, the links might be invalid and files not found. In this case, perform the splitting in the same environment used for the runs later.

For *summarizing* the dataset (i.e. computing summary statistics for input normalization), use script `summarize_data.py` in the same directory. Please modify line 85, `data_path_prefix` accordingly. It should point to the parent directory which hosts all the split, i.e. is equal to the `output_dir` from the above mentioned splitting script. Note that the summary script uses `mpi4py` for distributed computing, as the whole summarization on a single CPU can take a few hours. Once the `stats.h5` file is created, place it inside the training, test and validation directories.

### Previous dataset for ECP Annual Meeting 2019

This is a smaller dataset (~200GB total) available to get things started.
It is hosted via Globus:

https://app.globus.org/file-manager?origin_id=bf7316d8-e918-11e9-9bfc-0a19784404f4&origin_path=%2F

and also available via https:

https://portal.nersc.gov/project/dasrepo/deepcam/climseg-data-small/

## Before you run

Make sure you have a working python environment with `pytorch` and `h5py` setup. 
If you want to use learning rate warmup, you must also install the warmup-scheduler package
available at https://github.com/ildoonet/pytorch-gradual-warmup-lr.

## How to run the benchmark

Submission example scripts are in `src/deepCam/run_scripts`.

### Using Docker

The implementation comes with a Dockerfile optimized for NVIDIA workstations but usable on 
other NVIDIA multi-gpu systems. Use the Dockerfile 
`docker/Dockerfile.train` to build the container and the script `src/deepCam/run_scripts/run_training.sh`
for training.

## References

1. Wehner, M. F., Reed, K. A., Li, F., Bacmeister, J., Chen, C.-T., Paciorek, C., Gleckler, P. J., Sperber, K. R., Collins, W. D., Gettelman, A., et al.: The effect of horizontal resolution on simulation quality in the Community Atmospheric Model, CAM5. 1, Journal of Advances in Modeling Earth Systems, 6, 980-997, 2014.
2. Prabhat, Byna, S., Vishwanath, V., Dart, E., Wehner, M., Collins, W. D., et al.: TECA: Petascale pattern recognition for climate science, in: International Conference on Computer Analysis of Images and Patterns, pp. 426-436, Springer, 2015b.
