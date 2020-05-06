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

Unfortunately we don't yet have the dataset split into train/val/test nor a
recommended procedure for doing the split yourself. You can for now do uniform
splitting using a script similar to what is here:

https://gist.github.com/sparticlesteve/8a3e81a31e89fd1cccc81a3fae3fcf2d

### Previous dataset for ECP Annual Meeting 2019

This is a smaller dataset (~200GB total) available to get things started.
It is hosted via Globus:

https://app.globus.org/file-manager?origin_id=bf7316d8-e918-11e9-9bfc-0a19784404f4&origin_path=%2F

and also available via https:

https://portal.nersc.gov/project/dasrepo/deepcam/climseg-data-small/

## Before you run

Make sure you have a working python environment with `pytorch`, `h5py`, `basemap` and `wandb` setup. 
The training uses Weights & Biases (WandB/W&B, https://app.wandb.ai) as logging facility. 
In order to use it, please sign up, log in and create a new project. 
Create a file named `.wandbirc` containing the user login and the API key as follows:

```bash
<login> <API key>
```

Place this file in a directory accessible by the workers.

## How to run the benchmark

Submission scripts are in `src/deepCam/run_scripts`.

### Running at NERSC

To submit to the Cori KNL system, set up a conda env called
`mlperf_deepcam` which contains all the prereqs, such as `h5py`, `wandb` and `basemap`.
Please edit the entries

```bash
export PROJ_LIB=/global/homes/t/tkurth/.conda/envs/mlperf_deepcam/share/basemap
export PYTHONPATH=/global/homes/t/tkurth/.conda/envs/mlperf_deepcam/lib/python3.7/site-packages:${PYTHONPATH}
```

in `src/deepCam/run_scripts/run_training_cori.sh` to point to the correct paths and add 

```bash
--wandb_certdir <my-cert-dir>
```

and point it to the directory which contains the `.wandbirc` file.
Then run

```bash
# This example runs on 64 nodes.
cd src/deepCam/run_scripts
sbatch -N 64 run_training_cori.sh
```

### Using Docker

The implementation comes with a Dockerfile optimized for NVIDIA DGX-2 workstations. Use the Dockerfile 
`docker/Dockerfile.train` to build the container and the script `src/deepCam/run_scripts/run_training_dgx2.sh`
for training. Before building the docker container, please create a file `no-git/wandb_cert.key` 
formatted as mentioned above before building the image.

## References

1. Wehner, M. F., Reed, K. A., Li, F., Bacmeister, J., Chen, C.-T., Paciorek, C., Gleckler, P. J., Sperber, K. R., Collins, W. D., Gettelman, A., et al.: The effect of horizontal resolution on simulation quality in the Community Atmospheric Model, CAM5. 1, Journal of Advances in Modeling Earth Systems, 6, 980-997, 2014.
2. Prabhat, Byna, S., Vishwanath, V., Dart, E., Wehner, M., Collins, W. D., et al.: TECA: Petascale pattern recognition for climate science, in: International Conference on Computer Analysis of Images and Patterns, pp. 426-436, Springer, 2015b.
