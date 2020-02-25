#!/bin/bash

#nvidia-docker build -t tkurth/pytorch-bias_gan:latest .
nvidia-docker build -t tkurth/pytorch-deepcam_mlperf:latest .
#docker push registry.services.nersc.gov/tkurth/pytorch-bias_gan:latest

#run docker test
#docker run --device=/dev/nvidia-fs0 --workdir "/opt/pytorch/numpy_reader/scripts" -it tkurth/pytorch-numpy_reader:latest ./reader_test.sh
#nvidia-docker run --workdir "/opt/pytorch/numpy_reader/scripts" -it tkurth/pytorch-numpy_reader:latest ./reader_test.sh
