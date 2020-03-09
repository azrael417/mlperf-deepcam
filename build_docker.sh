#!/bin/bash

#login
#docker login -u tkurth gitlab-master.nvidia.com:5005

#nvidia-docker build -t tkurth/pytorch-bias_gan:latest .
nvidia-docker build -t gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:debug .
docker push gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:debug

#tag for NERSC registry
docker tag gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:debug registry.services.nersc.gov/tkurth/mlperf-deepcam:debug
docker push registry.services.nersc.gov/tkurth/mlperf-deepcam:debug

