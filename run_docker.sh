#!/bin/bash

data_root=/raid/tkurth/cam5_data

#inference runs
nvidia-docker run \
	      --ipc host \
	      --volume "${data_root}:/data:rw" \
	      --workdir "/opt/deepCam" -it tkurth/pytorch-deepcam_mlperf:latest /bin/bash
