#!/bin/bash

data_root=/raid/tkurth/cam5_data

#inference runs
nvidia-docker run \
	      --ipc host \
	      --volume "${data_root}:/data:rw" \
	      --workdir "/opt/deepCam" -it gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:debug /bin/bash
