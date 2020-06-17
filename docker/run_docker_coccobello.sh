#!/bin/bash

data_root=/raid/tkurth/cam5_data

#inference runs
nvidia-docker run \
	      --ipc host \
	      --cap-add=SYS_ADMIN \
	      --volume "${data_root}:/data:rw" \
	      --workdir "/opt/deepCam/run_scripts" -it gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:profile_internal /bin/bash
