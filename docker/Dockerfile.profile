#FROM gitlab-master.nvidia.com:5005/dl/dgx/pytorch:19.10-py3-devel
FROM nvcr.io/nvidia/pytorch:20.01-py3
#FROM gitlab-master.nvidia.com:5005/dl/dgx/pytorch:20.01-py3

#Install conda prereqs
RUN conda install h5py

#install pycuda
RUN pip install --no-cache-dir pycuda

#install mpi4py
RUN pip install mpi4py

#copy additional stuff
COPY src/deepCam /opt/deepCam
COPY src/utils /opt/utils

#init empty git repo so that wandb works
RUN cd /opt/deepCam && git init

#create additional folders for mapping data in
RUN mkdir -p /data
