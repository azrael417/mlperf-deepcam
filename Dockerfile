#FROM gitlab-master.nvidia.com:5005/dl/dgx/pytorch:19.10-py3-devel
FROM nvcr.io/nvidia/pytorch:20.01-py3
#FROM gitlab-master.nvidia.com:5005/dl/dgx/pytorch:20.01-py3

#Install conda prereqs
RUN conda config --add channels conda-forge \
    && conda install matplotlib basemap basemap-data-hires pillow h5py
ENV PROJ_LIB /opt/conda/share/proj

#install mpi4py
RUN pip install mpi4py

#pip install more python modules
RUN pip install wandb

#copy additional stuff
COPY ./src/deepCam /opt/deepCam
COPY ./src/utils /opt/utils

#init empty git repo so that wandb works
RUN cd /opt/deepCam && git init

#copy cert:
RUN mkdir -p /opt/certs
COPY no-git/wandb_cert.key /opt/certs/.wandbirc

#create additional folders for mapping data in
RUN mkdir -p /data
