FROM nvcr.io/nvidia/pytorch:20.03-py3

#Install newer version of nsight
COPY sys/nsight.tgz /opt
RUN cd /opt && \
    tar -xzf nsight.tgz && rm -rf nsight.tgz
ENV PATH /opt/NsightCompute:${PATH}
ENV LD_LIBRARY_PATH /opt/conda/lib/python3.6/site-packages/torch/lib:${LD_LIBRARY_PATH}

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
