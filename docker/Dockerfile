FROM tensorflow/tensorflow:1.9.0-devel-gpu-py3
ENV VALOHAI_CONFIG_DIR /work/.valohai
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install tflearn
RUN pip install argparse
RUN pip install h5py
RUN pip install valohai-cli
RUN apt-get -y install zip unzip
RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

