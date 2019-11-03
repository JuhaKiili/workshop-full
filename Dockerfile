FROM tensorflow/tensorflow:latest-gpu-py3
ENV VALOHAI_CONFIG_DIR /work/.valohai
USER root
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 zip unzip
RUN python -m pip install --upgrade pip setuptools wheel
COPY requirements.txt /work/requirements.txt
RUN pip install -r /work/requirements.txt