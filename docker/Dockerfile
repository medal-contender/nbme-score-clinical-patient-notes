# docker-compose build --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)"

FROM nvidia/cuda:11.1.1-cudnn8-devel

# To resolve stuck during tzdata install
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y \
  vim tmux git wget locales libgl1-mesa-glx libglib2.0-0
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build

# For pygame 1.9.6 install
# Comment following lines if there is no need to use typogen
RUN apt-get update && apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev poppler-utils

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Essential package install
RUN pip install tensorboard cmake   # cmake from apt-get is too old
RUN pip install torch==1.9 torchvision==0.10 -f https://download.pytorch.org/whl/cu111/torch_stable.html

ARG DETECTRON2_VERSION
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN cd detectron2 && git checkout ${DETECTRON2_VERSION}
RUN cd detectron2 && python3 setup.py build develop

# Project Env
RUN pip install regex==2018.01.10 msgpack==0.5.6
RUN pip install pytorch-lightning==1.4.0 \
                hydra-core==1.1.0 \
                timm==0.4.12 \
                numpy==1.19.2 \
                pyarrow==2.0.0 \
                lmdb==1.2.1 \
                fvcore==0.1.5.post20210630 \
                pandas==1.3.0 \
                transformers==4.11.1 \
                linformer==0.2.1 \
                nuitka==0.6.17.3 \
                pyarmor==6.8.1 \
                munch==2.5.0 \
                seqeval==1.2.2\
                cryptography==35.0.0\
                openpyxl==3.0.9

# Set ssh environment & git clone typogen
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh \
  && echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa \
  && chmod 600 /root/.ssh/id_rsa \
  && touch /root/.ssh/known_hosts \
  && ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN apt-get install locales
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.utf8

# Remove ssh key
RUN rm -rf /root/.ssh

RUN mkdir /jigsaw
WORKDIR /jigsaw
