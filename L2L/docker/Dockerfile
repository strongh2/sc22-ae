FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

EXPOSE 1-65535

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin/:${PATH}
ENV CUDA_HOME=/usr/local/cuda/

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

#RUN rm -f /etc/apt/sources.list.d/* && \
#sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
#sed -i "s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y locales locales-all && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en


##############################################################################
# build essential tools
##############################################################################
RUN apt-get install -y --no-install-recommends \
    make cmake build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

# common software tools
RUN apt-get install -y --no-install-recommends \
    libboost-all-dev libomp-dev zsh wget vim curl git sudo \
    libaio-dev htop python3-pip


##############################################################################
# pyenv, for nogil python version
##############################################################################
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

ENV _PYENV_INIT='export PYENV_ROOT="/root/.pyenv" && \
    export PATH="$PYENV_ROOT/bin:$PATH" && \
    eval "$(pyenv init -)" && \
    eval "$(pyenv virtualenv-init -)" '

RUN echo ${_PYENV_INIT} >> ~/.bashrc

ENV PYENV=/root/.pyenv/bin/pyenv

RUN ${PYENV} install nogil-3.9.10 && \
    ${PYENV} virtualenv nogil-3.9.10 nogilpy39 && \
    ${PYENV} install 3.9.10 && \
    ${PYENV} virtualenv 3.9.10 py39

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/github/
RUN mkdir -p ${STAGE_DIR}
ENV PIP_INSTALL='pip --no-cache-dir install --default-timeout=10000'

# download torch
RUN git clone --recursive https://github.com/pytorch/pytorch ${STAGE_DIR}/pytorch && \
    cd ${STAGE_DIR}/pytorch && \
    git fetch --tags && git checkout tags/v1.10.2 -b v1.10.2 && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0

# download torchvision
RUN git clone https://github.com/pytorch/vision.git ${STAGE_DIR}/vision && \
    cd ${STAGE_DIR}/vision && \
    git fetch --tags && git checkout tags/v0.11.3 -b v0.11.3

# download apex
RUN git clone https://github.com/NVIDIA/apex.git ${STAGE_DIR}/apex && \
    cd ${STAGE_DIR}/apex && \
    git fetch --tags && git checkout tags/22.03 -b 22.03

##############################################################################
# Installation/Basic Utilities
##############################################################################

# # manually install the following softwares

# # install
# ARG PY39=/root/.pyenv/versions/py39/bin/

# RUN ${PY39}/pip install deepspeed==0.5.7 regex pybind11 pyyaml && \
#     cd ${STAGE_DIR}/pytorch && \
#         ${PY39}/pip install -r requirements.txt && \
#         ${PY39}/python setup.py clean && \
#         ${PY39}/python setup.py develop --cmake && \
#     cd ${STAGE_DIR}/vision && \
#         ${PY39}/python setup.py clean && \
#         ${PY39}/python setup.py install && \
#     cd ${STAGE_DIR}/apex && \
#          ${PY39}/pip install -r ./requirements_dev.txt && \
#          ${PY39}/pip install -v --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' .

# # @scxs.
# # for supporting nogil-version cpython3.9.10, since the original file includes
# # underlying function call '_PyWeakref_ClearRef' and 'self->ob_refcnt' that not exits
# # in 'Modules/gcmodule.c' of cpython3.9.10
# COPY nogil.torch.csrc.autograd.python_variable.cpp ${STAGE_DIR}/pytorch/torch/csrc/autograd/python_variable.cpp

# ARG NOGILPY39=/root/.pyenv/versions/nogilpy39/bin/

# RUN ${NOGILPY39}/pip install deepspeed==0.5.7 ninja regex pybind11 pyyaml Pillow==8.4.0 && \
#     cd ${STAGE_DIR}/pytorch && \
#         ${NOGILPY39}/pip install -r requirements.txt && \
#         ${NOGILPY39}/python setup.py clean && \
#         ${NOGILPY39}/python setup.py develop --cmake && \
#     cd ${STAGE_DIR}/vision && \
#         ${NOGILPY39}/python setup.py clean && \
#         ${NOGILPY39}/python setup.py install && \
#     cd ${STAGE_DIR}/apex && \
#         ${NOGILPY39}/pip install -r ./requirements_dev.txt && \
#         ${NOGILPY39}/pip install -v --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' .


# # pip install ninja regex pybind11 pyyaml Pillow==8.4.0 && \
# # cd ../pytorch && pip install -r requirements.txt && \
# #     python setup.py clean && \
# #     python setup.py develop --cmake && \
# #     cd ../vision && python setup.py clean && python setup.py install && \
# #     cd ../apex && pip install -r ./requirements_dev.txt && pip install -v --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' .



##RUN useradd --create-home --uid 1000 --shell /bin/bash deepspeed
#RUN usermod -aG sudo deepspeed
#RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
## Change to non-root privilege
#USER deepspeed

###############################################################################
## Client Liveness & Uncomment Port 22 for SSH Daemon
###############################################################################
## Keep SSH client alive from server side
#RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
#RUN cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
