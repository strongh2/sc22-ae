#!/bin/bash

name=${1-scxs/cu11.4-torch1.10-deepspeed}

path=`dirname $0`

docker pull nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04
#docker pull nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

docker rm -f `docker ps -a | grep $name | awk '{print $1}'`
eval "docker images | grep $name" && docker image rm $name

docker build --force-rm --network=host -t ${name} -f ${path}/Dockerfile .
