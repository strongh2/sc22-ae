#!/bin/bash

name=${1-offload-container}

image=${2-scxs/cu11.4-torch1.10-deepspeed}

eval "docker ps -a | grep $name" && docker rm -f $name

homedir=/home/scxs

echo "starting docker image named $name"
docker run -d -t \
    --name $name \
    --network host \
    --ipc=host \
    --gpus=all \
    -v ${homedir}:/home/scxs/ \
    $image bash -c 'sleep infinity'
