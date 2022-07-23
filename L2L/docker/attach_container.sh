#!/bin/bash

name=${1-offload-container}

docker exec -i -w /home/scxs/ml-accl-offloading -t $name /bin/bash
