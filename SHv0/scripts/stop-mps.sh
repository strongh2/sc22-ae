#!/bin/bash

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/nvidia-mps # Select a location that’s
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps/nvidia-log # Select a location that’s

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Select GPU 0.

echo quit | nvidia-cuda-mps-control
# ps aux | grep nvidia-cuda-mps-control | grep -v 'grep' | awk '{print $2}' | xargs kill -9
# ps aux | grep nvidia-cuda-mps-server | grep -v 'grep' | awk '{print $2}' | xargs kill -9

# Set the GPUs to exclusive mode
for ((i=0;i<8;i++))
{  
    sudo nvidia-smi -i $i -c 0 
}

rm -rf ${CUDA_MPS_PIPE_DIRECTORY}
rm -rf ${CUDA_MPS_LOG_DIRECTORY}


cp ./scripts/distributed_c10d._v1.10.0_.py /usr/local/lib/python3.6/dist-packages/torch/distributed/distributed_c10d.py
