#!/bin/bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/nvidia-mps # Select a location
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps/nvidia-log # Select a location 

mkdir -p ${CUDA_MPS_PIPE_DIRECTORY}
mkdir -p ${CUDA_MPS_LOG_DIRECTORY}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Select GPU 0.
#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=4, 5, 6, 7

# Set the GPUs to exclusive mode
for((i=0;i<8;i++)); do
    sudo nvidia-smi -i $i -c EXCLUSIVE_PROCESS #> /dev/null 2>&1 
done


# Start the control daemon on CPU core #0 per Section 2.3.5.2
ps -ef | grep -v grep | grep "nvidia-cuda-mps-control" 
if [ $? -ne 0 ]; then 
    #taskset -c 0 nvidia-cuda-mps-control -d 
    nvidia-cuda-mps-control -d # Start the daemon.
fi

cp ./scripts/distributed_c10d._gl_.py /usr/local/lib/python3.6/dist-packages/torch/distributed/distributed_c10d.py