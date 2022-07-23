#!/bin/bash

# CUDA
export CUDA_HOME=/usr/local/cuda-10.2/
# MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps/nvidia-log
# 0 or 1 (default is 0)	Disables (when set to 1) or enables (when set to 0) asynchronous kernel launches.
export CUDA_LAUNCH_BLOCKING=0
# 1 to 32 (default is 8)	Sets the number of compute and copy engine concurrent connections (work queues) from the host to each device of compute capability 3.5 and above.
export CUDA_DEVICE_MAX_CONNECTIONS=32

# torch==1.9.0 is ok, but 1.10.0 fails to run the native megatron-lm source code
# JIT cannot deal with input tensor without concrete number of dimensions
export PYTORCH_JIT=0

# Distributed Env
export MASTER_ADDR=localhost
export MASTER_PORT=6001

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# DISTRIBUTED_ARGS="\
#        --nproc_per_node $GPUS_PER_NODE \
#        --nnodes $NNODES --node_rank $NODE_RANK \
#        --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Data
DATA_PATH=./data/my-bert-en_text_sentence
CHECKPOINT_PATH=./checkpoints/bert_345m
VOCAB_FILE=./data/bert-large-cased-vocab.txt

# Todo. Hard code. @gl
cp ./scripts/distributed_c10d._gl_.py /usr/local/lib/python3.6/dist-packages/torch/distributed/distributed_c10d.py
cp ./scripts/deepspeed_cpu_adam._gl_.py /usr/local/lib/python3.6/dist-packages/deepspeed/ops/adam/cpu_adam.py
# cp ./scripts/function._gl_.py /usr/local/lib/python3.6/dist-packages/torch/autograd/function.py

# Model defination
NUM_LAYERS=${1-12}
HIDDEN_SIZE=${2-8192}
HEADS=${3-16}
SEQ_LEN=${4-512}
BATCH_SIZE=${5-32}
WINDOW_SIZE=${6-2}

# for data parallel
# GLOBAL_BATCH_SIZE=$((8 * ${BATCH_SIZE} * ${WORLD_SIZE}))
GLOBAL_BATCH_SIZE=${BATCH_SIZE} #64

RAY_MAX_CONCURRENCY=$((86 / ${GPUS_PER_NODE}))

PYTHON_CMD="python pretrain_bert.py \
    --tensor-model-parallel-size ${GPUS_PER_NODE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size  ${HIDDEN_SIZE} \
    --num-attention-heads ${HEADS} \
    --seq-length ${SEQ_LEN} \
    --micro-batch-size ${BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --max-position-embeddings ${SEQ_LEN} \
    --train-iters 60 \
    --lr-decay-iters 990000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.0001 \
    --min-lr 0.00001 \
    --lr-decay-style linear \
    --lr-warmup-fraction .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10000 \
    --activations-checkpoint-method 'uniform' \
    --activations-checkpoint-num-layers 1 \
    --enable-gl \
    --gl-world-size ${WORLD_SIZE} \
    --gl-enable-ddp \
    --gl-window-size ${WINDOW_SIZE} \
    --gl-debug-print \
    --gl-ray-max-concurrency ${RAY_MAX_CONCURRENCY}"

# --fp16 \
# --use-cpu-initialization \

# File "/usr/local/lib/python3.6/dist-packages/torch/nn/parallel/distributed.py", line 495, in __init__
# ValueError: DistributedDataParallel's input module must be on the same type of devices, but input module parameters locate in {'cpu', 'cuda'}.
# --DDP-impl torch \

# Launch distributed scripts
for r in $(echo ${CUDA_VISIBLE_DEVICES} | tr "," '\n'); do
    ENV="CUDA_VISIBLE_DEVICES=$r \
        WORLD_SIZE=${WORLD_SIZE}  \
        LOCAL_RANK=$r \
        NODE_RANK=${NODE_RANK} \
        RANK=$(($r + $GPUS_PER_NODE * $NODE_RANK))"

    CMD=$(echo $ENV $PYTHON_CMD | tr -s [:space:])

    echo -e $CMD '\n\n'
    eval $CMD &
done

wait < <(jobs -p)
