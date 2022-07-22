#!/bin/bash

rm -rf /tmp/*.pt
rm -rf /tmp/ray_ssd/*
rm -rf ./checkpoints/* 

# CUDA
export CUDA_HOME=/usr/local/cuda/
# MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps/nvidia-log

# torch==1.9.0 is ok, but 1.10.0 fails to run the native megatron-lm source code
# JIT cannot deal with input tensor without concrete number of dimensions
export PYTORCH_JIT=0

# Distributed Env
RANK=0
WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=6001
GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=`seq -s ',' 0 1 $(( $GPUS_PER_NODE-1 ))`

# Data
_BASE=/home/sys/STRONGHOLD/data
DATA_PATH=${_BASE}/my-gpt2-en_text_document
VOCAB_PATH=${_BASE}/gpt2-vocab.json
MERGE_PATH=${_BASE}/gpt2-merges.txt
CHECKPOINT_PATH=./checkpoints/gpt2

# Todo. Hard code. @gl
PYTHON_LIB=/usr/local/lib/python3.8/dist-packages
cp ./scripts/distributed_c10d._gl_.py ${PYTHON_LIB}/torch/distributed/distributed_c10d.py
cp ./scripts/deepspeed_cpu_adam._gl_.py ${PYTHON_LIB}/deepspeed/ops/adam/cpu_adam.py

# Model defination
NUM_LAYERS=${1-24}
HIDDEN_SIZE=${2-2560}
HEADS=${3-16}
SEQ_LEN=${4-1024}
BATCH_SIZE=${5-4}

WINDOW_SIZE=${6-4}

# GLOBAL_BATCH_SIZE=$((8 * ${BATCH_SIZE} * ${WORLD_SIZE}))
GLOBAL_BATCH_SIZE=${BATCH_SIZE}

CMD="PYTHONGIL=1 python pretrain_gpt.py \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${HEADS} \
       --seq-length ${SEQ_LEN} \
       --micro-batch-size ${BATCH_SIZE} \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --max-position-embeddings ${SEQ_LEN} \
       --train-iters 50 \
       --log-interval 10 \
       --exit-interval 50 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH \
       --merge-file ${MERGE_PATH} \
       --data-impl mmap \
       --distributed-backend nccl \
       --split 949,50,1 \
       --lr 0.00015 \
       --min-lr 0.00001 \
       --lr-decay-style cosine \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 10 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 1000 \
       --checkpoint-activations \
       --activations-checkpoint-method 'uniform' \
       --activations-checkpoint-num-layers 1 \
       --enable-gl \
       --use-cpu-initialization \
       --gl-world-size ${WORLD_SIZE} \
       --gl-window-size ${WINDOW_SIZE} \
       --gl-ray-max-concurrency 12
       "

echo $CMD
eval $CMD
