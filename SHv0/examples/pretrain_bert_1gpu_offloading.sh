#!/bin/bash

# CUDA
export CUDA_HOME=/usr/local/cuda-10.2/
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
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7

# Data
DATA_PATH=./data/my-bert-en_text_sentence
CHECKPOINT_PATH=./checkpoints/bert_345m
VOCAB_FILE=./data/bert-large-cased-vocab.txt

# Todo. Hard code. @gl
cp ./scripts/distributed_c10d._gl_.py /usr/local/lib/python3.6/dist-packages/torch/distributed/distributed_c10d.py
cp ./scripts/deepspeed_cpu_adam._gl_.py /usr/local/lib/python3.6/dist-packages/deepspeed/ops/adam/cpu_adam.py
# cp ./scripts/function._gl_.py /usr/local/lib/python3.6/dist-packages/torch/autograd/function.py

# For DLC tasks. A10
# pip3 install --force-reinstall deepspeed

# Model defination
NUM_LAYERS=${1-12}
HIDDEN_SIZE=${2-1024}
HEADS=${3-16}
SEQ_LEN=${4-512}
BATCH_SIZE=${5-32}
WINDOW_SIZE=${6-6}

# GLOBAL_BATCH_SIZE=$((8 * ${BATCH_SIZE} * ${WORLD_SIZE}))
GLOBAL_BATCH_SIZE=64

CMD="python pretrain_bert.py \
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
       --lr-decay-iters 990000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 1000 \
       --activations-checkpoint-method 'uniform' \
       --activations-checkpoint-num-layers 1 \
       --enable-gl \
       --use-cpu-initialization \
       --gl-world-size ${WORLD_SIZE} \
       --gl-window-size ${WINDOW_SIZE} \
       --gl-debug-print \
       --gl-ray-max-concurrency 30
       "
# --fp16 \
#--gl-enable-ddp \

echo $CMD
eval $CMD
