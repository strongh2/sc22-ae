#!/bin/bash

# CUDA
export CUDA_HOME=/usr/local/cuda-10.2/
# MPS
# export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/nvidia-mps
# export CUDA_MPS_LOG_DIRECTORY=/tmp/mps/nvidia-log

# torch==1.9.0 is ok, but 1.10.0 fails to run the native megatron-lm source code
# JIT cannot deal with input tensor without concrete number of dimensions
export PYTORCH_JIT=0

# Distributed Env
export MASTER_ADDR=localhost
export MASTER_PORT=6001
#export CUDA_VISIBLE_DEVICES=1,4,0,5,2,3,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS_PER_NODE=${6-8}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((${GPUS_PER_NODE} * ${NNODES}))

DISTRIBUTED_ARGS="\
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

# Data
DATA_PATH=./data/my-bert-en_text_sentence
CHECKPOINT_PATH=./checkpoints/bert_345m
VOCAB_FILE=./data/bert-large-cased-vocab.txt

# Todo. Hard code. @gl
cp ./scripts/distributed_c10d._v1.10.0_.py /usr/local/lib/python3.6/dist-packages/torch/distributed/distributed_c10d.py
cp ./scripts/deepspeed_cpu_adam._v0.5.8_.py /usr/local/lib/python3.6/dist-packages/deepspeed/ops/adam/cpu_adam.py
cp ./scripts/function._v1.10.0_.py /usr/local/lib/python3.6/dist-packages/torch/autograd/function.py

# Model defination
NUM_LAYERS=${1-12}
HIDDEN_SIZE=${2-8192}
HEADS=${3-16}
SEQ_LEN=${4-512}
BATCH_SIZE=${5-32}

# for data parallel
# GLOBAL_BATCH_SIZE=$((8 * $BATCH_SIZE * $WORLD_SIZE))
GLOBAL_BATCH_SIZE=64

if ((${GPUS_PER_NODE} <= 2)); then
       python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
              pretrain_bert.py \
              --pipeline-model-parallel-size ${GPUS_PER_NODE} \
              --num-layers ${NUM_LAYERS} \
              --hidden-size ${HIDDEN_SIZE} \
              --num-attention-heads ${HEADS} \
              --seq-length ${SEQ_LEN} \
              --micro-batch-size ${BATCH_SIZE} \
              --global-batch-size ${GLOBAL_BATCH_SIZE} \
              --max-position-embeddings ${SEQ_LEN} \
              --train-iters 60 \
              --save ${CHECKPOINT_PATH} \
              --load ${CHECKPOINT_PATH} \
              --data-path ${DATA_PATH} \
              --vocab-file ${VOCAB_FILE} \
              --data-impl mmap \
              --split 949,50,1 \
              --distributed-backend nccl \
              --lr 0.0001 \
              --lr-decay-style linear \
              --min-lr 1.0e-5 \
              --lr-decay-iters 990000 \
              --weight-decay 1e-2 \
              --clip-grad 1.0 \
              --lr-warmup-fraction .01 \
              --log-interval 10 \
              --save-interval 10000 \
              --eval-interval 10000 \
              --eval-iters 10000 \
              --DDP-impl local \
              --gl-enable-ddp \
              --activations-checkpoint-method 'uniform' \
              --activations-checkpoint-num-layers 1
else
       # --tensor-model-parallel-size ${GPUS_PER_NODE} \
       python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
              pretrain_bert.py \
              --pipeline-model-parallel-size ${GPUS_PER_NODE} \
              --num-layers-per-virtual-pipeline-stage $(($NUM_LAYERS / $GPUS_PER_NODE)) \
              --num-layers ${NUM_LAYERS} \
              --hidden-size ${HIDDEN_SIZE} \
              --num-attention-heads ${HEADS} \
              --seq-length ${SEQ_LEN} \
              --micro-batch-size ${BATCH_SIZE} \
              --global-batch-size ${GLOBAL_BATCH_SIZE} \
              --max-position-embeddings ${SEQ_LEN} \
              --train-iters 60 \
              --save ${CHECKPOINT_PATH} \
              --load ${CHECKPOINT_PATH} \
              --data-path ${DATA_PATH} \
              --vocab-file ${VOCAB_FILE} \
              --data-impl mmap \
              --split 949,50,1 \
              --distributed-backend nccl \
              --lr 0.0001 \
              --lr-decay-style linear \
              --min-lr 1.0e-5 \
              --lr-decay-iters 990000 \
              --weight-decay 1e-2 \
              --clip-grad 1.0 \
              --lr-warmup-fraction .01 \
              --log-interval 10 \
              --save-interval 10000 \
              --eval-interval 10000 \
              --eval-iters 10000 \
              --DDP-impl local \
              --gl-enable-ddp \
              --activations-checkpoint-method 'uniform' \
              --activations-checkpoint-num-layers 1
       #--fp16 \
fi
