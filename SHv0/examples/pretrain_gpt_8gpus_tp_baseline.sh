#!/bin/bash

rm -rf /tmp/*.pt

# CUDA
export CUDA_HOME=/usr/local/cuda/
# MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps/nvidia-log

# torch==1.9.0 is ok, but 1.10.0 fails to run the native megatron-lm source code
# JIT cannot deal with input tensor without concrete number of dimensions
export PYTORCH_JIT=0

# Distributed Env
export MASTER_ADDR=localhost
export MASTER_PORT=6001

GPUS_PER_NODE=2
#GPUS_PER_NODE=8                                                                                                                                       
export CUDA_VISIBLE_DEVICES=`seq -s ',' 0 1 $(( $GPUS_PER_NODE-1 ))`

NNODES=1
NODE_RANK=0
WORLD_SIZE=$((${GPUS_PER_NODE} * ${NNODES}))

DISTRIBUTED_ARGS="\
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

# Data
#_BASE=/home/admin/workspace/gpt-dataset
_BASE=/home/scxs/data 
DATA_PATH=${_BASE}/my-gpt2-en_text_document
VOCAB_PATH=${_BASE}/gpt2-vocab.json
MERGE_PATH=${_BASE}/gpt2-merges.txt
CHECKPOINT_PATH=./checkpoints/gpt2_345m_ds

# Todo. Hard code. @gl
PYTHON_LIB=/usr/local/lib/python3.8/dist-packages
cp ./scripts/distributed_c10d._v1.10.0_.py ${PYTHON_LIB}/torch/distributed/distributed_c10d.py
cp ./scripts/deepspeed_cpu_adam._v0.5.8_.py ${PYTHON_LIB}/deepspeed/ops/adam/cpu_adam.py
cp ./scripts/function._v1.10.0_.py ${PYTHON_LIB}/torch/autograd/function.py

# Model defination
NUM_LAYERS=${1-19}
HIDDEN_SIZE=${2-2560}
HEADS=${3-16}
SEQ_LEN=${4-1024}
BATCH_SIZE=${5-4}

# for data parallel
TENSOR_PARALLEL=${GPUS_PER_NODE}

# GLOBAL_BATCH_SIZE=$((8 * $BATCH_SIZE * $WORLD_SIZE))
GLOBAL_BATCH_SIZE=$((4 * ${GPUS_PER_NODE} / $TENSOR_PARALLEL))

python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
      pretrain_gpt.py \
      --tensor-model-parallel-size ${TENSOR_PARALLEL} \
      --pipeline-model-parallel-size 1 \
      --num-layers ${NUM_LAYERS} \
      --hidden-size ${HIDDEN_SIZE} \
      --num-attention-heads ${HEADS} \
      --seq-length ${SEQ_LEN} \
      --micro-batch-size ${BATCH_SIZE} \
      --global-batch-size ${GLOBAL_BATCH_SIZE} \
      --max-position-embeddings 1024 \
      --train-iters 50 \
      --exit-interval 100 \
      --save ${CHECKPOINT_PATH} \
      --load ${CHECKPOINT_PATH} \
      --data-path ${DATA_PATH} \
      --vocab-file ${VOCAB_PATH} \
      --merge-file ${MERGE_PATH} \
      --data-impl mmap \
      --split 949,50,1 \
      --distributed-backend nccl \
      --lr 0.00015 \
      --lr-decay-style cosine \
      --min-lr 1.0e-5 \
      --lr-decay-iters 320000 \
      --weight-decay 1e-2 \
      --clip-grad 1.0 \
      --lr-warmup-fraction .01 \
      --log-interval 10 \
      --save-interval 10000 \
      --eval-interval 100 \
      --eval-iters 100 \
      --DDP-impl 'local' \
      --gl-enable-ddp \
      --activations-checkpoint-method 'uniform' \
      --activations-checkpoint-num-layers 1  \
      --enable-l2l
   
       # --num-layers-per-virtual-pipeline-stage $(($NUM_LAYERS / $GPUS_PER_NODE)) \
       #--fp16 \
