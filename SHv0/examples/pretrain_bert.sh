#!/bin/bash

RANK=0
WORLD_SIZE=1

DATA_PATH=./data/my-bert-en_text_sentence
CHECKPOINT_PATH=./checkpoints/bert_345m
VOCAB_FILE=./data/bert-large-cased-vocab.txt 

export MASTER_ADDR=localhost
export MASTER_PORT=6001
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/nvidia-mps # Select a location that’s
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps/nvidia-log # Select a location that’s

# torch==1.9.0 is ok, but 1.10.0 fails to run the native megatron-lm source code
# JIT cannot deal with input tensor without concrete number of dimensions
export PYTORCH_JIT=0

cp ./scripts/distributed_c10d._gl_.py /usr/local/lib/python3.6/dist-packages/torch/distributed/distributed_c10d.py

python pretrain_bert.py \
       --num-layers 240 \
       --hidden-size  1024 \
       --num-attention-heads 16 \
       --micro-batch-size 32 \
       --global-batch-size 32 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 2000 \
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
       --log-interval 10 \
       --save-interval 10000 \
       --eval-interval 10000 \
       --eval-iters 10000 \
       --fp16 \
       --activations-checkpoint-method 'uniform' \
       --activations-checkpoint-num-layers 12 \
       --enable-gl \
       --use-cpu-initialization \
       --gl-world-size ${WORLD_SIZE} \
       --gl-enable-ddp \
       --gl-window-size 24 \
       --gl-debug-print
