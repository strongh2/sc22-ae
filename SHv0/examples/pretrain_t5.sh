#!/bin/bash

RANK=0
WORLD_SIZE=1
DATA_PATH=./data/my-t5-en_text_sentence
VOCAB_FILE=./data/bert-large-uncased-vocab.txt
CHECKPOINT_PATH=./checkpoints/t5_base

export MASTER_ADDR=localhost
export MASTER_PORT=6000

python pretrain_t5.py \
       --num-layers 2 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --kv-channels 64 \
       --ffn-hidden-size 3072 \
       --encoder-seq-length 512 \
       --decoder-seq-length 128 \
       --micro-batch-size 16 \
       --global-batch-size 16 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --lr-decay-iters 1000000 \
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
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --vocab-extra-ids 100 \
       --dprint
