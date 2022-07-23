#!/bin/bash

./examples/run.sh -m "megatron-lm" -l 32 -h 2048 && \
./examples/run.sh -m "l2l" -l 78 -h 2048 && \
./examples/run.sh -m "zero-offload" -l 48 -h 2048 && \
./examples/run.sh -m "zero-infinity" -l 48 -h 2048 && \
./examples/run.sh -m "stronghold" -l 100 -h 2048
