#!/bin/bash

./examples/run.sh -m "l2l" -l 32 -h 2048 && \
./examples/run.sh -m "zero-offload" -l 32 -h 2048 && \
./examples/run.sh -m "zero-infinity" -l 32 -h 2048
