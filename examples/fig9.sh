#!/bin/bash

./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 2 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 4 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 6 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 8 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 10 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 12 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 14
