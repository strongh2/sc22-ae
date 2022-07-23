#!/bin/bash

# model configuration
NUM_LAYERS=16
HIDDEN_SIZE=2048
HEADS=16
SEQ_LEN=1024
BATCH_SIZE=4

WINDOW_SIZE=4


script_path=$(realpath $0)
script_dir=$(dirname $script_path)

while getopts 'm:l:h:b:w:' flag
do
    case "${flag}" in
        m) METHOD=${OPTARG};;
        l) NUM_LAYERS=${OPTARG};;
        h) HIDDEN_SIZE=${OPTARG};;
        b) BATCH_SIZE=${OPTARG};;
        w) WINDOW_SIZE=${OPTARG};;
    esac
done

_LOG_DIR=${script_dir}/../results


if [[ 'megatron-lm' = $METHOD ]]; then
    _SRC_DIR=${script_dir}/../Megatron-LM/examples
    _SCRIPT=sc22-gpt-megatron.sh

    echo -e "\n\n !!! The training model size in megatron-lm might be much smaller than others, such as zero-offload, stronghold, etc. !!! \n\n "

elif [[ 'zero-offload' = $METHOD ]]; then
    _SRC_DIR=${script_dir}/../DeepSpeedExample/examples
    _SCRIPT=sc22-gpt-zero-offloading.sh

elif [[ 'zero-infinity' = $METHOD ]]; then
    _SRC_DIR=${script_dir}/../DeepSpeedExample/examples
    _SCRIPT=sc22-gpt-zero-infinity-cpu.sh

elif [[ 'zero-infinity-nvme' = $METHOD ]]; then
    _SRC_DIR=${script_dir}/../DeepSpeedExample/examples
    _SCRIPT=ds_pretrain_gpt2-infinity-nvme.sh

elif [[ 'l2l' = $METHOD ]]; then
    _SRC_DIR=${script_dir}/../L2L/examples
    _SCRIPT=sc22-gpt-l2l.sh

elif [[ 'stronghold' = $METHOD ]]; then
    _SRC_DIR=${script_dir}/../SHv0/examples
    _SCRIPT=sc22-gpt-sh.sh

elif [[ 'all' = $METHOD ]]; then
    ${script_dir}/run.sh -m 'megatron-lm' -l ${NUM_LAYERS} -h ${HIDDEN_SIZE} -b ${BATCH_SIZE} -w ${WINDOW_SIZE}
    ${script_dir}/run.sh -m 'zero-offload' -l ${NUM_LAYERS} -h ${HIDDEN_SIZE} -b ${BATCH_SIZE} -w ${WINDOW_SIZE}
    ${script_dir}/run.sh -m 'zero-infinity' -l ${NUM_LAYERS} -h ${HIDDEN_SIZE} -b ${BATCH_SIZE} -w ${WINDOW_SIZE}
    ${script_dir}/run.sh -m 'stronghold' -l ${NUM_LAYERS} -h ${HIDDEN_SIZE} -b ${BATCH_SIZE} -w ${WINDOW_SIZE}
    ${script_dir}/run.sh -m 'l2l' -l ${NUM_LAYERS} -h ${HIDDEN_SIZE} -b ${BATCH_SIZE} -w ${WINDOW_SIZE}

    #PS: a little time-consuming!
    #${script_dir}/run.sh -m 'zero-infinity-nvme'
    exit 0
else
    echo "the value of '-m' is illegal: $METHOD"
    echo "please choose one value from ['megatron-lm', 'zero-offload', 'zero-infinity', 'l2l', 'stronghold', and 'all']. "
    exit 0
fi

CMD="cd ${_SRC_DIR}/.. && \
    ${_SRC_DIR}/${_SCRIPT} ${NUM_LAYERS} ${HIDDEN_SIZE} ${HEADS} ${SEQ_LEN} ${BATCH_SIZE} ${WINDOW_SIZE} 2>&1 | \
        tee ${_LOG_DIR}/log_${METHOD}_l-${NUM_LAYERS}_hs-${HIDDEN_SIZE}_bs-${BATCH_SIZE}_ws-${WINDOW_SIZE}_$(date '+%Y-%m-%d.%s').txt && \
    cd -"

echo $CMD
eval $CMD
