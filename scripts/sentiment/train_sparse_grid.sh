#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$(dirname $( dirname $SCRIPT_DIR ))

STACK_SIZE=1
PATIENCE=5
CPU_CORES=`sysctl hw.ncpu`
CPU_CORES="${CPU_CORES: -1}"
CPU_CORES=$((CPU_CORES+1))
echo "Commencing Grid Search"
echo "* running on ${CPU_CORES} cores"
DATA_DIR="${PROJECT_DIR}/data/sentiment/"
VECTOR_FILE="${PROJECT_DIR}/data/glove/glove.6B.300d.txt"
PROGRAM="${PROJECT_DIR}/build/examples/sparse_lstm_sentiment"

if [ "$#" -ne 1 ]; then
    echo "usage: $0 [results_dir] "
    exit
fi

if [ ! -d "$1" ]; then
    echo "Could not find directory \"$1\""
    exit
fi

if [ ! -f "${DATA_DIR}train.txt" ]; then
    echo "Training data not present. Downloading it now."
    python3 ${DATA_DIR}generate.py
fi

function ensure_dir {
    if [ "${1: -1}" != "/" ]; then
        echo "${1}/"
    else
        echo $1
    fi
}

SAVE_FOLDER="$(ensure_dir $1)saved_models"
RESULTS_FILE="$(ensure_dir $1)results.txt"
BASE_FLAGS="--results_file=${RESULTS_FILE}"
BASE_FLAGS="${BASE_FLAGS} --validation_metric=1"
BASE_FLAGS="${BASE_FLAGS} --stack_size=${STACK_SIZE} --patience=${PATIENCE}"
BASE_FLAGS="${BASE_FLAGS} --epochs=2000 -j=1"
BASE_FLAGS="${BASE_FLAGS} --nofast_dropout --noshortcut --noaverage_gradient"
# use some pretrained vectors:
# BASE_FLAGS="${BASE_FLAGS} --pretrained_vectors=${VECTOR_FILE}"
BASE_FLAGS="${BASE_FLAGS} --embedding_learning_rate -1."
BASE_FLAGS="${BASE_FLAGS} --train=${DATA_DIR}train.txt "
BASE_FLAGS="${BASE_FLAGS} --validation=${DATA_DIR}dev.txt "
BASE_FLAGS="${BASE_FLAGS} --test=${DATA_DIR}test.txt "
echo "* saving results to ${RESULTS_FILE}"

# start out clean
if [ -f $RESULTS_FILE ]; then
    rm $RESULTS_FILE
fi

# create the results file
touch $RESULTS_FILE

# build the save folder
if [ ! -d "$SAVE_FOLDER" ]; then
    mkdir $SAVE_FOLDER
fi

function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 5
    done
}

for hidden in 150
do
    for dropout in 0.3
    do
        for reg in 0
        do
            for lr in 0.035
            do
                for penalty in 0.0 0.0001 0.001 0.01 0.1
                do
                    for curve in flat linear square
                    do
                        # previously saved models are no longer useful for this grid tile
                        if [ ! -d "${SAVE_FOLDER}/model${curve}_${penalty}/" ]; then
                            mkdir "${SAVE_FOLDER}/model${curve}_${penalty}/"
                        fi
                        rm -rf "${SAVE_FOLDER}/model${curve}_${penalty}/*"
                        $PROGRAM $BASE_FLAGS --dropout $dropout --save_location="${SAVE_FOLDER}/model${lr}_${reg}" --memory_penalty_curve $curve --memory_penalty $penalty --learning_rate $lr --hidden $hidden --minibatch 100 --solver adagrad --reg $reg &
                        pwait $CPU_CORES
                    done
                done
            done
        done
    done
done
wait