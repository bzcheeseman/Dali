#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$(dirname $( dirname $SCRIPT_DIR ))

source PROJECT_DIR/scripts/utils.sh

STACK_SIZE=1
PATIENCE=5
echo "Commencing Grid Search"
echo "* running on ${CPU_CORES} cores"
DATA_DIR="${PROJECT_DIR}/data/sentiment/"
VECTOR_FILE="${PROJECT_DIR}/data/glove/glove.6B.300d.txt"
PROGRAM="${PROJECT_DIR}/build/examples/lstm_sentiment"

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

for hidden in 150
do
    for dropout in 0.3 0.4 0.5
    do
        for reg in 0.0004 0.001
        do
            for lr in 0.01 0.015 0.032 0.035 0.04 0.05
            do
                # previously saved models are no longer useful for this grid tile
                if [ ! -d "${SAVE_FOLDER}/model${lr}_${reg}/" ]; then
                    mkdir "${SAVE_FOLDER}/model${lr}_${reg}/"
                fi
                rm -rf "${SAVE_FOLDER}/model${lr}_${reg}/*"
                $PROGRAM $BASE_FLAGS --dropout $dropout --save_location="${SAVE_FOLDER}/model${lr}_${reg}" --learning_rate $lr --hidden $hidden --minibatch 25 --solver adagrad --reg $reg &
                pwait $CPU_CORES
            done
        done
    done
done
wait
