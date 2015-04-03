#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$( dirname $SCRIPT_DIR )

PROGRAM_DIR=~/Desktop/Coding/python_packages/recurrentjs/cpp/
STACK_SIZE=1
PATIENCE=5
CPU_CORES=`sysctl hw.ncpu`
CPU_CORES="${CPU_CORES: -1}"
CPU_CORES=$((CPU_CORES+1))
echo "Commencing Grid Search"
echo "* running on ${CPU_CORES} cores"
DATA_DIR="${PROGRAM_DIR}data/sentiment/"
PROGRAM="${PROGRAM_DIR}build/examples/lstm_sentiment"
SAVE_FOLDER="${SCRIPT_DIR}/saved_models"
RESULTS_FILE="${SCRIPT_DIR}/results.txt"
BASE_FLAGS="--results_file=${RESULTS_FILE}"
BASE_FLAGS="${BASE_FLAGS} --save_location=${SAVE_FOLDER}/model --stack_size=${STACK_SIZE} --patience=${PATIENCE} -epochs=2000 -j=${CPU_CORES} --fast_dropout --noshortcut "
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

# change the variance of the gradients using minibatch sizes:
for minibatch in 2 50 100
do
    # models keeps improving at these hidden sizes:
    for hidden in 300 350 400
    do
        # higher dropout values are subpar
        for dropout in 0.0 0.1 0.2 0.3 0.4
        do
            # previously saved models are no longer useful for this grid tile
            rm -rf $SAVE_FOLDER/*
            $PROGRAM $BASE_FLAGS --hidden $hidden --dropout $dropout --minibatch $minibatch
        done
    done
done
