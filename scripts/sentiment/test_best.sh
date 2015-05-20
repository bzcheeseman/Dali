#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "usage: $0 [saved_model_dir] "
    exit
fi

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$(dirname $( dirname $SCRIPT_DIR ))

source PROJECT_DIR/scripts/utils.sh

LOAD_DIR=`ensure_dir $1`

if [ ! -d "$LOAD_DIR" ]; then
    echo "Could not find \"$LOAD_DIR\""
    exit
fi

if [ ! -f "${LOAD_DIR}/config.md" ]; then
    echo "Could not find a configuration file under \"${LOAD_DIR}config.md\""
    exit
fi

# get the number of computer cores:
echo "running on ${NUM_THREADS} cores"

DATA_DIR="${PROJECT_DIR}data/sentiment/"

if [ ! -f "${DATA_DIR}test.txt" ]; then
    echo "Testing data not present. Downloading it now."
    python3 ${DATA_DIR}generate.py
fi

PROGRAM="${PROJECT_DIR}build/examples/lstm_sentiment"
BASE_FLAGS="-epochs=0 -j=${NUM_THREADS}"
BASE_FLAGS="${BASE_FLAGS} --train=${DATA_DIR}train.txt "
BASE_FLAGS="${BASE_FLAGS} --validation=${DATA_DIR}dev.txt "
BASE_FLAGS="${BASE_FLAGS} --test=${DATA_DIR}test.txt "
echo "Loading model from $LOAD_DIR"

$PROGRAM $BASE_FLAGS --load $LOAD_DIR --minibatch 100
