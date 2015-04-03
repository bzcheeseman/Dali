#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 [saved_model_dir] "
    exit
fi

function ensure_dir {
    if [ "${1: -1}" != "/" ]; then
        echo "${1}/"
    else
        echo $1
    fi
}

LOAD_DIR=`ensure_dir $1`

if [ ! -d "$LOAD_DIR" ]; then
    echo "Could not find \"$LOAD_DIR\""
    exit
fi

if [ ! -f "${LOAD_DIR}/config.md" ]; then
    echo "Could not find a configuration file under \"${LOAD_DIR}config.md\""
    exit
fi

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$( dirname $SCRIPT_DIR )

# get the number of computer cores:
CPU_CORES=`sysctl hw.ncpu`
CPU_CORES="${CPU_CORES: -1}"
CPU_CORES=$((CPU_CORES+1))
echo "running on ${CPU_CORES} cores"

PROGRAM_DIR=~/Desktop/Coding/python_packages/recurrentjs/cpp/
DATA_DIR="${PROGRAM_DIR}data/sentiment/"
PROGRAM="${PROGRAM_DIR}build/examples/lstm_sentiment"
BASE_FLAGS="-epochs=0 -j=${CPU_CORES}"
BASE_FLAGS="${BASE_FLAGS} --train=${DATA_DIR}train.txt "
BASE_FLAGS="${BASE_FLAGS} --validation=${DATA_DIR}dev.txt "
BASE_FLAGS="${BASE_FLAGS} --test=${DATA_DIR}test.txt "
echo "Loading model from $LOAD_DIR"

$PROGRAM $BASE_FLAGS --load $LOAD_DIR --minibatch 100
