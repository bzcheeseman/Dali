#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$( dirname $SCRIPT_DIR )

pushd $PROJECT_DIR/build
    rm -rf ./*
    cmake -DCMAKE_BUILD_TYPE=debug ..
    make -j 9
popd
