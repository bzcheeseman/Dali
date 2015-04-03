#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$( dirname $SCRIPT_DIR )

cat $SCRIPT_DIR/data/ascii_delorean.txt

pushd $PROJECT_DIR/build
    rm -rf ./*
    echo "Preparing the Ferrari,"
    cmake -DCMAKE_BUILD_TYPE=release ..
    echo "This builds the Ferrari,"
    make -j 9
    echo "Now you drive the Ferrari."
popd
