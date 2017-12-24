#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

mkdir build
cd build

CC=gcc-4.9 CXX=g++-4.9 cmake -DWITH_CUDA=$WITH_CUDA -DCMAKE_BUILD_TYPE=nooptimize -DARRAY_ONLY=1 ..

make -j9 dali

if [[ "$WITH_CUDA" == "FALSE" ]]; then
    # gperftools required for heapcheck
    # HEAPCHECK=normal
    make -j9 run_tests
else
    echo "Compiled GPU version, skipping tests..."
fi
