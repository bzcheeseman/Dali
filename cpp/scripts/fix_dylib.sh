#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$( dirname $SCRIPT_DIR )

INTEL_THREAD_FILE=/opt/intel/mkl/lib/libmkl_intel_thread.dylib
IOMP5_FILE=/opt/intel/composer_xe_2015.1.108/compiler/lib/libiomp5.dylib

file=${1:-}
if [[ -z "$file" ]]
then
    echo "Usage $0 FILE"
    exit 1
fi

echo "Fixing \"$file\""
if [ -f $file ];
then
    for lib in libmkl_intel_lp64.dylib libmkl_intel_thread.dylib libmkl_core.dylib
    do
        install_name_tool -change $lib /opt/intel/mkl/lib/$lib $file
    done
    install_name_tool -change libiomp5.dylib $IOMP5_FILE $file
else
    echo "install_name_tool can't find \"$file\""
fi