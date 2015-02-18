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
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

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

if [ -w "$INTEL_THREAD_FILE" ]
then
    install_name_tool -change libiomp5.dylib $IOMP5_FILE $INTEL_THREAD_FILE
    echo "${green}"
    echo "    ┌───────────────────────┐"
    echo "    │Achievement Unlocked !!│"
    echo "    └───────────────────────┘"
    echo "${reset}"
    echo " -> Linking fixed on all files"
else
    echo "${red}"
    echo "    ┌────────┐"
    echo "    │FAILURE!│"
    echo "    └────────┘"
    echo "${reset}"
    echo " -> Could not fix linking for $INTEL_THREAD_FILE"
    echo "    rerun with 'sudo'"
    exit 1
fi
