#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$( dirname $SCRIPT_DIR )


build_directory=${1:-}
if [[ -z "$build_directory" ]]
then
    echo "Usage $0 BUILD_DIRECTORY"
fi

BUILD_DIR=$( cd "$( dirname "$build_directory" )" && pwd )


PATH=$PATH:$BUILD_DIR/examples
complete -o bashdefault -o default -o nospace -C '$(which gflags_completions.sh) --tab_completion_columns $COLUMNS' language_model
