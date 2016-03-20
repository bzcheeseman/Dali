#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

function usage {
    echo "Usage $0 SOURCE TARGET PACKAGE \"PATTERN1:CONTENT1@PATTERN2:CONTENT2\""
    exit 1
}

SOURCE=${1:-}
if [[ -z "$SOURCE" ]]; then usage; fi

TARGET=${2:-}
if [[ -z "$TARGET" ]]; then usage; fi

PACKAGE=${3:-}
if [[ -z "$PACKAGE" ]]; then usage; fi

cp $SOURCE $TARGET

shift 3

for REPLACEMENTS_PAIR in "$@"
do
    KEY_VALUE=(${REPLACEMENTS_PAIR//:/ })
    PATTERN=${KEY_VALUE[0]}
    CONTENTS=${KEY_VALUE[@]:1}
    echo "REPLACING $PATTERN by $CONTENTS"
    sed -i -e "s/\@$PATTERN\@/$CONTENTS/g" $TARGET
done

PACKAGE_SHA=$(shasum -a 256 $PACKAGE | awk '{print $1;}')
sed -i -e "s/\@CMAKE_PACKAGE_SHA\@/$PACKAGE_SHA/g" $TARGET
