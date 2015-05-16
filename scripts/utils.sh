# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 5
    done
}
