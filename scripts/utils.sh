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

# can detect cores on Linux and Mac
function num_cores {
    local RET='you will never see me'
    if `which nproc` > /dev/null; then
        RET=$(nproc)
    else
        mac_cpus=`sysctl hw.ncpu`
        RET=${mac_cpus: -1}
    fi
    echo $RET
}

CPU_CORES=$(num_cores)
NUM_THREADS=$((CPU_CORES+1))

function ensure_dir {
    if [ "${1: -1}" != "/" ]; then
        echo "${1}/"
    else
        echo $1
    fi
}
