#!/bin/bash

# Install the CUDA toolkit
# inspiration: https://github.com/tmcdonell/cuda

if [[ "$WITH_CUDA" == "TRUE" ]]
then
    echo "Installing CUDA library"
    #travis_retry
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_${CUDA}_amd64.deb
    #travis_retry
    sudo dpkg -i cuda-repo-ubuntu1404_${CUDA}_amd64.deb
    #travis_retry
    # sudo apt-get update -qq
    sudo apt-get install cuda
    export CUDA_APT=${CUDA%-*}
    export CUDA_APT=${CUDA_APT/./-}
    #travis_retry
    # sudo apt-get install -y cuda-drivers
    # sudo apt-get install -y cuda-core-${CUDA_APT}
    # sudo apt-get install -y cuda-cublas-${CUDA_APT}
    # sudo apt-get install -y cuda-cublas-dev-${CUDA_APT}
    # sudo apt-get install -y cuda-cudart-${CUDA_APT}
    # sudo apt-get install -y cuda-cudart-dev-${CUDA_APT}
    # sudo apt-get install -y cuda-curand-${CUDA_APT}
    # sudo apt-get install -y cuda-curand-dev-${CUDA_APT}
    #travis_retry
    # sudo apt-get clean
    export CUDA_HOME=/usr/local/cuda-${CUDA%%-*}
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
else
    echo "Skipping GPU installation - CPU only."
fi
