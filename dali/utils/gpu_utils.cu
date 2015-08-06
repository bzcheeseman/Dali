#include "dali/utils/gpu_utils.h"

namespace gpu_utils {
    void set_default_gpu(int device) {
        cudaSetDevice(device);
    }

    std::string get_gpu_name(int device) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);

        return std::string(props.name);
    }

    int num_gpus() {
        int devices;
        cudaGetDeviceCount(&devices);
        return devices;
    }
}
