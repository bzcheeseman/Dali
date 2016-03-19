#ifndef DALI_UTILS_GPU_UTILS_H
#define DALI_UTILS_GPU_UTILS_H

#include <string>


namespace gpu_utils {
    void set_default_gpu(int device);

    std::string get_gpu_name(int device);

    int num_gpus();
}

#endif
