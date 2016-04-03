#ifndef DALI_ARRAY_MEMORY_MEMORY_OPS_H
#define DALI_ARRAY_MEMORY_MEMORY_OPS_H

#include <cstdlib>
#include <map>
#include <string>

#include "dali/config.h"

namespace memory {
    struct Device;
    struct DevicePtr;

    // allocates amount bytes of memory
    DevicePtr allocate(Device device, int amount, int inner_dimension);

    // frees amount * inner_dimension bytes of memory starting at addr
    void free(DevicePtr addr, int amount, int inner_dimension);

    // sets amount bytes of memory starting at ptr to zero
    void clear(DevicePtr addr, int amount, int inner_dimension);

    // copies the contents of amount bytes of memory from source to dest
    void copy(DevicePtr dest, DevicePtr source, int amount, int inner_dimension);

    #ifdef DALI_USE_CUDA
        // returns available memory on gpu in bytes
        size_t cuda_available_memory();
    #endif
};

#endif
