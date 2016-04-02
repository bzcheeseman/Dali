#ifndef DALI_ARRAY_MEMORY_MEMORY_OPERATIONS_H
#define DALI_ARRAY_MEMORY_MEMORY_OPERATIONS_H

#include "dali/config.h"

#include <cstdlib>
#include <map>
#include <string>


namespace memory_ops {
    enum Device {
        DEVICE_OF_DOOM=0,
        DEVICE_CPU=1,
#ifdef DALI_USE_CUDA
        DEVICE_GPU=2,
#endif
    };

    // allows conversion between a device enum
    // and its printable name (e.g. cpu, gpu)
    extern std::map<int, std::string> device_to_name;

    struct DevicePtr {
        Device device;
        void* ptr;
        DevicePtr(Device _device, void* _ptr);
    };

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
