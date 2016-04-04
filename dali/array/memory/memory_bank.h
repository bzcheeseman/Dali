#ifndef DALI_ARRAY_MEMORY_MEMORY_BANK_H
#define DALI_ARRAY_MEMORY_MEMORY_BANK_H

#include <atomic>
#include <vector>
#include <iostream>
#include <unordered_map>

#include "dali/config.h"
#include "dali/array/memory/memory_ops.h"



namespace memory {
    struct Device;
    struct DevicePtr;

    namespace bank {

        void deposit(DevicePtr dev_ptr, int amount, int inner_dimension);
        DevicePtr allocate(Device device, int amount, int inner_dimension);
        void clear(Device device);
    };
}
#endif
