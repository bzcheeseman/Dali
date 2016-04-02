#ifndef DALI_MATH_MEMORY_MEMORY_BANK_H
#define DALI_MATH_MEMORY_MEMORY_BANK_H

#include <atomic>
#include <vector>
#include <iostream>
#include <unordered_map>

#include "dali/config.h"
#include "dali/math/memory/memory_ops.h"



namespace memory_bank {
    void deposit(memory_ops::DevicePtr dev_ptr, int amount, int inner_dimension);
    memory_ops::DevicePtr allocate(memory_ops::Device device, int amount, int inner_dimension);
    void clear(memory_ops::Device device);
};

#endif
