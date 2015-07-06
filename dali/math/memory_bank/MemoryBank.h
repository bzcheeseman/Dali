#ifndef DALI_MATH_MEMORY_BANK_MEMORY_BANK_H
#define DALI_MATH_MEMORY_BANK_MEMORY_BANK_H
#include <vector>
#include <atomic>
#include <mshadow/tensor.h>
#include <mutex>

#include "dali/math/memory_bank/MemoryBankInternal.h"

template<typename R>
struct memory_bank {
    static std::atomic<long long> num_cpu_allocations;

    static void deposit_cpu(int amount, int inner_dimension, R* ptr);
    static R* allocate_cpu(int amount, int inner_dimension);

    #ifdef DALI_USE_CUDA
        static std::atomic<long long> num_gpu_allocations;
        static void deposit_gpu(int amount, int inner_dimension, R* ptr);
        static R* allocate_gpu(int amount, int inner_dimension);
    #endif
};

#endif
