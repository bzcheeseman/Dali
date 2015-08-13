#ifndef DALI_MATH_MEMORY_BANK_MEMORY_BANK_H
#define DALI_MATH_MEMORY_BANK_MEMORY_BANK_H
#include <atomic>
#include <vector>
#include <mutex>
#include <iostream>
#include <unordered_map>
#include <cuckoohash_map.hh>

#include "dali/math/memory_bank/MemoryBankInternal.h"

template<typename R>
struct memory_bank {
    static cuckoohash_map<unsigned long long,std::vector<R*>> cpu_memory_bank;
    static std::atomic<long long> num_cpu_allocations;
    static std::atomic<long long> total_cpu_memory;

    static void deposit_cpu(int amount, int inner_dimension, R* ptr);
    static R* allocate_cpu(int amount, int inner_dimension);
    static void clear_cpu();

    #ifdef DALI_USE_CUDA
        // find out how many bytes of memory are still available
        // on the device
        static cuckoohash_map<unsigned long long,std::vector<R*>> gpu_memory_bank;
        static std::atomic<long long> num_gpu_allocations;
        static std::atomic<long long> total_gpu_memory;
        static void clear_gpu();

        static void deposit_gpu(int amount, int inner_dimension, R* ptr);
        static size_t cuda_available_memory();
        static R* allocate_gpu(int amount, int inner_dimension);

    #endif
};

#endif
