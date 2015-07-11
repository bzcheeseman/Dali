#ifndef DALI_MATH_MEMORY_BANK_MEMORY_BANK_H
#define DALI_MATH_MEMORY_BANK_MEMORY_BANK_H
#include <atomic>
#include <vector>
#include <mutex>
#include <iostream>
#include <unordered_map>

#include "dali/math/memory_bank/MemoryBankInternal.h"
#ifdef DALI_USE_CUDA
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
#endif

template<typename R>
struct memory_bank {
    static std::unordered_map<unsigned long long,std::vector<R*>> cpu_memory_bank;
    static std::atomic<long long> num_cpu_allocations;
    static std::atomic<long long> total_cpu_memory;

    static void deposit_cpu(int amount, int inner_dimension, R* ptr);
    static R* allocate_cpu(int amount, int inner_dimension);

    #ifdef DALI_USE_CUDA
        // find out how many bytes of memory are still available
        // on the device
        static std::unordered_map<unsigned long long,std::vector<R*>> gpu_memory_bank;
        static std::atomic<long long> num_gpu_allocations;
        static std::atomic<long long> total_gpu_memory;

        static void deposit_gpu(int amount, int inner_dimension, R* ptr);
        static size_t cuda_available_memory();
        static R* allocate_gpu(int amount, int inner_dimension);

    #endif
};

#ifdef DALI_USE_CUDA
// inspiration from
// https://parallel-computing.pro/index.php/9-cuda/34-thrust-cuda-tip-reuse-temporary-buffers-across-transforms
template<typename R>
struct cached_allocator : thrust::device_malloc_allocator<R>{
    typedef thrust::device_malloc_allocator<R> super_t;
    typedef typename super_t::pointer   pointer;
    typedef typename super_t::size_type size_type;
    cached_allocator();
    ~cached_allocator();
    pointer allocate(size_type num_bytes);
    void deallocate(pointer p, size_type n);
};

#endif

#endif
