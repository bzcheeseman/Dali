#ifndef DALI_MATH_MEMORY_BANK_MEMORY_BANK_INTERNAL_H
#define DALI_MATH_MEMORY_BANK_MEMORY_BANK_INTERNAL_H

#include "dali/config.h"

#include <vector>

template<typename R>
struct memory_operations {

    static R* allocate_cpu_memory(int amount, int inner_dimension);

    static void free_cpu_memory(R* addr, int amount, int inner_dimension);


    static void copy_memory_cpu_to_cpu(R* dest, R* source, int amount, int inner_dimension);

    static void clear_cpu_memory(R* ptr, int amount, int inner_dimension);

    #ifdef DALI_USE_CUDA
        size_t cuda_available_memory();


        static R* allocate_gpu_memory(int amount, int inner_dimension);


        static void free_gpu_memory(R* addr, int amount, int inner_dimension);


        static void copy_memory_gpu_to_cpu(R* dest, R* source, int amount, int inner_dimension);


        static void copy_memory_gpu_to_gpu(R* dest, R* source, int amount, int inner_dimension);


        static void copy_memory_cpu_to_gpu(R* dest, R* source, int amount, int inner_dimension);


        static void clear_gpu_memory(R* ptr, int amount, int inner_dimension);
    #endif
};

#endif
