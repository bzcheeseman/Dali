#ifndef DALI_MAT_MATH_SYNCHRONIZED_MEMORY_H
#define DALI_MAT_MATH_SYNCHRONIZED_MEMORY_H

#include <functional>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <ostream>
#include <mshadow/tensor.h>
#include "dali/utils/core_utils.h"

/*
Synchronized Memory
-------------------

SynchronizedMemory wraps two tensors for GPU and CPU and
remembers the freshness of each version.
This class helps minimize the transfers between GPU and CPU
by remembering if the master copy is available on either device
or if it should be copied over.
*/

void dali_init();
enum Device {
    DEVICE_GPU,
    DEVICE_CPU
};

template<typename R> class SynchronizedMemory;

// Test where an operation should happen
// Uses poor man's heuristics such as
// where memory is located
// In the future this method could also
// use the size of the memory and the amount
// that should be copied to make smarter decisions
template<typename R>
bool should_compute_on_gpu(const std::vector<const SynchronizedMemory<R>*>& sts);

// Set the default device to GPU or CPU based on build type
#ifdef DALI_USE_CUDA
    static const Device default_preferred_device = DEVICE_GPU;
#else
    static const Device default_preferred_device = DEVICE_CPU;
#endif

// Make mshadow::Shape printable:
template<int dimension>
std::ostream& operator<<(std::ostream&, const mshadow::Shape<dimension>&);

template<typename R>
class SynchronizedMemory {
    public:
        Device preferred_device;
        // total amount of memory expressed in number or Dtypes
        int total_memory;
        // hint for inner dimension. Must divide total_memory.
        const int inner_dimension;
        // whether memory must be filled with zeros on allocation
        // defaults to false
        bool clear_on_allocation;

        mutable bool allocated_cpu;
        mutable bool cpu_fresh;
        mutable R* cpu_ptr;

        void free_cpu() const;
        // Ensure a fresh copy of the memory is on the cpu
        void to_cpu() const;
        bool prefers_cpu() const;
        bool prefers_gpu() const;

        bool allocate_cpu() const;

        SynchronizedMemory& operator=(const SynchronizedMemory&) = delete;

        SynchronizedMemory(int total_size,
                           int inner_dimension = 1,
                           Device preferred_device = default_preferred_device);
        // inherits preferred device and copies memory to it.
        SynchronizedMemory(const SynchronizedMemory& other);
        ~SynchronizedMemory();

        // depending on how memory is accessed, its freshess changes:
        // calling cpu_data without mutable asks for a fresh copy
        // of the memory, but promises to not modify it
        R* cpu_data() const;
        // mutable_cpu_data on the other hand declares that the memory
        // will be modified, and thus will need to be resynchronized
        // if a different device needs it (cpu vs. gpu freshness)
        R* mutable_cpu_data();
    private:
        mshadow::Tensor<mshadow::cpu, 2, R> dummy_cpu() const;
        // only used by copy constructor.
        template<typename SourceType>
        void copy_data_from(SourceType& src);
#ifdef DALI_USE_CUDA
    public:
        mutable bool gpu_fresh;
        mutable bool allocated_gpu;
        mutable R* gpu_ptr;

        bool allocate_gpu() const;
        void free_gpu() const;
        // Ensure a fresh copy of the memory is on the gpu
        void to_gpu() const;
        // tie-breaker for operations involving multiple tensors
        // on mixed devices.
        static Device tie_breaker_device;

        // see cpu_data for explanation
        R* gpu_data() const;
        R* mutable_gpu_data();
    private:
        mshadow::Tensor<mshadow::gpu, 2, R> dummy_gpu() const;
#endif
};


#endif



