#ifndef DALI_MAT_MATH_SYNCHRONIZED_MEMORY_H
#define DALI_MAT_MATH_SYNCHRONIZED_MEMORY_H

#include <functional>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <ostream>
#include <mshadow/tensor.h>
#include "dali/utils/core_utils.h"

// SynchronizedMemory wraps two tensors for GPU and CPU and
// remembers the freshness of each version.
// This class helps minimize the transfers between GPU and CPU
// by remembering if the master copy is available on either device
// or if it should be copied over.

void dali_init();

enum Device {
    DEVICE_GPU,
    DEVICE_CPU
};

template<typename R> class SynchronizedMemory;

template<typename R>
bool should_compute_on_gpu(const std::vector<const SynchronizedMemory<R>*>& sts);

#ifdef DALI_USE_CUDA
    static const Device default_preferred_device = DEVICE_GPU;
#else
    static const Device default_preferred_device = DEVICE_CPU;
#endif


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

        mutable bool allocated_cpu;
        mutable bool cpu_fresh;
        mutable R* cpu_ptr;

        void to_cpu() const;
        bool prefers_cpu() const;
        bool prefers_gpu() const;

        SynchronizedMemory& operator=(const SynchronizedMemory&) = delete;

        SynchronizedMemory(int total_size,
                           int inner_dimension = 1,
                           Device preferred_device = default_preferred_device);
        // inherits preferred device and copies memory to it.
        SynchronizedMemory(const SynchronizedMemory& other);
        ~SynchronizedMemory();

        R* cpu_data() const;
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

        void to_gpu() const;
        // tie-breaker for operations involving multiple tensors
        // on mixed devices.
        static Device tie_breaker_device;

        R* gpu_data() const;
        R* mutable_gpu_data();
    private:
        mshadow::Tensor<mshadow::gpu, 2, R> dummy_gpu() const;
#endif

};


#endif



