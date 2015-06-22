#ifndef DALI_MAT_MATH_SYNCHRONIZED_MEMORY_H
#define DALI_MAT_MATH_SYNCHRONIZED_MEMORY_H

#include <functional>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <mshadow/tensor.h>

// This a small file keeping track of freshness of memory on CPU.
// The whole reason this is done is because some operations are
// implemented for GPU while others only exist for CPU.
// To minimize transfers between host and GPU device, we
// keep track of which device has the master copy.

void dali_init();

enum PreferredDevice {
    DEVICE_GPU,
    DEVICE_CPU
};

class MemoryMover {
    protected:
        PreferredDevice preferred_device;
    public:
        virtual void to_cpu() const = 0;
        mutable bool cpu_fresh;
        bool prefers_cpu() const;
        bool prefers_gpu() const;
#ifdef DALI_USE_CUDA
    public:
        virtual void to_gpu() const = 0;
        mutable bool gpu_fresh;

        // tie-breaker for operations involving multiple tensors
        // on mixed devices.
        static PreferredDevice tie_breaker_device;
#endif

#ifdef DALI_USE_CUDA
    MemoryMover(bool cpu_fresh, bool gpu_fresh, PreferredDevice preferred_device);
#else
    MemoryMover(bool cpu_fresh, PreferredDevice preferred_device);
#endif
};

bool should_compute_on_gpu(const std::vector<const MemoryMover*>& sts);

#ifdef DALI_USE_CUDA
    static const PreferredDevice default_preferred_device = DEVICE_GPU;
#else
    static const PreferredDevice default_preferred_device = DEVICE_CPU;
#endif

template<typename R, int dimension>
class SynchronizedMemory : public MemoryMover {
    private:
        mutable bool allocated_cpu;
        // only used by copy constructor.
        template<typename SourceType>
        void copy_data_from(SourceType& src);
    public:
        void to_cpu() const override;
        typedef mshadow::Tensor<mshadow::cpu, dimension, R> cpu_tensor_t;
        mutable cpu_tensor_t mem_cpu;
        const cpu_tensor_t&   cpu_data() const;
        cpu_tensor_t& mutable_cpu_data();

        SynchronizedMemory& operator=(const SynchronizedMemory&) = delete;

        SynchronizedMemory(mshadow::Shape<dimension> dim, PreferredDevice preferred_device = default_preferred_device);
        // inherits preferred device and copies memory to it.
        SynchronizedMemory(const SynchronizedMemory& other);
        ~SynchronizedMemory();

        unsigned int number_of_elements() const;

        mshadow::Shape<dimension> shape() const;
#ifdef DALI_USE_CUDA
    private:
        mutable bool allocated_gpu;
    public:
        void to_gpu() const override;
        typedef mshadow::Tensor<mshadow::gpu, dimension, R> gpu_tensor_t;

        mutable gpu_tensor_t mem_gpu;
        const gpu_tensor_t&   gpu_data() const;
        gpu_tensor_t&       mutable_gpu_data();
#endif
};

#endif
