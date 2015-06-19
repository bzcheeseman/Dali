#ifndef DALI_MAT_MATH_SYNCHRONIZED_TENSOR_H
#define DALI_MAT_MATH_SYNCHRONIZED_TENSOR_H

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

template<typename R>
class SynchronizedTensor;

template<typename R>
bool should_compute_on_gpu(
        std::initializer_list<std::reference_wrapper<SynchronizedTensor<R>>> sts);

template<typename R>
bool should_compute_on_gpu(const std::vector<std::reference_wrapper<SynchronizedTensor<R>>>& sts);

enum PreferredDevice {
    DEVICE_GPU,
    DEVICE_CPU
};
#ifdef DALI_USE_CUDA
    template<typename R>
    class SynchronizedTensor {
      private:
        PreferredDevice preferred_device;
      public:
        typedef mshadow::Tensor<mshadow::cpu, 2, R> cpu_tensor_t;
        typedef mshadow::Tensor<mshadow::gpu, 2, R> gpu_tensor_t;

        mutable cpu_tensor_t mem_cpu;
        mutable gpu_tensor_t mem_gpu;
        mutable bool cpu_fresh;
        mutable bool gpu_fresh;
        SynchronizedTensor(int n, int d, PreferredDevice preferred_device);
        // inherits preferred device and copies memory to it.
        SynchronizedTensor(const SynchronizedTensor& other);
        ~SynchronizedTensor();

        const cpu_tensor_t&   cpu_data() const;
        cpu_tensor_t& mutable_cpu_data();

        const gpu_tensor_t&   gpu_data() const;
        gpu_tensor_t& mutable_gpu_data();

        bool prefers_cpu() const;
        bool prefers_gpu() const;

        SynchronizedTensor& operator=(const SynchronizedTensor&) = delete;

        template <template <typename, typename, typename, int> class wrapper_t, typename TA, typename TB, typename DType, int ta>
        SynchronizedTensor& operator=(const wrapper_t<TA, TB, DType, ta>& expr) {
            auto participants = expr.sync_tensors;
            std::cout << "participants.size() => " << participants.size() << std::endl;
            if (should_compute_on_gpu(participants)) {
                // refresh the gpu memory
                for (auto& participant : participants) {
                    participant.get().gpu_data();
                    std::cout << "mat with n="
                              << participant.get().mem_gpu.shape_[0]
                              << ", d=" << participant.get().mem_gpu.shape_[1]
                              << std::endl;
                }
                mshadow::expr::ExpEngine<mshadow::sv::saveto, gpu_tensor_t, R>::Eval(&mutable_gpu_data(), expr.right.self());
                //mutable_gpu_data() = expr.right;
            } else {
                // refresh the cpu memory
                for (auto& participant : participants)
                    participant.get().cpu_data();
                mshadow::expr::ExpEngine<mshadow::sv::saveto, cpu_tensor_t, R>::Eval(&mutable_cpu_data(), expr.left.self());
                //mutable_cpu_data() = expr.left;
            };
            return *this;
        }

        // tie-breaker for operations involving multiple tensors
        // on mixed devices.
        static PreferredDevice tie_breaker_device;
      private:
        void to_gpu() const;
        void to_cpu() const;

        // only used by copy constructor.
        template<typename SourceType>
        void copy_data_from(SourceType& src);
    };
#else
    template<typename R>
    class SynchronizedTensor {
      private:
        PreferredDevice preferred_device;
      public:
        typedef mshadow::Tensor<mshadow::cpu, 2, R> cpu_tensor_t;
        mutable cpu_tensor_t mem_cpu;
        mutable bool cpu_fresh;
        SynchronizedTensor(int n, int d, PreferredDevice preferred_device);
        // inherits preferred device and copies memory to it.
        SynchronizedTensor(const SynchronizedTensor& other);
        ~SynchronizedTensor();

        const cpu_tensor_t&   cpu_data() const;
        cpu_tensor_t& mutable_cpu_data();

        bool prefers_cpu() const;
        bool prefers_gpu() const;

        SynchronizedTensor& operator=(const SynchronizedTensor&) = delete;

        template <template <typename, typename, int> class wrapper_t, typename TA, typename DType, int ta>
        SynchronizedTensor& operator=(const wrapper_t<TA, DType, ta>& expr) {
            auto participants = expr.sync_tensors;
            std::cout << "participants.size() => " << participants.size() << std::endl;
            // refresh the cpu memory
            for (auto& participant : participants)
                participant.get().cpu_data();
            mshadow::expr::ExpEngine<mshadow::sv::saveto, cpu_tensor_t, R>::Eval(&mutable_cpu_data(), expr.left.self());
            //mutable_cpu_data() = expr.left;
            return *this;
        }

      private:
        void to_cpu() const;
        template<typename SourceType>
        void copy_data_from(SourceType& src);
    };

#endif




#endif
