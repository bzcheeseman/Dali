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

template<typename R, int dimension>
class SynchronizedMemory;

template<typename R, int dimension>
bool should_compute_on_gpu(
        std::initializer_list<std::reference_wrapper<const SynchronizedMemory<R, dimension>>> sts);

#ifdef DALI_USE_CUDA
    template<typename LeftType, typename RightType, typename DType, int ktype>
    class LazyTensor;
#else
    template<typename LeftType, typename DType, int ktype>
    class LazyTensor;
#endif

template<typename R, int dimension>
bool should_compute_on_gpu(const std::vector<std::reference_wrapper<const SynchronizedMemory<R, dimension>>>& sts);

enum PreferredDevice {
    DEVICE_GPU,
    DEVICE_CPU
};

template<typename R, int dimension>
class SynchronizedMemory {
    private:
        PreferredDevice preferred_device;
        mutable bool allocated_cpu;
        void to_cpu() const;
        // only used by copy constructor.
        template<typename SourceType>
        void copy_data_from(SourceType& src);
    public:
        typedef mshadow::Tensor<mshadow::cpu, dimension, R> cpu_tensor_t;
        mutable cpu_tensor_t mem_cpu;
        const cpu_tensor_t&   cpu_data() const;
        cpu_tensor_t& mutable_cpu_data();

        bool prefers_cpu() const;
        bool prefers_gpu() const;
        SynchronizedMemory& operator=(const SynchronizedMemory&) = delete;

        SynchronizedMemory(int n, int d, PreferredDevice preferred_device);
        // inherits preferred device and copies memory to it.
        SynchronizedMemory(const SynchronizedMemory& other);
        ~SynchronizedMemory();

        unsigned int number_of_elements() const;

        R sum() const;
        bool allclose(const SynchronizedMemory<R, dimension>& other, R tol) const;
        bool operator==(const SynchronizedMemory<R, dimension>& other) const;

        mshadow::Shape<dimension> shape() const;

        mutable bool cpu_fresh;
#ifdef DALI_USE_CUDA
    private:
        mutable bool allocated_gpu;
        void to_gpu() const;
    public:
        typedef mshadow::Tensor<mshadow::gpu, dimension, R> gpu_tensor_t;

        LazyTensor<cpu_tensor_t, gpu_tensor_t, R, mshadow::expr::type::kRValue> wrapper();

        mutable gpu_tensor_t mem_gpu;
        mutable bool gpu_fresh;
        const gpu_tensor_t&   gpu_data() const;
        gpu_tensor_t&       mutable_gpu_data();

        // tie-breaker for operations involving multiple tensors
        // on mixed devices.
        static PreferredDevice tie_breaker_device;

        #define DALI_SYNC_TENSOR_ASSIGN_OP(op_symbol) \
            template <template <typename, typename, typename, int> class wrapper_t, typename TA, typename TB, int ta> \
            SynchronizedMemory& operator op_symbol (const wrapper_t<TA, TB, R, ta>& expr) { \
                if (should_compute_on_gpu(expr.sync_tensors)) { \
                    /* refresh the gpu memory from cpu*/ \
                    for (auto& participant : expr.sync_tensors) { \
                        const auto& data = participant.get().gpu_data(); \
                    } \
                    mutable_gpu_data() op_symbol expr.right; \
                } else {/* refresh the cpu memory from gpu*/ \
                    for (auto& participant : expr.sync_tensors) { \
                        const auto& data = participant.get().cpu_data(); \
                    }\
                    mutable_cpu_data() op_symbol expr.left;\
                };\
                return *this;\
            }

        #define DALI_SYNC_TENSOR_ASSIGN_SCALAR_OP(op_symbol) \
        SynchronizedMemory& operator op_symbol (R scalar) { \
            if (should_compute_on_gpu({std::cref(*this)})) {\
                mutable_gpu_data() op_symbol scalar;\
            } else {\
                mutable_cpu_data() op_symbol scalar;\
            }\
            return *this;\
        }
#else
    public:
        LazyTensor<cpu_tensor_t, R, mshadow::expr::type::kRValue> wrapper();
    #define DALI_SYNC_TENSOR_ASSIGN_OP(op_symbol) \
        template <template <typename, typename, int> class wrapper_t, typename TA, int ta> \
        SynchronizedMemory& operator op_symbol (const wrapper_t<TA, R, ta>& expr) { \
            for (auto& participant : expr.sync_tensors) { \
                const auto& data = participant.get().cpu_data(); \
            }\
            mutable_cpu_data() op_symbol expr.left;\
            return *this;\
        }

    #define DALI_SYNC_TENSOR_ASSIGN_SCALAR_OP(op_symbol) \
        SynchronizedMemory& operator op_symbol (R scalar) { \
            mutable_cpu_data() op_symbol scalar;\
            return *this;\
        }
#endif
    public:
        DALI_SYNC_TENSOR_ASSIGN_OP(=)
        DALI_SYNC_TENSOR_ASSIGN_OP(+=)
        DALI_SYNC_TENSOR_ASSIGN_OP(-=)
        DALI_SYNC_TENSOR_ASSIGN_OP(/=)
        DALI_SYNC_TENSOR_ASSIGN_OP(*=)

        DALI_SYNC_TENSOR_ASSIGN_SCALAR_OP(=)
        DALI_SYNC_TENSOR_ASSIGN_SCALAR_OP(+=)
        DALI_SYNC_TENSOR_ASSIGN_SCALAR_OP(-=)
        DALI_SYNC_TENSOR_ASSIGN_SCALAR_OP(/=)
        DALI_SYNC_TENSOR_ASSIGN_SCALAR_OP(*=)
};

#endif
