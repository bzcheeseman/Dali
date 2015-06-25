#ifndef DALI_MAT_MATH_TENSOR_INTERNAL_H
#define DALI_MAT_MATH_TENSOR_INTERNAL_H

#include "dali/math/SynchronizedMemory.h"
#include "dali/utils/core_utils.h"
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>

// Defines mathematical operations on Synchronized Memory
// and also interfaces / and handles assignment from LazyTensor

typedef unsigned int dim_t;

template<typename R, int dimension>
class TensorInternal;

#ifdef DALI_USE_CUDA
    #define DALI_SYNC_TENSOR_ASSIGN_OP(op_symbol) \
        template <typename TA, typename TB, int ta> \
        TensorInternal& operator op_symbol (const LazyTensor<TA, TB, R, dimension, ta>& expr) { \
            if (should_compute_on_gpu(expr.sync_tensors)) { \
                /* refresh the gpu memory from cpu*/ \
                for (auto participant : expr.sync_tensors) { \
                    participant->to_gpu(); \
                } \
                for (auto participant : expr.dependent_tensors) { \
                    participant->update_tensor(); \
                } \
                this->mutable_gpu_data() op_symbol expr.right; \
            } else {/* refresh the cpu memory from gpu*/ \
                for (auto participant : expr.sync_tensors) { \
                    participant->to_cpu(); \
                }\
                for (auto participant : expr.dependent_tensors) { \
                    participant->update_tensor(); \
                } \
                this->mutable_cpu_data() op_symbol expr.left;\
            };\
            return *this;\
        }
    #define DALI_SYNC_TENSOR_ASSIGN_SCALAR_OP(op_symbol) \
        TensorInternal& operator op_symbol (R scalar) { \
            if (compute_me_on_gpu()) {\
                this->mutable_gpu_data() op_symbol scalar;\
            } else {\
                this->mutable_cpu_data() op_symbol scalar;\
            }\
            return *this;\
        }
#else
    #define DALI_SYNC_TENSOR_ASSIGN_OP(op_symbol) \
        template <typename TA, int ta> \
        TensorInternal& operator op_symbol (const LazyTensor<TA, R, dimension,ta>& expr) { \
            for (auto& participant : expr.sync_tensors) { \
                participant->to_cpu(); \
            }\
            for (auto& participant : expr.dependent_tensors) { \
                participant->update_tensor(); \
            }\
            this->mutable_cpu_data() op_symbol expr.left;\
            return *this;\
        }

    #define DALI_SYNC_TENSOR_ASSIGN_SCALAR_OP(op_symbol) \
        TensorInternal& operator op_symbol (R scalar) { \
            this->mutable_cpu_data() op_symbol scalar;\
            return *this;\
        }
#endif

#ifdef DALI_USE_CUDA
    template<typename LeftType, typename RightType, typename DType, int dimension, int ktype>
    class LazyTensor;
#else
    template<typename LeftType, typename DType, int dimension, int ktype>
    class LazyTensor;
#endif

template<typename R, int dimension>
class TensorInternal : public SynchronizedMemory<R, dimension> {
    public:
        static const int ndimensions = dimension;
        // inherit SynchronizedMemory's constructors (C++11)
        using SynchronizedMemory<R, dimension>::SynchronizedMemory;

        typedef SynchronizedMemory<R, dimension> parent_t;

        typedef typename SynchronizedMemory<R, dimension>::cpu_tensor_t cpu_tensor_t;
        #ifdef DALI_USE_CUDA
            typedef typename SynchronizedMemory<R, dimension>::gpu_tensor_t gpu_tensor_t;
            typedef LazyTensor<cpu_tensor_t, gpu_tensor_t, R, dimension, mshadow::expr::type::kRValue> lazy_t;
        #else
            typedef LazyTensor<cpu_tensor_t, R, dimension, mshadow::expr::type::kRValue> lazy_t;
        #endif
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

        R sum() const;
        R L2_norm() const;
        bool allclose(const TensorInternal<R, dimension>& other, R tol) const;
        bool operator==(const TensorInternal<R, dimension>& other) const;

        bool compute_me_on_gpu() const;

        operator lazy_t() const;
        lazy_t wrapper() const;

        R& operator()(int i, int j);
        R operator()(int i, int j) const;

        std::vector<int> argmin(int dim) const;
        std::vector<int> argmax(int dim) const;

        R& operator()(int i);
        R operator()(int i) const;

        const R* data() const;
        R* data();

        void print() const;
        void clear();

        static TensorInternal<R,dimension> zeros(mshadow::Shape<dimension>);
};

#endif
