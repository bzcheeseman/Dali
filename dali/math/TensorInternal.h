#ifndef DALI_MAT_MATH_TENSOR_INTERNAL_H
#define DALI_MAT_MATH_TENSOR_INTERNAL_H

#include "dali/math/SynchronizedMemory.h"
#include "dali/utils/core_utils.h"
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <string>
#include <chrono>
#include <memory>
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
            if (should_compute_on_gpu(extact_memory(expr.dependent_tensors))) { \
                /* refresh the gpu memory from cpu*/ \
                for (auto participant : expr.dependent_tensors) { \
                    participant->update_tensor(DEVICE_GPU); \
                } \
                this->mutable_gpu_data() op_symbol expr.right; \
            } else {/* refresh the cpu memory from gpu*/ \
                for (auto participant : expr.dependent_tensors) { \
                    participant->update_tensor(DEVICE_CPU); \
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
            for (auto participant : expr.dependent_tensors) { \
                participant->update_tensor(DEVICE_CPU); \
            } \
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
class TensorInternal {
    public:
        std::shared_ptr<SynchronizedMemory<R>> memory;
        static const int ndimensions = dimension;

        const mshadow::Shape<dimension> shape;
        int offset;

        TensorInternal(mshadow::Shape<dimension> shape);


        TensorInternal(mshadow::Shape<dimension> shape,
                       std::shared_ptr<SynchronizedMemory<R>> memory,
                       int offset);

        TensorInternal(const TensorInternal& other, bool copy_memory=false);

        typedef mshadow::Tensor<mshadow::cpu, dimension, R> cpu_tensor_t;
        #ifdef DALI_USE_CUDA
            typedef mshadow::Tensor<mshadow::gpu, dimension, R> gpu_tensor_t;
            typedef LazyTensor<cpu_tensor_t, gpu_tensor_t, R, dimension, mshadow::expr::type::kRValue> lazy_t;
        #else
            typedef LazyTensor<cpu_tensor_t, R, dimension, mshadow::expr::type::kRValue> lazy_t;
        #endif

        const cpu_tensor_t   cpu_data() const;
        cpu_tensor_t mutable_cpu_data();

        #ifdef DALI_USE_CUDA
            const gpu_tensor_t   gpu_data() const;
            gpu_tensor_t       mutable_gpu_data();
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

        int argmin() const;
        int argmax() const;

        int argmax_slice(int lower, int upper) const;
        int argmin_slice(int lower, int upper) const;

        std::vector<int> argsort() const;

        R& operator()(int i);
        R operator()(int i) const;

        TensorInternal<R, dimension - 1> operator[](mshadow::index_t idx) const;
        TensorInternal<R, 1> ravel() const;
        TensorInternal<R, dimension> Slice(mshadow::index_t begin, mshadow::index_t end) const;
        const R* data() const;
        R* data();

        void print(int indent=0) const;

        void clear();

        int number_of_elements() const;

        static TensorInternal<R,dimension> zeros(mshadow::Shape<dimension>);
};

template <> void TensorInternal<float, 1>::print(int indent) const;
template <> void TensorInternal<double, 1>::print(int indent) const;

#endif
