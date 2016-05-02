#ifndef DALI_ARRAY_GET_MSHADOW_H
#define DALI_ARRAY_GET_MSHADOW_H

#include "dali/config.h"
#include <mshadow/tensor.h>
#include <vector>
#ifdef DALI_USE_CUDA
    #include <thrust/device_vector.h>
#endif

#include "dali/array/array.h"

template<typename Device, int srcdim, typename DType>
class DaliWrapperExp;

#include "dali/array/function/args/mshadow_wrapper.h"

namespace internal {
    template<int dstdim>
    mshadow::Shape<dstdim> canonical_reshape(const std::vector<int>& src_shape) {
        mshadow::Shape<dstdim> res;
        for (int i = 0; i < dstdim; i++) res[i] = 1;

        int residual_shape = 1;
        for (int i = 0; i < src_shape.size(); ++i) {
            residual_shape *= src_shape[i];
            int dst_index = i - src_shape.size() + dstdim;
            if (dst_index >= 0) {
                res[dst_index] = residual_shape;
                residual_shape = 1;
            }
        }
        return res;
    }
}

template<int devT, typename T>
struct TypedArray {
    // some people use this but not sure who...
};

template<typename T>
struct TypedArray<memory::DEVICE_T_CPU, T> {
    mutable Array array;
    memory::Device device;

    T* ptr(memory::AM access_mode=memory::AM_READONLY) const;

    mshadow::Tensor<mshadow::cpu, 1, T> contiguous_d1(memory::AM access_mode=memory::AM_READONLY) const;
    mshadow::Tensor<mshadow::cpu, 2, T> contiguous_d2(memory::AM access_mode=memory::AM_READONLY) const;

    DaliWrapperExp<mshadow::cpu, 1, T> d1(memory::AM access_mode=memory::AM_READONLY) const;
    DaliWrapperExp<mshadow::cpu, 2, T> d2(memory::AM access_mode=memory::AM_READONLY) const;

    template<int dim>
    mshadow::Tensor<mshadow::cpu, dim, T> d(memory::AM access_mode) const {
        return mshadow::Tensor<mshadow::cpu, dim, T>(
            ptr(access_mode),
            internal::canonical_reshape<dim>(array.shape())
        );
    }

    template<int dim>
    mshadow::Tensor<mshadow::cpu, dim, T> contiguous_d(memory::AM access_mode) const {
        return mshadow::Tensor<mshadow::cpu, dim, T>(
            ptr(access_mode),
            internal::canonical_reshape<dim>(array.shape())
        );
    }

    TypedArray(const Array& _array, const memory::Device& _device);
};

#ifdef DALI_USE_CUDA
    template<typename T>
    struct TypedArray<memory::DEVICE_T_GPU, T> {
        mutable Array array;
        memory::Device device;

        T* ptr(memory::AM access_mode=memory::AM_READONLY) const;

        thrust::device_ptr<T> to_thrust(memory::AM access_mode=memory::AM_READONLY) const;

        mshadow::Tensor<mshadow::gpu, 1, T> contiguous_d1(memory::AM access_mode=memory::AM_READONLY) const;
        mshadow::Tensor<mshadow::gpu, 2, T> contiguous_d2(memory::AM access_mode=memory::AM_READONLY) const;

        DaliWrapperExp<mshadow::gpu, 1, T> d1(memory::AM access_mode=memory::AM_READONLY) const;
        DaliWrapperExp<mshadow::gpu, 2, T> d2(memory::AM access_mode=memory::AM_READONLY) const;

        template<int dim>
        mshadow::Tensor<mshadow::gpu, dim, T> d(memory::AM access_mode) const {
            return mshadow::Tensor<mshadow::gpu, dim, T>(
                ptr(access_mode),
                internal::canonical_reshape<dim>(array.shape())
            );
        }

        template<int dim>
        mshadow::Tensor<mshadow::gpu, dim, T> contiguous_d(memory::AM access_mode) const {
            return mshadow::Tensor<mshadow::gpu, dim, T>(
                ptr(access_mode),
                internal::canonical_reshape<dim>(array.shape())
            );
        }

        TypedArray(const Array& _array, const memory::Device& _device);
    };
#endif
#endif
