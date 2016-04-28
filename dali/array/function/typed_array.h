#ifndef DALI_ARRAY_GET_MSHADOW_H
#define DALI_ARRAY_GET_MSHADOW_H

#include "dali/config.h"
#include <mshadow/tensor.h>

#ifdef DALI_USE_CUDA
    #include <thrust/device_vector.h>
#endif

#include "dali/array/array.h"

template<int devT, typename T>
struct TypedArray {
    mutable Array array;
    memory::Device device;
    void ptr(memory::AM access_mode=memory::AM_READONLY) const;
    void d1(memory::AM access_mode=memory::AM_READONLY) const;

    TypedArray(const Array& _array, const memory::Device& _device);
};

template<typename T>
struct TypedArray<memory::DEVICE_T_CPU, T> {
    mutable Array array;
    memory::Device device;

    T* ptr(memory::AM access_mode=memory::AM_READONLY) const;

    mshadow::Tensor<mshadow::cpu, 1, T> d1(memory::AM access_mode=memory::AM_READONLY) const;

    TypedArray(const Array& _array, const memory::Device& _device);
};

#ifdef DALI_USE_CUDA
    template<typename T>
    struct TypedArray<memory::DEVICE_T_GPU, T> {
        mutable Array array;
        memory::Device device;

        T* ptr(memory::AM access_mode=memory::AM_READONLY) const;

        thrust::device_ptr<T> to_thrust(memory::AM access_mode=memory::AM_READONLY) const;

        mshadow::Tensor<mshadow::gpu, 1, T> d1(memory::AM access_mode=memory::AM_READONLY) const;

        TypedArray(const Array& _array, const memory::Device& _device);
    };
#endif
#endif
