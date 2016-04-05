#ifndef DALI_ARRAY_GET_MSHADOW_H
#define DALI_ARRAY_GET_MSHADOW_H

#include "dali/config.h"
#include <mshadow/tensor.h>

#ifdef DALI_USE_CUDA
    #include <thrust/device_vector.h>
#endif

#include "dali/array/Array.h"


template<int devT, typename T>
struct MArray {
    Array array;
    memory::Device device;
    void ptr(memory::AM access_mode=memory::AM_READONLY);
    void d1(memory::AM access_mode=memory::AM_READONLY);
};

template<typename T>
struct MArray<memory::DEVICE_T_CPU, T> {
    Array array;
    memory::Device device;

    T* ptr(memory::AM access_mode=memory::AM_READONLY);

    mshadow::Tensor<mshadow::cpu, 1, T> d1(memory::AM access_mode=memory::AM_READONLY);
};

#ifdef DALI_USE_CUDA
    template<typename T>
    struct MArray<memory::DEVICE_T_GPU, T> {
        Array array;
        memory::Device device;

        T* ptr(memory::AM access_mode=memory::AM_READONLY);

        thrust::device_ptr<T> to_thrust(memory::AM access_mode=memory::AM_READONLY);

        mshadow::Tensor<mshadow::gpu, 1, T> d1(memory::AM access_mode=memory::AM_READONLY);
    };
#endif

#endif
