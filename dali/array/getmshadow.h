#ifndef DALI_ARRAY_GET_MSHADOW
#define DALI_ARRAY_GET_MSHADOW

#include <mshadow/tensor.h>
#ifdef DALI_USE_CUDA
    #include <thrust/device_vector.h>
#endif
#include "dali/config.h"

template<int devT, typename T>
struct MArray {
    Array array;
    memory::Device device;
    void ptr(memory::AM access_mode=memory::AM_READONLY) {}
    void d1(memory::AM access_mode=memory::AM_READONLY) {}
};

template<typename T>
struct MArray<memory::DEVICE_T_CPU, T> {
    Array array;
    memory::Device device;

    T* ptr(memory::AM access_mode=memory::AM_READONLY) {
        return (T*)(array.memory()->data(device, access_mode));
    }

    mshadow::Tensor<mshadow::cpu, 1, T> d1(memory::AM access_mode=memory::AM_READONLY) {
        return mshadow::Tensor<mshadow::cpu, 1, T>(ptr(access_mode), mshadow::Shape1(array.number_of_elements()));
    }
};

#ifdef DALI_USE_CUDA
    template<typename T>
    struct MArray<memory::DEVICE_T_GPU, T> {
        Array array;
        memory::Device device;

        T* ptr(memory::AM access_mode=memory::AM_READONLY) {
            return (T*)(array.memory()->data(device, access_mode));
        }

        thrust::device_ptr<T> to_thrust(memory::AM access_mode=memory::AM_READONLY) {
           return thrust::device_pointer_cast(ptr(access_mode));
        }

        mshadow::Tensor<mshadow::gpu, 1, T> d1(memory::AM access_mode=memory::AM_READONLY) {
            return mshadow::Tensor<mshadow::gpu, 1, T>(ptr(access_mode), mshadow::Shape1(array.number_of_elements()));
        }
    };
#endif

#endif
