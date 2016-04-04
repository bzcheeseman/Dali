#ifndef DALI_ARRAY_GET_MSHADOW
#define DALI_ARRAY_GET_MSHADOW

#include <mshadow/tensor.h>

#include "dali/config.h"

template<int devT, typename T>
struct getmshadow {
    memory::Device d;
    void d1(Array a) {}
};

template<typename T>
struct getmshadow<memory::DEVICE_T_CPU, T> {
    memory::Device d;

    mshadow::Tensor<mshadow::cpu, 1, T> d1(Array a) {
        return mshadow::Tensor<mshadow::cpu, 1, T>((T*)(a.memory()->data(d)), mshadow::Shape1(a.number_of_elements()));
    }
};

#ifdef DALI_USE_CUDA
    template<typename T>
    struct getmshadow<memory::DEVICE_T_GPU, T> {
        memory::Device d;

        mshadow::Tensor<mshadow::gpu, 1, T> d1(Array a) {
            return mshadow::Tensor<mshadow::gpu, 1, T>((T*)(a.memory()->data(d)),  mshadow::Shape1(a.number_of_elements()));
        }
    };
#endif

#endif
