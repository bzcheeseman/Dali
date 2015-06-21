#ifndef DALI_MAT_MATH_LAZY_UTILS_H
#define DALI_MAT_MATH_LAZY_UTILS_H

template<typename T>
struct extract_tensor_arguments {
    typedef float DType;
    typedef T  tensor_t;
    typedef T  sub_tensor_t;
    typedef mshadow::cpu device_t;
    const static int dimension = 2;
    const static int subdim = 1;
};

template<template <typename, int, typename> class tensor_cls_t, typename Device, int dims, typename tensor_DType>
struct extract_tensor_arguments<tensor_cls_t<Device, dims, tensor_DType>> {
    typedef tensor_DType DType;
    typedef Device device_t;
    typedef tensor_cls_t<Device, dims, DType> tensor_t;
    const static int dimension = dims;
    const static int subdim = dims - 1;
    typedef tensor_cls_t<Device, subdim, DType> sub_tensor_t;
};

#endif
