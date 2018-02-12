#ifndef DALI_ARRAY_OP_CUDNN_UTILS_H
#define DALI_ARRAY_OP_CUDNN_UTILS_H

#include "dali/config.h"
#include "dali/utils/assert2.h"
#include "dali/array/dtype.h"
#include <vector>

#ifdef DALI_USE_CUDNN
#include <cudnn.h>

template<typename T>
struct DescriptorHolder {
};

template<>
struct DescriptorHolder<cudnnFilterDescriptor_t> {
    cudnnFilterDescriptor_t descriptor_;
    DescriptorHolder(const std::vector<int>& shape, DType dtype, bool nchw);
    ~DescriptorHolder();
};

template<>
struct DescriptorHolder<cudnnTensorDescriptor_t> {
    cudnnTensorDescriptor_t descriptor_;
    DescriptorHolder(const std::vector<int>& shape, DType dtype, bool nchw);
    ~DescriptorHolder();
};

template<>
struct DescriptorHolder<cudnnConvolutionDescriptor_t> {
    cudnnConvolutionDescriptor_t descriptor_;
    DescriptorHolder(DType dtype, int padding_h, int padding_w, int stride_h, int stride_w);
    ~DescriptorHolder();
};

template<>
struct DescriptorHolder<cudnnPoolingDescriptor_t> {
    cudnnPoolingDescriptor_t descriptor_;
    DescriptorHolder(cudnnPoolingMode_t pooling_mode,
                     int window_h, int window_w, int padding_h,
                     int padding_w, int stride_h, int stride_w);
    ~DescriptorHolder();
};

cudnnHandle_t* get_handle();

#define CUDNN_CHECK_RESULT(status, MESSAGE)\
    ASSERT2(status == CUDNN_STATUS_SUCCESS,\
        utils::make_message( ( MESSAGE ), cudnnGetErrorString(status), "."));
#endif  // DALI_USE_CUDNN
#endif  // DALI_ARRAY_OP_CUDNN_UTILS_H
