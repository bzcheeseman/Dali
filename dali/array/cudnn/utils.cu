#include "utils.h"
#include "dali/utils/make_message.h"

#include <atomic>
#ifdef DALI_USE_CUDNN
static cudnnHandle_t handle;
std::atomic<bool> handle_created(false);
// TODO(szymon): this should be stream specific handle I think.
cudnnHandle_t* get_handle() {
    bool expected = false;
    bool desired  = true;
    if (handle_created.compare_exchange_strong(expected, desired)) {
       cudnnCreate(&handle);
    }
    return &handle;
}

inline cudnnDataType_t cudnn_dtype(DType dtype) {
    if (dtype == DTYPE_FLOAT) {
        return CUDNN_DATA_FLOAT;
    } else if (dtype == DTYPE_DOUBLE) {
        return CUDNN_DATA_DOUBLE;
    } else {
        ASSERT2(false, utils::make_message(
            "CuDNN dtypes are only availabe for float and double but got ",
            dtype, "."));
        return CUDNN_DATA_FLOAT;
    }
}

DescriptorHolder<cudnnFilterDescriptor_t>::DescriptorHolder(const std::vector<int>& shape, DType dtype, bool nchw) {
    CUDNN_CHECK_RESULT(cudnnCreateFilterDescriptor(&descriptor_),
                       "when creating filter descriptor ");
    cudnnTensorFormat_t tensor_format = nchw ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    int n, c, h, w;
    if (nchw) {
        n = shape[0]; c = shape[1]; h = shape[2]; w = shape[3];
    } else {
        n = shape[0]; c = shape[3]; h = shape[1]; w = shape[2];
    }
    CUDNN_CHECK_RESULT(cudnnSetFilter4dDescriptor(
        descriptor_, cudnn_dtype(dtype), tensor_format, n, c, h, w),
        "when setting filter descriptor ");
}

DescriptorHolder<cudnnTensorDescriptor_t>::DescriptorHolder(const std::vector<int>& shape, DType dtype, bool nchw) {
    CUDNN_CHECK_RESULT(cudnnCreateTensorDescriptor(&descriptor_),
                       "when creating tensor descriptor ");
    cudnnTensorFormat_t tensor_format = nchw ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    int n, c, h, w;
    if (nchw) {
        n = shape[0]; c = shape[1]; h = shape[2]; w = shape[3];
    } else {
        n = shape[0]; c = shape[3]; h = shape[1]; w = shape[2];
    }
    CUDNN_CHECK_RESULT(cudnnSetTensor4dDescriptor(
        descriptor_, tensor_format, cudnn_dtype(dtype), n, c, h, w),
        "when setting tensor descriptor ");
}

DescriptorHolder<cudnnFilterDescriptor_t>::~DescriptorHolder() {
    CUDNN_CHECK_RESULT(cudnnDestroyFilterDescriptor(descriptor_),
                       "when destroying filter descriptor ");
}

DescriptorHolder<cudnnTensorDescriptor_t>::~DescriptorHolder() {
    CUDNN_CHECK_RESULT(cudnnDestroyTensorDescriptor(descriptor_),
                       "when destroying tensor descriptor ");
}

DescriptorHolder<cudnnConvolutionDescriptor_t>::DescriptorHolder(DType dtype,
                                                                 int padding_h,
                                                                 int padding_w,
                                                                 int stride_h,
                                                                 int stride_w) {
    CUDNN_CHECK_RESULT(cudnnCreateConvolutionDescriptor(&descriptor_),
                       "when creating convolution descriptor ");
    CUDNN_CHECK_RESULT(cudnnSetConvolution2dDescriptor(descriptor_,
                /*pad_h=*/padding_h,
                /*pad_w=*/padding_w,
                /*u=*/stride_h,
                /*v=*/stride_w,
                /*dilation_h=*/1,
                /*dilation_w=*/1,
                CUDNN_CROSS_CORRELATION,
                cudnn_dtype(dtype)), "when setting convolution descriptor ");
}

DescriptorHolder<cudnnConvolutionDescriptor_t>::~DescriptorHolder() {
    CUDNN_CHECK_RESULT(cudnnDestroyConvolutionDescriptor(descriptor_),
                       "when destroying convolution descriptor ");
}

DescriptorHolder<cudnnPoolingDescriptor_t>::DescriptorHolder(cudnnPoolingMode_t pooling_mode,
                                                             int window_h,
                                                             int window_w,
                                                             int padding_h,
                                                             int padding_w,
                                                             int stride_h,
                                                             int stride_w) {
    CUDNN_CHECK_RESULT(cudnnCreatePoolingDescriptor(&descriptor_),
                       "when creating pooling descriptor ");
    CUDNN_CHECK_RESULT(cudnnSetPooling2dDescriptor(descriptor_,
                pooling_mode,
                CUDNN_PROPAGATE_NAN,
                /*windowHeight=*/ window_h,
                /*windowWidth=*/  window_w,
                /*pad_h=*/        padding_h,
                /*pad_w=*/        padding_w,
                /*stride_h=*/     stride_h,
                /*stride_w=*/     stride_w
            ), "when setting pooling descriptor ");
}

DescriptorHolder<cudnnPoolingDescriptor_t>::~DescriptorHolder() {
    CUDNN_CHECK_RESULT(cudnnDestroyPoolingDescriptor(descriptor_),
                       "when destroying pooling descriptor ");
}

#endif
