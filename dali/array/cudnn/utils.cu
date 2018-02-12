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
    }

}

template<>
DescriptorHolder<cudnnFilterDescriptor_t>::DescriptorHolder(const Array& array, bool nchw) {
    CUDNN_CHECK_RESULT(cudnnCreateFilterDescriptor(&descriptor_),
                       "when creating filter descriptor ");
    cudnnTensorFormat_t tensor_format = nchw ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    CUDNN_CHECK_RESULT(cudnnSetFilter4dDescriptor(
        descriptor_, tensor_format, cudnn_dtype(array.dtype()),
        array.shape()[nchw ? 0 : 1], array.shape()[nchw ? 1 : 3],
        array.shape()[nchw ? 2 : 1], array.shape()[nchw ? 3 : 2]),
        "when setting filter descriptor ");
}

template<>
DescriptorHolder<cudnnTensorDescriptor_t>::DescriptorHolder(const Array& array, bool nchw) {
    CUDNN_CHECK_RESULT(cudnnCreateTensorDescriptor(&descriptor_),
                       "when creating tensor descriptor ");
    cudnnTensorFormat_t tensor_format = nchw ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    CUDNN_CHECK_RESULT(cudnnSetTensor4dDescriptor(
        descriptor_, tensor_format, cudnn_dtype(array.dtype()),
        array.shape()[nchw ? 0 : 1], array.shape()[nchw ? 1 : 3],
        array.shape()[nchw ? 2 : 1], array.shape()[nchw ? 3 : 2]),
        "when setting tensor descriptor ");
}

template<>
~DescriptorHolder<cudnnFilterDescriptor_t>::DescriptorHolder() {
    CUDNN_CHECK_RESULT(cudnnDestroyFilterDescriptor(&descriptor_),
                       "when destroying filter descriptor ");
}

template<>
~DescriptorHolder<cudnnTensorDescriptor_t>::DescriptorHolder() {
    CUDNN_CHECK_RESULT(cudnnDestroyTensorDescriptor(&descriptor_),
                       "when destroying tensor descriptor ");
}

template<>
DescriptorHolder<cudnnConvolutionDescriptor_t>::DescriptorHolder(int padding_h,
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
                /*upscalex=*/1,
                /*upscaley=*/1,
                CUDNN_CROSS_CORRELATION // Theano issue author claims its twice as fast:
                                        // https://github.com/Theano/Theano/issues/3632
            ), "when setting convolution descriptor ");
}

template<>
~DescriptorHolder<cudnnConvolutionDescriptor_t>::DescriptorHolder() {
    CUDNN_CHECK_RESULT(cudnnDestroyConvolutionDescriptor(&descriptor_),
                       "when destroying convolution descriptor ");
}

template<>
DescriptorHolder<cudnnPoolingDescriptor_t>::DescriptorHolder(cudnnPoolingMode_t pooling_mode,
                                                             int window_h,
                                                             int window_w,
                                                             int padding_h,
                                                             int padding_w,
                                                             int stride_h,
                                                             int stride_w) {
    CUDNN_CHECK_RESULT(cudnnCreatePooling2dDescriptor(&descriptor_),
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

template<>
~DescriptorHolder<cudnnPoolingDescriptor_t>::DescriptorHolder() {
    CUDNN_CHECK_RESULT(cudnnDestroyPooling2dDescriptor(&descriptor_),
                       "when destroying pooling descriptor ");
}

#endif
