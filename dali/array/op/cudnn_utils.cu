#include "cudnn_utils.h"

#include <atomic>

template<typename T>
struct TensorWrapperApi {
};

template<>
struct TensorWrapperApi<cudnnTensorDescriptor_t> {
    static void create(cudnnTensorDescriptor_t* doodle) {
        cudnnCreateTensorDescriptor(doodle);
    }

    static void destroy(cudnnTensorDescriptor_t doodle) {
        cudnnDestroyTensorDescriptor(doodle);
    }

    static void set(cudnnTensorDescriptor_t desc,
                    cudnnTensorFormat_t     tensor_format,
                    cudnnDataType_t         dtype,
                    int shape1,
                    int shape2,
                    int shape3,
                    int shape4) {
        cudnnSetTensor4dDescriptor(
            desc,
            tensor_format,
            dtype,
            shape1,
            shape2,
            shape3,
            shape4);
    }
};

template<>
struct TensorWrapperApi<cudnnFilterDescriptor_t> {
    static void create(cudnnFilterDescriptor_t* doodle) {
        cudnnCreateFilterDescriptor(doodle);
    }

    static void destroy(cudnnFilterDescriptor_t doodle) {
        cudnnDestroyFilterDescriptor(doodle);
    }

    static void set(cudnnFilterDescriptor_t desc,
                    cudnnTensorFormat_t     tensor_format,
                    cudnnDataType_t         dtype,
                    int shape1,
                    int shape2,
                    int shape3,
                    int shape4) {
        cudnnSetFilter4dDescriptor(
            desc,
            dtype,
            tensor_format,
            shape1,
            shape2,
            shape3,
            shape4);
    }
};


template<typename Descriptor>
template<typename T>
DaliCudnnWrapper<Descriptor>::DaliCudnnWrapper(
        TypedArray<memory::DEVICE_T_GPU,T> tensor,
        std::string data_format,
        memory::AM access_mode) {
    cudnnTensorFormat_t data_format_cudnn;
    if (data_format == "NCHW") {
        data_format_cudnn = CUDNN_TENSOR_NCHW;
    } else if (data_format == "NHWC") {
        data_format_cudnn = CUDNN_TENSOR_NHWC;
    }
    cudnnDataType_t cudnn_dtype;
    if (template_to_dtype<T>() == DTYPE_FLOAT) {
        cudnn_dtype = CUDNN_DATA_FLOAT;
    } else if (template_to_dtype<T>() == DTYPE_DOUBLE) {
        cudnn_dtype = CUDNN_DATA_DOUBLE;
    } else {
        ASSERT2(false, "unsupported dtype");
    }
    TensorWrapperApi<Descriptor>::create(&description);
    TensorWrapperApi<Descriptor>::set(
        description,
        data_format_cudnn,
        cudnn_dtype,
        tensor.array.shape()[0],
        tensor.array.shape()[1],
        tensor.array.shape()[2],
        tensor.array.shape()[3]
    );
    // TODO(szymon): add striding support and assert maybe???
    data = tensor.ptr(access_mode);
}

template<typename Descriptor>
DaliCudnnWrapper<Descriptor>::~DaliCudnnWrapper() {
    TensorWrapperApi<Descriptor>::destroy(description);
}

template class DaliCudnnWrapper<cudnnTensorDescriptor_t>;
template class DaliCudnnWrapper<cudnnFilterDescriptor_t>;

template DaliCudnnWrapper<cudnnTensorDescriptor_t>::DaliCudnnWrapper(TypedArray<memory::DEVICE_T_GPU,float>, std::string, memory::AM);
template DaliCudnnWrapper<cudnnTensorDescriptor_t>::DaliCudnnWrapper(TypedArray<memory::DEVICE_T_GPU,double>, std::string, memory::AM);

template DaliCudnnWrapper<cudnnFilterDescriptor_t>::DaliCudnnWrapper(TypedArray<memory::DEVICE_T_GPU,float>, std::string, memory::AM);
template DaliCudnnWrapper<cudnnFilterDescriptor_t>::DaliCudnnWrapper(TypedArray<memory::DEVICE_T_GPU,double>, std::string, memory::AM);


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

namespace cudnn_utils {
    void cudnn_conv2d(std::shared_ptr<DaliCudnnTensor>  out,
                      std::shared_ptr<DaliCudnnTensor>  in,
                      std::shared_ptr<DaliCudnnFilters> filters,
                      int stride_w,
                      int stride_h,
                      int padding_h,
                      int padding_w,
                      double alpha,
                      double beta,
                      DType dtype) {
        void* alpha_ptr;
        void* beta_ptr;
        float alpha_f = alpha, beta_f = beta;

        if (dtype == DTYPE_FLOAT) {
            alpha_ptr = (void*)&alpha_f;
            beta_ptr  = (void*)&beta_f;
        } else if (dtype == DTYPE_DOUBLE) {
            alpha_ptr = (void*)&alpha;
            beta_ptr  = (void*)&beta;
        } else {
            ASSERT2(false, "unsupported dtype");
        }

        cudnnConvolutionDescriptor_t conv_desc;
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnSetConvolution2dDescriptor(
            conv_desc,
            /*pad_h=*/   padding_h,
            /*pad_w=*/   padding_w,
            /*u=*/       stride_h,
            /*v=*/       stride_w,
            /*upscalex=*/1,
            /*upscaley=*/1,
            CUDNN_CROSS_CORRELATION // theano people say its fast.
        );

        // TODO(szymon): automatically choose best algorithm.
        cudnnConvolutionFwdAlgo_t algo =
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        void* working_memory    = NULL;
        int working_memory_size = 0;

        cudnnConvolutionForward(
            *get_handle(),
            alpha_ptr,
            in->description,
            in->data,
            filters->description,
            filters->data,
            conv_desc,
            algo,
            working_memory,
            working_memory_size,
            beta_ptr,
            out->description,
            out->data
        );

        cudnnDestroyConvolutionDescriptor(conv_desc);
    }
};
