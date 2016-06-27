#include "cudnn_utils.h"

#include <atomic>

///////////////////////////////////////////////////////////////////////////////
//                              UTILS                                        //
///////////////////////////////////////////////////////////////////////////////

#define CUDNN_CHECK_RESULT(status, message) \
        ASSERT2(status == CUDNN_STATUS_SUCCESS, \
            utils::MS() << message << ", cudnn error: " << cudnnGetErrorString(status))

std::string cudnnGetTensorFormatString(cudnnTensorFormat_t tf) {
    if (tf == CUDNN_TENSOR_NCHW) {
        return "NCHW";
    } else if (tf == CUDNN_TENSOR_NHWC) {
        return "NHWC";
    } else {
        return "unknown";
    }
}

std::string cudnnGetDateTypeString(cudnnDataType_t dt) {
    if (dt == CUDNN_DATA_HALF) {
        return "float16";
    } else if (dt == CUDNN_DATA_FLOAT) {
        return "float32";
    } else if (dt == CUDNN_DATA_DOUBLE) {
        return "float64";
    } else {
        return "unknown";
    }
}


template<typename T>
struct TensorWrapperApi {
};

template<>
struct TensorWrapperApi<cudnnTensorDescriptor_t> {
    static void create(cudnnTensorDescriptor_t* doodle) {
        auto result = cudnnCreateTensorDescriptor(doodle);
        CUDNN_CHECK_RESULT(result, "when creating tensor descriptor");
    }

    static void destroy(cudnnTensorDescriptor_t doodle) {
        auto result = cudnnDestroyTensorDescriptor(doodle);
        CUDNN_CHECK_RESULT(result, "when destroying tensor descriptor");

    }

    static void set(cudnnTensorDescriptor_t desc,
                    cudnnTensorFormat_t     tensor_format,
                    cudnnDataType_t         dtype,
                    int n,
                    int c,
                    int h,
                    int w) {
        auto result = cudnnSetTensor4dDescriptor(
            desc,
            tensor_format,
            dtype,
            n,
            c,
            h,
            w);

        CUDNN_CHECK_RESULT(result, "when setting tensor descriptor with "
                << "shape = [n=" << n << ",c=" << c << ",h=" << h << ",w=" << w << "], "
                << "data format = " << cudnnGetTensorFormatString(tensor_format) << ", "
                << "dtype = " << cudnnGetDateTypeString(dtype));
    }
};

template<>
struct TensorWrapperApi<cudnnFilterDescriptor_t> {
    static void create(cudnnFilterDescriptor_t* doodle) {
        auto result = cudnnCreateFilterDescriptor(doodle);
        CUDNN_CHECK_RESULT(result, "when creating filter descriptor");

    }

    static void destroy(cudnnFilterDescriptor_t doodle) {
        auto result = cudnnDestroyFilterDescriptor(doodle);
        CUDNN_CHECK_RESULT(result, "when destroying filter descriptor");
    }

    static void set(cudnnFilterDescriptor_t desc,
                    cudnnTensorFormat_t     tensor_format,
                    cudnnDataType_t         dtype,
                    int n,
                    int c,
                    int h,
                    int w) {
        auto result = cudnnSetFilter4dDescriptor(
            desc,
            dtype,
            tensor_format,
            n,
            c,
            h,
            w);

        CUDNN_CHECK_RESULT(result, "when setting filter descriptor with "
                << "shape = [n=" << n << ",c=" << c << ",h=" << h << ",w=" << w << "], "
                << "data format = " << cudnnGetTensorFormatString(tensor_format) << ", "
                << "dtype = " << cudnnGetDateTypeString(dtype));
    }
};

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

namespace cudnn {
    ///////////////////////////////////////////////////////////////////////////
    //                              wrappers                                 //
    ///////////////////////////////////////////////////////////////////////////


    namespace wrapper {
        template<typename Descriptor>
        template<typename T, int devT>
        BaseTensor<Descriptor>::BaseTensor(
                TypedArray<devT,T> tensor,
                std::string data_format,
                memory::AM access_mode) {
            ASSERT2(devT == memory::DEVICE_T_GPU,
                    "cudnn Tensor/Filters wrapper must be "
                    "constructed from GPU TypedArray.");
            cudnnTensorFormat_t data_format_cudnn;
            if (data_format == "NCHW") {
                data_format_cudnn = CUDNN_TENSOR_NCHW;
            } else if (data_format == "NHWC") {
                data_format_cudnn = CUDNN_TENSOR_NHWC;
            } else {
                ASSERT2(false, "unsupported data format");
            }
            cudnnDataType_t cudnn_dtype;
            if (template_to_dtype<T>() == DTYPE_FLOAT) {
                cudnn_dtype = CUDNN_DATA_FLOAT;
            } else if (template_to_dtype<T>() == DTYPE_DOUBLE) {
                cudnn_dtype = CUDNN_DATA_DOUBLE;
            } else {
                ASSERT2(false, "unsupported dtype");
            }
            int shape0, shape1, shape2, shape3;
            if (tensor.array.ndim() == 1) {
                if (data_format == "NCHW") {
                    shape0 = shape2 = shape3 = 1;
                    shape1 = tensor.array.shape()[0];
                } else if (data_format == "NHWC") {
                    shape0 = shape1 = shape2 = 1;
                    shape3 = tensor.array.shape()[0];
                }
            } else if (tensor.array.ndim() == 4) {
                shape0 = tensor.array.shape()[0];
                shape1 = tensor.array.shape()[1];
                shape2 = tensor.array.shape()[2];
                shape3 = tensor.array.shape()[3];
            } else {
                ASSERT2(false, "cudnn::wrapper::Tensor can only support 1D and 4D tensors.");
            }

            int n,c,h,w;
            if (data_format == "NCHW") {
                n = shape0; c = shape1; h = shape2; w = shape3;
            } else {
                n = shape0; c = shape3; h = shape1; w = shape2;
            }

            TensorWrapperApi<Descriptor>::create(&description);
            TensorWrapperApi<Descriptor>::set(
                description,
                data_format_cudnn,
                cudnn_dtype,
                n,
                c,
                h,
                w
            );
            // TODO(szymon): add striding support and assert maybe???
            data = tensor.ptr(access_mode);
        }

        template<typename Descriptor>
        BaseTensor<Descriptor>::~BaseTensor() {
            TensorWrapperApi<Descriptor>::destroy(description);
        }

        template class BaseTensor<cudnnTensorDescriptor_t>;
        template class BaseTensor<cudnnFilterDescriptor_t>;

        template BaseTensor<cudnnTensorDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_GPU,float>, std::string, memory::AM);
        template BaseTensor<cudnnTensorDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_GPU,double>, std::string, memory::AM);
        template BaseTensor<cudnnTensorDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_GPU,int>, std::string, memory::AM);
        template BaseTensor<cudnnTensorDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_CPU,float>, std::string, memory::AM);
        template BaseTensor<cudnnTensorDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_CPU,double>, std::string, memory::AM);
        template BaseTensor<cudnnTensorDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_CPU,int>, std::string, memory::AM);

        template BaseTensor<cudnnFilterDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_GPU,float>, std::string, memory::AM);
        template BaseTensor<cudnnFilterDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_GPU,double>, std::string, memory::AM);
        template BaseTensor<cudnnFilterDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_GPU,int>, std::string, memory::AM);
        template BaseTensor<cudnnFilterDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_CPU,float>, std::string, memory::AM);
        template BaseTensor<cudnnFilterDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_CPU,double>, std::string, memory::AM);
        template BaseTensor<cudnnFilterDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_CPU,int>, std::string, memory::AM);


        Convolution::Convolution(int padding_h, int padding_w,
                                 int stride_h, int stride_w) {
            auto result = cudnnCreateConvolutionDescriptor(&description);
            CUDNN_CHECK_RESULT(result, "when creating convolution descriptor");
            result = cudnnSetConvolution2dDescriptor(
                description,
                /*pad_h=*/   padding_h,
                /*pad_w=*/   padding_w,
                /*u=*/       stride_h,
                /*v=*/       stride_w,
                /*upscalex=*/1,
                /*upscaley=*/1,
                CUDNN_CROSS_CORRELATION // Theano issue author claims its twice as fast:
                                        // https://github.com/Theano/Theano/issues/3632
            );


            CUDNN_CHECK_RESULT(result, "when setting convolution descriptor with "
                << "padding_h = "  << padding_h << ", padding_w = " << padding_w
                << ", stride_h = " << stride_h  << ", stride_w = "  << stride_w);
        }

        Convolution::~Convolution() {
            auto result = cudnnDestroyConvolutionDescriptor(description);
            CUDNN_CHECK_RESULT(result, "when destroying convolution descriptor");
        }

        Pooling::Pooling(int window_h,  int window_w,
                         int padding_h, int padding_w,
                         int stride_h,  int stride_w,
                         POOLING_T pooling_mode) {
            auto result = cudnnCreatePoolingDescriptor(&description);
            CUDNN_CHECK_RESULT(result, "when creating pooling descriptor");

            cudnnPoolingMode_t cudnn_pooling_mode;
            if (pooling_mode == POOLING_T_MAX) {
                cudnn_pooling_mode = CUDNN_POOLING_MAX;
            } else if (pooling_mode == POOLING_T_AVG) {
                // Following what TensorFlow does:
                //   https://github.com/tensorflow/tensorflow/blob/
                //   6431560b7ec3565154cb9cdc9c827db78ccfebe7/
                //   tensorflow/stream_executor/cuda/cuda_dnn.cc
                cudnn_pooling_mode =
                        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            } else {
                ASSERT2(false, utils::MS() << "unknown POOLING_T ("
                                           << pooling_mode << ").");
            }

            result = cudnnSetPooling2dDescriptor(
                description,
                cudnn_pooling_mode,
                CUDNN_PROPAGATE_NAN,
                /*windowHeight=*/ window_h,
                /*windowWidth=*/  window_w,
                /*pad_h=*/        padding_h,
                /*pad_w=*/        padding_w,
                /*stride_h=*/     stride_h,
                /*stride_w=*/     stride_w
            );
            CUDNN_CHECK_RESULT(result, "when setting Pooling descriptor");
        }

        Pooling::~Pooling() {
            auto result = cudnnDestroyPoolingDescriptor(description);
            CUDNN_CHECK_RESULT(result, "when destroying Pooling descriptor");
        }


        Operator::Operator(OPERATOR_T operator_type, DType dtype) {
            switch (operator_type) {
                case OPERATOR_T_EQL:
                    alpha_d = 1.0;
                    beta_d  = 0.0;
                    break;
                case OPERATOR_T_ADD:
                    alpha_d = 1.0;
                    beta_d  = 1.0;
                    break;
                case OPERATOR_T_SUB:
                    alpha_d = -1.0;
                    beta_d  = 1.0;
                    break;
                default:
                    ASSERT2(false, "Cudnn only supports =, + and - operators");
            }

            alpha_f = alpha_d;
            beta_f  = beta_d;

            if (dtype == DTYPE_FLOAT) {
                alpha_ptr = (void*)&alpha_f;
                beta_ptr  = (void*)&beta_f;
            } else if (dtype == DTYPE_DOUBLE) {
                alpha_ptr = (void*)&alpha_d;
                beta_ptr  = (void*)&beta_d;
            } else {
                ASSERT2(false, "Cudnn only supports floating point types");
            }
        }
    }  // namespace wrapper


    ///////////////////////////////////////////////////////////////////////////
    //                              CONVOLUTIONS                             //
    ///////////////////////////////////////////////////////////////////////////

    void conv2d(std::shared_ptr<wrapper::Tensor>  out,
                std::shared_ptr<wrapper::Tensor>  in,
                std::shared_ptr<wrapper::Filters> filters,
                std::shared_ptr<wrapper::Convolution> conv,
                const wrapper::Operator& update_operator) {
        // TODO(szymon): automatically choose best algorithm.
        cudnnConvolutionFwdAlgo_t algo =
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        void* working_memory    = NULL;
        int working_memory_size = 0;

        auto result = cudnnConvolutionForward(
            *get_handle(),
            update_operator.alpha_ptr,
            in->description,
            in->data,
            filters->description,
            filters->data,
            conv->description,
            algo,
            working_memory,
            working_memory_size,
            update_operator.beta_ptr,
            out->description,
            out->data
        );
        CUDNN_CHECK_RESULT(result, "when running cudnnConvolutionForward");
    }

    void conv2d_bwd_input(std::shared_ptr<wrapper::Tensor>  in_dw,
                          std::shared_ptr<wrapper::Filters> filters,
                          std::shared_ptr<wrapper::Tensor>  out_dw,
                          std::shared_ptr<wrapper::Convolution> conv,
                          const wrapper::Operator& update_operator) {
        // TODO(szymon): automatically choose best algorithm.
        cudnnConvolutionBwdDataAlgo_t algo =
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        void* working_memory    = NULL;
        int working_memory_size = 0;

        auto result = cudnnConvolutionBackwardData(
            *get_handle(),
            update_operator.alpha_ptr,
            filters->description,
            filters->data,
            out_dw->description,
            out_dw->data,
            conv->description,
            algo,
            working_memory,
            working_memory_size,
            update_operator.beta_ptr,
            in_dw->description,
            in_dw->data
        );
        CUDNN_CHECK_RESULT(result, "when computing convolution's data gradient");
    }


    void conv2d_bwd_filters(std::shared_ptr<wrapper::Filters> filters_dw,
                            std::shared_ptr<wrapper::Tensor>  input,
                            std::shared_ptr<wrapper::Tensor>  out_dw,
                            std::shared_ptr<wrapper::Convolution> conv,
                            const wrapper::Operator& update_operator) {
        // TODO(szymon): automatically choose best algorithm.
        cudnnConvolutionBwdFilterAlgo_t algo =
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        void* working_memory    = NULL;
        int working_memory_size = 0;

        auto result = cudnnConvolutionBackwardFilter(
            *get_handle(),
            update_operator.alpha_ptr,
            input->description,
            input->data,
            out_dw->description,
            out_dw->data,
            conv->description,
            algo,
            working_memory,
            working_memory_size,
            update_operator.beta_ptr,
            filters_dw->description,
            filters_dw->data
        );
        CUDNN_CHECK_RESULT(result, "when computing convolution's filter gradient");
    }

    void conv2d_bwd_bias(std::shared_ptr<wrapper::Tensor> bias_dw,
                         std::shared_ptr<wrapper::Tensor> out_dw,
                         const wrapper::Operator& update_operator) {
        auto result = cudnnConvolutionBackwardBias(
            *get_handle(),
            update_operator.alpha_ptr,
            out_dw->description,
            out_dw->data,
            update_operator.beta_ptr,
            bias_dw->description,
            bias_dw->data
        );
        CUDNN_CHECK_RESULT(result, "when computing convolution bias gradient");
    }


    void pool2d(std::shared_ptr<wrapper::Tensor> out,
                std::shared_ptr<wrapper::Tensor>  in,
                std::shared_ptr<wrapper::Pooling> pooling,
                const wrapper::Operator& update_operator) {

        auto result = cudnnPoolingForward(
            *get_handle(),
            pooling->description,
            update_operator.alpha_ptr,
            in->description,
            in->data,
            update_operator.beta_ptr,
            out->description,
            out->data
        );

        CUDNN_CHECK_RESULT(result, "when computing pooling forward");
    }

        void pool2d_bwd(std::shared_ptr<wrapper::Tensor> in_dw,
                        std::shared_ptr<wrapper::Tensor> out,
                        std::shared_ptr<wrapper::Tensor> out_dw,
                        std::shared_ptr<wrapper::Tensor> in,
                        std::shared_ptr<wrapper::Pooling> pooling,
                        const wrapper::Operator& update_operator) {
            auto result = cudnnPoolingBackward(
                *get_handle(),
                pooling->description,
                update_operator.alpha_ptr,
                out->description,
                out->data,
                out_dw->description,
                out_dw->data,
                in->description,
                in->data,
                update_operator.beta_ptr,
                in_dw->description,
                in_dw->data
            );

            CUDNN_CHECK_RESULT(result, "when computing pooling forward");
        }

}  // namespace cudnn
