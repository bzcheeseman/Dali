#include "cudnn_utils.h"

#include <atomic>

///////////////////////////////////////////////////////////////////////////////
//                              UTILS                                        //
///////////////////////////////////////////////////////////////////////////////

void cudnn_check_result(cudnnStatus_t status, const std::string& msg) {
    ASSERT2(status == CUDNN_STATUS_SUCCESS,
            utils::MS() << msg << ", cudnn error: " << cudnnGetErrorString(status));
}

template<typename T>
struct TensorWrapperApi {
};

template<>
struct TensorWrapperApi<cudnnTensorDescriptor_t> {
    static void create(cudnnTensorDescriptor_t* doodle) {
        auto result = cudnnCreateTensorDescriptor(doodle);
        cudnn_check_result(result, "when creating tensor descriptor");
    }

    static void destroy(cudnnTensorDescriptor_t doodle) {
        auto result = cudnnDestroyTensorDescriptor(doodle);
        cudnn_check_result(result, "when destroying tensor descriptor");

    }

    static void set(cudnnTensorDescriptor_t desc,
                    cudnnTensorFormat_t     tensor_format,
                    cudnnDataType_t         dtype,
                    int shape1,
                    int shape2,
                    int shape3,
                    int shape4) {
        auto result = cudnnSetTensor4dDescriptor(
            desc,
            tensor_format,
            dtype,
            shape1,
            shape2,
            shape3,
            shape4);
        cudnn_check_result(result, "when setting tensor descriptor");
    }
};

template<>
struct TensorWrapperApi<cudnnFilterDescriptor_t> {
    static void create(cudnnFilterDescriptor_t* doodle) {
        auto result = cudnnCreateFilterDescriptor(doodle);
        cudnn_check_result(result, "when creating filter descriptor");

    }

    static void destroy(cudnnFilterDescriptor_t doodle) {
        auto result = cudnnDestroyFilterDescriptor(doodle);
        cudnn_check_result(result, "when destroying filter descriptor");
    }

    static void set(cudnnFilterDescriptor_t desc,
                    cudnnTensorFormat_t     tensor_format,
                    cudnnDataType_t         dtype,
                    int shape1,
                    int shape2,
                    int shape3,
                    int shape4) {
        auto result = cudnnSetFilter4dDescriptor(
            desc,
            dtype,
            tensor_format,
            shape1,
            shape2,
            shape3,
            shape4);
        cudnn_check_result(result, "when setting filter descriptor");
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
        template<typename T>
        BaseTensor<Descriptor>::BaseTensor(
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
        BaseTensor<Descriptor>::~BaseTensor() {
            TensorWrapperApi<Descriptor>::destroy(description);
        }

        template class BaseTensor<cudnnTensorDescriptor_t>;
        template class BaseTensor<cudnnFilterDescriptor_t>;

        template BaseTensor<cudnnTensorDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_GPU,float>, std::string, memory::AM);
        template BaseTensor<cudnnTensorDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_GPU,double>, std::string, memory::AM);

        template BaseTensor<cudnnFilterDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_GPU,float>, std::string, memory::AM);
        template BaseTensor<cudnnFilterDescriptor_t>::BaseTensor(TypedArray<memory::DEVICE_T_GPU,double>, std::string, memory::AM);




        Convolution::Convolution(int padding_h, int padding_w, int stride_h, int stride_w) {
            auto result = cudnnCreateConvolutionDescriptor(&description);
            cudnn_check_result(result, "when creating convolution descriptor");
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
            cudnn_check_result(result, "when setting convolution descriptor");
        }

        Convolution::~Convolution() {
            auto result = cudnnDestroyConvolutionDescriptor(description);
            cudnn_check_result(result, "when destroying convolution descriptor");
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

    void cudnn_conv2d(std::shared_ptr<wrapper::Tensor>  out,
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
        cudnn_check_result(result, "when setting convolution descriptor");

    }

    void cudnn_conv2d_bwd_data(std::shared_ptr<wrapper::Tensor>  in_dw,
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
        cudnn_check_result(result, "when computing convolution forward");

    }

}  // namespace cudnn
