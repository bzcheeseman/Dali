#include "conv.h"
#include "dali/config.h"
#include "dali/array/cudnn/utils.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/array/op/spatial_utils.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/array/expression/computation.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/expression/control_flow.h"

namespace op {
    struct CudnnExpression : public Expression {
        using Expression::copy;
        const bool nchw_;
        virtual bool supports_operator(OPERATOR_T operator_t) const {
            return (operator_t == OPERATOR_T_EQL ||
                    operator_t == OPERATOR_T_ADD ||
                    operator_t == OPERATOR_T_SUB);
        }
        CudnnExpression(const std::vector<int>& shape, DType dtype,
                        const std::vector<Array>& arguments, bool nchw) :
            Expression(shape, dtype, arguments), nchw_(nchw) {}
    };

    struct CudnnConv2dBackwardBias : public CudnnExpression {
        CudnnConv2dBackwardBias(const Array& input, bool nchw)
            : CudnnExpression(nchw ? std::vector<int>{input.shape()[1]} :
                              std::vector<int>{input.shape()[3]},
                              input.dtype(), {input}, nchw) {}
        memory::Device preferred_device() const override {
            return arguments_[0].preferred_device();
        }
        virtual expression_ptr copy() const override {
            return std::make_shared<CudnnConv2dBackwardBias>(arguments_[0], nchw_);
        }
    };

    struct CudnnConvExpression : public CudnnExpression {
        const ConvFunctionInfo info_;
        CudnnConvExpression(const std::vector<int>& shape, DType dtype,
                            const std::vector<Array>& arguments, ConvFunctionInfo info, bool nchw)
            : CudnnExpression(shape, dtype, arguments, nchw), info_(info) {}
        memory::Device preferred_device() const override {
            return device_promotion(arguments_[0], arguments_[1]);
        }
    };

    struct CudnnConv2d : public CudnnConvExpression {
        CudnnConv2d(const Array& input, const Array& filters, ConvFunctionInfo info, bool nchw)
            : CudnnConvExpression(nchw ? std::vector<int>({info.batch_size, info.out_channels, info.out_h, info.out_w}) :
                                  std::vector<int>({info.batch_size, info.out_h, info.out_w, info.out_channels}),
                                  input.dtype(), {input, filters}, info, nchw) {}
        virtual expression_ptr copy() const override {
            return std::make_shared<CudnnConv2d>(arguments_[0], arguments_[1], info_, nchw_);
        }
    };

    struct CudnnConv2dBackwardInput : public CudnnConvExpression {
        CudnnConv2dBackwardInput(const Array& filters, const Array& out_dw, const std::vector<int>& input_shape, ConvFunctionInfo info, bool nchw)
            : CudnnConvExpression(input_shape, filters.dtype(), {filters, out_dw}, info, nchw) {}
        virtual expression_ptr copy() const override {
            return std::make_shared<CudnnConv2dBackwardInput>(arguments_[0], arguments_[1], shape_, info_, nchw_);
        }
    };

    struct CudnnConv2dBackwardFilters : public CudnnConvExpression {
        CudnnConv2dBackwardFilters(const Array& input, const Array& out_dw, const std::vector<int>& filters_shape, ConvFunctionInfo info, bool nchw)
            : CudnnConvExpression(filters_shape, input.dtype(), {input, out_dw}, info, nchw) {}
        virtual expression_ptr copy() const override {
            return std::make_shared<CudnnConv2dBackwardFilters>(arguments_[0], arguments_[1], shape_, info_, nchw_);
        }
    };

    struct CudnnPoolExpression : public CudnnExpression {
        const PoolFunctionInfo info_;
        const POOLING_T pooling_mode_;
        CudnnPoolExpression(const std::vector<int>& shape, DType dtype,
                            const std::vector<Array>& arguments, PoolFunctionInfo info,
                            POOLING_T pooling_mode, bool nchw)
            : CudnnExpression(shape, dtype, arguments, nchw),
              info_(info), pooling_mode_(pooling_mode) {}
    };

    struct CudnnPool2d : public CudnnPoolExpression {
        using Expression::copy;
        CudnnPool2d(const Array& input, PoolFunctionInfo info, POOLING_T pooling_mode, bool nchw)
            : CudnnPoolExpression(nchw ? std::vector<int>({info.batch_size, info.in_channels, info.out_h, info.out_w}) :
                                  std::vector<int>({info.batch_size, info.out_h, info.out_w, info.in_channels}),
                                  input.dtype(), {input}, info, pooling_mode, nchw) {}
        virtual expression_ptr copy() const override {
            return std::make_shared<CudnnPool2d>(arguments_[0], info_, pooling_mode_, nchw_);
        }
        memory::Device preferred_device() const override {
            return arguments_[0].preferred_device();
        }
    };

    struct CudnnPool2dBackward : public CudnnPoolExpression {
        using Expression::copy;
        CudnnPool2dBackward(const Array& out, const Array& out_dw, const Array& in,
                            PoolFunctionInfo info, POOLING_T pooling_mode, bool nchw)
            : CudnnPoolExpression(in.shape(), out.dtype(), {out, out_dw, in}, info, pooling_mode, nchw) {}
        virtual expression_ptr copy() const override {
            return std::make_shared<CudnnPool2dBackward>(
                arguments_[0], arguments_[1], arguments_[2], info_, pooling_mode_, nchw_);
        }
        memory::Device preferred_device() const override {
            return arguments_[2].preferred_device();
        }
    };
}
#ifdef DALI_USE_CUDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


namespace {
    struct Operator {
        float alpha_f_, beta_f_;
        double alpha_d_, beta_d_;
        void* alpha_ptr_;
        void* beta_ptr_;
        Operator(OPERATOR_T operator_t, DType dtype) {
            switch (operator_t) {
                case OPERATOR_T_EQL:
                    alpha_d_ = 1.0;
                    beta_d_  = 0.0;
                    break;
                case OPERATOR_T_ADD:
                    alpha_d_ = 1.0;
                    beta_d_  = 1.0;
                    break;
                case OPERATOR_T_SUB:
                    alpha_d_ = -1.0;
                    beta_d_  = 1.0;
                    break;
                default:
                    ASSERT2(false, utils::make_message(
                        "Cudnn only supports =, += and -= operators but got ",
                        operator_to_name(operator_t), "."));
            }
            alpha_f_ = alpha_d_;
            beta_f_  = beta_d_;
            if (dtype == DTYPE_FLOAT) {
                alpha_ptr_ = (void*)&alpha_f_;
                beta_ptr_  = (void*)&beta_f_;
            } else if (dtype == DTYPE_DOUBLE) {
                alpha_ptr_ = (void*)&alpha_d_;
                beta_ptr_  = (void*)&beta_d_;
            } else {
                ASSERT2(false, utils::make_message(
                    "Cudnn only supports float or double, but got ", dtype, "."));
            }
        }
    };

    struct CudnnComputation : public Computation {
        using Computation::Computation;
        void* destination_data(memory::Device device) {
            memory::AM access_mode = operator_t_ == OPERATOR_T_EQL && left_.spans_entire_memory() ?
                memory::AM_OVERWRITE : memory::AM_MUTABLE;
            return (void*)(((char*)left_.memory()->data(device, access_mode)) + left_.offset() * size_of_dtype(left_.dtype()));
        }

        void* argument_data(memory::Device device, int idx) {
            const auto& arg = right_.expression()->arguments()[idx];
            return (void*)(((char*)arg.memory()->readonly_data(device)) + arg.offset() * size_of_dtype(arg.dtype()));
        }

        void run() {
            run_internal(Operator(operator_t_, left_.dtype()),
                         static_cast<op::CudnnExpression*>(right_.expression().get())->nchw_,
                         left_.preferred_device());
        }
        virtual void run_internal(const Operator&, bool nchw, const memory::Device&) = 0;
    };

    struct CudnnConv2dImpl : public CudnnComputation {
        using CudnnComputation::CudnnComputation;
        void run_internal(const Operator& update_operator, bool nchw, const memory::Device& device) override {
            auto conv = static_cast<op::CudnnConv2d*>(right_.expression().get());
            DescriptorHolder<cudnnTensorDescriptor_t> inputs_description(conv->arguments()[0], nchw);
            DescriptorHolder<cudnnTensorDescriptor_t> out_description(left_, nchw);
            DescriptorHolder<cudnnFilterDescriptor_t> filters_description(conv->arguments()[1], nchw);
            DescriptorHolder<cudnnConvolutionDescriptor_t> conv_description(left_.dtype(),
                                                                            conv->info_.padding_h,
                                                                            conv->info_.padding_w,
                                                                            conv->info_.stride_h,
                                                                            conv->info_.stride_w);
            // TODO(jonathan) autotune algo and working space selection
            // (See: SuperNeurons: Dynamic GPU Memory Management for Training
            // Deep Neural Networks, Wang et al.)
            cudnnConvolutionFwdAlgo_t conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            void* working_memory = NULL;
            int working_memory_size = 0;
            auto status = cudnnConvolutionForward(
                *get_handle(),
                update_operator.alpha_ptr_,
                inputs_description.descriptor_,
                argument_data(device, 0),
                filters_description.descriptor_,
                argument_data(device, 1),
                conv_description.descriptor_,
                conv_algo,
                working_memory,
                working_memory_size,
                update_operator.beta_ptr_,
                out_description.descriptor_,
                destination_data(device));
            CUDNN_CHECK_RESULT(status, utils::make_message(
                "Error when running cudnnConvolutionForward with ", conv->info_, " "));
        }
    };

    int conv2d_impl = register_implementation(
        typeid(op::CudnnConv2d).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            return std::make_shared<CudnnConv2dImpl>(dest, operator_t, x, assignment);
        }
    );

    struct CudnnConv2dBackwardInputImpl : public CudnnComputation {
        using CudnnComputation::CudnnComputation;
        void run_internal(const Operator& update_operator, bool nchw, const memory::Device& device) override {
            auto conv = static_cast<op::CudnnConv2dBackwardInput*>(right_.expression().get());
            DescriptorHolder<cudnnFilterDescriptor_t> filters_description(conv->arguments()[0], nchw);
            DescriptorHolder<cudnnTensorDescriptor_t> out_dw_description(conv->arguments()[1], nchw);
            DescriptorHolder<cudnnTensorDescriptor_t> out_description(left_, nchw);
            DescriptorHolder<cudnnConvolutionDescriptor_t> conv_description(left_.dtype(),
                                                                            conv->info_.padding_h,
                                                                            conv->info_.padding_w,
                                                                            conv->info_.stride_h,
                                                                            conv->info_.stride_w);
            cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            void* working_memory    = NULL;
            int working_memory_size = 0;
            auto status = cudnnConvolutionBackwardData(
                *get_handle(),
                update_operator.alpha_ptr_,
                filters_description.descriptor_,
                argument_data(device, 0),
                out_dw_description.descriptor_,
                argument_data(device, 1),
                conv_description.descriptor_,
                conv_bwd_data_algo,
                working_memory,
                working_memory_size,
                update_operator.beta_ptr_,
                out_description.descriptor_,
                destination_data(device));
            CUDNN_CHECK_RESULT(status, "Error when running cudnnConvolutionBackwardData ");
        }
    };

    int conv2d_backward_input_impl = register_implementation(
        typeid(op::CudnnConv2dBackwardInput).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            return std::make_shared<CudnnConv2dBackwardInputImpl>(dest, operator_t, x, assignment);
        }
    );

    struct CudnnConv2dBackwardFiltersImpl : public CudnnComputation {
        using CudnnComputation::CudnnComputation;
        void run_internal(const Operator& update_operator, bool nchw, const memory::Device& device) override {
            auto conv = static_cast<op::CudnnConv2dBackwardFilters*>(right_.expression().get());
            DescriptorHolder<cudnnTensorDescriptor_t> in_description(conv->arguments()[0], nchw);
            DescriptorHolder<cudnnTensorDescriptor_t> out_dw_description(conv->arguments()[1], nchw);
            DescriptorHolder<cudnnFilterDescriptor_t> out_description(left_, nchw);
            DescriptorHolder<cudnnConvolutionDescriptor_t> conv_description(left_.dtype(),
                                                                            conv->info_.padding_h,
                                                                            conv->info_.padding_w,
                                                                            conv->info_.stride_h,
                                                                            conv->info_.stride_w);
            cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
            void* working_memory    = NULL;
            int working_memory_size = 0;
            auto status = cudnnConvolutionBackwardFilter(
                *get_handle(),
                update_operator.alpha_ptr_,
                in_description.descriptor_,
                argument_data(device, 0),
                out_dw_description.descriptor_,
                argument_data(device, 1),
                conv_description.descriptor_,
                conv_bwd_filter_algo,
                working_memory,
                working_memory_size,
                update_operator.beta_ptr_,
                out_description.descriptor_,
                destination_data(device));
            CUDNN_CHECK_RESULT(status, utils::make_message(
                "Error when running cudnnConvolutionBackwardFilter "));
        }
    };

    int conv2d_backward_filters_impl = register_implementation(
        typeid(op::CudnnConv2dBackwardFilters).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            return std::make_shared<CudnnConv2dBackwardFiltersImpl>(dest, operator_t, x, assignment);
        }
    );

    struct CudnnConv2dBackwardBiasImpl : public CudnnComputation {
        using CudnnComputation::CudnnComputation;
        void run_internal(const Operator& update_operator, bool nchw, const memory::Device& device) override {
            DescriptorHolder<cudnnTensorDescriptor_t> out_description(left_, nchw);
            ELOG(left_.shape());
            DescriptorHolder<cudnnTensorDescriptor_t> out_dw_description(
                right_.expression()->arguments()[0], nchw);
            CUDNN_CHECK_RESULT(cudnnConvolutionBackwardBias(
                *get_handle(),
                update_operator.alpha_ptr_,
                out_dw_description.descriptor_,
                argument_data(device, 0),
                update_operator.beta_ptr_,
                out_description.descriptor_,
                destination_data(device)),
                "Error when computing convolution bias gradient ");
        }
    };

    int conv2d_backward_bias_impl = register_implementation(
        typeid(op::CudnnConv2dBackwardBias).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            return std::make_shared<CudnnConv2dBackwardBiasImpl>(dest, operator_t, x, assignment);
        }
    );

    struct CudnnPool2dImpl : public CudnnComputation {
        using CudnnComputation::CudnnComputation;
        void run_internal(const Operator& update_operator, bool nchw, const memory::Device& device) override {
            auto pool = static_cast<op::CudnnPool2d*>(right_.expression().get());
            // choice of pooling mode follows what TensorFlow does:
            //   https://github.com/tensorflow/tensorflow/blob/
            //   6431560b7ec3565154cb9cdc9c827db78ccfebe7/
            //   tensorflow/stream_executor/cuda/cuda_dnn.cc
            auto cudnn_pooling_mode = (
                pool->pooling_mode_ == POOLING_T_MAX ?
                CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);

            DescriptorHolder<cudnnTensorDescriptor_t> out_description(left_, nchw);
            DescriptorHolder<cudnnTensorDescriptor_t> in_description(pool->arguments()[0], nchw);
            DescriptorHolder<cudnnPoolingDescriptor_t> pooling(cudnn_pooling_mode,
                pool->info_.window_h, pool->info_.window_w, pool->info_.padding_h,
                pool->info_.padding_w, pool->info_.stride_h, pool->info_.stride_w);
            CUDNN_CHECK_RESULT(cudnnPoolingForward(
                *get_handle(),
                pooling.descriptor_,
                update_operator.alpha_ptr_,
                in_description.descriptor_,
                argument_data(device, 0),
                update_operator.beta_ptr_,
                out_description.descriptor_,
                destination_data(device)),
                "Error when computing pooling ");
        }
    };

    int conv2d_pool2d_impl = register_implementation(
        typeid(op::CudnnPool2d).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            return std::make_shared<CudnnPool2dImpl>(dest, operator_t, x, assignment);
        }
    );
}
#endif

namespace {
    #define CUDNN_CHECK_DATA_FORMAT(name, data_format)\
        ASSERT2(data_format == "NCHW" | data_format == "NHWC", utils::make_message(\
            name "'s data_format must be NCHW or NHWC but got ", data_format,\
            " instead."));
    #define CUDNN_CHECK_NDIM(name, array_name, array)\
        ASSERT2(( array ).ndim() == 4, utils::make_message(\
            name "'s " array_name " must be 4 dimensional but got " array_name " ",\
            ( array ).full_expression_name(), " with ndim = ", ( array ).ndim(), "."));

    std::tuple<Array, Array> ensure_floating_type(Array left, Array right) {
        // perform type promotion:
        if (left.dtype() != right.dtype()) {
            auto new_type = type_promotion(left, right);
            if (new_type != DTYPE_DOUBLE && new_type != DTYPE_FLOAT) {
                new_type = DTYPE_FLOAT;
            }
            left = op::astype(left, new_type);
            right = op::astype(right, new_type);
        } else if (left.dtype() != DTYPE_DOUBLE && left.dtype() != DTYPE_FLOAT) {
            auto new_type = DTYPE_FLOAT;
            left = op::astype(left, new_type);
            right = op::astype(right, new_type);
        }
        return std::make_tuple(left, right);
    }
}

namespace op {
    Array cudnn_conv2d(Array input,
                       Array filters,
                       int stride_h,
                       int stride_w,
                       PADDING_T padding,
                       const std::string& data_format) {
        // TODO(jonathan): validate input sizes (filter, odd, etc...)
        CUDNN_CHECK_NDIM("cudnn_conv2d", "input", input);
        CUDNN_CHECK_NDIM("cudnn_conv2d", "filters", filters);
        CUDNN_CHECK_DATA_FORMAT("cudnn_conv2d", data_format);
        std::tie(input, filters) = ensure_floating_type(input, filters);
        auto info = op::compute_conv2d_info(input.shape(),
                                            filters.shape(),
                                            stride_h,
                                            stride_w,
                                            padding,
                                            data_format);
        bool nchw = data_format == "NCHW";
        if (info.odd_padding_h || info.odd_padding_w) {
            // ODD PADDING SUPPORT (no odd padding support in cudnn)
            auto padded_shape = input.shape();
            if (info.odd_padding_h) padded_shape[nchw ? 2 : 1] += 1;
            if (info.odd_padding_w) padded_shape[nchw ? 3 : 2] += 1;
            auto padded = Array::zeros(
                padded_shape, input.dtype(), input.preferred_device());
            // copy values from source array over
            Array padded_slice = padded;
            if (info.odd_padding_h) {
                padded_slice = padded_slice.pluck_axis(
                    nchw ? 2 : 1, Slice(0, padded_shape[nchw ? 2 : 1] - 1));
            }
            if (info.odd_padding_w) {
                padded_slice = padded_slice.pluck_axis(
                    nchw ? 3 : 2, Slice(0, padded_shape[nchw ? 3 : 2] - 1));
            }
            input = op::control_dependency(
                op::assign(padded_slice, OPERATOR_T_EQL, input), padded);
        }
        return Array(std::make_shared<CudnnConv2d>(input, filters, info, nchw));
    }

    Array cudnn_conv2d_backward_input(Array filters,
                                      Array out_dw,
                                      int stride_h,
                                      int stride_w,
                                      const std::vector<int>& input_shape,
                                      PADDING_T padding,
                                      const std::string& data_format) {
        CUDNN_CHECK_NDIM("cudnn_conv2d_backward_input", "out_dw", out_dw);
        CUDNN_CHECK_NDIM("cudnn_conv2d_backward_input", "filters", filters);
        CUDNN_CHECK_DATA_FORMAT("cudnn_conv2d_backward_input", data_format);
        ASSERT2(input_shape.size() == 4, utils::make_message(
            "cudnn_conv2d_backward_input's input_shape must be of size 4, "
            "but got ", input_shape, "."));
        std::tie(out_dw, filters) = ensure_floating_type(out_dw, filters);
        auto info = op::compute_conv2d_info(input_shape,
                                            filters.shape(),
                                            stride_h,
                                            stride_w,
                                            padding,
                                            data_format);
        return Array(std::make_shared<CudnnConv2dBackwardInput>(
            filters, out_dw, input_shape, info, data_format == "NCHW"));
    }

    Array cudnn_conv2d_backward_filters(Array input,
                                        Array out_dw,
                                        int stride_h,
                                        int stride_w,
                                        const std::vector<int>& filters_shape,
                                        PADDING_T padding,
                                        const std::string& data_format) {
        CUDNN_CHECK_NDIM("cudnn_conv2d_backward_filters", "out_dw", out_dw);
        CUDNN_CHECK_NDIM("cudnn_conv2d_backward_filters", "input", input);
        CUDNN_CHECK_DATA_FORMAT("cudnn_conv2d_backward_filters", data_format);
        ASSERT2(filters_shape.size() == 4, utils::make_message(
            "cudnn_conv2d_backward_filters's filters_shape must be of size 4, "
            "but got ", filters_shape, "."));
        std::tie(input, out_dw) = ensure_floating_type(input, out_dw);
        auto info = op::compute_conv2d_info(input.shape(),
                                            filters_shape,
                                            stride_h,
                                            stride_w,
                                            padding,
                                            data_format);
        return Array(std::make_shared<CudnnConv2dBackwardFilters>(
            input, out_dw, filters_shape, info, data_format == "NCHW"));
    }

    Array cudnn_conv2d_backward_bias(Array out_dw,
                                     const std::string& data_format) {
        CUDNN_CHECK_NDIM("cudnn_conv2d_backward_bias", "out_dw", out_dw);
        CUDNN_CHECK_DATA_FORMAT("cudnn_conv2d_backward_bias", data_format);
        return Array(std::make_shared<CudnnConv2dBackwardBias>(out_dw, data_format == "NCHW"));
    }

    Array cudnn_pool2d(const Array& input,
                       int window_h,
                       int window_w,
                       int stride_h,
                       int stride_w,
                       POOLING_T pooling_mode,
                       PADDING_T padding,
                       const std::string& data_format) {
        // TODO(jonathan): check window, stride
        CUDNN_CHECK_NDIM("cudnn_pool2d", "input", input);
        CUDNN_CHECK_DATA_FORMAT("cudnn_pool2d", data_format);
        auto info = op::compute_pool_info(input.shape(),
                                          window_h,
                                          window_w,
                                          stride_h,
                                          stride_w,
                                          padding,
                                          data_format);
        return Array(std::make_shared<CudnnPool2d>(
            input, info, pooling_mode, data_format == "NCHW"));
    }

    Array cudnn_pool2d_backward(const Array& out,
                                const Array& out_dw,
                                const Array& in,
                                int window_h,
                                int window_w,
                                int stride_h,
                                int stride_w,
                                POOLING_T pooling_mode,
                                PADDING_T padding,
                                const std::string& data_format) {
        CUDNN_CHECK_NDIM("cudnn_pool2d_backward", "out", out);
        CUDNN_CHECK_NDIM("cudnn_pool2d_backward", "out_dw", out_dw);
        CUDNN_CHECK_NDIM("cudnn_pool2d_backward", "in", in);
        CUDNN_CHECK_DATA_FORMAT("cudnn_pool2d_backward", data_format);
        auto info = op::compute_pool_info(in.shape(),
                                          window_h,
                                          window_w,
                                          stride_h,
                                          stride_w,
                                          padding,
                                          data_format);
        return Array(std::make_shared<CudnnPool2dBackward>(
            out, out_dw, in, info, pooling_mode, data_format == "NCHW"));
    }
}  // namespace op
