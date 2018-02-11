#include "conv.h"
#include "dali/config.h"
#include "dali/array/cudnn/utils.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/array/op/spatial_utils.h"
#include "dali/array/op/elementwise_operation.h"
namespace op {
    struct CudnnExpression : public Expression {
        const ConvFunctionInfo info_;
        const bool nchw_;
        CudnnExpression(const std::vector<int>& shape, DType dtype,
                        const std::vector<Array>& arguments, ConvFunctionInfo info, bool nchw)
            : Expression(shape, dtype, arguments), info_(info), nchw_(nchw) {}
        memory::Device preferred_device() const override {
            return device_promotion(arguments_[0], arguments_[1]);
        }
        virtual bool supports_operator(OPERATOR_T operator_t) const {
            return (operator_t == OPERATOR_T_EQL ||
                    operator_t == OPERATOR_T_ADD ||
                    operator_t == OPERATOR_T_SUB);
        }
    };

    struct CudnnConv2d : public CudnnExpression {
        CudnnConv2d(const Array& input, const Array& filters, ConvFunctionInfo info, bool nchw)
            : CudnnExpression(nchw ? std::vector<int>({info.batch_size, info.out_channels, info.out_h, info.out_w}) :
                              std::vector<int>({info.batch_size, info.out_h, info.out_w, info.out_channels}),
                              input.dtype(), {input, filters}, info, nchw) {}
        using Expression::copy;
        virtual expression_ptr copy() const override {
            return std::make_shared<CudnnConv2d>(arguments_[0], arguments_[1], info_, nchw_);
        }
    };

    struct CudnnConv2dBackwardInput : public CudnnExpression {
        CudnnConv2dBackwardInput(const Array& filters, const Array& out_dw, const std::vector<int>& input_shape, ConvFunctionInfo info, bool nchw)
            : CudnnExpression(input_shape, filters.dtype(), {filters, out_dw}, info, nchw) {}
        using Expression::copy;
        virtual expression_ptr copy() const override {
            return std::make_shared<CudnnConv2dBackwardInput>(arguments_[0], arguments_[1], shape_, info_, nchw_);
        }
    };

    struct CudnnConv2dBackwardFilters : public CudnnExpression {
        CudnnConv2dBackwardFilters(const Array& input, const Array& out_dw, const std::vector<int>& filters_shape, ConvFunctionInfo info, bool nchw)
            : CudnnExpression(filters_shape, input.dtype(), {input, out_dw}, info, nchw) {}
        using Expression::copy;
        virtual expression_ptr copy() const override {
            return std::make_shared<CudnnConv2dBackwardFilters>(arguments_[0], arguments_[1], shape_, info_, nchw_);
        }
    };
}
#ifdef DALI_USE_CUDNN
namespace {
    struct Operator {
        float alpha_f_, beta_f_;
        double alpha_d_, beta_d_;
        void* alpha_ptr_;
        void* beta_ptr_;
        Operator(OPERATOR_T operator_type, DType dtype) {
            switch (operator_type) {
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
        void* destination_data(memory::Device device) {
            memory::AM access_mode = operator_t_ == OPERATOR_T_EQL && left_.spans_entire_memory() ?
                memory::AM_OVERWRITE : memory::AM_MUTABLE;
            return left_.memory().data(device, access_mode) + left_.offset();
        }

        void* argument_data(memory::Device device, int idx) {
            const auto& arg = right_.expression()->arguments_()[idx];
            return arg.memory().readonly_data(device) + arg.offset();
        }

        void run() {
            Operator update_operator(operator_t_, left_.dtype());
            run_internal(update_operator);
        }
    };

    struct CudnnConv2dImpl : public CudnnComputation {
        using Computation::Computation;
        void run_internal(Operator& update_operator) {
            auto conv = static_cast<CudnnConv2d>(right_.expression().get());
            DescriptorHolder<cudnnTensorDescriptor_t> inputs_description(conv->arguments_()[0], conv->nchw_);
            DescriptorHolder<cudnnTensorDescriptor_t> out_description(left_, conv->nchw_);
            DescriptorHolder<cudnnFilterDescriptor_t> filters_description(conv->arguments_()[1], conv->nchw_);
            DescriptorHolder<cudnnConvolutionDescriptor_t> conv_description(conv->info.padding_h,
                                                                            conv->info.padding_w,
                                                                            conv->info.stride_h,
                                                                            conv->info.stride_w);
            auto device = left_.preferred_device();
            // TODO(jonathan) autotune algo and working space selection
            // (See: SuperNeurons: Dynamic GPU Memory Management for Training
            // Deep Neural Networks, Wang et al.)
            cudnnConvolutionFwdAlgo_t conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            void* working_memory    = NULL;
            int working_memory_size = 0;
            auto status = cudnnConvolutionForward(
                *get_handle(),
                update_operator.alpha_ptr_,
                in_description.descriptor_,
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
            ASSERT2(status == CUDNN_STATUS_SUCCESS, utils::make_message(
                "Error when running cudnnConvolutionForward with \n"
                "CONVOLUTION: ", *conv, "\n"
                "OUTPUT:      ", *out, "\n"
                "INPUT:       ", *in, "\n"
                "FILTERS:     ", *filters, "\n"
                ": ", cudnnGetErrorString(status)));
        }
    };

    int conv2d_impl = register_implementation(
        typeid(op::CudnnConv2d).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            return std::make_shared<CudnnConv2dImpl>(dest, operator_t, x, assignment);
        }
    );

    struct CudnnConv2dBackwardInputImpl : public CudnnComputation {
        using Computation::Computation;
        void run_internal(Operator& update_operator) {
            auto conv = static_cast<CudnnConv2dBackwardInput>(right_.expression().get());
            DescriptorHolder<cudnnFilterDescriptor_t> filters_description(conv->arguments_()[0], conv->nchw_);
            DescriptorHolder<cudnnTensorDescriptor_t> out_dw_description(conv->arguments_()[1], conv->nchw_);
            DescriptorHolder<cudnnTensorDescriptor_t> out_description(left_, conv->nchw_);
            DescriptorHolder<cudnnConvolutionDescriptor_t> conv_description(conv->info.padding_h,
                                                                            conv->info.padding_w,
                                                                            conv->info.stride_h,
                                                                            conv->info.stride_w);
            auto device = left_.preferred_device();
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
            ASSERT2(status == CUDNN_STATUS_SUCCESS, utils::make_message(
                "Error when running cudnnConvolutionBackwardData with \n"
                "CONVOLUTION: ", *conv, "\n"
                "OUTPUT:      ", *out, "\n"
                "INPUT:       ", *in, "\n"
                "FILTERS:     ", *filters, "\n"
                ": ", cudnnGetErrorString(status)));
        }
    };

    int conv2d_backward_input_impl = register_implementation(
        typeid(op::CudnnConv2dBackwardInput).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            return std::make_shared<CudnnConv2dBackwardInputImpl>(dest, operator_t, x, assignment);
        }
    );

    struct CudnnConv2dBackwardInputImpl : public CudnnComputation {
        using Computation::Computation;
        void run_internal(Operator& update_operator) {
            auto conv = static_cast<CudnnConv2dBackwardInput>(right_.expression().get());
            DescriptorHolder<cudnnTensorDescriptor_t> in_description(conv->arguments_()[0], conv->nchw_);
            DescriptorHolder<cudnnTensorDescriptor_t> out_dw_description(conv->arguments_()[1], conv->nchw_);
            DescriptorHolder<cudnnFilterDescriptor_t> out_description(left_, conv->nchw_);
            DescriptorHolder<cudnnConvolutionDescriptor_t> conv_description(conv->info.padding_h,
                                                                            conv->info.padding_w,
                                                                            conv->info.stride_h,
                                                                            conv->info.stride_w);
            auto device = left_.preferred_device();
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
            ASSERT2(status == CUDNN_STATUS_SUCCESS, utils::make_message(
                "Error when running cudnnConvolutionBackwardFilter with \n"
                "CONVOLUTION: ", *conv, "\n"
                "OUTPUT:      ", *out, "\n"
                "INPUT:       ", *in, "\n"
                "FILTERS:     ", *filters, "\n"
                ": ", cudnnGetErrorString(status)));
        }
    };

    int conv2d_backward_filters_impl = register_implementation(
        typeid(op::CudnnConv2dBackwardFilters).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            return std::make_shared<CudnnConv2dBackwardFiltersImpl>(dest, operator_t, x, assignment);
        }
    );
}
#endif

// ODD PADDING SUPPORT
// This whole shenanigans is needed because
// cudnn does not support odd padding.
// If it is any consolation TF does it as well.
// if (info.odd_padding_h || info.odd_padding_w) {
//     // compute padded shape
//     DataFormatDimMapping mapping(data_format);

//     auto padded_shape = input.array.shape();
//     if (info.odd_padding_h) padded_shape[mapping.h_dim] += 1;
//     if (info.odd_padding_w) padded_shape[mapping.w_dim] += 1;

//     // create temporary storage for padded array.
//     auto padded_input_arr = Array::zeros(padded_shape,
//                                          input.array.dtype(),
//                                          input.device);
//     TypedArray<devT,T> padded_input(padded_input_arr, input.device, padded_shape);
//     maybe_copied_input = padded_input;

//     // copy values from source array over
//     Array padded_input_slice_arr = padded_input_arr;
//     if (info.odd_padding_h) {
//         padded_input_slice_arr = padded_input_slice_arr.pluck_axis(
//                 mapping.h_dim, Slice(0, padded_shape[mapping.h_dim] -1));
//     }
//     if (info.odd_padding_w) {
//         padded_input_slice_arr = padded_input_slice_arr.pluck_axis(
//                 mapping.w_dim, Slice(0, padded_shape[mapping.w_dim] -1));
//     }

//     TypedArray<devT,T> padded_input_slice(padded_input_slice_arr,
//                                           padded_input.device,
//                                           input.array.shape());
//     padded_input_slice.d2(memory::AM_MUTABLE) =
//             mshadow::expr::F<mshadow::op::identity>(input.d2());
// }

namespace op {
    Array cudnn_conv2d(Array input,
                       Array filters,
                       int stride_h,
                       int stride_w,
                       PADDING_T padding,
                       const std::string& data_format) {
        // TODO(jonathan): validate input sizes (filter, odd, etc...)
        ASSERT2(input.ndim() == 4, utils::make_message(
            "cudnn_conv2d's input must be 4 dimensional but got input ",
            input.full_expression_name(), " with ndim = ", input.ndim(), "."));
        ASSERT2(filters.ndim() == 4, utils::make_message(
            "cudnn_conv2d's filters must be 4 dimensional but got filters ",
            filters.full_expression_name(), " with ndim = ", filters.ndim(), "."));
        ASSERT2(data_format == "NCHW" | data_format == "NHWC", utils::make_message(
            "cudnn_conv2d's data_format must be NCHW or NHWC but got ", data_format,
            " instead."));
        // perform type promotion:
        DType new_type = input.dtype();
        if (input.dtype() != filters.dtype()) {
            auto new_type = type_promotion(input, filters);
            if (new_type != DTYPE_DOUBLE && new_type != DTYPE_FLOAT) {
                new_type = DTYPE_FLOAT;
            }
            input = astype(input, new_type);
            filters = astype(filters, new_type);
        } else if (new_type != DTYPE_DOUBLE && new_type != DTYPE_FLOAT) {
            new_type = DTYPE_FLOAT;
        }
        input = astype(input, new_type);
        filters = astype(filters, new_type);

        auto info = op::compute_conv2d_info(input.shape(),
                                            filters.shape(),
                                            stride_h,
                                            stride_w,
                                            padding,
                                            data_format);
        return Array(std::make_shared<CudnnConv2d>(input, filters, info, data_format == "NCHW"));
    }

    Array cudnn_conv2d_backward_input(Array filters,
                                      Array out_dw,
                                      int stride_h,
                                      int stride_w,
                                      const std::vector<int>& input_shape,
                                      PADDING_T padding,
                                      const std::string& data_format) {
        ASSERT2(out_dw.ndim() == 4, utils::make_message(
            "cudnn_conv2d_backward_input's out_dw must be 4 dimensional but got out_dw ",
            out_dw.full_expression_name(), " with ndim = ", out_dw.ndim(), "."));
        ASSERT2(filters.ndim() == 4, utils::make_message(
            "cudnn_conv2d_backward_input's filters must be 4 dimensional but got filters ",
            filters.full_expression_name(), " with ndim = ", filters.ndim(), "."));
        ASSERT2(data_format == "NCHW" | data_format == "NHWC", utils::make_message(
            "cudnn_conv2d_backward_input's data_format must be NCHW or NHWC but got ", data_format,
            " instead."));
        ASSERT2(input_shape.size() == 4, utils::make_message(
            "cudnn_conv2d_backward_input's input_shape must be of size 4, "
            "but got ", input_shape, "."));
        // perform type promotion:
        DType new_type = filters.dtype();
        if (out_dw.dtype() != filters.dtype()) {
            auto new_type = type_promotion(out_dw, filters);
            if (new_type != DTYPE_DOUBLE && new_type != DTYPE_FLOAT) {
                new_type = DTYPE_FLOAT;
            }
            out_dw = astype(out_dw, new_type);
            filters = astype(filters, new_type);
        } else if (new_type != DTYPE_DOUBLE && new_type != DTYPE_FLOAT) {
            new_type = DTYPE_FLOAT;
        }
        out_dw = astype(out_dw, new_type);
        filters = astype(filters, new_type);
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
        ASSERT2(out_dw.ndim() == 4, utils::make_message(
            "cudnn_conv2d_backward_filters's out_dw must be 4 dimensional but got out_dw ",
            out_dw.full_expression_name(), " with ndim = ", out_dw.ndim(), "."));
        ASSERT2(input.ndim() == 4, utils::make_message(
            "cudnn_conv2d_backward_filters's input must be 4 dimensional but got input ",
            input.full_expression_name(), " with ndim = ", input.ndim(), "."));
        ASSERT2(data_format == "NCHW" | data_format == "NHWC", utils::make_message(
            "cudnn_conv2d_backward_filters's data_format must be NCHW or NHWC but got ", data_format,
            " instead."));
        ASSERT2(filters_shape.size() == 4, utils::make_message(
            "cudnn_conv2d_backward_filters's filters_shape must be of size 4, "
            "but got ", filters_shape, "."));
        // perform type promotion:
        DType new_type = input.dtype();
        if (out_dw.dtype() != input.dtype()) {
            auto new_type = type_promotion(out_dw, input);
            if (new_type != DTYPE_DOUBLE && new_type != DTYPE_FLOAT) {
                new_type = DTYPE_FLOAT;
            }
            out_dw = astype(out_dw, new_type);
            input = astype(input, new_type);
        } else if (new_type != DTYPE_DOUBLE && new_type != DTYPE_FLOAT) {
            new_type = DTYPE_FLOAT;
        }
        out_dw = astype(out_dw, new_type);
        input = astype(input, new_type);
        auto info = op::compute_conv2d_info(input.shape(),
                                            filters_shape,
                                            stride_h,
                                            stride_w,
                                            padding,
                                            data_format);
        return Array(std::make_shared<CudnnConv2dBackwardFilters>(
            input, out_dw, filters_shape, info, data_format == "NCHW"));
    }
}  // namespace op
