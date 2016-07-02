#include "conv_backward.h"
#include "dali/config.h"
#include "dali/runtime_config.h"
#include "dali/array/array.h"
#include "dali/array/function/function.h"

///////////////////////////////////////////////////////////////////////////////
//                    Conv2dBwdInputFunction                                 //
///////////////////////////////////////////////////////////////////////////////
struct Conv2dBwdInputFunction : public Function<Conv2dBwdInputFunction,
                                        Array,
                                        Array,
                                        Array,
                                        int,
                                        int,
                                        std::vector<int>,
                                        PADDING_T,
                                        std::string> {
    static std::vector<int> deduce_output_bshape(const Array& filters,
                                                 const Array& out_dw,
                                                 int stride_h,
                                                 int stride_w,
                                                 const std::vector<int>& result_shape,
                                                 PADDING_T padding,
                                                 const std::string& data_format) {
        // TODO(szymon): potentially some checks are
        //               should be performed here.
        // idea: use convforward bshape computation and see if you get input back out.
        return result_shape;
    }

    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> in_dw,
                    TypedArray<memory::DEVICE_T_CPU, T> filters,
                    TypedArray<memory::DEVICE_T_CPU, T> out_dw,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    PADDING_T padding,
                    const std::string& data_format) {
        throw std::runtime_error("not implemented!");
    }

#ifdef DALI_USE_CUDA
    template<OPERATOR_T operator_t, typename T, DALI_FUNC_ENABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> in_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> filters,
                    TypedArray<memory::DEVICE_T_GPU, T> out_dw,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    PADDING_T padding,
                    const std::string& data_format) {
        ASSERT2(false, "integer convolution backward is not implemented for GPU.");
    }

    template<OPERATOR_T operator_t, typename T, DALI_FUNC_DISABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> in_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> filters,
                    TypedArray<memory::DEVICE_T_GPU, T> out_dw,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    PADDING_T padding,
                    const std::string data_format) {

        auto info = internal::compute_conv_info(in_dw.array.shape(),
                                                filters.array.shape(),
                                                stride_h,
                                                stride_w,
                                                padding,
                                                data_format);

        ASSERT2(info.odd_padding_h == 0 && info.odd_padding_w == 0,
                "Conv2d odd sized padding is presently unsupported.");


        auto out_access_mode = internal::OperatorAM<operator_t>::get(in_dw);

        cudnn::conv2d_bwd_input(
                std::make_shared<cudnn::wrapper::Tensor>(
                    in_dw, data_format, out_access_mode),
                std::make_shared<cudnn::wrapper::Filters>(
                    filters, data_format),
                std::make_shared<cudnn::wrapper::Tensor>(
                    out_dw, data_format),
                std::make_shared<cudnn::wrapper::Convolution>(
                    info.padding_h, info.padding_w, stride_h, stride_w),
                cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
        );
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////
//                    Conv2dBwdFiltersFunction                               //
///////////////////////////////////////////////////////////////////////////////
struct Conv2dBwdFiltersFunction : public Function<Conv2dBwdFiltersFunction,
                                        Array,
                                        Array,
                                        Array,
                                        int,
                                        int,
                                        std::vector<int>,
                                        PADDING_T,
                                        std::string> {
    static std::vector<int> deduce_output_bshape(const Array& input,
                                                 const Array& out_dw,
                                                 int stride_h,
                                                 int stride_w,
                                                 const std::vector<int>& result_shape,
                                                 PADDING_T padding,
                                                 const std::string& data_format) {
        // TODO(szymon): potentially some checks are
        //               should be performed here.
        //               idea: use convforward bshape
        //               computation and
        //               see if you get input back out.
        return result_shape;
    }

    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_ENABLE_IF_MUL_DIV>
    void typed_eval(TypedArray<devT, T> filters_dw,
                    TypedArray<devT, T> input,
                    TypedArray<devT, T> out_dw,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    PADDING_T padding,
                    const std::string& data_format) {
        ASSERT2(!(var_operator_t == OPERATOR_T_MUL) && !(var_operator_t == OPERATOR_T_MUL),
                "Convolution's result cannot be inplace-multiplied or inplace-divided.");
        ASSERT2(false, "If asserts above are complete this message should never be displayed");
    }

    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void typed_eval(TypedArray<devT, T> filters_dw,
                    TypedArray<devT, T> input,
                    TypedArray<devT, T> out_dw,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    PADDING_T padding,
                    const std::string& data_format) {
#ifdef DALI_USE_CUDNN
        if (use_cudnn && devT == memory::DEVICE_T_GPU && template_to_dtype<T>() != DTYPE_INT32) {
            cudnn_conv_backward<operator_t,T,devT>(filters_dw,
                                                   input,
                                                   out_dw,
                                                   stride_h,
                                                   stride_w,
                                                   result_shape,
                                                   padding,
                                                   data_format);
            return;
        }
#endif
        blas_conv_backward<operator_t,T,devT>(filters_dw,
                                              input,
                                              out_dw,
                                              stride_h,
                                              stride_w,
                                              result_shape,
                                              padding,
                                              data_format);
    }

    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void blas_conv_backward(TypedArray<devT, T> filters_dw,
                            TypedArray<devT, T> input,
                            TypedArray<devT, T> out_dw,
                            int stride_h,
                            int stride_w,
                            const std::vector<int>& result_shape,
                            PADDING_T padding,
                            const std::string& data_format) {
        ASSERT2(false, "not implemented");
    }

#ifdef DALI_USE_CUDNN
    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void cudnn_conv_backward(TypedArray<devT, T> filters_dw,
                             TypedArray<devT, T> input,
                             TypedArray<devT, T> out_dw,
                             int stride_h,
                             int stride_w,
                             const std::vector<int>& result_shape,
                             PADDING_T padding,
                             const std::string& data_format) {

        auto info = internal::compute_conv_info(input.array.shape(),
                                                filters_dw.array.shape(),
                                                stride_h,
                                                stride_w,
                                                padding,
                                                data_format);

        ASSERT2(info.odd_padding_h == 0 && info.odd_padding_w == 0,
                "Conv2d odd sized padding is presently unsupported.");

        auto out_access_mode = internal::OperatorAM<operator_t>::get(filters_dw);

        cudnn::conv2d_bwd_filters(
                std::make_shared<cudnn::wrapper::Filters>(
                        filters_dw, data_format, out_access_mode),
                std::make_shared<cudnn::wrapper::Tensor>(
                        input, data_format),
                std::make_shared<cudnn::wrapper::Tensor>(
                        out_dw, data_format),
                std::make_shared<cudnn::wrapper::Convolution>(
                        info.padding_h, info.padding_w, stride_h, stride_w),
                cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
        );
    }
#endif
};

namespace op {
    Assignable<Array> conv2d_backward_input(
                     const Array& filters,
                     const Array& out_dw,
                     int stride_h,
                     int stride_w,
                     const std::vector<int>& result_shape,
                     PADDING_T padding,
                     const std::string& data_format) {
        return Conv2dBwdInputFunction::run(filters,
                                           out_dw,
                                           stride_h,
                                           stride_w,
                                           result_shape,
                                           padding,
                                           data_format);
    }

    Assignable<Array> conv2d_backward_filters(
                     const Array& input,
                     const Array& out_dw,
                     int stride_h,
                     int stride_w,
                     const std::vector<int>& result_shape,
                     PADDING_T padding,
                     const std::string& data_format) {
        return Conv2dBwdFiltersFunction::run(input,
                                             out_dw,
                                             stride_h,
                                             stride_w,
                                             result_shape,
                                             padding,
                                             data_format);
    }
}  // namespace op
