#include "conv_backward.h"

#include "dali/config.h"

#include "dali/array/array.h"
#include "dali/array/function/function.h"
#include "dali/array/lazy/im2col.h"
#include "dali/array/mshadow_extension/dali_gemm_engine_exp.h"
#include "dali/array/op/cudnn_utils.h"
#include "dali/array/op/spatial/utils.h"
#include "dali/runtime_config.h"

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

    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void typed_eval(TypedArray<devT, T> in_dw,
                    TypedArray<devT, T> filters,
                    TypedArray<devT, T> out_dw,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    PADDING_T padding,
                    const std::string& data_format) {
#ifdef DALI_USE_CUDNN
        if (use_cudnn && devT == memory::DEVICE_T_GPU &&
                !std::is_same<T, int>::value &&
                data_format != "NHWC") {
            cudnn_conv_backward<operator_t,T,devT>(in_dw,
                                                   filters,
                                                   out_dw,
                                                   stride_h,
                                                   stride_w,
                                                   padding,
                                                   data_format);
            return;
        }
#endif
        blas_conv_backward<operator_t,T,devT>(in_dw,
                                              filters,
                                              out_dw,
                                              stride_h,
                                              stride_w,
                                              padding,
                                              data_format);
    }

    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_ENABLE_IF_MUL_DIV>
    void typed_eval(TypedArray<devT, T> in_dw,
                    TypedArray<devT, T> filters,
                    TypedArray<devT, T> out_dw,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    PADDING_T padding,
                    const std::string& data_format) {
        ASSERT2(!(var_operator_t == OPERATOR_T_MUL) && !(var_operator_t == OPERATOR_T_MUL),
                "Conv2dBwdInput's result cannot be inplace-multiplied or inplace-divided.");
        ASSERT2(false, "If asserts above are complete this message should never be displayed.");
    }

    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void blas_conv_backward(TypedArray<devT, T> in_dw,
                            TypedArray<devT, T> filters,
                            TypedArray<devT, T> out_dw,
                            int stride_h,
                            int stride_w,
                            PADDING_T padding,
                            const std::string data_format) {

        auto info = internal::compute_conv_info(in_dw.array.shape(),
                                                filters.array.shape(),
                                                stride_h,
                                                stride_w,
                                                padding,
                                                data_format);

        typedef decltype(filters.contiguous_d2()) mshadow_tensor_t;
        filters.array = filters.array.copyless_reshape({filters.array.shape()[0], -1});
        if (data_format == "NCHW") {
            filters.array = filters.array.transpose();
        } else {
            filters.array = filters.array.transpose();
        }

        bool filters_transposed;
        mshadow_tensor_t filters_tensor;
        std::tie(filters_transposed, filters_tensor) = filters.blas_friendly_tensor();

        if (data_format == "NCHW") {
            auto out_dw_cnhw_shape = out_dw.array.shape();
            std::swap(out_dw_cnhw_shape[0], out_dw_cnhw_shape[1]);
            Array out_dw_cnhw_arr(out_dw_cnhw_shape, template_to_dtype<T>(), in_dw.device);
            TypedArray<devT, T> out_dw_cnhw(out_dw_cnhw_arr, in_dw.device, out_dw_cnhw_shape);

            operator_assign_contiguous<OPERATOR_T_EQL, 4>(
                out_dw_cnhw,
                mshadow::expr::swapaxis<1, 0>(
                    out_dw.contiguous_d4()
                )
            );
            out_dw_cnhw.array = out_dw_cnhw.array.copyless_reshape({out_dw_cnhw_shape[0], -1});
            // make temporary storage information compatible with gemm:
            bool out_dw_cnhw_transposed;
            mshadow_tensor_t out_dw_cnhw_tensor;
            std::tie(out_dw_cnhw_transposed, out_dw_cnhw_tensor) = out_dw_cnhw.blas_friendly_tensor();

            Array in_dw_im2coled_arr(
                {filters.array.shape()[0], out_dw_cnhw.array.shape()[1]},
                template_to_dtype<T>(), in_dw.device
            );
            TypedArray<devT, T> in_dw_im2coled(in_dw_im2coled_arr, in_dw.device, in_dw_im2coled_arr.shape());

            // now that error is in 2D, and with channels swapped with N
            // we can use gemm to complete the gradient propagation:
            operator_assign_contiguous<OPERATOR_T_EQL, 2>(
                in_dw_im2coled,
                dali_gemm(
                    filters_tensor,
                    out_dw_cnhw_tensor,
                    filters_transposed,
                    out_dw_cnhw_transposed,
                    (T)1.0f
                ),
                /*collapse_leading=*/false
            );


            operator_assign<operator_t, 4>(
                in_dw,
                mshadow::expr::pack_col2patch<mshadow::expr::DATA_FORMAT_NCHW>(
                    in_dw_im2coled.contiguous_d2(),
                    vector2shape<4>(in_dw.array.shape()),
                    info.filter_h,
                    info.filter_w,
                    stride_h,
                    stride_w,
                    /*dilate_h=*/1,
                    /*dilate_w=*/1,
                    /*prepad_h=*/info.padding_h,
                    /*prepad_w=*/info.padding_w,
                    /*postpad_h=*/info.padding_h + info.odd_padding_h,
                    /*postpad_w=*/info.padding_w + info.odd_padding_w
                ),
                /*collapse_leading=*/false
            );
        } else {
            /* NHWC forward pass is:
             *
             *   output = (Im2col(Input))^T * Filters^T
             *
             * NHWC backward pass is:
             *
             *   ∂Im2col(Input)/∂E = Filters^T * ∂output/∂E^T
             *
             * Our 2d shapes into gemm are as follows:
             *
             *   ∂Im2col(Input)/∂E => (window_h * window_w * c) x (n * h * w)
             *
             *   Filters^T => (window_h * window_w * c) x (channels_out)
             *
             *   ∂output/∂E^T => (channels_out) x (n * h * w)
             *
             */
            // ∂output/∂E^T
            out_dw.array = out_dw.array.copyless_reshape({-1, out_dw.array.shape()[3]});
            out_dw.array = out_dw.array.transpose();

            bool out_dw_transposed;
            mshadow_tensor_t out_dw_tensor;
            std::tie(out_dw_transposed, out_dw_tensor) = out_dw.blas_friendly_tensor();

            // ∂Im2col(Input)/∂E
            Array im2col_dw_arr = Array::zeros(
                {info.filter_h * info.filter_w * info.in_channels, info.batch_size * info.out_h * info.out_w},
                template_to_dtype<T>(), out_dw.device
            );
            TypedArray<devT, T> im2col_dw(im2col_dw_arr, in_dw.device, im2col_dw_arr.shape());

            operator_assign_contiguous<OPERATOR_T_EQL, 2>(
                im2col_dw,
                dali_gemm(
                    filters_tensor,
                    out_dw_tensor,
                    filters_transposed,
                    out_dw_transposed,
                    (T)1.0f
                )
            );

            operator_assign<operator_t, 4>(
                in_dw,
                mshadow::expr::pack_col2patch<mshadow::expr::DATA_FORMAT_NHWC>(
                    im2col_dw.contiguous_d2(),
                    vector2shape<4>(in_dw.array.shape()),
                    info.filter_h,
                    info.filter_w,
                    stride_h,
                    stride_w,
                    /*dilate_h=*/1,
                    /*dilate_w=*/1,
                    /*prepad_h=*/info.padding_h,
                    /*prepad_w=*/info.padding_w,
                    /*postpad_h=*/info.padding_h + info.odd_padding_h,
                    /*postpad_w=*/info.padding_w + info.odd_padding_w
                ),
                /*collapse_leading=*/false
            );
        }
    }

#ifdef DALI_USE_CUDNN
    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void cudnn_conv_backward(TypedArray<devT, T> in_dw,
                             TypedArray<devT, T> filters,
                             TypedArray<devT, T> out_dw,
                             int stride_h,
                             int stride_w,
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
        if (use_cudnn && devT == memory::DEVICE_T_GPU &&
                !std::is_same<T, int>::value &&
                data_format != "NHWC") {
            cudnn_conv_backward<operator_t,T,devT>(filters_dw,
                                                   input,
                                                   out_dw,
                                                   stride_h,
                                                   stride_w,
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
                                              padding,
                                              data_format);
    }

    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void blas_conv_backward(TypedArray<devT, T> filters_dw,
                            TypedArray<devT, T> input,
                            TypedArray<devT, T> out_dw,
                            int stride_h,
                            int stride_w,
                            PADDING_T padding,
                            const std::string& data_format) {
        auto im2col_storage = internal::compute_im2col(
            input, filters_dw.array.shape(), stride_h, stride_w, padding, data_format
        );
        typedef decltype(im2col_storage.contiguous_d2()) mshadow_tensor_t;
        filters_dw.array   = filters_dw.array.copyless_reshape({filters_dw.array.shape()[0], -1});
        if (data_format == "NCHW") {
            im2col_storage.array = im2col_storage.array.transpose();
        } else {
            im2col_storage.array = im2col_storage.array.transpose();
        }

        mshadow_tensor_t im2col_tensor;
        bool im2col_transposed;
        std::tie(im2col_transposed, im2col_tensor) = im2col_storage.blas_friendly_tensor();

        if (data_format == "NCHW") {
            // make errors be 2D, and swap C with N:
            auto out_dw_cnhw_shape = out_dw.array.shape();
            std::swap(out_dw_cnhw_shape[0], out_dw_cnhw_shape[1]);
            Array out_dw_cnhw_arr(out_dw_cnhw_shape, template_to_dtype<T>(), out_dw.device);
            TypedArray<devT, T> out_dw_cnhw(out_dw_cnhw_arr, out_dw.device, out_dw_cnhw_arr.shape());
            // assigned the swapped data onto a temporary storage:
            operator_assign_contiguous<OPERATOR_T_EQL, 4>(
                out_dw_cnhw,
                mshadow::expr::swapaxis<1,0>(out_dw.contiguous_d4())
            );
            bool out_dw_cnhw_transposed;
            mshadow_tensor_t out_dw_cnhw_tensor;

            out_dw_cnhw.array = out_dw_cnhw.array.copyless_reshape({out_dw_cnhw.array.shape()[0], -1});

            // make temporary storage information compatible with gemm:
            std::tie(out_dw_cnhw_transposed, out_dw_cnhw_tensor) = out_dw_cnhw.blas_friendly_tensor();
            // now that error is in 2D, and with channels swapped with N
            // we can use gemm to complete the gradient propagation:
            operator_assign_contiguous<operator_t, 2>(
                filters_dw,
                dali_gemm(
                    out_dw_cnhw_tensor,
                    im2col_tensor,
                    out_dw_cnhw_transposed,
                    im2col_transposed,
                    (T)1.0f
                ),
                /*collapse_leading=*/false
            );
        } else {
            /* NHWC forward pass is:
             *
             *   output = (Im2col(Input))^T * Filters^T
             *
             * NHWC backward pass is:
             *
             *   ∂Filters/∂E = ∂output/∂E^T * Im2col(Input)^T
             *
             * Our 2d shapes into gemm are as follows:
             *
             *   Im2col(Input)^T => (n * h * w) x (window_h * window_w * c)
             *
             *   ∂output/∂E^T => (channels_out) x (n * h * w)
             *
             *   ∂Filters/∂E =>  (channels_out) x (window_h * window_w * c)
             *
             */
            out_dw.array = out_dw.array.copyless_reshape({-1, out_dw.array.shape()[3]});
            out_dw.array = out_dw.array.transpose();

            bool out_dw_transposed;
            mshadow_tensor_t out_dw_tensor;
            // make temporary storage information compatible with gemm:
            std::tie(out_dw_transposed, out_dw_tensor) = out_dw.blas_friendly_tensor();

            operator_assign_contiguous<operator_t, 2>(
                filters_dw,
                dali_gemm(
                    out_dw_tensor,
                    im2col_tensor,
                    out_dw_transposed,
                    im2col_transposed,
                    (T)1.0f
                )
            );
        }
    }

#ifdef DALI_USE_CUDNN
    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void cudnn_conv_backward(TypedArray<devT, T> filters_dw,
                             TypedArray<devT, T> input,
                             TypedArray<devT, T> out_dw,
                             int stride_h,
                             int stride_w,
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
