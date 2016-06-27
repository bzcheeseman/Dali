#include "spatial.h"

#include "dali/config.h"
#include "dali/runtime_config.h"
#include "dali/array/array.h"
#include "dali/array/function/function.h"
#include "dali/array/function/operator.h"
#include "dali/array/functor.h"
#include "dali/array/lazy/im2col.h"
#include "dali/array/mshadow_extension/dali_gemm_engine_exp.h"
#include "dali/utils/random.h"
#ifdef DALI_USE_CUDA
    #include "dali/array/op/cudnn_utils.h"
#endif


///////////////////////////////////////////////////////////////////////////////
//                                    UTILS                                  //
///////////////////////////////////////////////////////////////////////////////

int int_ceil(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

std::tuple<int, int> convolution_padding(
        const std::vector<int>& input_shape,
        const std::vector<int>& filters_shape,
        const std::vector<int>& output_shape,
        int stride_h,
        int stride_w,
        const std::string&      data_format,
        PADDING_T           padding) {

    int h_dim, w_dim;
    if (data_format == "NCHW") {
        h_dim = 2;
        w_dim = 3;
    } else if (data_format == "NHWC") {
        h_dim = 1;
        w_dim = 2;
    }

    int in_h     = input_shape[h_dim];
    int in_w     = input_shape[w_dim];
    int out_h    = output_shape[h_dim];
    int out_w    = output_shape[w_dim];
    int filter_h = filters_shape[h_dim];
    int filter_w = filters_shape[w_dim];;

    int padding_h, padding_w;

    if (padding == PADDING_T_SAME) {
        padding_h = (out_h - 1) * stride_h + filter_h - in_h;
        padding_w = (out_w - 1) * stride_w + filter_w - in_w;
        ASSERT2(padding_h % 2 == 0 && padding_w % 2 == 0,
                "Conv2d odd sized padding is presently unsupported.");
        padding_h /= 2;
        padding_w /= 2;
    } else if (padding == PADDING_T_VALID) {
        padding_h = 0;
        padding_w = 0;
    }

    return std::make_tuple(padding_h, padding_w);
}

void check_data_format(const std::string& data_format) {
    ASSERT2(data_format == "NCHW" || data_format == "NHWC",
            utils::MS() << "data_format must be one of NCHW, NHWC (was " << data_format << ")");
}

memory::AM operator_to_output_am(OPERATOR_T operator_t) {
    if (operator_t == OPERATOR_T_EQL) {
        return memory::AM_OVERWRITE;
    } else {
        return memory::AM_MUTABLE;
    }
}

std::vector<int> fake_padding_shape(int window_h, int window_w,
                                    const std::string& data_format) {
    if (data_format == "NHWC") {
        return std::vector<int>{1, window_h, window_w, 1};
    } else if (data_format == "NCHW") {
        return std::vector<int>{1, 1, window_h, window_w};
    } else {
        ASSERT2(false, "unknown data format");
        return std::vector<int>{};
    }
}

struct Conv2dFunctionInputInfo {
    int batch_size;
    int in_channels;
    int in_h;
    int in_w;
    int filter_in_channels;
    int filter_h;
    int filter_w;
    int out_channels;
    int out_w;
    int out_h;
};

static Conv2dFunctionInputInfo compute_conv_info(const std::vector<int>& input_shape,
                                                 const std::vector<int>& filters_shape,
                                                 const int& stride_h,
                                                 const int& stride_w,
                                                 PADDING_T padding,
                                                 const std::string& data_format) {
    Conv2dFunctionInputInfo info;

    if (data_format == "NCHW") {
        info.batch_size         = input_shape[0];
        info.in_channels        = input_shape[1];
        info.in_h               = input_shape[2];
        info.in_w               = input_shape[3];

        info.out_channels       = filters_shape[0];
        info.filter_in_channels = filters_shape[1];
        info.filter_h           = filters_shape[2];
        info.filter_w           = filters_shape[3];

    } else if (data_format == "NHWC") {
        info.batch_size         = input_shape[0];
        info.in_h               = input_shape[1];
        info.in_w               = input_shape[2];
        info.in_channels        = input_shape[3];

        info.out_channels       = filters_shape[0];
        info.filter_h           = filters_shape[1];
        info.filter_w           = filters_shape[2];
        info.filter_in_channels = filters_shape[3];
    }
    if (padding == PADDING_T_SAME) {
        info.out_h = int_ceil(info.in_h, stride_h);
        info.out_w = int_ceil(info.in_w, stride_w);
    } else if (padding == PADDING_T_VALID) {
        info.out_h = int_ceil(info.in_h - info.filter_h + 1, stride_h);
        info.out_w = int_ceil(info.in_w - info.filter_w + 1, stride_w);
    } else {
        ASSERT2(false, utils::MS() << "Unrecognized value of padding passed to Conv2dFunction (" << padding << ")");
    }
    return info;
}

///////////////////////////////////////////////////////////////////////////////
//                            Conv2dFunction                                 //
///////////////////////////////////////////////////////////////////////////////

struct Conv2dFunction : public Function<Conv2dFunction,
                                        Array,
                                        Array,
                                        Array,
                                        int,
                                        int,
                                        PADDING_T,
                                        std::string> {

    static std::vector<int> deduce_output_bshape(const Array& input,
                                                 const Array& filters,
                                                 int stride_h,
                                                 int stride_w,
                                                 PADDING_T padding,
                                                 const std::string& data_format) {

        auto info = compute_conv_info(input.shape(),
                                      filters.shape(),
                                      stride_h,
                                      stride_w,
                                      padding,
                                      data_format);

        ASSERT2_SHAPE_ND(input.shape(),   4, "Conv2dFunction input");
        ASSERT2_SHAPE_ND(filters.shape(), 4, "Conv2dFunction filters");
        check_data_format(data_format);

        ASSERT2_EQ(info.in_channels, info.filter_in_channels,
            "Conv2dFunction input and filters need to have the same number of input channels"
        );

        if (data_format == "NCHW") {
            return std::vector<int> {info.batch_size, info.out_channels, info.out_h, info.out_w};
        } else { // then data_format == "NHWC":
            return std::vector<int> {info.batch_size, info.out_h, info.out_w, info.out_channels};
        }
    }


    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_ENABLE_IF_MUL_DIV>
    void typed_eval(TypedArray<devT, T> out,
                    TypedArray<devT, T> input,
                    TypedArray<devT, T> filters,
                    int stride_h,
                    int stride_w,
                    PADDING_T padding,
                    const std::string& data_format) {
        ASSERT2(!(var_operator_t == OPERATOR_T_MUL) && !(var_operator_t == OPERATOR_T_MUL),
                "Convolution's result cannot be inplace-multiplied or inplace-divided.");
        ASSERT2(false, "If asserts above are complete this message should never be displayed");
    }

    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void typed_eval(TypedArray<devT, T> out,
                    TypedArray<devT, T> input,
                    TypedArray<devT, T> filters,
                    int stride_h,
                    int stride_w,
                    PADDING_T padding,
                    const std::string& data_format) {
#ifdef DALI_USE_CUDNN
        if (use_cudnn && devT == memory::DEVICE_T_GPU && template_to_dtype<T>() != DTYPE_INT32) {
            cudnn_conv<operator_t,T,devT>(out, input, filters, stride_h, stride_w, padding, data_format);
            return;
        }
#endif
        blas_conv<operator_t,T,devT>(out, input, filters, stride_h, stride_w, padding, data_format);
    }


    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void blas_conv(TypedArray<devT, T> out,
                   TypedArray<devT, T> input,
                   TypedArray<devT, T> filters,
                   int stride_h,
                   int stride_w,
                   PADDING_T padding,
                   const std::string& data_format) {
        auto info = compute_conv_info(input.array.shape(),
                                      filters.array.shape(),
                                      stride_h,
                                      stride_w,
                                      padding,
                                      data_format);
        filters.array   = filters.array.copyless_reshape({filters.array.shape()[0], -1});

        std::vector<int> temp_bshape;
        check_data_format(data_format);
        if (data_format == "NCHW") {
            temp_bshape = deduce_im2col_shape<mshadow::expr::DATA_FORMAT_NCHW>(
                input.array.shape(),
                info.filter_h, info.filter_w,
                stride_h, stride_w,
                /*dilate_h=*/1,
                /*dilate_w=*/1);
        } else {
            // when data_format is equal to the string containing
            // letters NHWC.
            temp_bshape = deduce_im2col_shape<mshadow::expr::DATA_FORMAT_NHWC>(
                input.array.shape(),
                info.filter_h, info.filter_w,
                stride_h, stride_w,
                /*dilate_h=*/1,
                /*dilate_w=*/1);
        }

        Array im2col_storage_arr(temp_bshape, template_to_dtype<T>(), out.device);
        TypedArray<devT, T> im2col_storage(
                im2col_storage_arr, input.device, temp_bshape);

        if (data_format == "NCHW") {
            im2col_storage.contiguous_d2(memory::AM_OVERWRITE) =
                    mshadow::expr::unpack_patch2col<mshadow::expr::DATA_FORMAT_NCHW>(
                input.d4(),
                info.filter_h,
                info.filter_w,
                stride_h,
                stride_w,
                /*dilate_h=*/1,
                /*dilate_w=*/1
            );
        } else { // then data_format = "NHWC"
            im2col_storage.contiguous_d2(memory::AM_OVERWRITE) =
                    mshadow::expr::unpack_patch2col<mshadow::expr::DATA_FORMAT_NHWC>(
                input.d4(),
                info.filter_h,
                info.filter_w,
                stride_h,
                stride_w,
                /*dilate_h=*/1,
                /*dilate_w=*/1
            );
        }

        typedef decltype(im2col_storage.contiguous_d2()) mshadow_tensor_t;

        if (data_format == "NCHW") {
            // do nothing
        } else {
            im2col_storage.array = im2col_storage.array.transpose();
            filters.array        = filters.array.transpose();
        }


        bool             im2col_transposed, filters_transposed;
        mshadow_tensor_t im2col_tensor,     filters_tensor;
        std::tie(im2col_transposed,   im2col_tensor)  = im2col_storage.blas_friendly_tensor();
        std::tie(filters_transposed, filters_tensor) = filters.blas_friendly_tensor();

        if (data_format == "NCHW") {
            auto out_cnhw_shape = out.array.shape();
            std::swap(out_cnhw_shape[0], out_cnhw_shape[1]);
            Array out_cnhw_arr(out_cnhw_shape, template_to_dtype<T>(), out.device);
            TypedArray<devT, T> out_cnhw(out_cnhw_arr, input.device, out_cnhw_shape);

            operator_assign_contiguous<OPERATOR_T_EQL, 2>(
                out_cnhw,
                dali_gemm(
                    filters_tensor,
                    im2col_tensor,
                    filters_transposed,
                    im2col_transposed,
                    (T)1.0f
                ),
                /*collapse_leading=*/false
            );


            operator_assign_contiguous<operator_t, 4>(
                out,
                mshadow::expr::swapaxis<1,0>(out_cnhw.contiguous_d4())
            );
        } else {
            auto out_2d_arr = out.array.copyless_reshape({-1, out.array.shape()[3]});
            TypedArray<devT, T> out_2d(out_2d_arr, out.device, out_2d_arr.shape());

            operator_assign_contiguous<operator_t, 2>(
                out_2d,
                dali_gemm(
                    im2col_tensor,
                    filters_tensor,
                    im2col_transposed,
                    filters_transposed,
                    (T)1.0f
                )
            );
        }
    }

#ifdef DALI_USE_CUDNN
    template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
    void cudnn_conv(TypedArray<devT, T> out,
                    TypedArray<devT, T> input,
                    TypedArray<devT, T> filters,
                    int stride_h,
                    int stride_w,
                    PADDING_T padding,
                    const std::string& data_format) {
        int padding_h, padding_w;
        std::tie(padding_h, padding_w) = convolution_padding(input.array.shape(),
                                                             filters.array.shape(),
                                                             out.array.shape(),
                                                             stride_h,
                                                             stride_w,
                                                             data_format,
                                                             padding);

        auto out_access_mode = operator_to_output_am(operator_t);

        cudnn::conv2d(
                std::make_shared<cudnn::wrapper::Tensor>(out, data_format, out_access_mode),
                std::make_shared<cudnn::wrapper::Tensor>(input, data_format),
                std::make_shared<cudnn::wrapper::Filters>(filters, data_format),
                std::make_shared<cudnn::wrapper::Convolution>(padding_h, padding_w, stride_h, stride_w),
                cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
        );
    }

#endif

};



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
        ASSERT2(false, "integer convolution is not implemented for GPU.");
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

        int padding_h, padding_w;
        std::tie(padding_h, padding_w) = convolution_padding(in_dw.array.shape(),
                                                             filters.array.shape(),
                                                             out_dw.array.shape(),
                                                             stride_h,
                                                             stride_w,
                                                             data_format,
                                                             padding);

        auto out_access_mode = operator_to_output_am(operator_t);

        cudnn::conv2d_bwd_input(
                std::make_shared<cudnn::wrapper::Tensor>(in_dw, data_format, out_access_mode),
                std::make_shared<cudnn::wrapper::Filters>(filters, data_format),
                std::make_shared<cudnn::wrapper::Tensor>(out_dw, data_format),
                std::make_shared<cudnn::wrapper::Convolution>(padding_h, padding_w, stride_h, stride_w),
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


    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> filters_dw,
                    TypedArray<memory::DEVICE_T_CPU, T> input,
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
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> filters_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> input,
                    TypedArray<memory::DEVICE_T_GPU, T> out_dw,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    PADDING_T padding,
                    const std::string& data_format) {
        ASSERT2(false, "integer convolution is not implemented for GPU.");
    }

    template<OPERATOR_T operator_t, typename T, DALI_FUNC_DISABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> filters_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> input,
                    TypedArray<memory::DEVICE_T_GPU, T> out_dw,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    PADDING_T padding,
                    const std::string& data_format) {

        int padding_h, padding_w;
        std::tie(padding_h, padding_w) = convolution_padding(input.array.shape(),
                                                             filters_dw.array.shape(),
                                                             out_dw.array.shape(),
                                                             stride_h,
                                                             stride_w,
                                                             data_format,
                                                             padding);

        auto out_access_mode = operator_to_output_am(operator_t);

        cudnn::conv2d_bwd_filters(
                std::make_shared<cudnn::wrapper::Filters>(filters_dw, data_format, out_access_mode),
                std::make_shared<cudnn::wrapper::Tensor>(input, data_format),
                std::make_shared<cudnn::wrapper::Tensor>(out_dw, data_format),
                std::make_shared<cudnn::wrapper::Convolution>(padding_h, padding_w, stride_h, stride_w),
                cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
        );
    }

#endif
};

///////////////////////////////////////////////////////////////////////////////
//                    Conv2dBwdBiasFunction                                  //
///////////////////////////////////////////////////////////////////////////////

struct Conv2dBwdBiasFunction : public Function<Conv2dBwdBiasFunction,
                                        Array,
                                        Array,
                                        std::string> {
    static std::vector<int> deduce_output_bshape(const Array& out_dw,
                                                 const std::string& data_format) {
        size_t channel_dim = data_format.find('C');
        ASSERT2(channel_dim != std::string::npos,
                utils::MS() << "data_format must be NCHW or NHWC (got " << data_format << ").");

        return {out_dw.shape()[channel_dim]};
    }


    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> bias_dw,
                    TypedArray<memory::DEVICE_T_CPU, T> out_dw,
                    const std::string& data_format) {
        throw std::runtime_error("not implemented!");
    }

#ifdef DALI_USE_CUDA
    template<OPERATOR_T operator_t, typename T, DALI_FUNC_ENABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> bias_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> out_dw,
                    const std::string& data_format) {
        ASSERT2(false, "integer Conv2dBwdBias is not implemented for GPU.");
    }

    template<OPERATOR_T operator_t, typename T, DALI_FUNC_DISABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> bias_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> out_dw,
                    const std::string& data_format) {

        auto out_access_mode = operator_to_output_am(operator_t);

        cudnn::conv2d_bwd_bias(
            std::make_shared<cudnn::wrapper::Tensor>(bias_dw, data_format, out_access_mode),
            std::make_shared<cudnn::wrapper::Tensor>(out_dw,  data_format),
            cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
        );
    }

#endif
};




///////////////////////////////////////////////////////////////////////////////
//                        Pool2dFunction                                     //
///////////////////////////////////////////////////////////////////////////////

struct Pool2dFunction : public Function<Pool2dFunction,
                                        Array,
                                        Array,
                                        int,
                                        int,
                                        int,
                                        int,
                                        POOLING_T,
                                        PADDING_T,
                                        std::string> {
    static std::vector<int> deduce_output_bshape(
                const Array& input,
                int window_h,
                int window_w,
                int stride_h,
                int stride_w,
                POOLING_T pooling_mode,
                PADDING_T padding,
                const std::string& data_format) {

        ASSERT2_SHAPE_ND(input.shape(),   4, "Pool2dFunction input");
        check_data_format(data_format);

        int out_w, out_h;

        int batch_size, in_channels, in_h, in_w;

        if (data_format == "NCHW") {
            batch_size         = input.shape()[0];
            in_channels        = input.shape()[1];
            in_h               = input.shape()[2];
            in_w               = input.shape()[3];
        } else if (data_format == "NHWC") {
            batch_size         = input.shape()[0];
            in_h               = input.shape()[1];
            in_w               = input.shape()[2];
            in_channels        = input.shape()[3];
        }

        if (padding == PADDING_T_SAME) {
            out_h = int_ceil(in_h, stride_h);
            out_w = int_ceil(in_w, stride_w);
        } else if (padding == PADDING_T_VALID) {
            out_h = int_ceil(in_h - window_h + 1, stride_h);
            out_w = int_ceil(in_w - window_w + 1, stride_w);
        } else {
            ASSERT2(false, utils::MS() << "Unrecognized value of padding passed to Pool2dFunction (" << padding << ")");
        }

        if (data_format == "NCHW") {
            return std::vector<int> {batch_size, in_channels, out_h, out_w};
        } else {
            return std::vector<int> {batch_size, out_h, out_w, in_channels};
        }
    }

    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> out,
                    TypedArray<memory::DEVICE_T_CPU, T> input,
                    int window_h,
                    int window_w,
                    int stride_h,
                    int stride_w,
                    POOLING_T pooling_mode,
                    PADDING_T padding,
                    const std::string& data_format) {
        throw std::runtime_error("not implemented!");
    }

#ifdef DALI_USE_CUDA
    template<OPERATOR_T operator_t, typename T, DALI_FUNC_ENABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> out,
                    TypedArray<memory::DEVICE_T_GPU, T> input,
                    int window_h,
                    int window_w,
                    int stride_h,
                    int stride_w,
                    POOLING_T pooling_mode,
                    PADDING_T padding,
                    const std::string& data_format) {
        ASSERT2(false, "integer convolution is not implemented for GPU.");
    }

    template<OPERATOR_T operator_t, typename T, DALI_FUNC_DISABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> out,
                    TypedArray<memory::DEVICE_T_GPU, T> input,
                    int window_h,
                    int window_w,
                    int stride_h,
                    int stride_w,
                    POOLING_T pooling_mode,
                    PADDING_T padding,
                    const std::string& data_format) {

        int padding_h, padding_w;
        std::tie(padding_h, padding_w) = convolution_padding(
                input.array.shape(),
                fake_padding_shape(window_h, window_w, data_format),
                out.array.shape(),
                stride_h,
                stride_w,
                data_format,
                padding);

        auto out_access_mode = operator_to_output_am(operator_t);

        cudnn::pool2d(
                std::make_shared<cudnn::wrapper::Tensor>(out, data_format, out_access_mode),
                std::make_shared<cudnn::wrapper::Tensor>(input, data_format),
                std::make_shared<cudnn::wrapper::Pooling>(window_h, window_w,
                                                          padding_h, padding_w,
                                                          stride_w, stride_h,
                                                          pooling_mode),
                cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
        );
    }

#endif

};



///////////////////////////////////////////////////////////////////////////////
//                        Pool2dFunction                                     //
///////////////////////////////////////////////////////////////////////////////

struct Pool2dBwdFunction : public Function<Pool2dBwdFunction,
                                           Array,
                                           Array,
                                           Array,
                                           Array,
                                           int,
                                           int,
                                           int,
                                           int,
                                           std::vector<int>,
                                           POOLING_T,
                                           PADDING_T,
                                           std::string> {
    static std::vector<int> deduce_output_bshape(
                const Array& out,
                const Array& out_dw,
                const Array& in,
                int window_h,
                int window_w,
                int stride_h,
                int stride_w,
                const std::vector<int>& result_shape,
                POOLING_T pooling_mode,
                PADDING_T padding,
                const std::string& data_format) {
        // TODO(szymon): perform validation
        return result_shape;
    }

    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> in_dw,
                    TypedArray<memory::DEVICE_T_CPU, T> out,
                    TypedArray<memory::DEVICE_T_CPU, T> out_dw,
                    TypedArray<memory::DEVICE_T_CPU, T> in,
                    int window_h,
                    int window_w,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    POOLING_T pooling_mode,
                    PADDING_T padding,
                    const std::string& data_format) {
        throw std::runtime_error("not implemented!");
    }

#ifdef DALI_USE_CUDA
    template<OPERATOR_T operator_t, typename T, DALI_FUNC_ENABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> in_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> out,
                    TypedArray<memory::DEVICE_T_GPU, T> out_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> in,
                    int window_h,
                    int window_w,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    POOLING_T pooling_mode,
                    PADDING_T padding,
                    const std::string& data_format) {
        ASSERT2(false, "integer Pool2dBwd is not implemented for GPU.");
    }

    template<OPERATOR_T operator_t, typename T, DALI_FUNC_DISABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> in_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> out,
                    TypedArray<memory::DEVICE_T_GPU, T> out_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> in,
                    int window_h,
                    int window_w,
                    int stride_h,
                    int stride_w,
                    const std::vector<int>& result_shape,
                    POOLING_T pooling_mode,
                    PADDING_T padding,
                    const std::string& data_format) {
        int padding_h, padding_w;
        std::tie(padding_h, padding_w) = convolution_padding(
                in.array.shape(),
                fake_padding_shape(window_h, window_w, data_format),
                out.array.shape(),
                stride_h,
                stride_w,
                data_format,
                padding);

        auto out_access_mode = operator_to_output_am(operator_t);

        cudnn::pool2d_bwd(
                std::make_shared<cudnn::wrapper::Tensor>(in_dw, data_format, out_access_mode),
                std::make_shared<cudnn::wrapper::Tensor>(out, data_format),
                std::make_shared<cudnn::wrapper::Tensor>(out_dw, data_format),
                std::make_shared<cudnn::wrapper::Tensor>(in, data_format),
                std::make_shared<cudnn::wrapper::Pooling>(window_h, window_w,
                                                          padding_h, padding_w,
                                                          stride_w, stride_h,
                                                          pooling_mode),
                cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
        );
    }

#endif

};

namespace op {
    Assignable<Array> conv2d(const Array& input,
                             const Array& filters,
                             int stride_h,
                             int stride_w,
                             PADDING_T padding,
                             const std::string& data_format) {
        return Conv2dFunction::run(input,
                                   filters,
                                   stride_h,
                                   stride_w,
                                   padding,
                                   data_format);
    }

    Assignable<Array> im2col(const Array& input,
                             int filter_h,
                             int filter_w,
                             int stride_h,
                             int stride_w,
                             const std::string& data_format) {
        check_data_format(data_format);
        if (data_format == "NCHW") {
            return lazy::im2col_nchw(input, filter_h, filter_w, stride_h, stride_w, 1, 1);
        } else {
            return lazy::im2col_nhwc(input, filter_h, filter_w, stride_h, stride_w, 1, 1);
        }
    }

    Assignable<Array> col2im(const Array& input,
                             const std::vector<int>& image_shape,
                             int filter_h,
                             int filter_w,
                             int stride_h,
                             int stride_w,
                             const std::string& data_format) {
        check_data_format(data_format);
        if (data_format == "NCHW") {
            return lazy::col2im_nchw(input, image_shape, filter_h, filter_w, stride_h, stride_w, 1, 1);
        } else {
            return lazy::col2im_nhwc(input, image_shape, filter_h, filter_w, stride_h, stride_w, 1, 1);
        }
    }

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

    Assignable<Array> conv2d_backward_bias(
                     const Array& out_dw,
                     const std::string& data_format) {
        return Conv2dBwdBiasFunction::run(out_dw,data_format);
    }

    Assignable<Array> pool2d(const Array& input,
                             int window_h,
                             int window_w,
                             int stride_h,
                             int stride_w,
                             POOLING_T pooling_mode,
                             PADDING_T padding,
                             const std::string& data_format) {
        return Pool2dFunction::run(input,
                                   window_h,
                                   window_w,
                                   stride_h,
                                   stride_w,
                                   pooling_mode,
                                   padding,
                                   data_format);
    }

    Assignable<Array> pool2d_backward(
                             const Array& out,
                             const Array& out_dw,
                             const Array& in,
                             int window_h,
                             int window_w,
                             int stride_h,
                             int stride_w,
                             const std::vector<int>& result_shape,
                             POOLING_T pooling_mode,
                             PADDING_T padding,
                             const std::string& data_format) {
        return Pool2dBwdFunction::run(out,
                                      out_dw,
                                      in,
                                      window_h,
                                      window_w,
                                      stride_h,
                                      stride_w,
                                      result_shape,
                                      pooling_mode,
                                      padding,
                                      data_format);

    }
};  // namespace op
