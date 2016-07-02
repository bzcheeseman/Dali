#include "pool.h"
#include "dali/config.h"
#include "dali/runtime_config.h"
#include "dali/array/array.h"
#include "dali/array/function/function.h"
#include <mshadow/extension/spatial_pool.h>
#include <mshadow/extension/spatial_unpool.h>

#include "dali/array/op/spatial/utils.h"

struct Pool2dFunctionInputInfo {
    int out_w;
    int out_h;
    int batch_size;
    int in_channels;
    int in_h;
    int in_w;
};

Pool2dFunctionInputInfo compute_pool_info(
            const std::vector<int>& input_shape,
            const int& window_h,
            const int& window_w,
            const int& stride_h,
            const int& stride_w,
            const PADDING_T& padding,
            const std::string& data_format
        ) {
    internal::check_data_format(data_format);
    Pool2dFunctionInputInfo info;

    if (data_format == "NCHW") {
        info.batch_size         = input_shape[0];
        info.in_channels        = input_shape[1];
        info.in_h               = input_shape[2];
        info.in_w               = input_shape[3];
    } else if (data_format == "NHWC") {
        info.batch_size         = input_shape[0];
        info.in_h               = input_shape[1];
        info.in_w               = input_shape[2];
        info.in_channels        = input_shape[3];
    }
    if (padding == PADDING_T_SAME) {
        info.out_h = internal::int_ceil(info.in_h, stride_h);
        info.out_w = internal::int_ceil(info.in_w, stride_w);
    } else if (padding == PADDING_T_VALID) {
        info.out_h = internal::int_ceil(info.in_h - window_h + 1, stride_h);
        info.out_w = internal::int_ceil(info.in_w - window_w + 1, stride_w);
    } else {
        ASSERT2(false, utils::MS() << "Unrecognized value of padding passed to Pool2dFunction (got " << padding << ").");
    }
    return info;
}

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
        auto info = compute_pool_info(
            input.shape(),
            window_h,
            window_w,
            stride_h,
            stride_w,
            padding,
            data_format
        );

        ASSERT2(pooling_mode == POOLING_T_MAX || pooling_mode == POOLING_T_AVG,
            utils::MS() << "Unrecognized pooling_mode value: pooling_mode must be max or avg (got "
                        << pooling_mode << ")."
        );

        if (data_format == "NCHW") {
            return std::vector<int> {info.batch_size, info.in_channels, info.out_h, info.out_w};
        } else {
            return std::vector<int> {info.batch_size, info.out_h, info.out_w, info.in_channels};
        }
    }

    template<OPERATOR_T operator_t, typename T, int devT>
    void typed_eval(TypedArray<devT, T> out,
                    TypedArray<devT, T> input,
                    int window_h,
                    int window_w,
                    int stride_h,
                    int stride_w,
                    POOLING_T pooling_mode,
                    PADDING_T padding,
                    const std::string& data_format) {
#ifdef DALI_USE_CUDNN
        if (use_cudnn && devT == memory::DEVICE_T_GPU && template_to_dtype<T>() != DTYPE_INT32
            && operator_t != OPERATOR_T_MUL && operator_t != OPERATOR_T_DIV) {
            cudnn_pool<operator_t,T,devT>(out, input, window_h, window_w, stride_h, stride_w, pooling_mode, padding, data_format);
            return;
        }
#endif
        mshadow_pool<operator_t,T,devT>(out, input, window_h, window_w, stride_h, stride_w, pooling_mode, padding, data_format);
    }

    template<OPERATOR_T operator_t, typename T, int devT>
    void mshadow_pool(TypedArray<devT, T> out,
                      TypedArray<devT, T> input,
                      int window_h,
                      int window_w,
                      int stride_h,
                      int stride_w,
                      POOLING_T pooling_mode,
                      PADDING_T padding,
                      const std::string& data_format) {
        if (pooling_mode == POOLING_T_AVG) {
            if (data_format == "NCHW") {
                operator_assign<operator_t, 4>(
                    out,
                    mshadow::expr::pool<mshadow::expr::DATA_FORMAT_NCHW, mshadow::red::sum>(
                        input.d4(), window_h, window_w, stride_h, stride_w
                    ) / ((double)window_h * window_w)
                );
            } else { // then data_format is NHWC
                operator_assign<operator_t, 4>(
                    out,
                    mshadow::expr::pool<mshadow::expr::DATA_FORMAT_NHWC, mshadow::red::sum>(
                        input.d4(), window_h, window_w, stride_h, stride_w
                    ) / ((double)window_h * window_w)
                );
            }
        } else if (pooling_mode == POOLING_T_MAX) {
            if (data_format == "NCHW") {
                operator_assign<operator_t, 4>(
                    out,
                    mshadow::expr::pool<mshadow::expr::DATA_FORMAT_NCHW, mshadow::red::maximum>(
                        input.d4(), window_h, window_w, stride_h, stride_w
                    )
                );
            } else { // then data_format is NHWC
                operator_assign<operator_t, 4>(
                    out,
                    mshadow::expr::pool<mshadow::expr::DATA_FORMAT_NHWC, mshadow::red::maximum>(
                        input.d4(), window_h, window_w, stride_h, stride_w
                    )
                );
            }
        }
    }
#ifdef DALI_USE_CUDNN
    template<OPERATOR_T operator_t, typename T, int devT>
    void cudnn_pool(TypedArray<devT, T> out,
                    TypedArray<devT, T> input,
                    int window_h,
                    int window_w,
                    int stride_h,
                    int stride_w,
                    POOLING_T pooling_mode,
                    PADDING_T padding,
                    const std::string& data_format) {

        auto info = internal::compute_conv_info(input.array.shape(),
                                                fake_padding_shape(window_h, window_w, data_format),
                                                stride_h,
                                                stride_w,
                                                padding,
                                                data_format);

        auto out_access_mode = internal::OperatorAM<operator_t>::get(out);
        cudnn::pool2d(
                std::make_shared<cudnn::wrapper::Tensor>(out, data_format, out_access_mode),
                std::make_shared<cudnn::wrapper::Tensor>(input, data_format),
                std::make_shared<cudnn::wrapper::Pooling>(window_h, window_w,
                                                          info.padding_h, info.padding_w,
                                                          stride_w, stride_h,
                                                          pooling_mode),
                cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
        );
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////
//                        Pool2dBwdFunction                                     //
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
                POOLING_T pooling_mode,
                PADDING_T padding,
                const std::string& data_format) {
        ASSERT2(pooling_mode == POOLING_T_MAX || pooling_mode == POOLING_T_AVG,
            utils::MS() << "Unrecognized pooling_mode value: pooling_mode must be max or avg (got "
                        << pooling_mode << ")."
        );
        ASSERT2_SHAPE_ND(out.shape(),   4, "Pool2dBackward out");
        ASSERT2_SHAPE_ND(out_dw.shape(),   4, "Pool2dBackward out_dw");
        ASSERT2_SHAPE_ND(in.shape(),   4, "Pool2dBackward in");
        auto info = compute_pool_info(
            in.shape(),
            window_h,
            window_w,
            stride_h,
            stride_w,
            padding,
            data_format
        );

        std::vector<int> expected_pooled_shape;

        if (data_format == "NCHW") {
            expected_pooled_shape = {info.batch_size, info.in_channels, info.out_h, info.out_w};
        } else {
            expected_pooled_shape = {info.batch_size, info.out_h, info.out_w, info.in_channels};
        }
        ASSERT2(out_dw.shape() == expected_pooled_shape,
            utils::MS() << "Pool2dBackward argument `out_dw` should have shape "
                        << expected_pooled_shape << " (got " << out_dw.shape() << ").");
        ASSERT2(out.shape() == expected_pooled_shape,
            utils::MS() << "Pool2dBackward argument `out` should have shape "
                        << expected_pooled_shape << " (got " << out.shape() << ").");
        return in.bshape();
    }

    template<OPERATOR_T operator_t, typename T, int devT>
    void typed_eval(TypedArray<devT, T> in_dw,
                    TypedArray<devT, T> out,
                    TypedArray<devT, T> out_dw,
                    TypedArray<devT, T> input,
                    int window_h,
                    int window_w,
                    int stride_h,
                    int stride_w,
                    POOLING_T pooling_mode,
                    PADDING_T padding,
                    const std::string& data_format) {
#ifdef DALI_USE_CUDNN
        if (use_cudnn && devT == memory::DEVICE_T_GPU && template_to_dtype<T>() != DTYPE_INT32
            && operator_t != OPERATOR_T_MUL && operator_t != OPERATOR_T_DIV) {
            cudnn_pool_backward<operator_t,T,devT>(
                in_dw,
                out,
                out_dw,
                input,
                window_h,
                window_w,
                stride_h,
                stride_w,
                pooling_mode,
                padding,
                data_format
            );
            return;
        }
#endif
        mshadow_pool_backward<operator_t,T,devT>(
            in_dw,
            out,
            out_dw,
            input,
            window_h,
            window_w,
            stride_h,
            stride_w,
            pooling_mode,
            padding,
            data_format
        );
    }

    template<OPERATOR_T operator_t, typename T, int devT>
    void mshadow_pool_backward(TypedArray<devT, T> in_dw,
                               TypedArray<devT, T> out,
                               TypedArray<devT, T> out_dw,
                               TypedArray<devT, T> in,
                               int window_h,
                               int window_w,
                               int stride_h,
                               int stride_w,
                               POOLING_T pooling_mode,
                               PADDING_T padding,
                               const std::string& data_format) {

        if (pooling_mode == POOLING_T_AVG) {
            if (data_format == "NCHW") {
                operator_assign<operator_t, 4>(
                    in_dw,
                    mshadow::expr::unpool<mshadow::expr::DATA_FORMAT_NCHW, mshadow::red::sum>(
                        in.d4(),
                        out.d4(),
                        out_dw.d4(),
                        window_h,
                        window_w,
                        stride_h,
                        stride_w
                    ) / ((double)window_h * window_w)
                );
            } else { // then data_format is NHWC
                operator_assign<operator_t, 4>(
                    in_dw,
                    mshadow::expr::unpool<mshadow::expr::DATA_FORMAT_NHWC, mshadow::red::sum>(
                        in.d4(),
                        out.d4(),
                        out_dw.d4(),
                        window_h,
                        window_w,
                        stride_h,
                        stride_w
                    ) / ((double)window_h * window_w)
                );
            }
        } else if (pooling_mode == POOLING_T_MAX) {
            if (data_format == "NCHW") {
                operator_assign<operator_t, 4>(
                    in_dw,
                    mshadow::expr::unpool<mshadow::expr::DATA_FORMAT_NCHW, mshadow::red::maximum>(
                        in.d4(),
                        out.d4(),
                        out_dw.d4(),
                        window_h,
                        window_w,
                        stride_h,
                        stride_w
                    )
                );
            } else { // then data_format is NHWC
                operator_assign<operator_t, 4>(
                    in_dw,
                    mshadow::expr::unpool<mshadow::expr::DATA_FORMAT_NHWC, mshadow::red::maximum>(
                        in.d4(),
                        out.d4(),
                        out_dw.d4(),
                        window_h,
                        window_w,
                        stride_h,
                        stride_w
                    )
                );
            }
        }
    }

#ifdef DALI_USE_CUDNN
    template<OPERATOR_T operator_t, typename T, int devT>
    void cudnn_pool_backward(TypedArray<devT, T> in_dw,
                             TypedArray<devT, T> out,
                             TypedArray<devT, T> out_dw,
                             TypedArray<devT, T> in,
                             int window_h,
                             int window_w,
                             int stride_h,
                             int stride_w,
                             POOLING_T pooling_mode,
                             PADDING_T padding,
                             const std::string& data_format) {
        auto info = internal::compute_conv_info(in.array.shape(),
                                                fake_padding_shape(window_h, window_w, data_format),
                                                stride_h,
                                                stride_w,
                                                padding,
                                                data_format);

        auto out_access_mode = internal::OperatorAM<operator_t>::get(in_dw);

        cudnn::pool2d_bwd(
            std::make_shared<cudnn::wrapper::Tensor>(in_dw, data_format, out_access_mode),
            std::make_shared<cudnn::wrapper::Tensor>(out, data_format),
            std::make_shared<cudnn::wrapper::Tensor>(out_dw, data_format),
            std::make_shared<cudnn::wrapper::Tensor>(in, data_format),
            std::make_shared<cudnn::wrapper::Pooling>(window_h, window_w,
                                                      info.padding_h, info.padding_w,
                                                      stride_w, stride_h,
                                                      pooling_mode),
            cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
        );
    }
#endif

};

namespace op {
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

    Assignable<Array> pool2d_backward(const Array& out,
                                      const Array& out_dw,
                                      const Array& in,
                                      int window_h,
                                      int window_w,
                                      int stride_h,
                                      int stride_w,
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
                                      pooling_mode,
                                      padding,
                                      data_format);

    }
}  // namespace op
