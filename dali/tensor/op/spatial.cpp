#include "binary.h"

#include "dali/array/op.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    Tensor conv2d(Tensor input,
                  Tensor filters,
                  int stride_h,
                  int stride_w,
                  PADDING_T padding,
                  const std::string& data_format) {
        Tensor out(op::conv2d(input.w,
                              filters.w,
                              stride_h,
                              stride_w,
                              padding,
                              data_format));

        if (graph::backprop_enabled() && (!input.constant || !filters.constant))
            graph::emplace_back([out, input, filters,
                                 stride_h,stride_w,
                                 padding,data_format]() mutable {
                MAYBE_GRAD(input) +=
                    op::conv2d_backward_input(filters.w,
                                              out.dw,
                                              stride_h,
                                              stride_w,
                                              input.shape(),
                                              padding,
                                              data_format);
                MAYBE_GRAD(filters) +=
                    op::conv2d_backward_filters(input.w,
                                                out.dw,
                                                stride_h,
                                                stride_w,
                                                filters.shape(),
                                                padding,
                                                data_format);
            });
        return out;
    }

    Tensor im2col(Tensor input,
                  int filter_h,
                  int filter_w,
                  int stride_h,
                  int stride_w,
                  const std::string& data_format) {
        Tensor out(
            op::im2col(
                input.w,
                filter_h,
                filter_w,
                stride_h,
                stride_w,
                data_format
            )
        );

        if (graph::backprop_enabled() && !input.constant) {
            auto out_dw = out.dw;
            auto input_dw = input.dw;
            graph::emplace_back([out_dw, input_dw, data_format,
                                 filter_h, filter_w, stride_h, stride_w]() mutable {
                input_dw <<= op::col2im(
                    out_dw,
                    input_dw.shape(),
                    filter_h,
                    filter_w,
                    stride_h,
                    stride_w,
                    data_format
                );
            });
        }
        return out;
    }

    Tensor col2im(Tensor input,
                  const std::vector<int>& image_shape,
                  int filter_h,
                  int filter_w,
                  int stride_h,
                  int stride_w,
                  const std::string& data_format) {
        Tensor out(
            op::col2im(
                input.w,
                image_shape,
                filter_h,
                filter_w,
                stride_h,
                stride_w,
                data_format
            )
        );

        if (graph::backprop_enabled() && !input.constant) {
            auto out_dw = out.dw;
            auto input_dw = input.dw;
            graph::emplace_back([out_dw, input_dw, data_format,
                                 filter_h, filter_w, stride_h, stride_w]() mutable {
                input_dw <<= op::im2col(
                    out_dw,
                    filter_h,
                    filter_w,
                    stride_h,
                    stride_w,
                    data_format
                );
            });
        }
        return out;
    }

    Tensor conv2d_add_bias(Tensor conv_out,
                           Tensor bias,
                           const std::string& data_format) {
        Array broadcasted_bias;
        if (data_format == "NCHW") {
            broadcasted_bias = bias.w[Broadcast()][Slice(0, bias.shape()[0])][Broadcast()][Broadcast()];
        } else if (data_format == "NHWC") {
            broadcasted_bias = bias.w[Broadcast()][Broadcast()][Broadcast()];
        } else {
            ASSERT2(false, utils::MS() << "data_format must be NHWC or NCHW (got " << data_format << ").");
        }

        Tensor out(conv_out.w + broadcasted_bias);

        if (graph::backprop_enabled())
            graph::emplace_back([conv_out, bias, out, data_format]() mutable {
                MAYBE_GRAD(conv_out) += out.dw;

                MAYBE_GRAD(bias) += op::conv2d_backward_bias(out.dw, data_format);
            });
        return out;
    }

    Tensor pool2d(Tensor input,
                  int window_h,
                  int window_w,
                  int stride_h,
                  int stride_w,
                  POOLING_T pooling_mode,
                  PADDING_T padding,
                  const std::string& data_format) {
        Tensor out(op::pool2d(input.w,
                              window_h,
                              window_w,
                              stride_h,
                              stride_w,
                              pooling_mode,
                              padding,
                              data_format));
        if (graph::backprop_enabled() && !input.constant)
            graph::emplace_back([out, input,
                                 window_h, window_w,
                                 stride_h, stride_w,
                                 pooling_mode,
                                 padding,
                                 data_format]() mutable {
                MAYBE_GRAD(input) +=
                    op::pool2d_backward(out.w,
                                        out.dw,
                                        input.w,
                                        window_h,
                                        window_w,
                                        stride_h,
                                        stride_w,
                                        pooling_mode,
                                        padding,
                                        data_format);
            });
        return out;
    }


    Tensor max_pool(Tensor input,
                    int window_h,
                    int window_w,
                    int stride_h=-1,
                    int stride_w=-1,
                    PADDING_T padding=PADDING_T_VALID,
                    const std::string& data_format="NCHW") {
        if (stride_h == -1) stride_h = window_h;
        if (stride_w == -1) stride_w = window_w;

        return pool2d(input, window_h, window_w,
                             stride_h, stride_w,
                             POOLING_T_MAX,
                             padding, data_format);
    }

    Tensor avg_pool(Tensor input,
                    int window_h,
                    int window_w,
                    int stride_h=-1,
                    int stride_w=-1,
                    PADDING_T padding=PADDING_T_VALID,
                    const std::string& data_format="NCHW") {
        if (stride_h == -1) stride_h = window_h;
        if (stride_w == -1) stride_w = window_w;

        return pool2d(input, window_h, window_w,
                             stride_h, stride_w,
                             POOLING_T_AVG,
                             padding, data_format);
    }

}  // namespace tensor_ops
