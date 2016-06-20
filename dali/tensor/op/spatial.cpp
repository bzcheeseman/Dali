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

        if (graph::backprop_enabled())
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
        if (graph::backprop_enabled())
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
                                        input.shape(),
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
