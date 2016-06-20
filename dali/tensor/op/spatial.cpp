#include "binary.h"

#include "dali/array/op.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"



namespace tensor_ops {
    Tensor conv2d(Tensor input,
                  Tensor filters,
                  int stride_h,
                  int stride_w,
                  op::PADDING_T padding,
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
                    conv2d_backward_input(filters.w,
                                          out.dw,
                                          stride_h,
                                          stride_w,
                                          input.shape(),
                                          padding,
                                          data_format);
                MAYBE_GRAD(filters) +=
                    conv2d_backward_filters(input.w,
                                            out.dw,
                                            stride_h,
                                            stride_w,
                                            filters.shape(),
                                            padding,
                                            data_format);
            });
        return out;
    }

}  // namespace tensor_ops
