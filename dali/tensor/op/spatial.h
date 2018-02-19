#ifndef DALI_TENSOR_OP_SPATIAL_H
#define DALI_TENSOR_OP_SPATIAL_H

#include "dali/tensor/tensor.h"
#include "dali/array/op/spatial_enums.h"

namespace tensor_ops {
    Tensor conv2d(Tensor input,
                  Tensor filters,
                  int stride_h,
                  int stride_w,
                  PADDING_T padding,
                  const std::string& data_format);

    Tensor im2col(Tensor input,
                  int filter_h,
                  int filter_w,
                  int stride_h,
                  int stride_w,
                  const std::string& data_format);

    Tensor col2im(Tensor input,
                  const std::vector<int>& image_shape,
                  int filter_h,
                  int filter_w,
                  int stride_h,
                  int stride_w,
                  const std::string& data_format);

    Tensor conv2d_add_bias(Tensor conv_out,
                           Tensor bias,
                           const std::string& data_format);

    Tensor pool2d(Tensor input,
                  int window_h,
                  int window_w,
                  int stride_h,
                  int stride_w,
                  POOLING_T pooling_mode,
                  PADDING_T padding,
                  const std::string& data_format);

    Tensor max_pool(Tensor input,
                    int window_h,
                    int window_w,
                    int stride_h=-1,
                    int stride_w=-1,
                    PADDING_T padding=PADDING_T_VALID,
                    const std::string& data_format="NCHW");

    Tensor avg_pool(Tensor input,
                    int window_h,
                    int window_w,
                    int stride_h=-1,
                    int stride_w=-1,
                    PADDING_T padding=PADDING_T_VALID,
                    const std::string& data_format="NCHW");
}

#endif  // DALI_TENSOR_OP_SPATIAL_H
