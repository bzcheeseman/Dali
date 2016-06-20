#ifndef DALI_TENSOR_OP_SPATIAL_H
#define DALI_TENSOR_OP_SPATIAL_H

#include "dali/tensor/tensor.h"
#include "dali/array/op/spatial.h"

namespace tensor_ops {
    Tensor conv2d(Tensor input,
                  Tensor filters,
                  int stride_h,
                  int stride_w,
                  PADDING_T padding,
                  const std::string& data_format);

    Tensor pool2d(Tensor input,
                  int window_h,
                  int window_w,
                  int stride_h,
                  int stride_w,
                  POOLING_T pooling_mode,
                  PADDING_T padding,
                  const std::string& data_format);
}

#endif  // DALI_TENSOR_OP_SPATIAL_H
