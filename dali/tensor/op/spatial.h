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
}

#endif  // DALI_TENSOR_OP_SPATIAL_H
