#ifndef DALI_ARRAY_OP_CONV_H
#define DALI_ARRAY_OP_CONV_H

#include "dali/array/array.h"
#include "dali/array/op/spatial_enums.h"

namespace op {
Array conv2d(const Array& input,
             const Array& filters,
             int stride_h,
             int stride_w,
             PADDING_T padding,
             const std::string& data_format);
}

#endif  // DALI_ARRAY_OP_CONV_H
