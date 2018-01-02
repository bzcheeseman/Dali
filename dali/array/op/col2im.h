#ifndef DALI_ARRAY_OP_COL2IM_H
#define DALI_ARRAY_OP_COL2IM_H

#include "dali/array/array.h"

namespace op {
Array col2im(
    const Array& input,
    const std::vector<int>& image_shape,
    int filter_h,
    int filter_w,
    int stride_h,
    int stride_w,
    const std::string& data_format
);
}  // namespace op

#endif  // DALI_ARRAY_OP_COL2IM_H
