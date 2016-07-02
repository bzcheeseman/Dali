#ifndef DALI_ARRAY_OP_SPATIAL_CONV_H
#define DALI_ARRAY_OP_SPATIAL_CONV_H

#include "dali/array/op/spatial/spatial_enums.h"
#include <vector>
#include <string>

class Array;
template<typename OutType>
class Assignable;

namespace op {
    // the type of padding algorithm to use.

    Assignable<Array> conv2d(const Array& input,
                             const Array& filters,
                             int stride_h,
                             int stride_w,
                             PADDING_T padding,
                             const std::string& data_format);

    Assignable<Array> im2col(const Array& input,
                             int filter_h,
                             int filter_w,
                             int stride_h,
                             int stride_w,
                             const std::string& data_format);

    Assignable<Array> col2im(const Array& input,
                             const std::vector<int>& image_shape,
                             int filter_h,
                             int filter_w,
                             int stride_h,
                             int stride_w,
                             const std::string& data_format);
}  // namespace op

#endif  // DALI_ARRAY_OP_SPATIAL_CONV_H
