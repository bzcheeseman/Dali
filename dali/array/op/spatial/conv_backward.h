#ifndef DALI_ARRAY_OP_SPATIAL_CONV_BACKWARD_H
#define DALI_ARRAY_OP_SPATIAL_CONV_BACKWARD_H

#include "dali/array/op/spatial/spatial_enums.h"
#include <vector>
#include <string>

class Array;
template<typename OutType>
class Assignable;

namespace op {
    // returns gradient with respect to the input.
    Assignable<Array> conv2d_backward_input(
        const Array& filters,
        const Array& out_dw,
        int stride_h,
        int stride_w,
        const std::vector<int>& result_shape,
        PADDING_T padding,
        const std::string& data_format
    );

    Assignable<Array> conv2d_backward_filters(
        const Array& input,
        const Array& out_dw,
        int stride_h,
        int stride_w,
        const std::vector<int>& result_shape,
        PADDING_T padding,
        const std::string& data_format
    );
}  // namespace op

#endif  // DALI_ARRAY_OP_SPATIAL_CONV_BACKWARD_H
