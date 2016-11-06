#ifndef DALI_ARRAY_OP2_CONV_H
#define DALI_ARRAY_OP2_CONV_H

#include <string>

#include "dali/array/op/spatial/spatial_enums.h"

namespace expression {
    struct Expression;
}  // namespace expression

namespace op {
    expression::Expression conv2d(
        const expression::Expression& input,
        const expression::Expression& filters,
        int stride_h,
        int stride_w,
        PADDING_T padding,
        const std::string& data_format
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_CONV_H
