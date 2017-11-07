#ifndef DALI_ARRAY_OP2_CONV_H
#define DALI_ARRAY_OP2_CONV_H

#include <string>

#include "dali/array/op/spatial/spatial_enums.h"

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
    expression::ExpressionGraph conv2d(
        const expression::ExpressionGraph& input,
        const expression::ExpressionGraph& filters,
        int stride_h,
        int stride_w,
        PADDING_T padding,
        const std::string& data_format
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_CONV_H
