#ifndef DALI_ARRAY_OP2_IM2COL_H
#define DALI_ARRAY_OP2_IM2COL_H

#include <string>

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
    expression::ExpressionGraph im2col(
        const expression::ExpressionGraph& input,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        int padding_h,
        int padding_w,
        const std::string& data_format
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_IM2COL_H
