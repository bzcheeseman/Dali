#ifndef DALI_ARRAY_OP2_COL2IM_H
#define DALI_ARRAY_OP2_COL2IM_H

#include <string>
#include <vector>

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
    expression::ExpressionGraph col2im(
        const expression::ExpressionGraph& input,
        const std::vector<int>& image_shape,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        const std::string& data_format
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_COL2IM_H
