#ifndef DALI_ARRAY_OP2_COL2IM_H
#define DALI_ARRAY_OP2_COL2IM_H

#include <string>
#include <vector>

namespace expression {
    struct Expression;
}  // namespace expression

namespace op {
    expression::Expression col2im(
        const expression::Expression& input,
        const std::vector<int>& image_shape,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        const std::string& data_format
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_COL2IM_H
