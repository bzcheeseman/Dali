#ifndef DALI_ARRAY_OP2_IM2COL_H
#define DALI_ARRAY_OP2_IM2COL_H

#include <string>

struct Operation;

namespace op {
    Operation im2col(
        const Operation& input,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        const std::string& data_format
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_IM2COL_H
