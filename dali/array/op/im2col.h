#ifndef DALI_ARRAY_OP_IM2COL_H
#define DALI_ARRAY_OP_IM2COL_H

#include "dali/array/array.h"

namespace op {
    std::vector<int> im2col_shape(
            const std::vector<int>& src_shape,
            const int& filter_h,
            const int& filter_w,
            const int& stride_h,
            const int& stride_w,
            const int& dilate_h,
            const int& dilate_w,
            const int& prepad_h,
            const int& prepad_w,
            const int& postpad_h,
            const int& postpad_w,
            const std::string& data_format);
    Array im2col(
        const Array& input,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        int padding_h,
        int padding_w,
        int postpad_h,
        int postpad_w,
        const std::string& data_format
    );
}  // namespace op

#endif  // DALI_ARRAY_OP_IM2COL_H
