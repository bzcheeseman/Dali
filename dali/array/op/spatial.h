#ifndef DALI_ARRAY_OP_SPATIAL_H
#define DALI_ARRAY_OP_SPATIAL_H

#include <string>
#include <vector>

class Array;
template<typename OutType>
class Assignable;


namespace op {
    // the type of padding algorithm to use.
    enum PADDING_T {
        PADDING_T_SAME  = 0,
        PADDING_T_VALID = 1
    };

    Assignable<Array> conv2d(const Array& input,
                             const Array& filters,
                             int stride_h,
                             int stride_w,
                             PADDING_T padding,
                             const std::string& data_format);

    // returns gradient with respect to the input.
    Assignable<Array> conv2d_backward_input(
                         const Array& filters,
                         const Array& out_dw,
                         int stride_h,
                         int stride_w,
                         const std::vector<int>& result_shape,
                         PADDING_T padding,
                         const std::string& data_format);

    Assignable<Array> conv2d_backward_filters(
                     const Array& input,
                     const Array& out_dw,
                     int stride_h,
                     int stride_w,
                     const std::vector<int>& result_shape,
                     PADDING_T padding,
                     const std::string& data_format);
}

#endif  // DALI_ARRAY_OP_SPATIAL_H
