#ifndef DALI_ARRAY_OP_SPATIAL_H
#define DALI_ARRAY_OP_SPATIAL_H

#include <string>
#include <vector>

#include "dali/array/op/spatial_enums.h"

class Array;
template<typename OutType>
class Assignable;

namespace internal {
    struct Conv2dFunctionInputInfo {
        int batch_size;
        int in_channels;
        int in_h;
        int in_w;
        int filter_in_channels;
        int filter_h;
        int filter_w;
        int out_channels;
        int out_w;
        int out_h;
    };

    Conv2dFunctionInputInfo compute_conv_info(
            const std::vector<int>& input_shape,
            const std::vector<int>& filters_shape,
            const int& stride_h,
            const int& stride_w,
            PADDING_T padding,
            const std::string& data_format);

    std::tuple<int, int> convolution_padding(
        const std::vector<int>& input_shape,
        const std::vector<int>& filters_shape,
        const std::vector<int>& output_shape,
        int stride_h,
        int stride_w,
        const std::string&      data_format,
        PADDING_T           padding);
}

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

    Assignable<Array> conv2d_backward_bias(const Array& out_dw,
                                           const std::string& data_format);

    Assignable<Array> pool2d(const Array& input,
                             int window_h,
                             int window_w,
                             int stride_h,
                             int stride_w,
                             POOLING_T pooling_mode,
                             PADDING_T padding,
                             const std::string& data_format);

    Assignable<Array> pool2d_backward(const Array& out,
                                      const Array& out_dw,
                                      const Array& in,
                                      int window_h,
                                      int window_w,
                                      int stride_h,
                                      int stride_w,
                                      POOLING_T pooling_mode,
                                      PADDING_T padding,
                                      const std::string& data_format);
}  // namespace op

#endif  // DALI_ARRAY_OP_SPATIAL_H
