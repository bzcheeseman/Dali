#include "conv.h"
#include "dali/array/op/im2col.h"
#include "dali/array/op/dot.h"
#include "dali/array/op/spatial_utils.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"

namespace op {
    Array conv2d(const Array& input,
                 const Array& filters,
                 int stride_h,
                 int stride_w,
                 PADDING_T padding,
                 const std::string& data_format) {
        ASSERT2(input.ndim() == 3 || input.ndim() == 4, utils::make_message(
            "Input argument to conv2d must be 3D or 4D (got input.ndim=",
            input.ndim(), ")."));
        ASSERT2(filters.ndim() == 4, utils::make_message(
            "Filters argument to conv2d must be 4D (got filters.ndim=",
            filters.ndim(), ")."));
        int n_dim, c_dim, h_dim, w_dim;
        check_data_format(data_format, &n_dim, &c_dim, &h_dim, &w_dim);

        auto input_shape = input.shape();
        if (input_shape.size() == 3) {
            // make the shape appear to have N-dimension be 1
            // if input is 3D:
            input_shape.insert(input_shape.begin() + n_dim, 1);
        }
        auto info = compute_conv2d_info(
            input_shape, filters.shape(),
            stride_h,
            stride_w,
            padding,
            data_format);

        auto filters_nxxx = filters.swapaxes(n_dim, 0);
        auto filters_nX = filters_nxxx.reshape({filters_nxxx.shape()[0], -1});

        // format is c * filt_h * filt_w x batch x out_h x out_w
        auto im2col_image = op::im2col(
            input,
            info.filter_h,
            info.filter_w,
            stride_h,
            stride_w,
            /*padding_h=*/info.padding_h,
            /*padding_w=*/info.padding_w,
            /*postpad_h=*/info.padding_h + info.odd_padding_h,
            /*postpad_w=*/info.padding_w + info.odd_padding_w,
            data_format);

        if (data_format == "NCHW") {
            return op::dot(filters_nX, im2col_image).transpose().reshape(
                {info.batch_size, info.out_channels, info.out_h, info.out_w});
        } else if (data_format == "NHWC") {
            return op::dot(im2col_image.transpose(),
                           filters_nX.transpose()).reshape(
                {info.batch_size, info.out_h, info.out_w, info.out_channels});
        } else {
            ASSERT2(false, utils::make_message(
                "conv2d only supports data_format NCHW and NHWC "
                "(got data_format = ", data_format, ")."));
        }
    }
}  // namespace op
