#include "conv.h"
#include "dali/array/op/im2col.h"
#include "dali/array/op/col2im.h"
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

    Array conv2d_backward_input(Array filters,
                                Array out_dw,
                                int stride_h,
                                int stride_w,
                                const std::vector<int>& result_shape,
                                PADDING_T padding,
                                const std::string& data_format) {
        ASSERT2(out_dw.ndim() == 4, utils::make_message(
            "conv2d_backward_input's out_dw must be 4 dimensional but got out_dw ",
            out_dw.full_expression_name(), " with ndim = ", out_dw.ndim(), "."));
        ASSERT2(filters.ndim() == 4, utils::make_message(
            "conv2d_backward_input's filters must be 4 dimensional but got filters ",
            filters.full_expression_name(), " with ndim = ", filters.ndim(), "."));
        ASSERT2(data_format == "NCHW" | data_format == "NHWC", utils::make_message(
            "conv2d_backward_input's data_format must be NCHW or NHWC but got ", data_format,
            " instead."));
        ASSERT2(result_shape.size() == 4, utils::make_message(
            "conv2d_backward_input's result_shape must be of size 4, "
            "but got ", result_shape, "."));
       auto info = op::compute_conv2d_info(result_shape,
                                           filters.shape(),
                                           stride_h,
                                           stride_w,
                                           padding,
                                           data_format);
        filters = filters.reshape({filters.shape()[0], -1}).transpose();
        if (data_format == "NCHW") {
            auto out_dw_cnhw = out_dw.swapaxes(0, 1);
            out_dw_cnhw = out_dw_cnhw.reshape({out_dw_cnhw.shape()[0], -1});
            // filters2D?
            return op::col2im(op::dot(filters, out_dw_cnhw),
                              result_shape,
                              info.filter_h,
                              info.filter_w,
                              stride_h,
                              stride_w,
                              data_format);
        } else {
            /* NHWC forward pass is:
             *
             *   output = (Im2col(Input))^T * Filters^T
             *
             * NHWC backward pass is:
             *
             *   ∂Im2col(Input)/∂E = Filters^T * ∂output/∂E^T
             *
             * Our 2d shapes into gemm are as follows:
             *
             *   ∂Im2col(Input)/∂E => (window_h * window_w * c) x (n * h * w)
             *
             *   Filters^T => (window_h * window_w * c) x (channels_out)
             *
             *   ∂output/∂E^T => (channels_out) x (n * h * w)
             *
             */
            out_dw = out_dw.reshape({-1, out_dw.shape()[3]}).transpose();
            return op::col2im(op::dot(filters, out_dw),
                              result_shape,
                              info.filter_h,
                              info.filter_w,
                              stride_h,
                              stride_w,
                              data_format);
        }
    }

    Array conv2d_backward_filters(Array input,
                                  Array out_dw,
                                  int stride_h,
                                  int stride_w,
                                  const std::vector<int>& filters_shape,
                                  PADDING_T padding,
                                  const std::string& data_format) {
        ASSERT2(out_dw.ndim() == 4, utils::make_message(
            "conv2d_backward_filters's out_dw must be 4 dimensional but got out_dw ",
            out_dw.full_expression_name(), " with ndim = ", out_dw.ndim(), "."));
        ASSERT2(input.ndim() == 4, utils::make_message(
            "conv2d_backward_filters's input must be 4 dimensional but got input ",
            input.full_expression_name(), " with ndim = ", input.ndim(), "."));
        ASSERT2(data_format == "NCHW" | data_format == "NHWC", utils::make_message(
            "conv2d_backward_filters's data_format must be NCHW or NHWC but got ", data_format,
            " instead."));
        ASSERT2(filters_shape.size() == 4, utils::make_message(
            "conv2d_backward_filters's filters_shape must be of size 4, "
            "but got ", filters_shape, "."));
        auto info = op::compute_conv2d_info(input.shape(),
                                            filters_shape,
                                            stride_h,
                                            stride_w,
                                            padding,
                                            data_format);
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
            data_format).transpose();
        if (data_format == "NCHW") {
            auto out_dw_cnhw = out_dw.swapaxes(1, 0);
            out_dw_cnhw = out_dw_cnhw.reshape({out_dw_cnhw.shape()[0], -1});
            // TODO(jonathan) ensure the reshape is done after the computation:
            return op::dot(out_dw_cnhw, im2col_image).reshape(filters_shape);
        } else {
            /* NHWC forward pass is:
             *
             *   output = (Im2col(Input))^T * Filters^T
             *
             * NHWC backward pass is:
             *
             *   ∂Filters/∂E = ∂output/∂E^T * Im2col(Input)^T
             *
             * Our 2d shapes into gemm are as follows:
             *
             *   Im2col(Input)^T => (n * h * w) x (window_h * window_w * c)
             *
             *   ∂output/∂E^T => (channels_out) x (n * h * w)
             *
             *   ∂Filters/∂E =>  (channels_out) x (window_h * window_w * c)
             */
            out_dw = out_dw.reshape({-1, out_dw.shape()[3]}).transpose();
            return op::dot(out_dw, im2col_image).reshape(filters_shape);
        }
    }
}  // namespace op
