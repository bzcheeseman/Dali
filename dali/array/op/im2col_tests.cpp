#include <gtest/gtest.h>
#include "dali/array/op/arange.h"
#include "dali/array/op/im2col.h"
#include "dali/array/expression/assignment.h"

Array reference_im2col(Array image,
                       int filter_h,
                       int filter_w,
                       int stride_h,
                       int stride_w,
                       const std::string& data_format) {
    ASSERT2(data_format == "NCHW" | data_format == "NHWC",
        "only supports NCHW or NHWC");
    if (data_format == "NCHW") {
        image = image.transpose({0, 2, 3, 1});
    }
    auto out = Array::zeros(op::im2col_shape(image.shape(),
        filter_h,
        filter_w,
        stride_h,
        stride_w,
        1,
        1,
        0,
        0,
        0,
        0,
        "NHWC"), image.dtype());
    int frame_idx = 0;
    for (int batch_idx = 0; batch_idx < image.shape()[0]; batch_idx++) {
        for (int height_idx = 0; height_idx + filter_h <= image.shape()[1]; height_idx += stride_h) {
            for (int width_idx = 0; width_idx + filter_w <= image.shape()[2]; width_idx += stride_w) {
                Array sliced = image[batch_idx][Slice(height_idx, height_idx+filter_h)][Slice(width_idx, width_idx+filter_w)];
                Array out_sliced = out[Slice()][frame_idx];
                op::assign(out_sliced, OPERATOR_T_EQL, sliced.ravel()).eval();
                frame_idx += 1;
            }
        }
    }
    return out;
}

TEST(JITTests, im2col_without_channels) {
    memory::WithDevicePreference dp(memory::Device::cpu());
    Array image_nchw({2, 1, 3, 4}, DTYPE_INT32);
    Array image_nhwc = image_nchw.transpose({0, 2, 3, 1});
    image_nchw = op::assign(image_nchw, OPERATOR_T_EQL, op::arange(2 * 1 * 3 * 4).reshape({2, 1, 3, 4}));
    // image_nchw[1] *= 10;
    image_nchw.eval();

    Array old_res_nhwc = reference_im2col(
        image_nhwc, 3, 3, 1, 1, "NHWC"
    );
    Array jit_res_nhwc = op::im2col(image_nhwc, 3, 3, 1, 1, 0, 0, "NHWC");
    EXPECT_TRUE(Array::equals(jit_res_nhwc, old_res_nhwc));

    Array old_res_nchw = reference_im2col(
        image_nchw, 3, 3, 1, 1, "NCHW"
    );
    Array jit_res_nchw = op::im2col(image_nchw, 3, 3, 1, 1, 0, 0, "NCHW");
    EXPECT_TRUE(Array::equals(jit_res_nchw, old_res_nchw));

    Array image_wnch = image_nchw.transpose({3, 0, 1, 2});

    // allow arbitrary data-formats:
    Array jit_res_wnch = op::im2col(image_wnch, 3, 3, 1, 1, 0, 0, "WNCH");

    // // break out the patch dimension
    auto jit_res_wnch_patches = jit_res_wnch.reshape({3, 1, 3, -1});

    // // make patch become c, h, w:
    auto jit_res_nchw_patches = jit_res_wnch_patches.transpose({1, 2, 0, 3}).reshape({9, -1});
    EXPECT_TRUE(Array::equals(jit_res_nchw_patches, old_res_nchw));
}
