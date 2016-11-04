#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/one_hot.h"
#include "dali/array/op.h"
#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/im2col.h"

TEST(RTCTests, im2col_without_channels) {
    Array image_nchw({2, 1, 3, 4}, DTYPE_INT32);
    Array image_nhwc = image_nchw.transpose({0, 2, 3, 1});
    image_nchw = initializer::arange(0, 1);
    image_nchw[1] *= 10;

    Array old_res_nhwc = old_op::im2col(
        image_nhwc, 3, 3, 1, 1, "NHWC"
    );
    Array rtc_res_nhwc = op::im2col(image_nhwc, 3, 3, 1, 1, "NHWC");
    EXPECT_TRUE(Array::equals(rtc_res_nhwc, old_res_nhwc));

    Array old_res_nchw = old_op::im2col(
        image_nchw, 3, 3, 1, 1, "NCHW"
    );
    Array rtc_res_nchw = op::im2col(image_nchw, 3, 3, 1, 1, "NCHW");
    EXPECT_TRUE(Array::equals(rtc_res_nchw, old_res_nchw));

    Array image_wnch = image_nchw.transpose({3, 0, 1, 2});
    // allow arbitrary data-formats:
    Array rtc_res_wnch = op::im2col(image_wnch, 3, 3, 1, 1, "WNCH");

    // break out the patch dimension
    auto rtc_res_wnch_patches = rtc_res_wnch.reshape({3, 1, 3, -1});
    // make patch become c, h, w:
    auto rtc_res_nchw_patches = rtc_res_wnch_patches.transpose({1, 2, 0, 3}).reshape({9, -1});

    EXPECT_TRUE(Array::equals(rtc_res_nchw_patches, old_res_nchw));
}
