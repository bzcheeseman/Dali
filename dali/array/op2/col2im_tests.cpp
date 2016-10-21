#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/one_hot.h"
#include "dali/array/op.h"
#include "dali/array/op2/operation.h"
#include "dali/array/op2/im2col.h"
#include "dali/array/op2/col2im.h"


TEST(RTCTests, col2im_without_channels) {
    Array image_nchw({2, 1, 3, 4}, DTYPE_INT32);
    Array image_nhwc = image_nchw.transpose({0, 2, 3, 1});

    image_nchw = initializer::arange(0, 1);
    image_nchw[1] *= -1.0;

    Array im2coled_image1 = op::im2col(
        image_nchw, 3, 3, 1, 1, "NCHW"
    );

    Array im2coled_image2 = op::im2col(
        image_nhwc, 3, 3, 1, 1, "NHWC"
    );

    Array col2imed_nchw = op::col2im(im2coled_image1, image_nchw.shape(), 3, 3, 1, 1, "NCHW");
    Array col2imed_nhwc = op::col2im(im2coled_image2, image_nhwc.shape(), 3, 3, 1, 1, "NHWC");
    EXPECT_TRUE(
        Array::equals(
            col2imed_nchw.transpose({0, 2, 3, 1}),
            col2imed_nhwc
        )
    );
}

TEST(RTCTests, col2im_with_channels) {
    Array image_nchw({2, 2, 3, 4}, DTYPE_INT32);
    Array image_nhwc = image_nchw.transpose({0, 2, 3, 1});

    image_nchw = initializer::arange(0, 1);
    image_nchw[1] *= 10;
    (Array)image_nchw[Slice(0, 2)][0] *= -1.0;

    Array im2coled_nchw = op::im2col(
        image_nchw, 3, 3, 1, 1, "NCHW"
    );

    Array im2coled_nhwc = op::im2col(
        image_nhwc, 3, 3, 1, 1, "NHWC"
    );

    Array col2imed_nchw = op::col2im(im2coled_nchw, image_nchw.shape(), 3, 3, 1, 1, "NCHW");
    Array col2imed_nchw_old = old_op::col2im(im2coled_nchw, image_nchw.shape(), 3, 3, 1, 1, "NCHW");
    Array col2imed_nhwc = op::col2im(im2coled_nhwc, image_nhwc.shape(), 3, 3, 1, 1, "NHWC");
    Array col2imed_nhwc_old = old_op::col2im(im2coled_nhwc, image_nhwc.shape(), 3, 3, 1, 1, "NHWC");

    EXPECT_TRUE(
        Array::equals(
            col2imed_nhwc,
            col2imed_nhwc_old
        )
    );

    // col2imed_nchw.print();
    // col2imed_nchw_old.print();

    EXPECT_TRUE(
        Array::equals(
            col2imed_nchw,
            col2imed_nchw_old
        )
    );

    EXPECT_TRUE(
        Array::equals(
            col2imed_nchw.transpose({0, 2, 3, 1}),
            col2imed_nhwc
        )
    );
}
