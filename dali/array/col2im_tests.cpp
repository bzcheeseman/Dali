#include <gtest/gtest.h>
#include "dali/array/array.h"
#include "dali/array/op/col2im.h"
#include "dali/array/op/im2col.h"
#include "dali/array/op/arange.h"
#include "dali/array/expression/assignment.h"

// TODO(jonathan): add reference col2im

TEST(JITTests, col2im_without_channels) {
    Array image_nchw({2, 1, 3, 4}, DTYPE_INT32);
    Array image_nhwc = image_nchw.transpose({0, 2, 3, 1});

    image_nchw = op::assign(image_nchw, OPERATOR_T_EQL, op::arange(2 * 1 * 3 * 4).reshape({2, 1, 3, 4}));
    auto sliced = image_nchw[1];
    (sliced *= -1.0).eval();

    Array im2coled_image1 = op::im2col(
        image_nchw, 3, 3, 1, 1, 0, 0, 0, 0, "NCHW"
    );

    Array im2coled_image2 = op::im2col(
        image_nhwc, 3, 3, 1, 1, 0, 0, 0, 0, "NHWC"
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

TEST(JITTests, col2im_with_channels) {
    Array image_nchw({2, 2, 3, 4}, DTYPE_INT32);
    Array image_nhwc = image_nchw.transpose({0, 2, 3, 1});

    image_nchw = op::assign(image_nchw, OPERATOR_T_EQL, op::arange(2 * 2 * 3 * 4).reshape({2, 2, 3, 4}));
    image_nchw[1] *= 10;
    (Array)image_nchw[Slice(0, 2)][0] *= -1.0;

    Array im2coled_nchw = op::im2col(
        image_nchw, 3, 3, 1, 1, 0, 0, 0, 0, "NCHW"
    );

    Array im2coled_nhwc = op::im2col(
        image_nhwc, 3, 3, 1, 1, 0, 0, 0, 0, "NHWC"
    );
}
