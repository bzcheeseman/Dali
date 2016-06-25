#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"
#include "dali/array/functor.h"
#include "dali/array/lazy_op.h"
#include "dali/array/lazy/im2col.h"


TEST(ArrayReshapeTests, im2col_without_channels) {
    Array image_nchw({2, 1, 3, 4}, DTYPE_INT32);
    Array image_nhwc = image_nchw.transpose({0, 2, 3, 1});

    image_nchw = initializer::arange();
    image_nchw[1] *= 10;

    Array im2coled_image = lazy::im2col_nchw(
        image_nchw, 3, 3, 1, 1
    );
    Array im2coled_image2 = lazy::im2col_nhwc(
        image_nhwc, 3, 3, 1, 1
    );
    EXPECT_TRUE(Array::equals(im2coled_image2, im2coled_image));
}


TEST(ArrayReshapeTests, im2col_with_channels) {
    Array image_nchw({2, 2, 3, 4}, DTYPE_INT32);
    Array image_nhwc = image_nchw.transpose({0, 2, 3, 1});

    image_nchw = initializer::arange();
    image_nchw[1] *= 10;
    (Array)image_nchw[Slice(0, 2)][0] *= -1.0;

    Array im2coled_image1 = lazy::im2col_nchw(
        image_nchw, 3, 3, 1, 1
    );
    Array im2coled_image1_ch1 = im2coled_image1[Slice(0, im2coled_image1.shape()[0] / 2)];
    Array im2coled_image1_ch2 = im2coled_image1[Slice(im2coled_image1.shape()[0] / 2, im2coled_image1.shape()[0])];

    Array im2coled_image2 = lazy::im2col_nhwc(
        image_nhwc, 3, 3, 1, 1
    );

    Array im2coled_image2_ch1 = im2coled_image2[Slice(0, im2coled_image2.shape()[0], 2)];
    Array im2coled_image2_ch2 = im2coled_image2[Slice(1, im2coled_image2.shape()[0], 2)];

    EXPECT_TRUE(Array::equals(im2coled_image1_ch1, im2coled_image2_ch1));
    EXPECT_TRUE(Array::equals(im2coled_image1_ch2, im2coled_image2_ch2));
}

TEST(ArrayReshapeTests, col2im_without_channels) {
    Array image_nchw({2, 1, 3, 4}, DTYPE_INT32);
    Array image_nhwc = image_nchw.transpose({0, 2, 3, 1});

    image_nchw = initializer::arange();
    image_nchw[1] *= -1.0;

    Array im2coled_image1 = lazy::im2col_nchw(
        image_nchw, 3, 3, 1, 1
    );

    Array im2coled_image2 = lazy::im2col_nhwc(
        image_nhwc, 3, 3, 1, 1
    );

    Array col2imed_nchw = lazy::col2im_nchw(im2coled_image1, image_nchw.shape(), 3, 3, 1, 1);
    Array col2imed_nhwc = lazy::col2im_nhwc(im2coled_image2, image_nhwc.shape(), 3, 3, 1, 1);

    EXPECT_TRUE(
        Array::equals(
            col2imed_nchw.transpose({0, 2, 3, 1}),
            col2imed_nhwc
        )
    );
}

TEST(ArrayReshapeTests, col2im_with_channels) {
    Array image_nchw({2, 2, 3, 4}, DTYPE_INT32);
    Array image_nhwc = image_nchw.transpose({0, 2, 3, 1});

    image_nchw = initializer::arange();
    image_nchw[1] *= 10;
    (Array)image_nchw[Slice(0, 2)][0] *= -1.0;

    Array im2coled_image1 = lazy::im2col_nchw(
        image_nchw, 3, 3, 1, 1
    );

    Array im2coled_image2 = lazy::im2col_nhwc(
        image_nhwc, 3, 3, 1, 1
    );

    Array col2imed_nchw = lazy::col2im_nchw(im2coled_image1, image_nchw.shape(), 3, 3, 1, 1);
    Array col2imed_nhwc = lazy::col2im_nhwc(im2coled_image2, image_nhwc.shape(), 3, 3, 1, 1);

    EXPECT_TRUE(
        Array::equals(
            col2imed_nchw.transpose({0, 2, 3, 1}),
            col2imed_nhwc
        )
    );
}
