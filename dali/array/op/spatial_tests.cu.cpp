#include <gtest/gtest.h>

#include "dali/array/test_utils.h"
#include "dali/array/op.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"


using namespace op;


TEST(ArraySpatialTests, conv_forward) {
    Array X = Array::arange({1, 1, 8, 8}, DTYPE_FLOAT);
    Array W = Array::ones({1, 1, 2, 2}, DTYPE_FLOAT);

    Array out = conv2d(
        X,
        W,
        /*stride_h=*/2,
        /*stride_w=*/2,
        PADDING_T_VALID,
        "NCHW");
    // TODO(szymon): add a test that compares this
    //               to reference implementation
}

TEST(ArraySpatialTests, pool2d_forward) {
    Array X = Array::arange({1, 1, 8, 8}, DTYPE_FLOAT);

    Array out = pool2d(
        X,
        /*window_h=*/2,
        /*window_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NCHW");
}

TEST(ArraySpatialTests, pool2d_backward) {
    Array X = Array::arange({1, 1, 8, 8}, DTYPE_FLOAT);

    Array out = pool2d(
        X,
        /*window_h=*/2,
        /*window_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NCHW");

    Array out_dw = Array::ones_like(out);

    Array in_dw = pool2d_backward(
        out,
        out_dw,
        X,
        /*window_h=*/2,
        /*window_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        X.shape(),
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NCHW"
        );
    X.print();
    out.print();
    in_dw.print();
}


TEST(ArraySpatialTests, conv_backward) {
    Array X = Array::arange({1, 1, 8, 8}, DTYPE_FLOAT);
    Array W = Array::ones({1, 1, 2, 2}, DTYPE_FLOAT);

    Array out = conv2d(
        X,
        W,
        /*stride_h=*/2,
        /*stride_w=*/2,
        PADDING_T_VALID,
        "NCHW");

    Array out_dw = Array::ones_like(out);

    Array in_dw = conv2d_backward_input(
        W,
        out_dw,
        /*stride_h=*/2,
        /*stride_w=*/2,
        X.shape(),
        PADDING_T_VALID,
        "NCHW");

    Array W_dw = conv2d_backward_filters(
        X,
        out_dw,
        /*stride_h=*/2,
        /*stride_w=*/2,
        W.shape(),
        PADDING_T_VALID,
        "NCHW");

    // TODO(szymon): add a test that compares this
    //               to reference implementation
}
