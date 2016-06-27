#include <gtest/gtest.h>

#include "dali/array/test_utils.h"
#include "dali/array/op.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"


using namespace op;

TEST(ArraySpatialTests, conv_forward_nchw) {
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

TEST(ArraySpatialTests, conv_forward_nhwc) {
    Array X = Array::arange({1, 8, 8, 1}, DTYPE_FLOAT);
    Array W = Array::ones({1, 2, 2, 1}, DTYPE_FLOAT);

    Array out = conv2d(
        X,
        W,
        /*stride_h=*/2,
        /*stride_w=*/2,
        PADDING_T_VALID,
        "NHWC");

    // TODO(szymon): add a test that compares this
    //               to reference implementation
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


TEST(ArraySpatialTests, conv_backward_bias) {
    Array X = Array::ones({2, 3, 4, 5}, DTYPE_FLOAT);

    Array out = conv2d_backward_bias(X, "NCHW");
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(2 * 4 * 5, (float)out[i]);
    }
}

Array reference_pool2d(const Array& x,
                       int window_h,
                       int window_w,
                       int stride_h,
                       int stride_w,
                       POOLING_T pooling_mode,
                       PADDING_T padding_mode,
                       const std::string& data_format) {
    ASSERT2(x.shape().size() == 4, "must be a 4D array");
    ASSERT2(padding_mode == PADDING_T_VALID, "reference only exists for valid padding");
    std::vector<int> out_shape(4, 0);

    int pos_n = data_format.find_first_of('N');
    int pos_c = data_format.find_first_of('C');
    int pos_h = data_format.find_first_of('H');
    int pos_w = data_format.find_first_of('W');

    ASSERT2(
        data_format.size() == 4 && pos_n != -1 && pos_c != -1 && pos_h != -1 && pos_w != -1,
        "data format must be a 4 letter string containing N, C, H and W"
    );
    int in_n = x.shape()[pos_n];
    int in_c = x.shape()[pos_c];
    out_shape[pos_n] = in_n;
    out_shape[pos_c] = in_c;

    int in_w = x.shape()[pos_w];
    int out_w = 1 + (in_w - window_w) / stride_w;
    out_shape[pos_w] = out_w;

    int in_h = x.shape()[pos_h];
    int out_h = 1 + (in_h - window_w) / stride_w;
    out_shape[pos_h] = out_h;
    Array out(out_shape, x.dtype());

    auto x_swapped = x.transpose({pos_n, pos_c, pos_h, pos_w});
    auto out_swapped = out.transpose({pos_n, pos_c, pos_h, pos_w});

    for (int i = 0; i < out_h; i++) {
        int h_start = i * stride_h;
        int h_end = h_start + window_h;
        for (int j = 0; j < out_w; j++) {
            int w_start = j * stride_w;
            int w_end = w_start + window_w;
            Array window = x_swapped[Slice(0, in_n)][Slice(0, in_c)][Slice(h_start, h_end)][Slice(w_start, w_end)];
            window = window.reshape({in_n, in_c, -1});
            if (pooling_mode == POOLING_T_MAX) {
                (Array)(out_swapped[Slice(0, in_n)][Slice(0, in_c)][i][j]) = window.max(-1);
            } else if (pooling_mode == POOLING_T_AVG) {
                (Array)(out_swapped[Slice(0, in_n)][Slice(0, in_c)][i][j]) = window.mean(-1);
            }
        }
    }
    return out;
}

TEST(ArraySpatialTests, pool2d_forward_nchw) {
    Array X = Array::arange({2, 2, 8, 8}, DTYPE_FLOAT);
    Array expected_out = reference_pool2d(
        X,
        /*window_h=*/2,
        /*window_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NCHW"
    );
    Array out = pool2d(
        X,
        /*window_h=*/2,
        /*window_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NCHW"
    );
    EXPECT_TRUE(Array::allclose(expected_out, out, 1e-3));
}

TEST(ArraySpatialTests, pool2d_forward_nhwc) {
    Array X = Array::arange({2, 8, 8, 2}, DTYPE_FLOAT);

    Array expected_out = reference_pool2d(
        X,
        /*window_h=*/2,
        /*window_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NHWC"
    );
    Array out = pool2d(
        X,
        /*window_h=*/2,
        /*window_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NHWC"
    );

    EXPECT_TRUE(Array::allclose(expected_out, out, 1e-3));
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
        "NCHW"
    );

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
}
