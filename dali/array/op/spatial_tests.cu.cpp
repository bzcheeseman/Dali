#include <gtest/gtest.h>

#include "dali/array/test_utils.h"
#include "dali/array/op.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"


using namespace op;

Array reference_conv2d(Array X, Array W,
                           int stride_h, int stride_w,
                           PADDING_T padding,
                           const std::string& data_format) {
    auto info = internal::compute_conv_info(
            X.shape(),
            W.shape(),
            stride_h,
            stride_w,
            padding,
            data_format);

    if (data_format == "NHWC") {
        X = X.transpose({0, 3, 1, 2});
        W = W.transpose({0, 3, 1, 2});
    }

    if (padding == PADDING_T_SAME) {
        int n = X.shape()[0];
        int c = X.shape()[1];
        int h = X.shape()[2];
        int w = X.shape()[3];
        int pad_h = info.padding_h;
        int pad_w = info.padding_w;
        int odd_pad_h = info.odd_padding_h;
        int odd_pad_w = info.odd_padding_w;

        std::vector<int> padded_shape =
                {n, c, h + 2 * pad_h + odd_pad_h, w + 2 * pad_w + odd_pad_w};
        auto X_padded = Array::zeros(padded_shape, X.dtype(), X.preferred_device());

        Array X_padded_content =
                X_padded[Slice(0, n)]
                        [Slice(0, c)]
                        [Slice(pad_h, h + pad_h)]
                        [Slice(pad_w, w + pad_w)];
        X_padded_content = op::identity(X);

        X = X_padded;
    }


    Array out = Array::zeros({info.batch_size,
                              info.out_channels,
                              info.out_h,
                              info.out_w}, X.dtype());

    auto normalize_h = [&](int h_idx) {
        return std::max(0, std::min(X.shape()[2], h_idx));
    };

    auto normalize_w = [&](int w_idx) {
        return std::max(0, std::min(X.shape()[3], w_idx));
    };

    for (int n_idx = 0; n_idx < info.batch_size; ++n_idx) {
        for (int out_c_idx = 0; out_c_idx < info.out_channels; ++out_c_idx) {
            for (int h_idx = 0; h_idx < info.out_h; ++h_idx) {
                for (int w_idx = 0; w_idx < info.out_w; ++w_idx) {

                    int in_h_idx = h_idx * stride_h; // TODO(szymon): +padding
                    int in_w_idx = w_idx * stride_w;
                    auto h_slice = Slice(normalize_h(in_h_idx),
                                         normalize_h(in_h_idx + info.filter_h));
                    auto w_slice = Slice(normalize_w(in_w_idx),
                                         normalize_w(in_w_idx + info.filter_w));

                    Array temp = X[n_idx][Slice(0, info.in_channels)][h_slice][w_slice] *
                                 W[out_c_idx];
                    out[n_idx][out_c_idx][h_idx][w_idx] = temp.sum();
                }
            }
        }
    }
    if (data_format == "NHWC") {
        return out.transpose({0, 2, 3, 1});
    } else {
        return out;
    }
}

TEST(ArraySpatialTests, conv_forward) {
    for (int stride_h = 1; stride_h <= 2; ++stride_h) {
        for (int stride_w = 1; stride_w <= 2; ++stride_w) {
            for (std::string data_format: {"NCHW", "NHWC"}) {
                for (PADDING_T padding : {PADDING_T_VALID, PADDING_T_SAME}) {
                    Array X, W;

                    if (data_format == "NCHW") {
                        X = Array({5, 3, 6, 8}, DTYPE_FLOAT);
                        W = Array({2, 3, 2, 4}, DTYPE_FLOAT);
                    } else {
                        X = Array({5, 6, 8, 3}, DTYPE_FLOAT);
                        W = Array({2, 2, 4, 3}, DTYPE_FLOAT);
                    }

                    X = initializer::uniform(-1.0, 1.0);
                    W = initializer::uniform(-1.0, 1.0);

                    auto padding_str = (padding == PADDING_T_VALID) ? "valid" : "same";
                    std::string scope_name = utils::MS() << "stride_h = " << stride_h
                                                         << ", stride_w = " << stride_w
                                                         << ", data_format = " << data_format
                                                         << ", padding = " << padding_str;
                    SCOPED_TRACE(scope_name);

                    Array expected =
                        reference_conv2d(
                            X,
                            W,
                            stride_h,
                            stride_w,
                            padding,
                            data_format);
                    // reference computation will be much faster on CPU, methinks.
                    X.to_device(memory::Device::cpu());
                    W.to_device(memory::Device::cpu());
                    Array actual = conv2d(
                            X,
                            W,
                            stride_h,
                            stride_w,
                            padding,
                            data_format);

                    ASSERT_TRUE(Array::allclose(expected, actual, 1e-3));
                }
            }
        }
    }
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

Array reference_pool2d_backward(const Array& out,
                                const Array& out_dw,
                                const Array& X,
                                int window_h,
                                int window_w,
                                int stride_h,
                                int stride_w,
                                POOLING_T pooling_mode,
                                PADDING_T padding_mode,
                                const std::string& data_format) {
    ASSERT2(out.shape().size() == 4, "must be a 4D array");
    ASSERT2(out_dw.shape().size() == 4, "must be a 4D array");
    ASSERT2(X.shape().size() == 4, "must be a 4D array");
    ASSERT2(padding_mode == PADDING_T_VALID, "reference only exists for valid padding");

    int pos_n = data_format.find_first_of('N');
    int pos_c = data_format.find_first_of('C');
    int pos_h = data_format.find_first_of('H');
    int pos_w = data_format.find_first_of('W');

    ASSERT2(
        data_format.size() == 4 && pos_n != -1 && pos_c != -1 && pos_h != -1 && pos_w != -1,
        "data format must be a 4 letter string containing N, C, H and W"
    );
    Array in_dw = Array::zeros_like(X);

    auto x_swapped = X.transpose({pos_n, pos_c, pos_h, pos_w});
    auto out_swapped = out.transpose({pos_n, pos_c, pos_h, pos_w});
    auto out_dw_swapped = out_dw.transpose({pos_n, pos_c, pos_h, pos_w});
    auto in_dw_swapped = in_dw.transpose({pos_n, pos_c, pos_h, pos_w});

    int out_h = in_dw.shape()[pos_h];
    int out_w = in_dw.shape()[pos_w];
    int in_n = in_dw.shape()[pos_n];
    int in_c = in_dw.shape()[pos_c];

    int pshape_h = out.shape()[pos_h];
    int pshape_w = out.shape()[pos_w];

    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            int ph_min = h < window_h ? 0 : (h - window_h + stride_h) / stride_h;
            int ph_max = std::min((h + stride_h) / stride_h, pshape_h);

            int pw_min = w < window_w ? 0 : (w - window_w + stride_w) / stride_w;
            int pw_max = std::min((w + stride_w) / stride_w, pshape_w);

            Array source_at_hw = x_swapped[Slice(0, in_n)][Slice(0, in_c)][h][w][Broadcast()][Broadcast()];

            Array out_impact = out_swapped[Slice(0, in_n)][Slice(0, in_c)][Slice(ph_min, ph_max)][Slice(pw_min, pw_max)];
            Array out_grad_impact = out_dw_swapped[Slice(0, in_n)][Slice(0, in_c)][Slice(ph_min, ph_max)][Slice(pw_min, pw_max)];

            Array grad_at_hw;
            if (pooling_mode == POOLING_T_MAX) {
                grad_at_hw = op::equals(source_at_hw, out_impact) * out_grad_impact;
            } else if (pooling_mode == POOLING_T_AVG) {
                grad_at_hw = out_grad_impact / ((double) window_h * window_w);
            }
            (Array)(in_dw_swapped[Slice(0, in_n)][Slice(0, in_c)][h][w]) = grad_at_hw.reshape({in_n, in_c, -1}).sum(-1);
        }
    }
    return in_dw;
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

TEST(ArraySpatialTests, pool2d_backward_nchw) {
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
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NCHW"
    );
    Array expected_in_dw = reference_pool2d_backward(
        out,
        out_dw,
        X,
        /*window_h=*/2,
        /*window_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NCHW"
    );
    EXPECT_TRUE(Array::allclose(expected_in_dw, in_dw, 1e-3));
}

TEST(ArraySpatialTests, pool2d_backward_nhwc) {
    Array X = Array::arange({1, 8, 8, 1}, DTYPE_FLOAT);

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

    Array out_dw = Array::ones_like(out);
    Array in_dw = pool2d_backward(
        out,
        out_dw,
        X,
        /*window_h=*/2,
        /*window_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NHWC"
    );
    Array expected_in_dw = reference_pool2d_backward(
        out,
        out_dw,
        X,
        /*window_h=*/2,
        /*window_w=*/2,
        /*stride_h=*/2,
        /*stride_w=*/2,
        POOLING_T_MAX,
        PADDING_T_VALID,
        "NHWC"
    );
    EXPECT_TRUE(Array::allclose(expected_in_dw, in_dw, 1e-3));
}
