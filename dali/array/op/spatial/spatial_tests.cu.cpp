#include <gtest/gtest.h>
#include <vector>

#include "dali/array/test_utils.h"
#include "dali/array/op.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"


using namespace op;

//////////////////////////////////////////////////////////////////////////////////////////////
//                           REFERENCE IMPLEMENTATIONS                                      //
//////////////////////////////////////////////////////////////////////////////////////////////

std::vector<int> permute_shape(const std::vector<int>& shape, const std::string& data_format) {
    ASSERT2(shape.size() == 4, "only 4d tensors are allowed.");

    internal::DataFormatDimMapping mapping(data_format);

    std::vector<int> res(4);

    res[mapping.n_dim] = shape[0];
    res[mapping.c_dim] = shape[1];
    res[mapping.h_dim] = shape[2];
    res[mapping.w_dim] = shape[3];

    return res;
}

Array pad_array(Array in, int prepad_h, int postpad_h, int prepad_w, int postpad_w) {
    int n = in.shape()[0];
    int c = in.shape()[1];
    int h = in.shape()[2];
    int w = in.shape()[3];

    std::vector<int> padded_shape =
            {n, c, h + prepad_h + postpad_h, w + prepad_w + postpad_w};
    auto X_padded = Array::zeros(padded_shape, in.dtype(), in.preferred_device());

    Array X_padded_content =
            X_padded[Slice(0, n)]
                    [Slice(0, c)]
                    [Slice(prepad_h, h + prepad_h)]
                    [Slice(prepad_w, w + prepad_w)];
    X_padded_content = op::identity(in);

    return X_padded;
}

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

    internal::DataFormatDimMapping mapping(data_format);

    X = X.transpose({mapping.n_dim, mapping.c_dim, mapping.h_dim, mapping.w_dim});
    W = W.transpose({mapping.n_dim, mapping.c_dim, mapping.h_dim, mapping.w_dim});



    if (padding == PADDING_T_SAME) {
        X = pad_array(X, info.padding_h, info.padding_h + info.odd_padding_h,
                         info.padding_w, info.padding_w + info.odd_padding_w);
    }

    auto out_shape = permute_shape({info.batch_size,
                                     info.out_channels,
                                     info.out_h,
                                     info.out_w},
                                    data_format);
    Array out_orig_data_format = Array::zeros(out_shape, X.dtype(), X.preferred_device());
    Array out = out_orig_data_format.transpose(
            {mapping.n_dim, mapping.c_dim, mapping.h_dim, mapping.w_dim});

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
    return out_orig_data_format;
}

Array reference_pool2d(Array X,
                       int window_h,
                       int window_w,
                       int stride_h,
                       int stride_w,
                       POOLING_T pooling_mode,
                       PADDING_T padding_mode,
                       const std::string& data_format) {
    ASSERT2(X.shape().size() == 4, "must be a 4D array");

    auto info = internal::compute_pool_info(
            X.shape(),
            window_h,
            window_w,
            stride_h,
            stride_w,
            padding_mode,
            data_format);

    internal::DataFormatDimMapping mapping(data_format);

    X = X.transpose({mapping.n_dim, mapping.c_dim, mapping.h_dim, mapping.w_dim});


    if (padding_mode == PADDING_T_SAME) {
        X = pad_array(X, info.padding_h, info.padding_h + info.odd_padding_h,
                         info.padding_w, info.padding_w + info.odd_padding_w);
    }

    auto out_shape = permute_shape({info.batch_size,
                                     info.in_channels,
                                     info.out_h,
                                     info.out_w},
                                    data_format);

    auto out_orig_data_format = Array(out_shape, X.dtype(), X.preferred_device());
    Array out = out_orig_data_format.transpose(
            {mapping.n_dim, mapping.c_dim, mapping.h_dim, mapping.w_dim});


    int in_c = info.in_channels, in_n = info.batch_size;

    for (int i = 0; i < info.out_h; i++) {
        int h_start = i * stride_h;
        int h_end = h_start + window_h;
        for (int j = 0; j < info.out_w; j++) {
            int w_start = j * stride_w;
            int w_end = w_start + window_w;
            Array window = X[Slice(0, in_n)][Slice(0, in_c)][Slice(h_start, h_end)][Slice(w_start, w_end)];
            window = window.reshape({in_n, in_c, -1});
            if (pooling_mode == POOLING_T_MAX) {
                (Array)(out[Slice(0, in_n)][Slice(0, in_c)][i][j]) = window.max(-1);
            } else if (pooling_mode == POOLING_T_AVG) {
                (Array)(out[Slice(0, in_n)][Slice(0, in_c)][i][j]) = window.mean(-1);
            }
        }
    }
    return out_orig_data_format;
}


//////////////////////////////////////////////////////////////////////////////////////////////
//                                   TESTS                                                  //
//////////////////////////////////////////////////////////////////////////////////////////////


TEST(ArraySpatialTests, conv2d_forward) {
    for (int stride_h = 1; stride_h <= 2; ++stride_h) {
        for (int stride_w = 1; stride_w <= 2; ++stride_w) {
            for (std::string data_format: {"NCHW", "NHWC"}) {
                for (PADDING_T padding : {PADDING_T_VALID, PADDING_T_SAME}) {

                    auto padding_str = (padding == PADDING_T_VALID) ? "valid" : "same";
                    std::string scope_name = utils::MS() << "stride_h = " << stride_h
                                                         << ", stride_w = " << stride_w
                                                         << ", data_format = " << data_format
                                                         << ", padding = " << padding_str;
                    SCOPED_TRACE(scope_name);

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

                    Array actual = conv2d(
                            X,
                            W,
                            stride_h,
                            stride_w,
                            padding,
                            data_format);

                    // reference computation will be much faster on CPU, methinks.
                    X.to_device(memory::Device::cpu());
                    W.to_device(memory::Device::cpu());

                    Array expected =
                        reference_conv2d(
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


TEST(ArraySpatialTests, pool2d_forward) {
    for (int window_h = 1; window_h <= 2; ++window_h) {
        for (int window_w = 1; window_w <= 2; ++window_w) {
            for (int stride_h = 1; stride_h <= 2; ++stride_h) {
                for (int stride_w = 1; stride_w <= 2; ++stride_w) {
                    for (std::string data_format: {"NCHW", "NHWC"}) {
                        for (PADDING_T padding : {PADDING_T_VALID, PADDING_T_SAME}) {
                            for (POOLING_T pooling: {POOLING_T_MAX, POOLING_T_AVG}) {
                                auto padding_str = (padding == PADDING_T_VALID) ? "valid" : "same";
                                auto pooling_str = (pooling == POOLING_T_MAX)   ? "max"   : "avg";
                                std::string scope_name = utils::MS() <<   "window_h = " << window_h
                                                                     << ", window_w = " << window_w
                                                                     << ", stride_h = " << stride_h
                                                                     << ", stride_w = " << stride_w
                                                                     << ", data_format = " << data_format
                                                                     << ", padding = " << padding_str
                                                                     << ", pooling = " << pooling_str;
                                SCOPED_TRACE(scope_name);

                                Array X, W;

                                if (data_format == "NCHW") {
                                    X = Array({5, 3, 6, 8}, DTYPE_FLOAT);
                                } else {
                                    X = Array({5, 6, 8, 3}, DTYPE_FLOAT);
                                }

                                X = initializer::uniform(-1.0, 1.0);
                                W = initializer::uniform(-1.0, 1.0);

                                Array out = pool2d(
                                    X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format
                                );

                                X.to_device(memory::Device::cpu());

                                Array expected_out = reference_pool2d(
                                    X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format
                                );

                                EXPECT_TRUE(Array::allclose(expected_out, out, 1e-3));

                            }
                        }
                    }
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
