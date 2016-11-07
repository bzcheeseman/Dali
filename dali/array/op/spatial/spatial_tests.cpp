#include <gtest/gtest.h>
#include <vector>

#include "dali/config.h"

#include "dali/array/test_utils.h"
#include "dali/array/op.h"
#include "dali/runtime_config.h"
#include "dali/utils/make_message.h"

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

                    int in_h_idx = h_idx * stride_h;
                    int in_w_idx = w_idx * stride_w;
                    auto h_slice = Slice(normalize_h(in_h_idx),
                                         normalize_h(in_h_idx + info.filter_h));
                    auto w_slice = Slice(normalize_w(in_w_idx),
                                         normalize_w(in_w_idx + info.filter_w));

                    auto temp = X[n_idx][Slice(0, info.in_channels)][h_slice][w_slice] *
                                 W[out_c_idx];
                    out[n_idx][out_c_idx][h_idx][w_idx] = op::sum(temp);
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

    int unpadded_x_h = X.shape()[2], unpadded_x_w = X.shape()[3];
    int prepad_h = info.padding_h;
    int prepad_w = info.padding_w;

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

        h_start = std::max(h_start - prepad_h, 0);
        h_end   = std::min(h_end - prepad_h, unpadded_x_h);

        for (int j = 0; j < info.out_w; j++) {
            int w_start = j * stride_w;
            int w_end = w_start + window_w;

            w_start = std::max(w_start - prepad_w, 0);
            w_end   = std::min(w_end - prepad_w, unpadded_x_w);

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


Array reference_pool2d_backward(Array out,
                                Array out_dw,
                                Array X,
                                int window_h,
                                int window_w,
                                int stride_h,
                                int stride_w,
                                POOLING_T pooling_mode,
                                PADDING_T padding_mode,
                                const std::string& data_format) {
    ASSERT2(out.shape().size() == 4,    "must be a 4D array");
    ASSERT2(out_dw.shape().size() == 4, "must be a 4D array");
    ASSERT2(X.shape().size() == 4,      "must be a 4D array");

    auto info = internal::compute_pool_info(
            X.shape(),
            window_h,
            window_w,
            stride_h,
            stride_w,
            padding_mode,
            data_format);

    internal::DataFormatDimMapping mapping(data_format);

    Array in_dw = Array::zeros_like(X);

    auto x_swapped      =      X.transpose({mapping.n_dim, mapping.c_dim, mapping.h_dim, mapping.w_dim});
    auto out_swapped    =    out.transpose({mapping.n_dim, mapping.c_dim, mapping.h_dim, mapping.w_dim});
    auto out_dw_swapped = out_dw.transpose({mapping.n_dim, mapping.c_dim, mapping.h_dim, mapping.w_dim});
    auto in_dw_swapped  =  in_dw.transpose({mapping.n_dim, mapping.c_dim, mapping.h_dim, mapping.w_dim});

    int out_h = out_swapped.shape()[2];
    int out_w = out_swapped.shape()[3];

    int in_n  = in_dw_swapped.shape()[0];
    int in_c  = in_dw_swapped.shape()[1];
    int in_h  = in_dw_swapped.shape()[2];
    int in_w  = in_dw_swapped.shape()[3];

    int prepad_h = info.padding_h;
    int prepad_w = info.padding_w;

    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            int unpadded_image_h_start = std::max(h * stride_h - prepad_h, 0);
            int unpadded_image_w_start = std::max(w * stride_w - prepad_w, 0);

            int unpadded_image_h_end = std::min(h * stride_h - prepad_h + window_h, in_h);
            int unpadded_image_w_end = std::min(w * stride_w - prepad_w + window_w, in_w);


            auto h_slice = Slice(unpadded_image_h_start, unpadded_image_h_end);
            auto w_slice = Slice(unpadded_image_w_start, unpadded_image_w_end);

            Array window = in_dw_swapped[Slice(0, in_n)][Slice(0, in_c)][h_slice][w_slice];

            if (pooling_mode == POOLING_T_AVG) {
                int window_size = window.shape()[2] * window.shape()[3];
                window += out_dw_swapped[Slice(0, in_n)][Slice(0, in_c)][h][w][Broadcast()][Broadcast()]
                        / window_size;
            } else {
                Array input_window = x_swapped[Slice(0, in_n)][Slice(0, in_c)][h_slice][w_slice];
                Array max_in_window = op::max(input_window, {-2, -1});

                Array max_locations = op::equals(
                        input_window, (Array)max_in_window[Slice(0, in_n)][Slice(0, in_c)][Broadcast()][Broadcast()]);
                window += max_locations *
                        out_dw_swapped[Slice(0, in_n)][Slice(0, in_c)][h][w][Broadcast()][Broadcast()];
            }
        }
    }
    return in_dw;
}


//////////////////////////////////////////////////////////////////////////////////////////////
//                                   TESTS                                                  //
//////////////////////////////////////////////////////////////////////////////////////////////


class ArraySpatialTests : public ::testing::Test,
                          public ::testing::WithParamInterface<bool> {
    bool old_use_cudnn;
    virtual void SetUp() {
        old_use_cudnn = use_cudnn;
        use_cudnn = GetParam();
    }

    virtual void TearDown() {
        use_cudnn = old_use_cudnn;
    }
};

TEST_P(ArraySpatialTests, pool2d_simple) {
    auto X = Array::arange({1,1, 5, 5}, DTYPE_FLOAT);
    Array O = op::pool2d(X, 3, 3, 1, 1, POOLING_T_AVG, PADDING_T_SAME, "NCHW");
    Array O2 = reference_pool2d(X, 3, 3, 1, 1, POOLING_T_AVG, PADDING_T_SAME, "NCHW");


    Array correct_O({1,1,5,5}, DTYPE_FLOAT);
    correct_O[0][0] = std::vector<std::vector<float>> {
       {  3.  ,  3.5   ,4.5   , 5.5 ,  6. },
       {  5.5 ,  6.    ,7.    , 8. ,   8.5},
       { 10.5 , 11.   ,12.   , 13. ,  13.5},
       { 15.5 , 16.   ,17.   , 18. ,  18.5},
       { 18.  , 18.5  ,19.5  , 20.5 , 21. }
    };

    ASSERT_TRUE(Array::allclose(correct_O, O, 1e-3));
    ASSERT_TRUE(Array::allclose(correct_O, O2, 1e-3));

    Array G  = op::pool2d_backward(O, Array::ones_like(O), X, 3, 3, 1, 1, POOLING_T_AVG, PADDING_T_SAME, "NCHW");
    Array G2 = reference_pool2d_backward(O, Array::ones_like(O), X, 3, 3, 1, 1, POOLING_T_AVG, PADDING_T_SAME, "NCHW");

    Array correct_G({1,1,5,5}, DTYPE_FLOAT);
    correct_G[0][0] = std::vector<std::vector<float>> {
        {0.69444448, 0.97222227, 0.83333331,  0.97222227,  0.69444448},
        {0.97222227, 1.36111128, 1.16666675,  1.36111116,  0.97222227},
        {0.83333337, 1.16666675, 1.,          1.16666663,  0.83333337},
        {0.97222227, 1.36111104, 1.16666663,  1.36111116,  0.97222227},
        {0.69444448, 0.97222227, 0.83333337,  0.97222227,  0.69444448}
    };

    ASSERT_TRUE(Array::allclose(correct_G, G, 1e-3));
    ASSERT_TRUE(Array::allclose(correct_G, G2, 1e-3));

}

TEST_P(ArraySpatialTests, conv2d_forward) {
    for (int stride_h = 1; stride_h <= 3; ++stride_h) {
        for (int stride_w = 1; stride_w <= 3; ++stride_w) {
            for (std::string data_format: {"NCHW", "NHWC"}) {
                for (PADDING_T padding : {PADDING_T_VALID, PADDING_T_SAME}) {

                    auto padding_str = (padding == PADDING_T_VALID) ? "valid" : "same";
                    std::string scope_name = utils::make_message(
                        "stride_h = ", stride_h, ", stride_w = ", stride_w, ", "
                        "data_format = ", data_format, ", padding = ", padding_str);
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

                    Array actual = op::conv2d(
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
                    if (!Array::allclose(expected, actual, 1e-3)) {
                        expected.print(); actual.print();
                    }
                    ASSERT_TRUE(Array::allclose(expected, actual, 1e-3));
                }
            }
        }
    }
}

TEST_P(ArraySpatialTests, pool2d_forward) {
    for (int window_h = 1; window_h <= 3; ++window_h) {
        for (int window_w = 1; window_w <= 3; ++window_w) {
            for (int stride_h = 1; stride_h <= 2; ++stride_h) {
                for (int stride_w = 1; stride_w <= 2; ++stride_w) {
                    for (std::string data_format: {"NCHW", "NHWC"}) {
                        for (PADDING_T padding : {PADDING_T_VALID, PADDING_T_SAME}) {
                            for (POOLING_T pooling: {POOLING_T_MAX, POOLING_T_AVG}) {
                                auto padding_str = (padding == PADDING_T_VALID) ? "valid" : "same";
                                auto pooling_str = (pooling == POOLING_T_MAX)   ? "max"   : "avg";
                                std::string scope_name = utils::make_message(
                                    "window_h = ", window_h, ", window_w = ", window_w, ", "
                                    "stride_h = ", stride_h, ", stride_w = ", stride_w, ", "
                                    "data_format = ", data_format, ", padding = ", padding_str, ", "
                                    "pooling = ", pooling_str);
                                SCOPED_TRACE(scope_name);

                                Array X;

                                if (data_format == "NCHW") {
                                    X = Array({5, 3, 6, 8}, DTYPE_FLOAT);
                                } else {
                                    X = Array({5, 6, 8, 3}, DTYPE_FLOAT);
                                }

                                X = initializer::uniform(-1.0, 1.0);

                                Array out = op::pool2d(
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


TEST_P(ArraySpatialTests, conv_backward) {
    Array X = Array::arange({1, 1, 8, 8}, DTYPE_FLOAT);
    Array W = Array::ones({1, 1, 2, 2}, DTYPE_FLOAT);

    Array out = op::conv2d(
        X,
        W,
        /*stride_h=*/2,
        /*stride_w=*/2,
        PADDING_T_VALID,
        "NCHW");

    Array out_dw = Array::ones_like(out);

    Array W_dw = op::conv2d_backward_filters(
        X,
        out_dw,
        /*stride_h=*/2,
        /*stride_w=*/2,
        W.shape(),
        PADDING_T_VALID,
        "NCHW");

    Array in_dw = op::conv2d_backward_input(
        W,
        out_dw,
        /*stride_h=*/2,
        /*stride_w=*/2,
        X.shape(),
        PADDING_T_VALID,
        "NCHW");

    // TODO(szymon): add a test that compares this
    //               to reference implementation
}


TEST_P(ArraySpatialTests, conv_backward_bias) {
    Array X = Array::ones({2, 3, 4, 5}, DTYPE_FLOAT);

    Array out = op::conv2d_backward_bias(X, "NCHW");
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(2 * 4 * 5, (float)out[i]);
    }
}


TEST_P(ArraySpatialTests, unpool2d_forward) {
    for (int window_h = 1; window_h <= 3; ++window_h) {
        for (int window_w = 1; window_w <= 3; ++window_w) {
            for (int stride_h = 1; stride_h <= 2; ++stride_h) {
                for (int stride_w = 1; stride_w <= 2; ++stride_w) {
                    for (std::string data_format: {"NCHW", "NHWC"}) {
                        for (PADDING_T padding : {PADDING_T_VALID, PADDING_T_SAME}) {
                            for (POOLING_T pooling: {POOLING_T_MAX, POOLING_T_AVG}) {
                                auto padding_str = (padding == PADDING_T_VALID) ? "valid" : "same";
                                auto pooling_str = (pooling == POOLING_T_MAX)   ? "max"   : "avg";
                                std::string scope_name = utils::make_message(
                                    "window_h = ", window_h, ", window_w = ", window_w, ", "
                                    "stride_h = ", stride_h, ", stride_w = ", stride_w, ", "
                                    "data_format = ", data_format, ", padding = ", padding_str, ", "
                                    "pooling = ", pooling_str);
                                SCOPED_TRACE(scope_name);
                                Array X;

                                if (data_format == "NCHW") {
                                    X = Array({5, 3, 6, 8}, DTYPE_FLOAT);
                                } else {
                                    X = Array({5, 6, 8, 3}, DTYPE_FLOAT);
                                }

                                X = initializer::uniform(-1.0, 1.0);

                                Array out = op::pool2d(
                                    X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format
                                );

                                Array out_dw = Array::ones_like(out);

                                Array in_dw = op::pool2d_backward(
                                    out,
                                    out_dw,
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

                                Array expected_in_dw = reference_pool2d_backward(
                                    out,
                                    out_dw,
                                    X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format
                                );

                                EXPECT_TRUE(Array::allclose(expected_in_dw, in_dw, 1e-3));
                            }
                        }
                    }
                }
            }
        }
    }
}

#ifdef DALI_USE_CUDNN
INSTANTIATE_TEST_CASE_P(with_cudnn,
                        ArraySpatialTests,
                        ::testing::Values(true));
#endif  // DALI_USE_CUDNN

INSTANTIATE_TEST_CASE_P(with_mshadow,
                        ArraySpatialTests,
                        ::testing::Values(false));
