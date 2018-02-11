#include <gtest/gtest.h>

#include "dali/array/op/conv.h"
#include "dali/array/op/uniform.h"
#include "dali/array/op/arange.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/spatial_utils.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/expression/control_flow.h"

namespace {
    int int_ceil(int numerator, int denominator) {
        return (numerator + denominator - 1) / denominator;
    }

    std::vector<int> permute_shape(const std::vector<int>& shape, const std::string& data_format) {
        ASSERT2(shape.size() == 4, "only 4d tensors are allowed.");
        int n_dim;
        int c_dim;
        int h_dim;
        int w_dim;
        op::check_data_format(data_format,
                          &n_dim,
                          &c_dim,
                          &h_dim,
                          &w_dim);
        std::vector<int> res(4);

        res[n_dim] = shape[0];
        res[c_dim] = shape[1];
        res[h_dim] = shape[2];
        res[w_dim] = shape[3];

        return res;
    }

    Array pad_array(Array in, int prepad_h, int postpad_h, int prepad_w, int postpad_w) {
        int n = in.shape()[0];
        int c = in.shape()[1];
        int h = in.shape()[2];
        int w = in.shape()[3];
        std::vector<int> shape = {n, c, h + prepad_h + postpad_h, w + prepad_w + postpad_w};
        auto out = Array::zeros(shape, in.dtype(), in.preferred_device());
        Array content = out[Slice(0, n)]
                           [Slice(0, c)]
                           [Slice(prepad_h, h + prepad_h)]
                           [Slice(prepad_w, w + prepad_w)];
        return op::control_dependency(
            op::assign(content, OPERATOR_T_EQL, in), out);
    }

    Array reference_conv2d(Array X, Array W,
                           int stride_h, int stride_w,
                           PADDING_T padding,
                           const std::string& data_format) {
        auto info = op::compute_conv2d_info(
                X.shape(),
                W.shape(),
                stride_h,
                stride_w,
                padding,
                data_format);
        int n_dim;
        int c_dim;
        int h_dim;
        int w_dim;
        op::check_data_format(data_format,
                          &n_dim,
                          &c_dim,
                          &h_dim,
                          &w_dim);

        X = X.transpose({n_dim, c_dim, h_dim, w_dim});
        W = W.transpose({n_dim, c_dim, h_dim, w_dim});



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
            {n_dim, c_dim, h_dim, w_dim});

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
                        op::assign(out[n_idx][out_c_idx][h_idx][w_idx], OPERATOR_T_EQL, temp.sum()).eval();
                    }
                }
            }
        }
        return out_orig_data_format;
    }
}

// TODO: add jit reshape, jit transpose, jit swapaxes etc...
// Also add transposed im2col
// figure out why padding SAME is different here from tensorflow

TEST(ConvTests, reference_valid_padding_conv2d) {
    /* Compare with tensorflow's conv2d:
    ```
    session = tf.InteractiveSession()
    x = np.arange(2 * 3 * 7 * 5).reshape((2, 3, 7, 5)).astype(np.float32)
    filters = np.arange(4 * 3 * 6 * 5).reshape((3, 6, 5, 4)).astype(np.float32)
    result = session.run(tf.nn.conv2d(x, filters, (1, 1, 1, 1), "VALID"))
    ```
    */
    int nfilters = 4;
    int in_channels = 5;
    int nbatch = 2;
    int stride_h = 1;
    int stride_w = 1;
    int in_h = 3;
    int in_w = 7;
    int filter_h = 3;
    int filter_w = 6;
    std::vector<float> expected_data = {
        1071960, 1076415, 1080870, 1085325, 1152060, 1156965,
        1161870, 1166775, 2754060, 2767965, 2781870, 2795775,
        2834160, 2848515, 2862870, 2877225};
    auto expected = Array::zeros({nbatch, int_ceil(in_h, filter_h), int_ceil(in_w, filter_w), nfilters}, DTYPE_FLOAT);
    auto data_ptr = static_cast<float*>(expected.memory()->overwrite_data(memory::Device::cpu()));
    for (int i = 0; i < expected_data.size(); i++) {
        data_ptr[i] = expected_data[i];
    }
    auto image = op::arange(nbatch * in_h * in_w * in_channels).reshape({nbatch, in_h, in_w, in_channels}).astype(DTYPE_FLOAT);
    auto filters = op::arange(nfilters * filter_h * filter_w * in_channels).reshape(
        {filter_h, filter_w, in_channels, nfilters}).transpose({3, 0, 1, 2}).astype(DTYPE_FLOAT);
    auto filtered = op::conv2d(image, filters,
        stride_h,
        stride_w,
        PADDING_T_VALID,
        "NHWC");
    // ELOG(filtered.full_expression_name());
    auto ref = reference_conv2d(image, filters, stride_h, stride_w, PADDING_T_VALID,
        "NHWC");
    ASSERT_TRUE(Array::equals(expected, ref));
    ASSERT_TRUE(Array::equals(expected, filtered));
}


TEST(ConvTests, reference_same_padding_conv2d) {
    /* Compare with tensorflow's conv2d:
```
import numpy as np, tensorflow as tf
session = tf.InteractiveSession()
x = np.arange(2 * 3 * 7 * 5).reshape((2, 3, 7, 5)).astype(np.float32)
filters = np.arange(4 * 3 * 6 * 5).reshape((3, 6, 5, 4)).astype(np.float32)
result = session.run(tf.nn.conv2d(x, filters, (1, 1, 1, 1), "SAME"))
```
    */
    int nfilters = 4;
    int in_channels = 5;
    int nbatch = 2;
    int stride_h = 1;
    int stride_w = 1;
    int in_h = 3;
    int in_w = 7;
    int filter_h = 3;
    int filter_w = 6;
    std::vector<float> expected_data = {
        325960, 327040, 328120, 329200, 428700, 430175, 431650, 433125, 537940, 539860, 541780, 543700, 609340, 611560, 613780, 616000, 513200, 515175, 517150, 519125, 413560, 415240, 416920, 418600, 311420, 312755, 314090, 315425, 704640, 707310, 709980, 712650, 888300, 891825, 895350, 898875, 1071960, 1076415, 1080870, 1085325, 1152060, 1156965, 1161870, 1166775, 943800, 948075, 952350, 956625, 740040, 743610, 747180, 750750, 542280, 545070, 547860, 550650, 389560, 392040, 394520, 397000, 475700, 478925, 482150, 485375, 555340, 559360, 563380, 567400, 590740, 595060, 599380, 603700, 465200, 468925, 472650, 476375, 349160, 352240, 355320, 358400, 243620, 246005, 248390, 250775, 1409560, 1414840, 1420120, 1425400, 1730700, 1737425, 1744150, 1750875, 2037340, 2045560, 2053780, 2062000, 2108740, 2117260, 2125780, 2134300, 1710200, 1717425, 1724650, 1731875, 1329160, 1335040, 1340920, 1346800, 966620, 971105, 975590, 980075, 1952040, 1961010, 1969980, 1978950, 2368800, 2380200, 2391600, 2403000, 2754060, 2767965, 2781870, 2795775, 2834160, 2848515, 2862870, 2877225, 2266800, 2278950, 2291100, 2303250, 1735440, 1745310, 1755180, 1765050, 1241580, 1249095, 1256610, 1264125, 969160, 975840, 982520, 989200, 1147700, 1156175, 1164650, 1173125, 1298740, 1309060, 1319380, 1329700, 1334140, 1344760, 1355380, 1366000, 1032200, 1041175, 1050150, 1059125, 760760, 768040, 775320, 782600, 520820, 526355, 531890, 537425};
    auto expected = Array::zeros({nbatch, in_h, in_w, nfilters}, DTYPE_FLOAT);
    auto data_ptr = static_cast<float*>(expected.memory()->overwrite_data(memory::Device::cpu()));
    for (int i = 0; i < expected_data.size(); i++) {
        data_ptr[i] = expected_data[i];
    }
    auto image = op::arange(nbatch * in_h * in_w * in_channels).reshape({nbatch, in_h, in_w, in_channels}).astype(DTYPE_FLOAT);
    auto filters = op::arange(nfilters * filter_h * filter_w * in_channels).reshape(
        {filter_h, filter_w, in_channels, nfilters}).transpose({3, 0, 1, 2}).astype(DTYPE_FLOAT);
    auto filtered = op::conv2d(image, filters,
        stride_h,
        stride_w,
        PADDING_T_SAME,
        "NHWC");
    auto image2 = op::arange(nbatch * in_h * in_w * in_channels).reshape({nbatch, in_h, in_w, in_channels}).astype(DTYPE_FLOAT);
    auto filters2 = op::arange(nfilters * filter_h * filter_w * in_channels).reshape(
        {filter_h, filter_w, in_channels, nfilters}).transpose({3, 0, 1, 2}).astype(DTYPE_FLOAT);
    auto ref = reference_conv2d(image2, filters2, stride_h, stride_w, PADDING_T_SAME,
        "NHWC");
    ASSERT_TRUE(Array::equals(expected, ref));
    ASSERT_TRUE(Array::equals(expected, filtered));
}


TEST(ConvTests, nchw_conv2d_backward) {
    Array X = op::arange(64).reshape({1, 1, 8, 8}).astype(DTYPE_FLOAT);
    Array W = Array::ones({1, 1, 2, 2}, DTYPE_FLOAT);
    Array out = op::conv2d(
        X, W,
        /*stride_h=*/2,
        /*stride_w=*/2,
        PADDING_T_VALID,
        "NCHW");
    out.eval();
    Array out_dw = Array::ones_like(out);
    Array W_dw = op::conv2d_backward_filters(
        X,
        out_dw,
        /*stride_h=*/2,
        /*stride_w=*/2,
        W.shape(),
        PADDING_T_VALID,
        "NCHW");
    W_dw.eval();
    Array in_dw = op::conv2d_backward_input(
        W,
        out_dw,
        /*stride_h=*/2,
        /*stride_w=*/2,
        X.shape(),
        PADDING_T_VALID,
        "NCHW");
    in_dw.eval();
}

TEST(ConvTests, nhwc_conv2d_backward) {
    Array X = op::arange(64).reshape({1, 8, 8, 1}).astype(DTYPE_FLOAT);
    Array W = Array::ones({1, 2, 2, 1}, DTYPE_FLOAT);
    Array out = op::conv2d(
        X, W,
        /*stride_h=*/2,
        /*stride_w=*/2,
        PADDING_T_VALID,
        "NHWC");
    out.eval();
    Array out_dw = Array::ones_like(out);
    Array W_dw = op::conv2d_backward_filters(
        X,
        out_dw,
        /*stride_h=*/2,
        /*stride_w=*/2,
        W.shape(),
        PADDING_T_VALID,
        "NHWC");
    W_dw.eval();
    Array in_dw = op::conv2d_backward_input(
        W,
        out_dw,
        /*stride_h=*/2,
        /*stride_w=*/2,
        X.shape(),
        PADDING_T_VALID,
        "NHWC");
    in_dw.eval();
}

#ifdef DALI_USE_CUDNN
TEST(ArraySpatialTests, small_cudnn_conv2d_forward) {
    for (int stride_h = 1; stride_h <= 1; ++stride_h) {
        for (int stride_w = 1; stride_w <= 1; ++stride_w) {
            for (std::string data_format: {"NCHW", "NHWC"}) {
                for (PADDING_T padding : {PADDING_T_VALID, PADDING_T_SAME}) {

                    auto padding_str = (padding == PADDING_T_VALID) ? "valid" : "same";
                    std::string scope_name = utils::make_message(
                        "stride_h = ", stride_h, ", stride_w = ", stride_w, ", "
                        "data_format = ", data_format, ", padding = ", padding_str);
                    SCOPED_TRACE(scope_name);

                    Array X, W;

                    if (data_format == "NCHW") {
                        X = Array({1, 2, 3, 2}, DTYPE_FLOAT);
                        W = Array({2, 2, 3, 2}, DTYPE_FLOAT);
                    } else {
                        X = Array({1, 3, 2, 2}, DTYPE_FLOAT);
                        W = Array({2, 3, 2, 2}, DTYPE_FLOAT);
                    }

                    X = initializer::uniform(-1.0, 1.0);
                    W = initializer::uniform(-1.0, 1.0);

                    Array actual = op::cudnn_conv2d(
                        X,
                        W,
                        stride_h,
                        stride_w,
                        padding,
                        data_format);

                    // reference computation will be much faster on CPU, methinks.
                    X.to_device(memory::Device::cpu());
                    W.to_device(memory::Device::cpu());

                    Array expected = reference_conv2d(
                        X,
                        W,
                        stride_h,
                        stride_w,
                        padding,
                        data_format);
                    if (!Array::allclose(expected, actual, 1e-3)) {
                        ELOG(data_format);
                        expected.print(); actual.print();
                    }
                    ASSERT_TRUE(Array::allclose(expected, actual, 1e-3));
                }
            }
        }
    }
}

TEST(ArraySpatialTests, cudnn_conv2d_forward) {
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
#endif
