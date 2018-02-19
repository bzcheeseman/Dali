#include <gtest/gtest.h>

#include "dali/array/tests/test_utils.h"
#include "dali/array/op/conv.h"
#include "dali/array/op/random.h"
#include "dali/array/op/arange.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/spatial_utils.h"
#include "dali/array/cudnn/conv.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/expression/control_flow.h"
#include "dali/utils/make_message.h"

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

class ConvTests : public ::testing::TestWithParam<bool> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.
    public:
        bool use_cudnn() const {return GetParam();}
};

TEST_P(ConvTests, small_conv2d_forward) {
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
                        X = op::uniform(-1.0f, 1.0f, {1, 2, 3, 2});
                        W = op::uniform(-1.0f, 1.0f, {2, 2, 3, 2});
                    } else {
                        X = op::uniform(-1.0f, 1.0f, {1, 3, 2, 2});
                        W = op::uniform(-1.0f, 1.0f, {2, 3, 2, 2});
                    }
                    // TODO(jonathan) don't resample:
                    X.eval();
                    W.eval();
                    Array actual;
                    if (use_cudnn()) {
                        actual = op::cudnn_conv2d(X,
                        W,
                        stride_h,
                        stride_w,
                        padding,
                        data_format);
                    } else {
                        actual = op::conv2d(X,
                        W,
                        stride_h,
                        stride_w,
                        padding,
                        data_format);
                    }
                        

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
                        expected.print(); actual.print();
                    }
                    ASSERT_TRUE(Array::allclose(expected, actual, 1e-3));
                }
            }
        }
    }
}

TEST_P(ConvTests, conv2d_forward) {
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
                        X = op::uniform(-1.0f, 1.0f, {5, 3, 6, 8});
                        W = op::uniform(-1.0f, 1.0f, {2, 3, 2, 4});
                    } else {
                        X = op::uniform(-1.0f, 1.0f, {5, 6, 8, 3});
                        W = op::uniform(-1.0f, 1.0f, {2, 2, 4, 3});
                    }
                    X.eval();
                    W.eval();
                    Array actual;
                    if (use_cudnn()) {
                        actual = op::cudnn_conv2d(X,
                        W,
                        stride_h,
                        stride_w,
                        padding,
                        data_format);
                    } else {
                        actual = op::conv2d(X,
                        W,
                        stride_h,
                        stride_w,
                        padding,
                        data_format);
                    }

                    // reference computation could be much faster on CPU.
                    X.to_device(memory::Device::cpu());
                    W.to_device(memory::Device::cpu());

                    Array expected = reference_conv2d(X,
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

Array reference_pool2d(Array X,
                       int window_h,
                       int window_w,
                       int stride_h,
                       int stride_w,
                       POOLING_T pooling_mode,
                       PADDING_T padding_mode,
                       const std::string& data_format) {
    ASSERT2(X.shape().size() == 4, "must be a 4D array");

    auto info = op::compute_pool_info(
            X.shape(),
            window_h,
            window_w,
            stride_h,
            stride_w,
            padding_mode,
            data_format);
    int n_dim, c_dim, h_dim, w_dim;
    op::check_data_format(data_format, &n_dim, &c_dim, &h_dim, &w_dim);
    X = X.transpose({n_dim, c_dim, h_dim, w_dim});
    int unpadded_x_h = X.shape()[2], unpadded_x_w = X.shape()[3];
    int prepad_h = info.padding_h, prepad_w = info.padding_w;

    auto out_shape = permute_shape({info.batch_size,
                                    info.in_channels,
                                    info.out_h,
                                    info.out_w},
                                    data_format);

    auto out_orig_data_format = Array(out_shape, X.dtype(), X.preferred_device());
    Array out = out_orig_data_format.transpose({n_dim, c_dim, h_dim, w_dim});

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
            out[Slice(0, in_n)][Slice(0, in_c)][i][j].assign(
                pooling_mode == POOLING_T_MAX ? window.max({-1}) : window.mean({-1})).eval();
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

    auto info = op::compute_pool_info(
            X.shape(),
            window_h,
            window_w,
            stride_h,
            stride_w,
            padding_mode,
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
    Array in_dw = Array::zeros_like(X);

    auto x_swapped      =      X.transpose({n_dim, c_dim, h_dim, w_dim});
    auto out_swapped    =    out.transpose({n_dim, c_dim, h_dim, w_dim});
    auto out_dw_swapped = out_dw.transpose({n_dim, c_dim, h_dim, w_dim});
    auto in_dw_swapped  =  in_dw.transpose({n_dim, c_dim, h_dim, w_dim});

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
            window.eval();
        }
    }
    return in_dw;
}

Array reference_pool2d_backward_result_view(Array out,
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

    auto info = op::compute_pool_info(
            X.shape(),
            window_h,
            window_w,
            stride_h,
            stride_w,
            padding_mode,
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
    Array in_dw = Array::zeros_like(X);

    auto x_swapped      =      X.transpose({n_dim, c_dim, h_dim, w_dim});
    auto out_swapped    =    out.transpose({n_dim, c_dim, h_dim, w_dim});
    auto out_dw_swapped = out_dw.transpose({n_dim, c_dim, h_dim, w_dim});
    auto in_dw_swapped  =  in_dw.transpose({n_dim, c_dim, h_dim, w_dim});

    int out_h = out_swapped.shape()[2];
    int out_w = out_swapped.shape()[3];

    int in_n  = in_dw_swapped.shape()[0];
    int in_c  = in_dw_swapped.shape()[1];
    int in_h  = in_dw_swapped.shape()[2];
    int in_w  = in_dw_swapped.shape()[3];

    int prepad_h = info.padding_h;
    int prepad_w = info.padding_w;
    

    for (int h = 0; h < in_h; h++) {
        for (int w = 0; w < in_w; w++) {
            Array gradient = in_dw_swapped[Slice(0, in_n)][Slice(0, in_c)][h][w];

            const int phstart = (h + info.padding_h < window_h) ? 0 : (h + info.padding_h - window_h) / stride_h + 1;
            const int pwstart = (w + info.padding_w < window_w) ? 0 : (w + info.padding_w - window_w) / stride_w + 1;

            const int phend = std::min((h + info.padding_h) / stride_h + 1, out_h);
            const int pwend = std::min((w + info.padding_w) / stride_w + 1, out_w);
            
            // for all batch and channel for this gradient element find relevant chunks
            for (int ph = phstart; ph < phend; ++ph) {
              for (int pw = pwstart; pw < pwend; ++pw) {
                // figure out the pooling size
                int hstart = ph * stride_h - info.padding_h;
                int wstart = pw * stride_w - info.padding_w;
                int hend = std::min(hstart + window_h, out_h);
                int wend = std::min(wstart + window_w, out_w);
                hstart = std::max(hstart, 0);
                wstart = std::max(wstart, 0);
                int pool_size = (hend - hstart) * (wend - wstart);
                gradient += out_dw_swapped[Slice(0, in_n)][Slice(0, in_c)][ph][pw] / pool_size;
              }
            }
            gradient.eval();
        }
    }
    return in_dw;
}

TEST_P(ConvTests, pool2d_forward) {
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
                                    X = op::uniform(-1.0f, 1.0f, {5, 3, 6, 8});
                                } else {
                                    X = op::uniform(-1.0f, 1.0f, {5, 6, 8, 3});
                                }
                                X.eval();

                                Array out;
                                if (use_cudnn()) {
                                    out = op::cudnn_pool2d(X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format);
                                } else {
                                    out = op::pool2d(X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format);
                                }

                                X.to_device(memory::Device::cpu());

                                Array expected_out = reference_pool2d(
                                    X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format);

                                // if (!Array::allclose(expected_out, out, 1e-3)) {
                                //     X.print();
                                //     expected_out.print();
                                //     out.print();
                                // }

                                // ASSERT_TRUE(Array::allclose(expected_out, out, 1e-3));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_P(ConvTests, conv_backward_bias) {
    Array X = Array::ones({2, 3, 4, 5}, DTYPE_FLOAT);
    Array out;
    if (use_cudnn()) {
        out = op::cudnn_conv2d_backward_bias(X, "NCHW");
    } else {
        out = op::conv2d_backward_bias(X, "NCHW");
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(2 * 4 * 5, (float)out[i]);
    }
}

TEST_P(ConvTests, unpool2d_forward) {
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
                                    X = op::uniform(-1.0f, 1.0f, {5, 3, 6, 8});
                                } else {
                                    X = op::uniform(-1.0f, 1.0f, {5, 6, 8, 3});
                                }
                                X.eval();

                                Array out;
                                if (use_cudnn()) {
                                    out = op::cudnn_pool2d(
                                    X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format);
                                } else {
                                    out = op::pool2d(
                                    X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format);
                                }


                                Array out_dw = Array::ones_like(out);

                                Array in_dw;
                                if (use_cudnn()) {
                                    in_dw = op::cudnn_pool2d_backward(
                                    out,
                                    out_dw,
                                    X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format);
                                } else {
                                    in_dw = op::pool2d_backward(
                                    out,
                                    out_dw,
                                    X,
                                    /*window_h=*/window_h,
                                    /*window_w=*/window_w,
                                    /*stride_h=*/stride_h,
                                    /*stride_w=*/stride_w,
                                    pooling,
                                    padding,
                                    data_format);
                                }

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
                                    data_format);
                                ASSERT_TRUE(Array::allclose(expected_in_dw, in_dw, 1e-3));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_P(ConvTests, pool2d_simple) {
    POOLING_T pooling_mode = POOLING_T_AVG;
    auto X = op::arange(25).reshape({1,1, 5, 5}).astype(DTYPE_FLOAT);
    Array O;
    if (use_cudnn()) {
        O = op::cudnn_pool2d(X, 3, 3, 1, 1, pooling_mode, PADDING_T_SAME, "NCHW");
    } else {
        O = op::pool2d(X, 3, 3, 1, 1, pooling_mode, PADDING_T_SAME, "NCHW");
    }
    Array O2 = reference_pool2d(X, 3, 3, 1, 1, pooling_mode, PADDING_T_SAME, "NCHW");


    Array correct_O({1,1,5,5}, DTYPE_FLOAT);
    assign_from_vec(correct_O[0][0], std::vector<std::vector<float>> {
       {  3.  ,  3.5   ,4.5   , 5.5 ,  6. },
       {  5.5 ,  6.    ,7.    , 8. ,   8.5},
       { 10.5 , 11.   ,12.   , 13. ,  13.5},
       { 15.5 , 16.   ,17.   , 18. ,  18.5},
       { 18.  , 18.5  ,19.5  , 20.5 , 21. }
    });
    ASSERT_TRUE(Array::allclose(correct_O, O, 1e-3));
    ASSERT_TRUE(Array::allclose(correct_O, O2, 1e-3));
    Array G;
    if (use_cudnn()) {
        G = op::cudnn_pool2d_backward(O, Array::ones_like(O), X, 3, 3, 1, 1, pooling_mode, PADDING_T_SAME, "NCHW");
    } else {
        G = op::pool2d_backward(O, Array::ones_like(O), X, 3, 3, 1, 1, pooling_mode, PADDING_T_SAME, "NCHW");
    }
    Array G2 = reference_pool2d_backward_result_view(O, Array::ones_like(O), X, 3, 3, 1, 1, pooling_mode, PADDING_T_SAME, "NCHW");
    Array correct_G({1,1,5,5}, DTYPE_FLOAT);
    assign_from_vec(correct_G[0][0], std::vector<std::vector<float>> {
        {0.69444448, 0.97222227, 0.83333331,  0.97222227,  0.69444448},
        {0.97222227, 1.36111128, 1.16666675,  1.36111116,  0.97222227},
        {0.83333337, 1.16666675, 1.,          1.16666663,  0.83333337},
        {0.97222227, 1.36111104, 1.16666663,  1.36111116,  0.97222227},
        {0.69444448, 0.97222227, 0.83333337,  0.97222227,  0.69444448}
    });
    ASSERT_TRUE(Array::allclose(correct_G, G, 1e-3));
    ASSERT_TRUE(Array::allclose(correct_G, G2, 1e-3));
}

#ifdef DALI_USE_CUDNN
INSTANTIATE_TEST_CASE_P(Cudnn,
                        ConvTests,
                        ::testing::Values(true));
#endif
INSTANTIATE_TEST_CASE_P(Blas,
                        ConvTests,
                        ::testing::Values(false));
