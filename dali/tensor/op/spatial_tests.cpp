#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>

#include "dali/config.h"
#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op/spatial.h"

using std::vector;

typedef MemorySafeTest TensorSpatialTests;

TEST_F(TensorSpatialTests, conv2d_add_bias) {
    EXPERIMENT_REPEAT {
        auto X = Tensor::uniform(10.0, {2, 3, 4, 5}, DTYPE_FLOAT);
        auto b = Tensor::uniform(10.0, {3,},         DTYPE_FLOAT);

        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::conv2d_add_bias(X, b, "NCHW");
        };
        ASSERT_TRUE(gradient_same(functor, {X, b}, 1e-2, 1e-2));
    }
}

TEST_F(TensorSpatialTests, conv2d) {
    for (int stride_h = 1; stride_h <= 2; ++stride_h) {
        for (int stride_w = 1; stride_w <= 2; ++stride_w) {
            for (std::string data_format: {"NCHW", "NHWC"}) {
                // TODO: add support for PADDING_T_SAME
                for (PADDING_T padding : {PADDING_T_VALID, PADDING_T_SAME }) {
                    for (DType dtype : {DTYPE_FLOAT, DTYPE_DOUBLE}) {
                        Tensor X, W;
                        if (data_format == "NCHW") {
                            X = Tensor::uniform(-1.0, 1.0, {5, 3, 6, 8}, dtype);
                            W = Tensor::uniform(-1.0, 1.0, {2, 3, 2, 4}, dtype);
                        } else {
                            X = Tensor::uniform(-1.0, 1.0, {5, 6, 8, 3}, dtype);
                            W = Tensor::uniform(-1.0, 1.0, {2, 2, 4, 3}, dtype);
                        }

                        auto padding_str = (padding == PADDING_T_VALID) ? "valid" : "same";
                        std::string scope_name = utils::MS() << "stride_h = " << stride_h
                                                             << ", stride_w = " << stride_w
                                                             << ", data_format = " << data_format
                                                             << ", padding = " << padding_str
                                                             << ", dtype " << dtype;
                        SCOPED_TRACE(scope_name);
                        auto functor = [&](vector<Tensor> Xs)-> Tensor {
                            return tensor_ops::conv2d(
                                X, W,
                                stride_h,
                                stride_w,
                                padding,
                                data_format);
                        };
                        ASSERT_TRUE(gradient_same(functor, {X, W}, 1e-2, 1e-2));
                    }
                }
            }
        }
    }
}

TEST_F(TensorSpatialTests, pool2d) {
    auto functor = [](vector<Tensor> Xs) -> Tensor {
        return tensor_ops::pool2d(
            Xs[0],
            /*window_h=*/2,
            /*window_w=*/2,
            /*stride_h=*/2,
            /*stride_w=*/2,
            POOLING_T_MAX,
            PADDING_T_VALID,
            "NCHW");
    };

    EXPERIMENT_REPEAT {
        Tensor X = Tensor::arange({1, 1, 8, 8}, DTYPE_FLOAT);
        ASSERT_TRUE(gradient_same(functor, {X}, 1e-3, 1e-2));
    }
}

TEST_F(TensorSpatialTests, im2col_nchw) {
    auto functor = [](vector<Tensor> Xs) -> Tensor {
        return tensor_ops::im2col(
            Xs[0],
            /*filter_h=*/3,
            /*filter_w=*/3,
            /*stride_h=*/1,
            /*stride_w=*/1,
            "NCHW");
    };
    EXPERIMENT_REPEAT {
        Tensor X = Tensor::arange({2, 2, 3, 4}, DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {X}));
    }
}

TEST_F(TensorSpatialTests, im2col_nhwc) {
    auto functor = [](vector<Tensor> Xs) -> Tensor {
        return tensor_ops::im2col(
            Xs[0],
            /*filter_h=*/3,
            /*filter_w=*/3,
            /*stride_h=*/1,
            /*stride_w=*/1,
            "NHWC");
    };
    EXPERIMENT_REPEAT {
        Tensor X = Tensor::arange({2, 3, 4, 2}, DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {X}));
    }
}

TEST_F(TensorSpatialTests, col2im_nchw) {
    auto functor = [](vector<Tensor> Xs) -> Tensor {
        return tensor_ops::col2im(
            Xs[0],
            {2, 2, 3, 4},
            /*filter_h=*/3,
            /*filter_w=*/3,
            /*stride_h=*/1,
            /*stride_w=*/1,
            "NCHW");
    };
    EXPERIMENT_REPEAT {
        Tensor X = Tensor::arange({2 * 3 * 3, 2 * 2}, DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {X}));
    }
}

TEST_F(TensorSpatialTests, col2im_nhwc) {
    auto functor = [](vector<Tensor> Xs) -> Tensor {
        return tensor_ops::col2im(
            Xs[0],
            {2, 3, 4, 2},
            /*filter_h=*/3,
            /*filter_w=*/3,
            /*stride_h=*/1,
            /*stride_w=*/1,
            "NHWC");
    };
    EXPERIMENT_REPEAT {
        Tensor X = Tensor::arange({3 * 3 * 2, 2 * 2}, DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {X}));
    }
}
