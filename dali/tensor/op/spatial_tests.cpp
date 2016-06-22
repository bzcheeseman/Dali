#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>

#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op/spatial.h"

using std::vector;

TEST(TensorSpatialTests, conv2d_add_bias) {
    EXPERIMENT_REPEAT {
        auto X = Tensor::uniform(10.0, {2, 3, 4, 5}, DTYPE_FLOAT);
        auto b = Tensor::uniform(10.0, {3,},         DTYPE_FLOAT);

        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::conv2d_add_bias(X, b, "NCHW");

        };
        ASSERT_TRUE(gradient_same(functor, {X, b}, 1e-2, 1e-2));
    }
}

TEST(TensorSpatialTests, conv2d) {
    EXPERIMENT_REPEAT {
        auto X = Tensor::arange({1, 1, 8, 8}, DTYPE_DOUBLE);
        auto W = Tensor::ones({1, 1, 2, 2}, DTYPE_DOUBLE);

        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::conv2d(
                X, W,
                2, 2,
                PADDING_T_VALID,
                "NCHW");
        };
        ASSERT_TRUE(gradient_same(functor, {X,W}, 1e-3, 1e-2));
    }
}

TEST(TensorSpatialTests, pool2d) {
    EXPERIMENT_REPEAT {
        Tensor X = Tensor::arange({1, 1, 8, 8}, DTYPE_FLOAT);

        auto functor = [&](vector<Tensor> Xs) -> Tensor {
            return tensor_ops::pool2d(
                X,
                /*window_h=*/2,
                /*window_w=*/2,
                /*stride_h=*/2,
                /*stride_w=*/2,
                POOLING_T_MAX,
                PADDING_T_VALID,
                "NCHW");
        };

        ASSERT_TRUE(gradient_same(functor, {X}, 1e-3, 1e-2));
    }
}
