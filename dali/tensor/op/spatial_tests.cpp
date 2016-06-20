#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>

#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op/spatial.h"

using std::vector;

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
