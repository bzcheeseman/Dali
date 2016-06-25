#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>

#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op/dot.h"
#include "dali/array/op/initializer.h"

using std::vector;

typedef MemorySafeTest TensorDotTests;

TEST_F(TensorDotTests, dot_2D) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::dot(Xs[0], Xs[1]);
    };
    EXPERIMENT_REPEAT {
        auto A = Tensor::uniform(-1.0, 1.0, {3, 4}, DTYPE_DOUBLE);
        auto B = Tensor::uniform(-1.0, 1.0, {4, 3}, DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-4, DEFAULT_GRAD_EPS, true));
    }

}

TEST_F(TensorDotTests, dot_3D) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::dot(Xs[0], Xs[1]);
    };

    EXPERIMENT_REPEAT {
        auto A = Tensor::uniform(-1.0, 1.0, {3, 5, 7}, DTYPE_DOUBLE);
        auto B = Tensor::uniform(-1.0, 1.0, {1, 7, 3}, DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-4, DEFAULT_GRAD_EPS, true));
    }
}
TEST_F(TensorDotTests, self_dot) {
    EXPERIMENT_REPEAT {
        auto W = Tensor::uniform(-20.0, 20.0, {5, 5}, DTYPE_DOUBLE);

        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return W.dot(W.transpose());
        };

        ASSERT_TRUE(gradient_same(functor, {W}, 1e-4, DEFAULT_GRAD_EPS, true));
    }
}
