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

TEST(TensorDotTests, dot_2D) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::dot(Xs[0], Xs[1]);
    };
    EXPERIMENT_REPEAT {
        auto A = Tensor({3, 4}, initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);
        auto B = Tensor({4, 3},  initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-4, DEFAULT_GRAD_EPS, true));
    }

}

TEST(TensorDotTests, dot_3D) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::dot(Xs[0], Xs[1]);
    };

    EXPERIMENT_REPEAT {
        auto A = Tensor({3, 5, 7}, initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);
        auto B = Tensor({1, 7, 3}, initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-4, DEFAULT_GRAD_EPS, true));
    }
}
