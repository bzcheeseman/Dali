#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>
#include <vector>

#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op/cost.h"
#include "dali/array/op/initializer.h"

using std::vector;

TEST(TesnorCostTests, binary_cross_entropy) {
    // We observe the KL divergence to 0 or 1 for each unit
    // in our input matrix with respect to the target.
    EXPERIMENT_REPEAT {
        auto A = Tensor::uniform(0.1, 0.9, {10, 20}, DTYPE_DOUBLE);
        R target = utils::randdouble(0.01, 0.99);
        auto functor = [target](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::binary_cross_entropy(Xs[0], target);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-2));
    }
}

TEST(TesnorCostTests, binary_cross_entropy_matrix_target) {
    // We observe the KL divergence to 0 or 1 for each unit
    // in our input matrix with respect to the target.
    EXPERIMENT_REPEAT {
        auto A      = Tensor::uniform(0.1, 0.9, {10, 20}, DTYPE_DOUBLE);
        auto target = Tensor::uniform(0.1, 0.9, {10, 20}, DTYPE_DOUBLE);
        target.constant = true;
        auto functor = [target](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::binary_cross_entropy(Xs[0], target);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-2));
    }
}
