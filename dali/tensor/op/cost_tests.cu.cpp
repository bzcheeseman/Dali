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

TEST(TensorCostTests, binary_cross_entropy) {
    // We observe the KL divergence to 0 or 1 for each unit
    // in our input matrix with respect to the target.
    EXPERIMENT_REPEAT {
        auto A = Tensor::uniform(0.1, 0.9, {3, 4}, DTYPE_DOUBLE);
        double target = utils::randdouble(0.01, 0.99);
        auto functor = [target](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::binary_cross_entropy(Xs[0], target);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-2));
    }
}

TEST(TensorCostTests, binary_cross_entropy_matrix_target) {
    // We observe the KL divergence to 0 or 1 for each unit
    // in our input matrix with respect to the target.
    EXPERIMENT_REPEAT {
        auto A      = Tensor::uniform(0.1, 0.9, {3, 4}, DTYPE_DOUBLE);
        auto target = Tensor::uniform(0.1, 0.9, {3, 4}, DTYPE_DOUBLE);
        auto functor = [target](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::binary_cross_entropy(Xs[0], target);
        };
        ASSERT_TRUE(gradient_same(functor, {A, target}, 1e-2));
    }
}

TEST(TensorCostTests, sigmoid_binary_cross_entropy) {
    // We observe the KL divergence to 0 or 1 for each unit
    // in our input matrix with respect to the target.
    EXPERIMENT_REPEAT {
        auto A = Tensor::uniform(-5.0, 5.0, {3, 4}, DTYPE_DOUBLE);
        double target = utils::randdouble(0.01, 0.99);
        auto functor = [target](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::sigmoid_binary_cross_entropy(Xs[0], target);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-2));
    }
}

TEST(TensorCostTests, sigmoid_binary_cross_entropy_matrix_target) {
    // We observe the KL divergence to 0 or 1 for each unit
    // in our input matrix with respect to the target.
    EXPERIMENT_REPEAT {
        auto A      = Tensor::uniform(-5.0, 5.0, {3, 4}, DTYPE_DOUBLE);
        auto target = Tensor::uniform(0.1, 0.9, {3, 4}, DTYPE_DOUBLE);
        auto functor = [target](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::sigmoid_binary_cross_entropy(Xs[0], target);
        };
        ASSERT_TRUE(gradient_same(functor, {A, target}, 1e-2));
    }
}

TEST(TensorCostTests, DISABLED_margin_loss_colwise) {
    utils::random::set_seed(100);
    // we can now extend the range of our random numbers to be beyond
    // 0 and 1 since sigmoid will clamp them to 0 or 1.
    EXPERIMENT_REPEAT {
        Tensor A({3, 4}, initializer::uniform(-5.0, 5.0));
        double margin = utils::randdouble(0.01, 0.1);
        int target = utils::randint(0, 2);
        auto functor = [target, margin](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::margin_loss(Xs[0], target, margin, 0);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-4));
    }
    utils::random::reseed();
}
