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

TEST(TensorCostTests, softmax_axis) {
    int row;
    int axis;
    double temperature;
    auto functor = [&row, &axis, &temperature](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::softmax(
            Xs[0],
            /*axis=*/axis,
            /*temperature=*/temperature
        ).pluck_axis(axis, row);
    };

    EXPERIMENT_REPEAT {
        Tensor A({2, 3, 2}, initializer::uniform(-3.0, 3.0), DTYPE_DOUBLE);
        for (axis = 0; axis < A.ndim(); axis++) {
            for (temperature = 0.5; temperature <= 1.5; temperature += 0.5) {
                row = utils::randint(0, A.shape()[axis] - 1);
                ASSERT_TRUE(gradient_same(functor, {A}, 1e-5, 1e-5));
                A.dw.clear();
            }
        }
    }
}

TEST(TensorCostTests, softmax_noncontig_axis) {
    int row;
    int axis;
    double temperature;
    auto functor = [&row, &axis, &temperature](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::softmax(
            Xs[0],
            /*axis=*/axis,
            /*temperature=*/temperature
        ).pluck_axis(axis, row);
    };

    EXPERIMENT_REPEAT {
        Tensor A({2, 3, 2}, initializer::uniform(-3.0, 3.0), DTYPE_DOUBLE);
        A = A.swapaxes(0, 2);
        for (axis = 0; axis < A.ndim(); axis++) {
            for (temperature = 0.5; temperature <= 1.5; temperature += 0.5) {
                row = utils::randint(0, A.shape()[axis] - 1);
                ASSERT_TRUE(gradient_same(functor, {A}, 1e-5, 1e-5));
                A.dw.clear();
            }
        }
    }
}



TEST(TensorCostTests, cross_entropy_grad_through_target) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::cross_entropy(
            tensor_ops::softmax(Xs[0], /*axis=*/0),
            Xs[1]
        );
    };

    EXPERIMENT_REPEAT {
        Tensor input({10}, initializer::uniform(-2.0, 2.0), DTYPE_DOUBLE);
        Tensor target({10}, initializer::uniform(0.15, 0.85), DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {input, target}, 1e-1));
    }
}

TEST(TensorCostTests, softmax_temperature) {
    graph::NoBackprop nb;

    Tensor logits({10}, initializer::arange(), DTYPE_DOUBLE);
    auto uniform = Tensor::fill_like(1.0 / logits.number_of_elements(), logits);
    // base entropy
    auto kl = (double) tensor_ops::cross_entropy(tensor_ops::softmax(logits), uniform).sum().w;
    // distribution becomes more uniform as temperature rises
    for (int i = 2; i < 11; i++) {
        double temperature = 1.0 * i;
        auto hotter_distribution = tensor_ops::softmax(logits, 0, temperature);
        auto new_kl = (double) tensor_ops::cross_entropy(hotter_distribution, uniform).sum().w;
        ASSERT_LT(new_kl, kl);
        kl = new_kl;
    }
}



