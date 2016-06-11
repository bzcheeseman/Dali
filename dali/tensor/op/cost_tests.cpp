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
        auto A = Tensor::uniform(-5.0, 5.0, {3, 4});
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
        auto A = Tensor::uniform(-3.0, 3.0, {2, 3, 2}, DTYPE_DOUBLE);
        for (axis = 0; axis < A.ndim(); axis++) {
            for (temperature = 0.5; temperature <= 1.5; temperature += 0.5) {
                row = utils::randint(0, A.shape()[axis] - 1);
                ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-3));
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
        auto A = Tensor::uniform(-3.0, 3.0, {2, 3, 2}, DTYPE_DOUBLE);
        A = A.swapaxes(0, 2);
        for (axis = 0; axis < A.ndim(); axis++) {
            for (temperature = 0.5; temperature <= 1.5; temperature += 0.5) {
                row = utils::randint(0, A.shape()[axis] - 1);
                ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-3));
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
        auto input  = Tensor::uniform(-2.0, 2.0,  {10}, DTYPE_DOUBLE);
        auto target = Tensor::uniform(0.15, 0.85, {10}, DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {input, target}, 1e-1));
    }
}

TEST(TensorCostTests, cross_entropy_with_idxes_forward) {
    auto input = Tensor::uniform(0.1, 0.9, {2, 2, 3}, DTYPE_DOUBLE);
    Tensor idxes({2, 2}, DTYPE_INT32);
    idxes.w = vector<vector<int>>({
        {2, 1},
        {0, 2},
    });

    auto res = tensor_ops::cross_entropy(input, idxes);

    EXPECT_NEAR((double)res[0][0].w, -std::log((double)input[0][0][2].w), 1e-3);
    EXPECT_NEAR((double)res[0][1].w, -std::log((double)input[0][1][1].w), 1e-3);
    EXPECT_NEAR((double)res[1][0].w, -std::log((double)input[1][0][0].w), 1e-3);
    EXPECT_NEAR((double)res[1][1].w, -std::log((double)input[1][1][2].w), 1e-3);
}

TEST(TensorCostTests, cross_entropy_with_idxes) {
    EXPERIMENT_REPEAT {
        auto input = Tensor::uniform(0.1, 0.9, {2, 2, 3}, DTYPE_DOUBLE);
        auto idxes = Tensor::uniform(0,   2,   {2, 2},    DTYPE_INT32);

        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::cross_entropy(input, idxes);
        };

        ASSERT_TRUE(gradient_same(functor, {input}, 1e-3));
    }
}

TEST(TensorCostTests,  softmax_cross_entropy_with_probs) {
    EXPERIMENT_REPEAT {
        auto input   = Tensor::uniform(-10, 10.0, {2, 3}, DTYPE_DOUBLE);
        auto targets = Tensor::uniform(0.1, 0.9, {2, 3}, DTYPE_DOUBLE);
        double temperature = utils::randdouble(0.1, 10.0);


        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::softmax_cross_entropy(input, targets, temperature);
        };

        ASSERT_TRUE(gradient_same(functor, {input, targets}, 1e-3));
    }
}

TEST(TensorCostTests,  softmax_cross_entropy_with_idxes) {
    EXPERIMENT_REPEAT {
        auto input = Tensor::uniform(-10, 10.0, {2, 3}, DTYPE_DOUBLE);
        auto idxes = Tensor::uniform(0, 2,      {2},    DTYPE_INT32);
        double temperature = utils::randdouble(0.1, 10.0);

        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::softmax_cross_entropy(input, idxes, temperature);
        };

        ASSERT_TRUE(gradient_same(functor, {input}, 1e-3));
    }
}


TEST(TensorCostTests, softmax_temperature) {
    graph::NoBackprop nb;

    Tensor logits({10}, DTYPE_DOUBLE);
    logits.w = initializer::arange();
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
