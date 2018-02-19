#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>
#include <vector>

#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op.h"

using std::vector;

typedef MemorySafeTest TensorCompositeTests;

TEST_F(TensorCompositeTests, DISABLED_matrix_dot_with_bias) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::dot_with_bias(Xs[0], Xs[1], Xs[2]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        auto X = Tensor::uniform(-10, 10, {num_examples, input_size});
        auto W = Tensor::uniform(-10, 10, {input_size, hidden_size});
        auto bias = Tensor::uniform(-2, 2,  {hidden_size})[Broadcast()];
        ASSERT_TRUE(gradient_same(functor, {X, W, bias}, 1e-4));
    }
}

TEST_F(TensorCompositeTests, DISABLED_matrix_multiple_dot_with_bias) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::multiple_dot_with_bias({Xs[0], Xs[2]}, {Xs[1], Xs[3]}, Xs[4]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    int other_input_size = 7;
    EXPERIMENT_REPEAT {
        auto X = Tensor::uniform(-10, 10, {num_examples, input_size});
        auto W = Tensor::uniform(-10, 10, {input_size, hidden_size});

        auto X_other = Tensor::uniform(-10, 10, {num_examples, other_input_size});
        auto W_other = Tensor::uniform(-10, 10, {other_input_size, hidden_size});

        auto bias = Tensor::uniform(-2, 2,  {hidden_size})[Broadcast()];
        ASSERT_TRUE(gradient_same(functor, { X, W, X_other, W_other, bias}, 0.0003));
    }
}

TEST_F(TensorCompositeTests, DISABLED_matrix_multiple_dot_with_bias_fancy_broadcast) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::multiple_dot_with_bias({Xs[0], Xs[2], Xs[4]}, {Xs[1], Xs[3], Xs[5]}, Xs[6]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    int other_input_size = 7;
    EXPERIMENT_REPEAT {
        auto X = Tensor::uniform(-10, 10, {num_examples, input_size});
        auto W = Tensor::uniform(-10, 10, {input_size, hidden_size});

        Tensor X_fancy   = Tensor::uniform(-10, 10, {input_size})[Broadcast()];
        auto W_fancy = Tensor::uniform(-10, 10, {input_size, hidden_size});

        auto X_other = Tensor::uniform(-10, 10, {num_examples, other_input_size});
        auto W_other = Tensor::uniform(-10, 10, {other_input_size, hidden_size});

        auto bias = Tensor::uniform(-2, 2,  {hidden_size})[Broadcast()];
        ASSERT_TRUE(gradient_same(functor, {X, W, X_fancy, W_fancy, X_other, W_other, bias}, 0.0003));
    }
}

TEST_F(TensorCompositeTests, DISABLED_quadratic_form) {
    EXPERIMENT_REPEAT {
        auto left = Tensor::uniform(-20.0, 20.0, {2, 4});
        auto middle = Tensor::uniform(-20.0, 20.0, {2, 3});
        auto right = Tensor::uniform(-20.0, 20.0, {3, 5});

        auto functor = [](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::quadratic_form(Xs[0], Xs[1], Xs[2]);
        };
        ASSERT_TRUE(gradient_same(functor, {left, middle, right}, 1e-3));
    }
}

TEST_F(TensorCompositeTests, DISABLED_quadratic_form_with_3D_input) {
    //TODO(jonathan): quadratic form in 3D / N-D suffers from weird LDA to dgemm
    EXPERIMENT_REPEAT {
        auto left = Tensor::uniform(-20.0, 20.0, {2, 4, 1});
        auto middle = Tensor::uniform(-20.0, 20.0, {2, 3});
        auto right = Tensor::uniform(-20.0, 20.0, {3, 1});

        auto functor = [](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::quadratic_form(Xs[0], Xs[1], Xs[2]);
        };
        ASSERT_TRUE(gradient_same(functor, {left, middle, right}, 1e-3));
    }
}

TEST_F(TensorCompositeTests, DISABLED_matrix_multiple_dot_with_bias_mini) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::multiple_dot_with_bias({Xs[0]}, {Xs[1]}, Xs[2]);
    };

    EXPERIMENT_REPEAT {
        auto X = Tensor::uniform(-10, 10, {4})[Broadcast()];
        auto W = Tensor::uniform(-10, 10, {4, 5});

        auto bias = Tensor::uniform(-2, 2,  {5})[Broadcast()];
        ASSERT_TRUE(gradient_same(functor, { X, W, bias}, 0.0003));
    }
}
