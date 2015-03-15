#include <chrono>
#include <vector>
#include <gtest/gtest.h>
#include <Eigen/Eigen>

#include "dali/mat/Mat.h"
#include "dali/mat/MatOps.h"

using std::vector;
using std::chrono::milliseconds;

typedef double R;

#define NUM_RETRIES 100
#define EXPERIMENT_REPEAT for(int __repetition=0; __repetition < NUM_RETRIES; ++__repetition)

template<typename T, typename K>
bool matrix_equals (const T& A, const K& B) {
    return (A.array() == B.array()).all();
}

template<typename R>
bool matrix_equals (Mat<R> A, Mat<R> B) {
    return (A.w().array() == B.w().array()).all();
}

template<typename T, typename K, typename J>
bool matrix_almost_equals (const T& A, const K& B, J eps) {
    return (A.array() - B.array()).abs().array().maxCoeff() < eps;
}

template<typename R>
bool matrix_almost_equals (Mat<R> A, Mat<R> B, R eps) {
    return (A.w().array() - B.w().array()).abs().array().maxCoeff() < eps;
}

#define ASSERT_MATRIX_EQ(A, B) ASSERT_TRUE(matrix_equals((A), (B)))
#define ASSERT_MATRIX_NEQ(A, B) ASSERT_FALSE(matrix_equals((A), (B)))
#define ASSERT_MATRIX_CLOSE(A, B, eps) ASSERT_TRUE(matrix_almost_equals((A), (B), (eps)))

TEST(Matrix, eigen_addition) {
    auto A = Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>(10, 20);
    A.fill(0);
    A.array() += 1;
    auto B = Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>(10, 20);
    B.fill(0);
    ASSERT_MATRIX_EQ(A, A)  << "A equals A.";
    ASSERT_MATRIX_NEQ(A, B) << "A different from B.";
}

TEST(Matrix, addition) {
    auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
    auto B = Mat<R>(10, 20, weights<R>::uniform(2.0));
    ASSERT_MATRIX_EQ(A, A)  << "A equals A.";
    ASSERT_MATRIX_NEQ(A, B) << "A different from B.";
}

/**
Gradient Same
-------------

Numerical gradient checking method. Performs
a finite difference estimation of the gradient
over each argument to a functor.

**/
template<typename R>
bool gradient_same(
        std::function<Mat<R>(std::vector<Mat<R>>)> functor,
        std::vector<Mat<R>> arguments,
        R tolerance    = 1e-5,
        R grad_epsilon = 1e-9) {

    auto error = functor(arguments).sum();
    error.grad();
    graph::backward();

    bool worked_out = true;

    // from now on gradient is purely numerical:
    graph::NoBackprop nb;

    for (auto& arg : arguments) {
        auto Arg_prime = Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>(arg.dims(0), arg.dims(1));
        for (int i = 0; i < arg.dims(0); i++) {
            for (int j = 0; j < arg.dims(1); j++) {
                auto prev_val = arg.w()(i, j);
                arg.w()(i, j) = prev_val +  grad_epsilon;
                auto obj_positive = functor(arguments).w().array().sum();
                arg.w()(i, j) = prev_val - grad_epsilon;
                auto obj_negative = functor(arguments).w().array().sum();
                arg.w()(i, j) = prev_val;
                Arg_prime(i,j) = (obj_positive - obj_negative) / (2.0 * grad_epsilon);
            }
        }

        worked_out = worked_out && matrix_almost_equals(Arg_prime, arg.dw(), tolerance);
        if (!worked_out) {
            break;
        }
    }

    return worked_out;
}

TEST(Matrix, sum_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].sum();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same<R>(functor, {A}));
    }
}

TEST(Matrix, addition_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] + Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        auto B = Mat<R>(10, 20,  weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same<R>(functor, {A, B}));
    }
}

TEST(Matrix, addition_broadcast_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] + Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        auto B = Mat<R>(10, 1,  weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same<R>(functor, {A, B}));
    }
}

TEST(Matrix, mean_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].mean();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same<R>(functor, {A}));
    }
}

TEST(Matrix, sigmoid_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].sigmoid();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same<R>(functor, {A}, 1e-4));
    }
}

TEST(Matrix, tanh_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].tanh();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same<R>(functor, {A}, 1e-4));
    }
}

TEST(Matrix, exp_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].tanh();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same<R>(functor, {A}, 1e-4));
    }
}

TEST(Matrix, log_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].tanh();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.001, 20.0));
        ASSERT_TRUE(gradient_same<R>(functor, {A}, 1e-4));
    }
}

TEST(Matrix, matrix_dot_plus_bias) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[1].dot(Xs[0]) + Xs[2];
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        auto X = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
        auto W = Mat<R>(hidden_size, input_size, weights<R>::uniform(2.0));
        auto bias = Mat<R>(hidden_size, 1, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same<R>(functor, {X, W, bias}, 1e-4));
    }
}

TEST(Matrix, matrix_mul_with_bias) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::mul_with_bias(Xs[1], Xs[0], Xs[2]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        auto X = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
        auto W = Mat<R>(hidden_size, input_size, weights<R>::uniform(2.0));
        auto bias = Mat<R>(hidden_size, 1, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same<R>(functor, {X, W, bias}, 1e-4));
    }
}

TEST(Matrix, matrix_mul_add_mul_with_bias) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::mul_add_mul_with_bias(Xs[0], Xs[1], Xs[2], Xs[3], Xs[4]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    int other_input_size = 7;
    EXPERIMENT_REPEAT {
        auto X       = Mat<R>(input_size, num_examples,      weights<R>::uniform(20.0));
        auto X_other = Mat<R>(other_input_size, num_examples,      weights<R>::uniform(20.0));
        auto W       = Mat<R>(hidden_size, input_size,       weights<R>::uniform(2.0));
        auto W_other = Mat<R>(hidden_size, other_input_size, weights<R>::uniform(2.0));
        auto bias    = Mat<R>(hidden_size, 1,                weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same<R>(functor, {W, X, W_other, X_other, bias}, 0.0003));
    }
}
