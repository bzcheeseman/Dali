#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>

#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op/binary.h"
#include "dali/array/op/initializer.h"

using std::vector;


void test_binary_function(std::function<Tensor(std::vector<Tensor>&)> functor) {
    EXPERIMENT_REPEAT {
        auto A = Tensor({10, 20}, initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);
        auto B = Tensor({10, 20}, initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);

        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-5, DEFAULT_GRAD_EPS, true));
    }
}

TEST(TensorBinaryTests, add) {
    test_binary_function([](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::add(Xs[0], Xs[1]);
    });
}

TEST(TensorBinaryTests, sub) {
    test_binary_function([](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::sub(Xs[0], Xs[1]);
    });
}

TEST(TensorBinaryTests, eltmul) {
    test_binary_function([](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::eltmul(Xs[0], Xs[1]);
    });
}

TEST(TensorBinaryTests, eltdiv) {
    test_binary_function([](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::eltmul(Xs[0], Xs[1]);
    });
}

TEST(TensorBinaryTests, pow) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::pow(Xs[0], Xs[1]);
    };

    EXPERIMENT_REPEAT {
        auto A = Tensor({10, 20}, initializer::uniform(0.5, 1.0), DTYPE_DOUBLE);
        auto B = Tensor({10, 20},  initializer::uniform(0.5, 1.0), DTYPE_DOUBLE);

        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-4, DEFAULT_GRAD_EPS, true));
    }
}

TEST(TensorBinaryTests, add_recursive) {
    auto functor = [](vector<Tensor>& Xs)-> Tensor {
        return tensor_ops::add(Xs[0], Xs[0]);
    };
    EXPERIMENT_REPEAT {
        auto A = Tensor({10, 20}, initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, DEFAULT_GRAD_EPS, true));
    }
}

TEST(TensorBinaryTests, DISABLED_dot) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::dot(Xs[0], Xs[1]);
    };
    EXPERIMENT_REPEAT {
        auto A = Tensor({3, 4}, initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);
        auto B = Tensor({4, 3},  initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-4, DEFAULT_GRAD_EPS, true));
    }

    EXPERIMENT_REPEAT {
        auto A = Tensor({2, 3, 4}, initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);
        auto B = Tensor({2, 4, 3},  initializer::uniform(-1.0, 1.0), DTYPE_DOUBLE);
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-4, DEFAULT_GRAD_EPS, true));
    }
}

// TEST_F(MatrixTests, recursive_sum) {
//     auto functor = [](vector<Mat<R>>& Xs)-> Mat<R> {
//         auto doubled = Xs[0] + Xs[0];
//         return doubled.sum();
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
//         ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, DEFAULT_GRAD_EPS, true));
//     }
// }
// TEST_F(MatrixTests, inplace_sum) {
//
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(3, 4, weights<R>::uniform(2.0));
//         auto B = Mat<R>(3, 4, weights<R>::uniform(2.0));
//
//         auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
//             auto A_temp = A;
//             auto B_temp = B;
//             A_temp += B_temp;
//             return A_temp;
//         };
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-2, DEFAULT_GRAD_EPS, true));
//     }
// }

// TEST_F(MatrixTests, inplace_substract) {
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(3, 4, weights<R>::uniform(1.0, 2.0));
//         auto B = Mat<R>(3, 4, weights<R>::uniform(1.0, 2.0));
//
//         auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
//             auto A_temp = A;
//             auto B_temp = B;
//             A_temp -= B_temp;
//             return A_temp;
//         };
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-5, DEFAULT_GRAD_EPS, true));
//     }
// }
//
// TEST_F(MatrixTests, inplace_multiply) {
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(3, 4, weights<R>::uniform(2.0));
//         auto B = Mat<R>(3, 4, weights<R>::uniform(2.0));
//
//         auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
//             auto A_temp = A;
//             auto B_temp = B;
//             A_temp *= B_temp;
//             return A_temp;
//         };
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-5, DEFAULT_GRAD_EPS, true));
//     }
// }
//
// TEST_F(MatrixTests, addition) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[0] + Xs[1];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
//         auto B = Mat<R>(10, 20,  weights<R>::uniform(0.5));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-5, DEFAULT_GRAD_EPS, true));
//     }
// }
//
// TEST_F(MatrixTests, subtraction) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[0] - Xs[1];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
//         auto B = Mat<R>(10, 20,  weights<R>::uniform(0.5));
//         ASSERT_TRUE(gradient_same(functor, {A, B}));
//     }
// }
//


// TEST_F(MatrixTests, addition_broadcast_rowwise) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[0] + Xs[1];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(20, 10, weights<R>::uniform(2.0));
//         auto B = Mat<R>(1, 10, weights<R>::uniform(0.5));
//         ASSERT_TRUE(gradient_same(functor, {A, B}));
//     }
// }
//

//
// TEST_F(MatrixTests, addition_broadcast_colwise) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return MatOps<R>::add_broadcast_colwise(Xs[0], Xs[1]);
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
//         auto B = Mat<R>(10, 1, weights<R>::uniform(0.5));
//         ASSERT_TRUE(gradient_same(functor, {A, B}));
//     }
// }
//
// TEST_F(MatrixTests, substraction_broadcast) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[0] - Xs[1];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
//         auto B = Mat<R>(10, 1, weights<R>::uniform(0.5));
//         ASSERT_TRUE(gradient_same(functor, {A, B}));
//     }
// }
//
// TEST_F(MatrixTests, substraction_reversed_broadcast) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[1] - Xs[0];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
//         auto B = Mat<R>(10, 1, weights<R>::uniform(0.5));
//         ASSERT_TRUE(gradient_same(functor, {A, B}));
//     }
// }
//
//

//
// TEST_F(MatrixTests, matrix_divide) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[0] / Xs[1];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(10.0, 20.0));
//         auto B = Mat<R>(10, 20, weights<R>::uniform(5.0, 15.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-3));
//     }
// }
//
// TEST_F(MatrixTests, matrix_divide_broadcast) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[0] / Xs[1];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(0.1, 20.0));
//         auto B = Mat<R>(10, 1,  weights<R>::uniform(0.5, 4.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 5e-3, 1e-3));
//     }
// }

// TEST_F(MatrixTests, matrix_divide) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[0] / Xs[1];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(10.0, 20.0));
//         auto B = Mat<R>(10, 20, weights<R>::uniform(5.0, 15.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-3));
//     }
// }
//
// TEST_F(MatrixTests, matrix_divide_broadcast) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[0] / Xs[1];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(0.1, 20.0));
//         auto B = Mat<R>(10, 1,  weights<R>::uniform(0.5, 4.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 5e-3, 1e-3));
//     }
// }
//
// TEST_F(MatrixTests, matrix_eltmul_broadcast_rowwise_default) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[0] * Xs[1];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(20, 10, weights<R>::uniform(0.1, 20.0));
//         auto B = Mat<R>(1,  10,  weights<R>::uniform(0.5, 4.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 5e-3, 1e-3));
//     }
// }
//
// TEST_F(MatrixTests, matrix_eltmul_broadcast_colwise) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return MatOps<R>::eltmul_broadcast_colwise(Xs[0], Xs[1]);
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(0.1, 20.0));
//         auto B = Mat<R>(10, 1,  weights<R>::uniform(0.5, 4.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 5e-3, 1e-3));
//     }
// }
//
// TEST_F(MatrixTests, matrix_divide_reversed_broadcast) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[1] / Xs[0];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(0.5, 5.0));
//         auto B = Mat<R>(10, 1,  weights<R>::uniform(0.1, 20.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 5e-3, 1e-3));
//     }
// }
//
// TEST_F(MatrixTests, matrix_eltmul_rowwise) {
//     // Operation of the form f(A,B) => A * B.T
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return MatOps<R>::eltmul_rowwise(Xs[0], Xs[1]);
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(4, 5, weights<R>::uniform(10.0));
//         auto B = Mat<R>(5, 4, weights<R>::uniform(10.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-3));
//     }
// }
//
// TEST_F(MatrixTests, matrix_eltmul_broadcast_rowwise) {
//     // Operation of the form f(A,B) => A * B.T
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         return MatOps<R>::eltmul_broadcast_rowwise(Xs[0], Xs[1]);
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(4, 5, weights<R>::uniform(10.0));
//         auto B = Mat<R>(1, 5, weights<R>::uniform(10.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-3));
//     }
// }
// TEST_F(MatrixTests, scalar_pow) {
//     int height = 3;
//     int width = 4;
//
//     EXPERIMENT_REPEAT {
//         auto mat = Mat<R>(height, width, weights<R>::uniform(1.0, 2.0));
//         R exponent = utils::randdouble(0.4, 2.5);
//
//         auto functor = [exponent](vector<Mat<R>> Xs)-> Mat<R> {
//             return Xs[0] ^ exponent;
//         };
//         ASSERT_TRUE(gradient_same(functor, {mat}, 1e-3));
//     }
// }
//

// TEST_F(MatrixTests, pow) {
//     int height = 3;
//     int width = 4;
//
//     EXPERIMENT_REPEAT {
//
//         auto mat = Mat<R>(height, width, weights<R>::uniform(0.5, 1.5));
//         auto exponent = Mat<R>(1,1);
//         exponent = MatOps<R>::fill(exponent, 2.4);
//
//         auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//             return Xs[0] ^ Xs[1];
//         };
//         ASSERT_TRUE(gradient_same(functor, {mat, exponent}, 1e-3));
//     }
// }
