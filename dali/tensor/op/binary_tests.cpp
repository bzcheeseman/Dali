#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>

#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op/binary.h"
#include "dali/array/op/arange.h"

typedef MemorySafeTest TensorBinaryTests;


TEST_F(TensorBinaryTests, single_evaluation) {
    Tensor left(op::arange(10).astype(DTYPE_DOUBLE));
    Tensor right(op::arange(10).astype(DTYPE_DOUBLE));
    auto res = left + right;
    res.grad();
    graph::backward();
    ELOG(res.dw.pretty_print_full_expression_name());
    res.dw.print();
    EXPECT_TRUE(Array::equals(res.dw, Array::ones_like(res.dw)));
    EXPECT_TRUE(Array::equals(left.dw, Array::ones_like(left.dw)));
    EXPECT_TRUE(Array::equals(right.dw, Array::ones_like(right.dw)));
}


TEST_F(TensorBinaryTests, uniform_single_evaluation) {
    auto A = Tensor::uniform(-1.0, 1.0, {10, 20});
    auto B = Tensor::uniform(-1.0, 1.0, {10, 20});

    auto functor = [](std::vector<Tensor> Xs)-> Tensor {
        return tensor_ops::add(Xs[0], Xs[1]);
    };
    std::vector<Tensor> arguments = {A, B};
    auto res = functor(arguments);
    res.grad();
    graph::backward();
    // for some reason not evaluating the objective causes a weird error:
    res.dw.eval();

    EXPECT_TRUE(Array::equals(res.dw, Array::ones_like(res.dw)));
    EXPECT_TRUE(Array::equals(A.dw, Array::ones_like(A.dw)));
    EXPECT_TRUE(Array::equals(B.dw, Array::ones_like(B.dw)));
}


void test_binary_function(std::function<Tensor(std::vector<Tensor>&)> functor) {
    EXPERIMENT_REPEAT {
        auto A = Tensor::uniform(-1.0, 1.0, {10, 20});
        auto B = Tensor::uniform(-1.0, 1.0, {10, 20});
        A.w.eval();
        B.w.eval();
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-5, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(TensorBinaryTests, DISABLED_add) {
    test_binary_function([](std::vector<Tensor> Xs)-> Tensor {
        return tensor_ops::add(Xs[0], Xs[1]);
    });
}

TEST_F(TensorBinaryTests, DISABLED_subtract) {
    test_binary_function([](std::vector<Tensor> Xs)-> Tensor {
        return tensor_ops::subtract(Xs[0], Xs[1]);
    });
}

TEST_F(TensorBinaryTests, DISABLED_eltmul) {
    test_binary_function([](std::vector<Tensor> Xs)-> Tensor {
        return tensor_ops::eltmul(Xs[0], Xs[1]);
    });
}

TEST_F(TensorBinaryTests, DISABLED_eltdiv) {
    test_binary_function([](std::vector<Tensor> Xs)-> Tensor {
        return tensor_ops::eltmul(Xs[0], Xs[1]);
    });
}

TEST_F(TensorBinaryTests, DISABLED_pow) {
    auto functor = [](std::vector<Tensor> Xs)-> Tensor {
        return tensor_ops::pow(Xs[0], Xs[1]);
    };

    EXPERIMENT_REPEAT {
        auto A = Tensor::uniform(0.5, 1.0, {10, 20});
        auto B = Tensor::uniform(0.5, 1.0, {10, 20});

        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-4, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(TensorBinaryTests, DISABLED_add_recursive) {
    auto functor = [](std::vector<Tensor>& Xs)-> Tensor {
        return tensor_ops::add(Xs[0], Xs[0]);
    };
    EXPERIMENT_REPEAT {
        auto A = Tensor::uniform(-1.0, 1.0, {10, 20});
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(TensorBinaryTests, DISABLED_circular_convolution) {
    auto functor = [](std::vector<Tensor> Xs)-> Tensor {
        return tensor_ops::circular_convolution(Xs[0], Xs[1]);
    };
    EXPERIMENT_REPEAT {
        auto matrix = Tensor::uniform(-20.0, 20.0, {4, 5});
        auto shift  = Tensor::uniform(-20.0, 20.0, {4, 5});
        ASSERT_TRUE(gradient_same(functor, {matrix, shift}, 1e-4));
    }
}

TEST_F(TensorBinaryTests, DISABLED_prelu) {
    auto functor = [](std::vector<Tensor> Xs)-> Tensor {
        return tensor_ops::prelu(Xs[0], Xs[1]);
    };
    EXPERIMENT_REPEAT {
        auto x = Tensor::uniform(-20.0, 20.0, {4, 5, 6});
        Tensor weights  = Tensor::uniform(0.5, 20.0, {6})[Broadcast()][Broadcast()];
        ASSERT_TRUE(gradient_same(functor, {x, weights}, 1e-4));
    }
}

// TEST_F(MatrixTests, recursive_sum) {
//     auto functor = [](std::vector<Mat<R>>& Xs)-> Mat<R> {
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
//         auto functor = [&A, &B](std::vector<Mat<R>>& Xs)-> Mat<R> {
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
//         auto functor = [&A, &B](std::vector<Mat<R>>& Xs)-> Mat<R> {
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
//         auto functor = [&A, &B](std::vector<Mat<R>>& Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
//         return Xs[0] / Xs[1];
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(0.1, 20.0));
//         auto B = Mat<R>(10, 1,  weights<R>::uniform(0.5, 4.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 5e-3, 1e-3));
//     }
// }

// TEST_F(MatrixTests, matrix_divide) {
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     // Expression of the form f(A,B) => A * B.T
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
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
//     // Expression of the form f(A,B) => A * B.T
//     auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
//         return MatOps<R>::eltmul_broadcast_rowwise(Xs[0], Xs[1]);
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(4, 5, weights<R>::uniform(10.0));
//         auto B = Mat<R>(1, 5, weights<R>::uniform(10.0));
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-3));
//     }
// }

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
//         auto functor = [](std::vector<Mat<R>> Xs)-> Mat<R> {
//             return Xs[0] ^ Xs[1];
//         };
//         ASSERT_TRUE(gradient_same(functor, {mat, exponent}, 1e-3));
//     }
// }
