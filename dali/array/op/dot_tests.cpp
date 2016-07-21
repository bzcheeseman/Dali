#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op.h"

using namespace op;

void check_dot_result(DType dtype, bool contiguous) {
    Array a = Array::ones({2, 3}, dtype);
    Array b({4, 3}, dtype);
    b = initializer::arange(0, 1);
    b = contiguous ? b.transpose().ascontiguousarray() : b.transpose();

    int dali_function_computations = 0;
    auto cb = make_scoped_callback([&](const std::string& ignored, int ignored2) {
        dali_function_computations += 1;
    }, &debug::dali_function_end);

    Array c = op::dot(a, b);

    std::vector<float> expected = {
        3, 12, 21, 30,
        3, 12, 21, 30
    };

    for (int i = 0; i < c.number_of_elements(); i++) {
        EXPECT_EQ_DTYPE(expected[i], c(i), dtype);
    }
    // make sure that the function is lazy - no extra dali functions are run
    // during computation.
    EXPECT_EQ(1, dali_function_computations);
}

void check_strided_dot_result(DType dtype) {
    Array a = Array::ones({2, 3}, dtype);
    Array b({8, 3}, dtype);
    b = 999;
    b = b[Slice(0, 8, 2)];
    for (int i = 0; i < b.number_of_elements(); ++i) {
        b(i) = i;
    }

    Array c = op::dot(a, b.transpose());

    std::vector<float> expected = {
        3, 12, 21, 30,
        3, 12, 21, 30
    };

    for (int i = 0; i < c.number_of_elements(); i++) {
        EXPECT_EQ_DTYPE(expected[i], c(i), dtype);
    }
}

TEST(ArrayDotTests, dot) {
    check_dot_result(DTYPE_FLOAT, true);
    check_dot_result(DTYPE_INT32, true);
}

TEST(ArrayDotTests, dot_T) {
    check_dot_result(DTYPE_FLOAT, false);
    check_dot_result(DTYPE_INT32, false);
}

TEST(ArrayDotTests, dot_strided) {
    check_strided_dot_result(DTYPE_FLOAT);
    check_strided_dot_result(DTYPE_INT32);
}

TEST(ArrayDotTests, tensordot) {
    Array a = Array::ones({3, 2, 1}, DTYPE_INT32);
    Array b = Array::ones({3, 1, 2}, DTYPE_INT32);
    Array c = dot(a, b);
    // c.print();
    EXPECT_EQ(std::vector<int>({3, 2, 3, 2}), c.shape());
}

struct ArrayCompatibilityCheck {
    std::vector<int> leftshape;
    std::vector<int> rightshape;
    bool worked_out;
    std::vector<int> outshape;

    ArrayCompatibilityCheck(
        const std::vector<int>& leftshape_,
        const std::vector<int>& rightshape_,
        bool worked_out_,
        const std::vector<int>& outshape_) :
            outshape(outshape_),
            worked_out(worked_out_),
            leftshape(leftshape_),
            rightshape(rightshape_) {}
};

TEST(ArrayDotTests, matrix_vector_dot) {
    auto left = Array::arange({4, 3}, DTYPE_FLOAT);
    auto right = Array::arange({3}, DTYPE_FLOAT);

    Array res = dot(left, right);
    EXPECT_EQ(std::vector<int>({4}), res.shape());

    std::vector<float> expected_result = {5, 14, 23, 32};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ_DTYPE(expected_result[i], res(i), DTYPE_FLOAT);
    }
}

TEST(ArrayDotTests, vector_matrix_dot) {
    auto left = Array::arange({4, 3}, DTYPE_FLOAT);
    auto right = Array::arange({3}, DTYPE_FLOAT);

    Array res = dot(right, left.transpose());
    EXPECT_EQ(std::vector<int>({4}), res.shape());

    std::vector<float> expected_result = {5, 14, 23, 32};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ_DTYPE(expected_result[i], res(i), DTYPE_FLOAT);
    }
}


Array reference_tensordot(const Array& a, const Array&b) {
    ASSERT2(a.ndim() == 3, "a must have ndim == 3");
    ASSERT2(b.ndim() == 4, "b must have ndim == 4");
    ASSERT2(a.dtype() == b.dtype(), "a.dtype() must equal b.dtype()");
    std::vector<int> outshape;
    for (int i = 0; i < a.ndim() - 2;i++) {
        outshape.emplace_back(a.shape()[i]);
    }

    for (int i = 0; i < b.ndim() - 2;i++) {
        outshape.emplace_back(b.shape()[i]);
    }

    auto cloop = Array::zeros(outshape, a.dtype());

    for (int i = 0; i < outshape[0]; i++) {
        for (int j = 0; j < outshape[1]; j++) {
            for (int k = 0; k < outshape[2]; k++) {
                // loop over summed indices -- these don't exist
                // in the tensor product.
                for (int l = 0; l < a.shape().at(a.ndim() - 2); l++) {
                    for (int m = 0; m < a.shape().at(a.ndim() - 1); m++) {
                        cloop[i][j][k] += a[i][l][m] * b[j][k][m][l];
                    }
                }
            }
        }
    }
    return cloop;
}

TEST(ArrayDotTests, broadcasted_args_explicit_reshape) {
    Array X = Array::arange({5}, DTYPE_DOUBLE)[Broadcast()];
    X = X.reshape_broadcasted({3,5});
    auto W = Array::arange({5, 10}, DTYPE_DOUBLE);

    auto out = Array({3, 10}, DTYPE_DOUBLE);

    out = X.dot(W);
}


TEST(ArrayDotTests, broadcasted_args_inner) {
    Array X = Array::arange({2}, DTYPE_DOUBLE)[Slice(0,2)][Broadcast()];
    auto W = Array::arange({5, 10}, DTYPE_DOUBLE);

    Array out = X.dot(W);
}


TEST(ArrayDotTests, tensordot_alignment_rules) {
    std::vector<ArrayCompatibilityCheck> checks = {
        ArrayCompatibilityCheck({4,}, {2,}, false, {}),
        ArrayCompatibilityCheck({4,}, {4,}, true, {}),

        ArrayCompatibilityCheck({2,}, {2,}, true, {}),
        ArrayCompatibilityCheck({2,}, {4,}, false, {}),

        ArrayCompatibilityCheck({2,}, {4, 2}, false, {}),
        ArrayCompatibilityCheck({2,}, {6, 4, 2}, false, {}),
        ArrayCompatibilityCheck({2,}, {8, 6, 4, 2}, false, {}),
        ArrayCompatibilityCheck({2,}, {10, 8, 6, 4, 2}, false, {}),

        ArrayCompatibilityCheck({2, 4}, {2,}, false, {}),
        ArrayCompatibilityCheck({2, 4}, {4,}, true, {2}),

        ArrayCompatibilityCheck({2, 4}, {4, 2}, true, {2, 2}),
        ArrayCompatibilityCheck({2, 4}, {6, 4, 2}, true, {2, 6, 2}),
        ArrayCompatibilityCheck({2, 4}, {8, 6, 4, 2}, true, {2, 8, 6, 2}),
        ArrayCompatibilityCheck({2, 4}, {10, 8, 6, 4, 2}, true, {2, 10, 8, 6, 2}),

        ArrayCompatibilityCheck({6, 2, 4}, {2,}, false, {}),
        ArrayCompatibilityCheck({6, 2, 4}, {4,}, true, {6, 2}),

        ArrayCompatibilityCheck({6, 2, 4}, {4, 2,}, true, {6, 2, 2}),
        ArrayCompatibilityCheck({6, 2, 4}, {6, 4, 2,}, true, {6, 2, 6, 2}),
        ArrayCompatibilityCheck({6, 2, 4}, {8, 6, 4, 2,}, true, {6, 2, 8, 6, 2}),
        ArrayCompatibilityCheck({6, 2, 4}, {10, 8, 6, 4, 2,}, true, {6, 2, 10, 8, 6, 2}),

        ArrayCompatibilityCheck({8, 6, 2, 4}, {2,}, false, {}),
        ArrayCompatibilityCheck({8, 6, 2, 4}, {4,}, true, {8, 6, 2}),

        ArrayCompatibilityCheck({8, 6, 2, 4}, {4, 2,}, true, {8, 6, 2, 2}),
        ArrayCompatibilityCheck({8, 6, 2, 4}, {6, 4, 2,}, true, {8, 6, 2, 6, 2}),
        ArrayCompatibilityCheck({8, 6, 2, 4}, {8, 6, 4, 2,}, true, {8, 6, 2, 8, 6, 2}),
        ArrayCompatibilityCheck({8, 6, 2, 4}, {10, 8, 6, 4, 2,}, true, {8, 6, 2, 10, 8, 6, 2}),

        ArrayCompatibilityCheck({10, 8, 6, 2, 4}, {2,}, false, {}),
        ArrayCompatibilityCheck({10, 8, 6, 2, 4}, {4,}, true, {10, 8, 6, 2}),

        ArrayCompatibilityCheck({10, 8, 6, 2, 4}, {4, 2,}, true, {10, 8, 6, 2, 2}),
        ArrayCompatibilityCheck({10, 8, 6, 2, 4}, {6, 4, 2,}, true, {10, 8, 6, 2, 6, 2}),
        ArrayCompatibilityCheck({10, 8, 6, 2, 4}, {8, 6, 4, 2,}, true, {10, 8, 6, 2, 8, 6, 2}),
        ArrayCompatibilityCheck({10, 8, 6, 2, 4}, {10, 8, 6, 4, 2,}, true, {10, 8, 6, 2, 10, 8, 6, 2}),
    };


    for (auto& check : checks) {
        if (check.worked_out) {
            EXPECT_NO_THROW({
                Array res = dot(
                    Array(check.leftshape),
                    Array(check.rightshape)
                );
                EXPECT_EQ(check.outshape, res.shape());
            });
        } else {
            EXPECT_THROW(
                (Array)dot(
                    Array(check.leftshape),
                    Array(check.rightshape)
                ),
                std::runtime_error
            );
        }
    }

    Array a({2, 3, 4}, DTYPE_FLOAT);
    Array b({5, 6, 4, 3}, DTYPE_FLOAT);
    a = initializer::gaussian(0.0, 1.0);
    b = initializer::gaussian(0.0, 1.0);

    Array c = tensordot(a, b, {1, 2}, {3, 2});

    Array expected_c = reference_tensordot(a, b);

    EXPECT_EQ(expected_c.shape(), c.shape());

    for (int i = 0; i < expected_c.number_of_elements(); i++) {
        EXPECT_NEAR_DTYPE((float)expected_c(i), c(i), (float)1e-5, c.dtype());
    }

}

TEST(ArrayDotTests, broadcast_outer) {
    auto X = Array::zeros({4}, DTYPE_DOUBLE)[Broadcast()];
    auto W = Array::zeros({4, 5}, DTYPE_DOUBLE);


    CountImplicitCopies implicit_copies;
    Array y = op::dot(X,W);
    ASSERT_EQ(0, implicit_copies.count);
}

TEST(ArrayDotTests, broadcast_inner) {
    auto X = Array::zeros({3}, DTYPE_DOUBLE)[Slice(0,3)][Broadcast()];
    auto W = Array::zeros({4, 5}, DTYPE_DOUBLE);

    CountImplicitCopies implicit_copies;
    Array y = op::dot(X,W);
    ASSERT_EQ(1, implicit_copies.count);
}

TEST(ArrayDotTests, broadcast_to_out) {
    Array X = Array::arange({5}, DTYPE_DOUBLE)[Broadcast()];
    auto W = Array::arange({5, 10}, DTYPE_DOUBLE);

    auto out = Array({3, 10}, DTYPE_DOUBLE);

    out = X.dot(W);
}

TEST(ArrayDotTests, broadcast_outer_with_jumps) {
    Array X = Array::arange({300, 5}, DTYPE_DOUBLE);
    X = X[Slice(0, 300, 100)];

    auto W = Array::arange({5, 10}, DTYPE_DOUBLE);

    auto out = Array({3, 10}, DTYPE_DOUBLE);

    CountImplicitCopies implicit_copies;
    out = X.dot(W);
    ASSERT_EQ(1, implicit_copies.count);
}
