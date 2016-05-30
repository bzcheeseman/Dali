#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#define DALI_USE_LAZY 0
#include "dali/array/op.h"

// This softmax causes many copies, but the logic is clear:
Array reference_softmax(const Array& input, int axis, const double& temperature) {
    auto exped = op::exp(input / temperature);
    Array denominator = op::sum(exped, axis);
    return exped / denominator.insert_broadcast_axis(axis);
}

TEST(ArraySoftmaxTests, softmax_axis) {
    Array A({2, 3, 1, 4, 5, 2}, DTYPE_DOUBLE);
    A = initializer::uniform(-20.0, 20.0);
    for (int axis = 0; axis < A.ndim(); axis++) {
        Array res = op::softmax(A, axis);
        Array ref = reference_softmax(A, axis, 1.0);
        EXPECT_TRUE(Array::allclose(res, ref, 1e-3));
    }
}

TEST(ArraySoftmaxTests, softmax_temperature) {
    Array A({4, 5}, DTYPE_DOUBLE);
    A = initializer::uniform(-20.0, 20.0);
    for (int axis = 0; axis < A.ndim(); axis++) {
        for (double temperature = 0.1; temperature < 3.0; temperature += 0.5) {
            Array res = op::softmax(A, axis, temperature);
            Array ref = reference_softmax(A, axis, temperature);
            EXPECT_TRUE(Array::allclose(res, ref, 1e-3));
        }
    }
}
