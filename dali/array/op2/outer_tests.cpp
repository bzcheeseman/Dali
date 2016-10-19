#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/outer.h"
// #include "dali/array/op2/reducers.h"
#include "dali/array/op.h"
#include "dali/array/op2/operation.h"

namespace {
    Array reference_outer_product(const Array& left, const Array& right) {
        ASSERT2(left.ndim() == 1 && right.ndim() == 1, "left and right should have ndim == 1");
        Array out = Array::zeros({left.shape()[0], right.shape()[0]}, left.dtype());
        for (int i = 0; i < out.shape()[0]; i++) {
            for (int j = 0; j < out.shape()[1]; j++) {
                out[i][j] = left[i] * right[j];
            }
        }
        return out;
    }
}

TEST(RTCTests, outer_product_chainable) {
    Array x = Array::arange({3});
    Array y = Array::arange({4});
    Array outer = op::outer(op::tanh(x - 3.0), op::tanh(y - 2.0));
    auto expected_outer = reference_outer_product(op::tanh(x - 3.0), op::tanh(y - 2.0));
    EXPECT_TRUE(Array::allclose(outer, expected_outer, 1e-5));
}

TEST(RTCTests, outer_product_chainable_with_sum) {
    Array x = Array::arange({3});
    Array y = Array::arange({4, 4});
    Array outer = op::outer(x, op::sum(y, {0}));
    auto expected_outer = reference_outer_product(x, op::sum(y, {0}));
    EXPECT_TRUE(Array::allclose(outer, expected_outer, 1e-5));
}
