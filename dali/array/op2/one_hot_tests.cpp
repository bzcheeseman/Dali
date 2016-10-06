#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/one_hot.h"
#include "dali/array/op.h"
#include "dali/array/op2/operation.h"

Array reference_one_hot(Array indices, int depth, double on_value, double off_value) {
    auto out_shape = indices.shape();
    out_shape.emplace_back(depth);
    auto res = Array::zeros(out_shape, DTYPE_DOUBLE);
    res = res.copyless_reshape({-1, depth});
    indices = indices.ravel();
    res = initializer::fill(off_value);
    for (int i = 0; i < indices.number_of_elements(); i++) {
        res[i][indices[i]] = on_value;
    }
    return res.reshape(out_shape);
}


TEST(RTCTests, one_hot) {
    auto a = Array::zeros({2, 3, 5}, DTYPE_INT32);
    a = initializer::uniform(0, 6);
    EXPECT_TRUE(Array::equals(reference_one_hot(a, 7, 112.2, 42.0), op2::one_hot(a, 7, 112.2, 42.0)));
}
