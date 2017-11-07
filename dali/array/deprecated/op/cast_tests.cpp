#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#define DALI_USE_LAZY 0
#include "dali/array/op.h"

using namespace op;

TEST(ArrayCastTests, astype) {
    Array integer_arange = Array::zeros({1, 2, 3}, DTYPE_INT32);
    integer_arange = initializer::arange(0, 1);

    EXPECT_EQ(DTYPE_INT32, integer_arange.dtype());

    Array float_arange_with_offset = (Array)integer_arange.astype(DTYPE_FLOAT) - 0.6;

    EXPECT_EQ(DTYPE_FLOAT, float_arange_with_offset.dtype());

    for (int i = 0; i < integer_arange.number_of_elements(); i++) {
        EXPECT_NEAR((float)integer_arange(i) - 0.6, (float)float_arange_with_offset(i), 1e-6);
    }

    Array int_arange_with_offset = float_arange_with_offset.astype(DTYPE_INT32);

    for (int i = 0; i < integer_arange.number_of_elements(); i++) {
        EXPECT_EQ(std::round((float)integer_arange(i) - 0.6), (int)int_arange_with_offset(i));
    }
}

TEST(ArrayCastTests, mean) {
    /* 0 1 2
       3 4 5 */
    Array integer_arange = Array::zeros({1, 2, 3}, DTYPE_INT32);
    integer_arange = initializer::arange(0, 1);
    Array mean = integer_arange.mean();
    EXPECT_EQ(DTYPE_DOUBLE, mean.dtype());
    EXPECT_EQ((0.0 + 1.0 + 2.0 + 3.0 + 4.0 + 5.0)/6.0, (double)mean(0));

    Array mean_axis = integer_arange.mean(-1);

    EXPECT_EQ(DTYPE_DOUBLE, mean_axis.dtype());
    EXPECT_EQ((0.0 + 1.0 + 2.0)/3.0, (double)mean_axis(0));
    EXPECT_EQ((3.0 + 4.0 + 5.0)/3.0, (double)mean_axis(1));
}
