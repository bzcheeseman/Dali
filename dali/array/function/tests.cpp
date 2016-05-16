#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"

#include "dali/runtime_config.h"
#include "dali/array/function/function.h"
#define DALI_USE_LAZY 0
#include "dali/array/op.h"

using namespace op;


TEST(FunctionTests, lse) {
    Array target = Array::zeros({3,4}, DTYPE_INT32);
    Array source = Array::arange({3,4}, DTYPE_INT32);

    target <<= source;

    for (int i = 0; i < target.number_of_elements(); ++i) {
        EXPECT_EQ(i, (int)target(i));
    }
}

TEST(FunctionTests, broadcasted_lse) {
    Array target = Array::zeros({3}, DTYPE_INT32)[Slice(0,3)][Broadcast()];
    Array source = Array::ones({3,4}, DTYPE_INT32);

    target <<= source;

    for (int i = 0; i < target.number_of_elements(); ++i) {
        EXPECT_EQ(4,  (int)target(i));
    }
}

TEST(FunctionTests, broadcasted_lse2) {
    Array target = Array::zeros({4}, DTYPE_INT32)[Broadcast()][Slice(0,4)];
    Array source = Array::ones({3,4}, DTYPE_INT32);

    target <<= source;

    for (int i = 0; i < target.number_of_elements(); ++i) {
        EXPECT_EQ(3,  (int)target(i));
    }
}
