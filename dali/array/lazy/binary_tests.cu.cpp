#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"
#include "dali/array/functor.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"


TEST(ArrayLazyOpsTests, lazy_binary_correctness) {
    Array x({2,1});
    Array y({2,1});
    Array z({2,1});

    x(0) = 2; y(0) = 3;  z(0) = 5;
    x(1) = 7; y(1) = 11; z(1) = 13;

    auto partial = x * y * z;
    Array res = partial;

    EXPECT_EQ((float)(res(0)), 2 * 3  * 5);
    EXPECT_EQ((float)(res(1)), 7 * 11 * 13);
}


TEST(ArrayLazyOpsTests, broadcasted_add) {
    auto out = Array::zeros({2,3,4}, DTYPE_INT32);
    auto A = Array::ones({2,3,4}, DTYPE_INT32);
    auto B = Array::ones({3},     DTYPE_INT32);

    B = B[Broadcast()][Slice(0,3)][Broadcast()];

    out = A +  2 * B;

    ASSERT_EQ((int)(Array)out.sum(), 2 * 3 * 4 * 3);
}
