#include <gtest/gtest.h>

// #include "dali/utils/print_utils.h"
// #include "dali/array/test_utils.h"
// #include "dali/runtime_config.h"
#include "dali/array/op2/reducers.h"
// #include "dali/array/op.h"
#include "dali/array/op2/fused_operation.h"

TEST(RTCTests, all_reduce_sum) {
    int size = 10;
    auto a = Array::arange({5, size}, DTYPE_FLOAT);
    // sum of n first numbers is (n * (n+1)) / 2:
    int expected_total = (a.number_of_elements() * (a.number_of_elements() - 1)) / 2;
    EXPECT_EQ(expected_total, (int)Array(op2::sum(a)));
}
