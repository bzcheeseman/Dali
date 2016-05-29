#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#define DALI_USE_LAZY 0
#include "dali/array/op/reshape.h"
#include "dali/array/op/initializer.h"

using namespace op;

TEST(ArrayReshapeTests, hstack) {
    Array a_b({2, 7}, DTYPE_INT32);
    a_b = initializer::arange();
    // a:
    // 0 1 2
    // 7 8 9
    // b:
    // 3  4  5  6
    // 10 11 12 13
    Array a = a_b[Slice(0, 2)][Slice(0, 3)];
    Array b = a_b[Slice(0, 2)][Slice(3, 7)];

    Array c = op::hstack({a, b});
    EXPECT_TRUE(Array::equals(a_b, c));
}

TEST(ArrayReshapeTests, concatenate_one_arg) {
    Array a({2, 7}, DTYPE_INT32);
    a = initializer::arange();

    Array b = op::concatenate({a}, 0);
    ASSERT_EQ(a.memory(), b.memory());
}

TEST(ArrayReshapeTests, concatenate_zero_arg) {
    EXPECT_THROW(Array b = op::concatenate({}, 0), std::runtime_error);
}

TEST(ArrayReshapeTests, vstack) {
    Array a_b({7, 2, 1}, DTYPE_INT32);
    a_b = initializer::arange();
    // a:
    // 0 1
    // 2 3
    // 4 5
    // b:
    // 6 7
    // 8 9
    // 10 11
    // 12 13
    Array a = a_b[Slice(0, 3)];
    Array b = a_b[Slice(3, 7)];
    Array c = op::vstack({a, b});
    EXPECT_TRUE(Array::equals(a_b, c));
}

TEST(ArrayReshapeTests, concatenate_middle_axis) {
    Array a_b({3, 2, 3}, DTYPE_INT32);
    a_b = initializer::arange();

    Array a = a_b[Slice(0, 3)][0][Broadcast()][Slice(0, 3)];
    Array b = a_b[Slice(0, 3)][1][Broadcast()][Slice(0, 3)];
    Array c = op::concatenate({a, b}, 1);

    EXPECT_TRUE(Array::equals(a_b, c));
}

TEST(ArrayReshapeTests, concatenate_keeps_broadcast) {
    Array a_b = Array({3, 2}, DTYPE_INT32)[Broadcast()];
    a_b = initializer::arange();
    EXPECT_EQ(std::vector<int>({-1, 3, 2}), a_b.bshape());

    Array a = a_b[Slice(0,1)][Slice(0, 3)][0][Broadcast()];
    Array b = a_b[Slice(0,1)][Slice(0, 3)][1][Broadcast()];
    // join along the last axis:
    Array c = op::concatenate({a, b}, 2);
    EXPECT_TRUE(Array::equals(a_b, c));
    // we can see that the first dimension which was originally a
    // broadcasted dimension remains so:
    EXPECT_EQ(std::vector<int>({-1, 3, 2}), c.bshape());
}
