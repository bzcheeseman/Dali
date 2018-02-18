#include <gtest/gtest.h>
#include "dali/config.h"

#include "dali/utils/print_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/tests/test_utils.h"
#include "dali/array/op/concatenate.h"
#include "dali/array/op/arange.h"

TEST(ArrayConcatenateTests, hstack) {
    Array a_b = op::arange(14).reshape({2, 7});
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

TEST(ArrayConcatenateTests, concatenate_one_arg) {
    Array a = op::arange(14).reshape({2, 7});

    Array b = op::concatenate({a}, 0);
    ASSERT_EQ(a.memory(), b.memory());
}

TEST(ArrayConcatenateTests, concatenate_zero_arg) {
    EXPECT_THROW(Array b = op::concatenate({}, 0), std::runtime_error);
}

TEST(ArrayConcatenateTests, vstack) {
    Array a_b = op::arange(14).reshape({7, 2, 1});
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

TEST(ArrayConcatenateTests, concatenate_middle_axis) {
    Array a_b = op::arange(18).reshape({3, 2, 3});

    Array a = a_b[Slice(0, 3)][0][Broadcast()][Slice(0, 3)];
    Array b = a_b[Slice(0, 3)][1][Broadcast()][Slice(0, 3)];
    Array c = op::concatenate({a, b}, 1);

    EXPECT_TRUE(Array::equals(a_b, c));
}

TEST(ArrayConcatenateTests, concatenate_keeps_broadcast) {
    Array a_b = op::arange(6).reshape({3, 2})[Broadcast()];
    EXPECT_EQ(std::vector<int>({1, 3, 2}), a_b.shape());

    Array a = a_b[Slice(0,1)][Slice(0, 3)][0][Broadcast()];
    Array b = a_b[Slice(0,1)][Slice(0, 3)][1][Broadcast()];
    // join along the last axis:
    Array c = op::concatenate({a, b}, 2);
    EXPECT_TRUE(Array::equals(a_b, c));
    EXPECT_EQ(std::vector<int>({1, 3, 2}), c.shape());
}

TEST(ArrayConcatenateTests, concatenate_with_broadcast) {
    Array broadcasted = op::arange(5)[Broadcast()];
    EXPECT_EQ(std::vector<int>({1, 5}), broadcasted.shape());
    Array other = op::arange(21).reshape({3, 7});
    Array res = op::hstack({broadcasted, other});
    EXPECT_EQ(std::vector<int>({3, 12}), res.shape());
}
