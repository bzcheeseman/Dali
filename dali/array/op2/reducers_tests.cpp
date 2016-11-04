#include <gtest/gtest.h>

#include "dali/array/op2/reducers.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op.h"
#include "dali/array/op2/expression/expression.h"

TEST(RTCTests, all_reduce_sum) {
    int rows = 5, cols = 10;
    auto a = Array::arange({rows, cols}, DTYPE_FLOAT);
    // sum of n first numbers is (n * (n+1)) / 2:
    int expected_total = (a.number_of_elements() * (a.number_of_elements() - 1)) / 2;
    EXPECT_EQ(expected_total, (int)Array(op::sum(a)));
}

TEST(RTCTests, all_reduce_prod) {
    auto a = Array::arange(1, 6, 1, DTYPE_INT32);
    int expected_total = 1 * 2 * 3 * 4 * 5;
    EXPECT_EQ(expected_total, (int)Array(op::prod(a)));
}

TEST(RTCTests, all_reduce_max_min) {
    auto a = Array::arange(-100, 42, 1, DTYPE_INT32);
    int expected_max = 41, expected_min = -100;
    EXPECT_EQ(expected_max, (int)Array(op::max(a)));
    EXPECT_EQ(expected_min, (int)Array(op::min(a)));
}

TEST(RTCTests, all_reduce_argmax_argmin) {
    auto a = Array::arange(-100, 42, 1, DTYPE_INT32);
    int expected_argmax = 141, expected_argmin = 0;
    EXPECT_EQ(expected_argmax, (int)Array(op::argmax(a)));
    EXPECT_EQ(expected_argmin, (int)Array(op::argmin(a)));
}

TEST(RTCTests, axis_reduce_argmax_argmin) {
    auto a = Array::arange({4, 5}, DTYPE_INT32);
    EXPECT_TRUE(Array::equals(Array::ones({5}, DTYPE_INT32) * 3, op::argmax(a, 0)));
    EXPECT_TRUE(Array::equals(Array::ones({4}, DTYPE_INT32) * 4, op::argmax(a, 1)));

    a = a * -1.0;

    EXPECT_TRUE(Array::equals(Array::ones({5}, DTYPE_INT32) * 3, op::argmin(a, 0)));
    EXPECT_TRUE(Array::equals(Array::ones({4}, DTYPE_INT32) * 4, op::argmin(a, 1)));
}

TEST(RTCTests, all_reduce_argmax_argmin_4d) {
    auto a = Array::arange({2, 3, 4, 5}, DTYPE_INT32);
    int expected_argmax = 2 * 3 * 4 * 5 - 1, expected_argmin = 0;
    EXPECT_EQ(expected_argmax, (int)Array(op::argmax(a)));
    EXPECT_EQ(expected_argmin, (int)Array(op::argmin(a)));
}

TEST(RTCTests, all_reduce_mean) {
    auto a = Array::arange(1, 3, 1, DTYPE_INT32);
    double expected_mean = 1.5;
    EXPECT_EQ(expected_mean, (double)Array(op::mean(a)));
}

TEST(RTCTests, all_reduce_l2_norm) {
    auto a = Array::ones({4}, DTYPE_INT32);
    EXPECT_EQ(2.0, (double)Array(op::L2_norm(a)));

    a = Array::ones({2}, DTYPE_INT32);
    EXPECT_EQ(std::sqrt(2.0), (double)Array(op::L2_norm(a)));
}

TEST(RTCTests, all_reduce_sum_with_broadcast) {
    int rows = 5, cols = 10;
    Array a = Array::arange({rows, cols}, DTYPE_FLOAT)[Slice()][Broadcast()][Slice()];
    // sum of n first numbers is (n * (n+1)) / 2:
    int expected_total = (a.number_of_elements() * (a.number_of_elements() - 1)) / 2;
    EXPECT_EQ(expected_total, (int)Array(op::sum(a)));
}

TEST(RTCTests, all_reduce_sum_with_strides) {
    int rows = 5, cols = 10, skip = 2, expected_total = 0, k=0;
    Array a = Array::arange({rows, cols}, DTYPE_FLOAT)[Slice()][Slice(0,cols,skip)];
    // sum of n first numbers, while skipping 1 out of 2 on last dim:
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j % skip == 0) {
                expected_total += k++;
            } else {
                ++k;
            }
        }
    }
    EXPECT_EQ(expected_total, (int)Array(op::sum(a)));
}

TEST(RTCTests, all_reduce_mixed_sum) {
    auto a = Array::arange({3, 4, 5}, DTYPE_INT32);
    Array b = Array::arange({2, 3, 4, 5}, DTYPE_INT32)[Slice()][Slice()][Slice(0, 4, 3)];
    int expected_result = (
        (a.number_of_elements() * (a.number_of_elements() - 1)) / 2 -
        (2 * 3 * 4 * 5 - 1) + 2.0
    );
    auto operation = op::add(2, (op::sub(op::sum(a), op::max(b))));
    EXPECT_EQ(expected_result, (int)Array(operation));
}

TEST(RTCTests, axis_reduce_sum_low_dim) {
    auto a = Array::ones({2, 3, 4, 5}, DTYPE_INT32);
    // same kernel is used in all these cases:
    EXPECT_TRUE(Array::equals(Array::ones({2, 3, 4}, DTYPE_INT32) * 5, op::sum(a, {-1})));
    EXPECT_TRUE(Array::equals(Array::ones({2, 3}, DTYPE_INT32) * 20, op::sum(a, {-2, -1})));
    EXPECT_TRUE(Array::equals(Array::ones({2}, DTYPE_INT32) * 60, op::sum(a, {-3, -2, -1})));
}

TEST(RTCTests, axis_reduce_sum_high_dim) {
    auto a = Array::ones({2, 3, 4, 5}, DTYPE_INT32);
    // same kernel is used in all these cases:
    EXPECT_TRUE(Array::equals(Array::ones({3, 4, 5}, DTYPE_INT32) * 2, op::sum(a, {0})));
    EXPECT_TRUE(Array::equals(Array::ones({4, 5}, DTYPE_INT32) * 6, op::sum(a, {1, 0})));
    EXPECT_TRUE(Array::equals(Array::ones({5}, DTYPE_INT32) * 24, op::sum(a, {2, 1, 0})));
}

TEST(RTCTests, axis_reduce_sum_middle_dim) {
    auto a = Array::ones({2, 3}, DTYPE_INT32);
    EXPECT_TRUE(Array::equals(Array::ones({3}, DTYPE_INT32) * 2, op::sum(a, {0})));
    EXPECT_TRUE(Array::equals(Array::ones({2}, DTYPE_INT32) * 3, op::sum(a, {1})));
    a = Array::ones({2, 3, 4}, DTYPE_INT32);
    EXPECT_TRUE(Array::equals(Array::ones({3, 4}, DTYPE_INT32) * 2, op::sum(a, {0})));
    EXPECT_TRUE(Array::equals(Array::ones({2, 4}, DTYPE_INT32) * 3, op::sum(a, {1})));
    EXPECT_TRUE(Array::equals(Array::ones({2, 3}, DTYPE_INT32) * 4, op::sum(a, {2})));
}

TEST(RTCTests, lse_reduce) {
    auto a = Array::zeros({2}, DTYPE_INT32).insert_broadcast_axis(1);
    a <<= Expression(Array::ones({2, 5}, DTYPE_INT32));
    EXPECT_EQ(5, int(a[0][0]));
    EXPECT_EQ(5, int(a[1][0]));
}
