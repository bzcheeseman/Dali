#include <gtest/gtest.h>

#include "dali/array/op2/reducers.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/fused_operation.h"

TEST(RTCTests, all_reduce_sum) {
    int rows = 5, cols = 10;
    auto a = Array::arange({rows, cols}, DTYPE_FLOAT);
    // sum of n first numbers is (n * (n+1)) / 2:
    int expected_total = (a.number_of_elements() * (a.number_of_elements() - 1)) / 2;
    EXPECT_EQ(expected_total, (int)Array(op2::sum(a)));
}

TEST(RTCTests, all_reduce_prod) {
    auto a = Array::arange(1, 6, 1, DTYPE_INT32);
    int expected_total = 1 * 2 * 3 * 4 * 5;
    EXPECT_EQ(expected_total, (int)Array(op2::prod(a)));
}

TEST(RTCTests, all_reduce_max_min) {
    auto a = Array::arange(-100, 42, 1, DTYPE_INT32);
    int expected_max = 41, expected_min = -100;
    EXPECT_EQ(expected_max, (int)Array(op2::max(a)));
    EXPECT_EQ(expected_min, (int)Array(op2::min(a)));
}

TEST(RTCTests, all_reduce_argmax_argmin) {
    auto a = Array::arange(-100, 42, 1, DTYPE_INT32);
    int expected_argmax = 141, expected_argmin = 0;
    EXPECT_EQ(expected_argmax, (int)Array(op2::argmax(a)));
    EXPECT_EQ(expected_argmin, (int)Array(op2::argmin(a)));
}

TEST(RTCTests, all_reduce_argmax_argmin_4d) {
    auto a = Array::arange({2, 3, 4, 5}, DTYPE_INT32);
    int expected_argmax = 2 * 3 * 4 * 5 - 1, expected_argmin = 0;
    EXPECT_EQ(expected_argmax, (int)Array(op2::argmax(a)));
    EXPECT_EQ(expected_argmin, (int)Array(op2::argmin(a)));
}

TEST(RTCTests, all_reduce_mean) {
    auto a = Array::arange(1, 3, 1, DTYPE_INT32);
    double expected_mean = 1.5;
    EXPECT_EQ(expected_mean, (double)Array(op2::mean(a)));
}

TEST(RTCTests, all_reduce_l2_norm) {
    auto a = Array::ones({4}, DTYPE_INT32);
    EXPECT_EQ(2.0, (double)Array(op2::L2_norm(a)));

    a = Array::ones({2}, DTYPE_INT32);
    EXPECT_EQ(std::sqrt(2.0), (double)Array(op2::L2_norm(a)));
}

TEST(RTCTests, all_reduce_sum_with_broadcast) {
    int rows = 5, cols = 10;
    Array a = Array::arange({rows, cols}, DTYPE_FLOAT)[Slice()][Broadcast()][Slice()];
    // sum of n first numbers is (n * (n+1)) / 2:
    int expected_total = (a.number_of_elements() * (a.number_of_elements() - 1)) / 2;
    EXPECT_EQ(expected_total, (int)Array(op2::sum(a)));
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
    EXPECT_EQ(expected_total, (int)Array(op2::sum(a)));
}

TEST(RTCTests, all_reduce_mixed_sum) {
    auto a = Array::arange({3, 4, 5}, DTYPE_INT32);
    Array b = Array::arange({2, 3, 4, 5}, DTYPE_INT32)[Slice()][Slice()][Slice(0, 4, 3)];
    int expected_result = (
        (a.number_of_elements() * (a.number_of_elements() - 1)) / 2 -
        (2 * 3 * 4 * 5 - 1) + 2.0
    );
    auto operation = op2::add(2, (op2::sub(op2::sum(a), op2::max(b))));
    EXPECT_EQ(expected_result, (int)Array(operation));
}
