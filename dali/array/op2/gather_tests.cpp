#include <gtest/gtest.h>

// #include "dali/array/test_utils.h"
#include "dali/array/op2/gather.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op2/gather_from_rows.h"
#include "dali/array/op.h"
#include "dali/array/op2/expression/expression.h"

TEST(RTCTests, gather_simple) {
    auto indices = Array::arange({5}, DTYPE_INT32);
    auto source = Array::arange({5, 6}, DTYPE_INT32);
    EXPECT_TRUE(Array::equals(op::gather(source, indices), old_op::gather(source, indices)));
    auto source2 = Array::arange({5, 6, 7}, DTYPE_INT32);
    EXPECT_TRUE(Array::equals(op::gather(source2, indices), old_op::gather(source2, indices)));
}

TEST(RTCTests, gather_simple_elementwise) {
    auto indices = Array::arange({5}, DTYPE_INT32);
    auto source = Array::arange({5, 6}, DTYPE_DOUBLE);
    EXPECT_TRUE(
        Array::allclose(
            op::gather(op::sigmoid(source), indices),
            old_op::gather(op::sigmoid(source), indices),
            1e-6
        )
    );
}

TEST(RTCTests, gather_from_rows_simple) {
    auto indices = Array::arange({5}, DTYPE_INT32);
    auto source = Array::arange({5, 6}, DTYPE_INT32);
    EXPECT_TRUE(Array::equals(op::gather_from_rows(source, indices), old_op::gather_from_rows(source, indices)));

    auto source2 = Array::arange({5, 6, 7}, DTYPE_INT32);
    Array result_2d = op::gather_from_rows(source2, indices);
    std::vector<std::vector<int>> expected_result({
        {  0,   1,   2,   3,   4,   5,   6},
        { 49,  50,  51,  52,  53,  54,  55},
        { 98,  99, 100, 101, 102, 103, 104},
        {147, 148, 149, 150, 151, 152, 153},
        {196, 197, 198, 199, 200, 201, 202}
   	});
    ASSERT_EQ(result_2d.shape(), std::vector<int>({int(expected_result.size()), int(expected_result[0].size())}));
    for (int i = 0; i < result_2d.shape()[0]; i++) {
        for (int j = 0; j < result_2d.shape()[1]; j++) {
            EXPECT_EQ(expected_result[i][j], int(result_2d[i][j]));
        }
    }
}

TEST(RTCTests, scatter_simple) {
    auto indices = Array::zeros({6}, DTYPE_INT32);
    std::vector<int> vals = {0, 0, 1, 1, 1, 2};
    for (int i = 0; i < vals.size(); i++) {
        indices[i] = vals[i];
    }
    auto dest = Array::zeros({3}, DTYPE_INT32);
    auto gathered = dest[indices];
    ASSERT_EQ(gathered.shape(), indices.shape());
    gathered += expression::Expression(1);
    EXPECT_EQ(2, int(dest[0]));
    EXPECT_EQ(3, int(dest[1]));
    EXPECT_EQ(1, int(dest[2]));
}

TEST(RTCTests, scatter_to_rows_simple) {
    auto indices = Array::zeros({6}, DTYPE_INT32);
    std::vector<int> vals = {0, 0, 1, 1, 1, 2};
    for (int i = 0; i < vals.size(); i++) {
        indices[i] = vals[i];
    }
    auto dest = Array::zeros({7, 3}, DTYPE_INT32);
    dest = 42;
    auto gathered = dest.gather_from_rows(indices);
    ASSERT_EQ(gathered.shape(), indices.shape());
    gathered += expression::Expression(1);

    for (int i = 0; i < vals.size(); i++) {
        for (int j = 0; j < dest.shape()[1]; j++) {
            if (j != vals[i]) {
                EXPECT_EQ(42, int(dest[i][j]));
            } else {
                EXPECT_EQ(43, int(dest[i][j]));
            }
        }
    }
}
