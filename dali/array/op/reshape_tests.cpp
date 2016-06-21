#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/lazy_op.h"

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

// TODO(jonathan): add back resizing to array from Mat<R>
//
// TEST_F(TensorOpsTests, resize_decrease_rows) {
//     int row_size = 3, col_size = 4;
//     // decrease number of rows by 1
//     auto A = Mat<R>(row_size, col_size);
//     for (int i = 0; i < 12; i++) {
//         A.w(i) = i;
//     }
//
//     auto new_shape = mshadow::Shape2(row_size - 1, col_size);
//     A.w().resize(new_shape);
//     for (int i = 0; i < (row_size - 1) * col_size ; i++) {
//         ASSERT_EQ(A.w(i), i);
//     }
//     ASSERT_EQ(A.w().shape, new_shape);
// }
//
// TEST_F(TensorOpsTests, resize_increase_rows) {
//     int row_size = 3, col_size = 4;
//     // increase number of rows by 1
//     auto A = Mat<R>(row_size, col_size);
//     for (int i = 0; i < row_size * col_size; i++) {
//         A.w(i) = i;
//     }
//     auto new_shape = mshadow::Shape2(row_size + 1, col_size);
//     A.w().resize(new_shape, 3.5);
//     for (int i = 0; i < row_size * col_size; i++) {
//         ASSERT_EQ(A.w(i), i);
//     }
//     for (int i = row_size * col_size; i < (row_size + 1) * col_size; i++) {
//         ASSERT_EQ(A.w(i), 3.5);
//     }
//     ASSERT_EQ(A.w().shape, new_shape);
// }
//
// TEST_F(TensorOpsTests, resize_decrease_cols) {
//     int row_size = 3, col_size = 4;
//     // decrease number of columns by 1
//     auto A = Mat<R>(row_size, col_size);
//     for (int i = 0; i < row_size * col_size; i++) {
//         A.w(i) = i;
//     }
//     auto new_shape = mshadow::Shape2(row_size, col_size - 1);
//     A.w().resize(new_shape);
//     for (int i = 0; i < row_size; i++) {
//         for (int j = 0; j < col_size - 1; j++) {
//             ASSERT_EQ(A.w(i,j), i * col_size + j);
//         }
//     }
//     ASSERT_EQ(A.w().shape, new_shape);
// }
//
// TEST_F(TensorOpsTests, resize_increase_cols) {
//     int row_size = 3, col_size = 4;
//     // increase number of columns by 1
//     auto A = Mat<R>(row_size, col_size);
//     for (int i = 0; i < row_size * col_size; i++) {
//         A.w(i) = i;
//     }
//     auto new_shape = mshadow::Shape2(row_size, col_size + 1);
//     A.w().resize(new_shape, 4.2);
//     for (int i = 0; i < row_size; i++) {
//         for (int j = 0; j < col_size; j++) {
//             ASSERT_EQ(A.w(i,j), i * col_size + j);
//         }
//     }
//     for (int i = 0; i < row_size; i++) {
//         for (int j = col_size; j < col_size + 1; j++) {
//             ASSERT_EQ(A.w(i,j), 4.2);
//         }
//     }
//     ASSERT_EQ(A.w().shape, new_shape);
// }
//
// TEST_F(TensorOpsTests, resize_increase_rows_and_cols) {
//     int row_size = 3, col_size = 4;
//     // increase number of rows and columns by 1
//     auto A = Mat<R>(row_size, col_size);
//     for (int i = 0; i < row_size * col_size; i++) {
//         A.w(i) = i;
//     }
//     auto new_shape = mshadow::Shape2(row_size + 1, col_size + 1);
//     A.w().resize(new_shape, 4.2);
//     for (int i = 0; i < row_size; i++) {
//         for (int j = 0; j < col_size; j++) {
//             ASSERT_EQ(A.w(i,j), i * col_size + j);
//         }
//     }
//     for (int i = 0; i < row_size; i++) {
//         for (int j = col_size; j < col_size + 1; j++) {
//             ASSERT_EQ(A.w(i,j), 4.2);
//         }
//     }
//     for (int i = row_size; i < row_size + 1; i++) {
//         for (int j = 0; j < col_size + 1; j++) {
//             ASSERT_EQ(A.w(i,j), 4.2);
//         }
//     }
//     ASSERT_EQ(A.w().shape, new_shape);
// }
//
// TEST_F(TensorOpsTests, resize_decrease_rows_and_cols) {
//     int row_size = 3, col_size = 4;
//     // decrease number of rows and columns by 1
//     auto A = Mat<R>(row_size, col_size);
//     for (int i = 0; i < row_size * col_size; i++) {
//         A.w(i) = i;
//     }
//     auto new_shape = mshadow::Shape2(row_size - 1, col_size - 1);
//     A.w().resize(new_shape, 4.2);
//     for (int i = 0; i < row_size - 1; i++) {
//         for (int j = 0; j < col_size - 1; j++) {
//             ASSERT_EQ(A.w(i,j), i * col_size + j);
//         }
//     }
//     ASSERT_EQ(A.w().shape, new_shape);
// }
//
// TEST_F(TensorOpsTests, resize_1D_decrease_rows) {
//     int row_size = 3;
//     // decrease number of rows by 1
//     TensorInternal<R,1> A(mshadow::Shape1(row_size));
//
//     for (int i = 0; i < row_size; i++) {
//         A(i) = i;
//     }
//
//     auto new_shape = mshadow::Shape1(row_size - 1);
//     A.resize(new_shape);
//     for (int i = 0; i < (row_size - 1); i++) {
//         ASSERT_EQ(A(i), i);
//     }
//     ASSERT_EQ(A.shape, new_shape);
// }
//
// TEST_F(TensorOpsTests, resize_1D_increase_rows) {
//     int row_size = 3;
//     // increase number of rows by 1
//     TensorInternal<R,1> A(mshadow::Shape1(row_size));
//
//     for (int i = 0; i < row_size; i++) {
//         A(i) = i;
//     }
//
//     auto new_shape = mshadow::Shape1(row_size + 1);
//     A.resize(new_shape, 666.0);
//     for (int i = 0; i < (row_size); i++) {
//         ASSERT_EQ(A(i), i);
//     }
//     ASSERT_EQ(A(row_size), 666.0);
//     ASSERT_EQ(A.shape, new_shape);
// }
//
