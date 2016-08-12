#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"
#include "dali/array/functor.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"


TEST(ArrayLazyOpsTests, imports) {
    ASSERT_TRUE(lazy::ops_loaded);
}



TEST(ArrayLazyOpsTests, sum_all) {
   auto z = Array::zeros({2,4}, DTYPE_FLOAT);
   auto o = Array::ones({2,4}, DTYPE_FLOAT);

   ASSERT_NEAR((float)(Array)z.sum(), 0.0, 1e-4);
   ASSERT_NEAR((float)(Array)o.sum(), 8.0, 1e-4);
}

// TODO(jonathan): make this test more gnarly
TEST(ArrayLazyOpsTests, argmax) {
   auto z = Array::zeros({2,4}, DTYPE_FLOAT);

   z[0][1] = 2.0;
   z[1][3] = 3.0;
   Array max_values = lazy::max(z, 1);
   Array max_indices = lazy::argmax(z, 1);

   ASSERT_NEAR(2.0, (float)max_values[0], 1e-6);
   ASSERT_NEAR(3.0, (float)max_values[1], 1e-6);

   ASSERT_EQ(DTYPE_INT32, max_indices.dtype());

   ASSERT_EQ(1, (int)max_indices[0]);
   ASSERT_EQ(3, (int)max_indices[1]);
}

TEST(ArrayLazyOpsTests, argmin) {
   auto z = Array::zeros({2,4}, DTYPE_FLOAT);
   z[0][1] = -2.0;
   z[1][3] = -3.0;
   Array min_values = lazy::min(z, 1);
   Array min_indices = lazy::argmin(z, 1);

   ASSERT_NEAR(-2.0, (float)min_values[0], 1e-6);
   ASSERT_NEAR(-3.0, (float)min_values[1], 1e-6);

   // ASSERT_EQ(DTYPE_INT32, min_indices.dtype());

   ASSERT_EQ(1, (int)min_indices[0]);
   ASSERT_EQ(3, (int)min_indices[1]);
}

TEST(ArrayTests, sum) {
    Array x = Array::ones({2,3,4}, DTYPE_INT32);
    for (int reduce_axis = 0; reduce_axis < x.ndim(); reduce_axis++) {
        Array y = lazy::sum(x, reduce_axis);
        std::vector<int> expected_shape;
        switch (reduce_axis) {
            case 0:
                expected_shape = {3, 4};
                break;
            case 1:
                expected_shape = {2, 4};
                break;
            case 2:
                expected_shape = {2, 3};
                break;
        }
        EXPECT_EQ(expected_shape, y.shape());
        for (int i = 0; i < y.number_of_elements(); i++) {
            EXPECT_EQ(x.shape()[reduce_axis], (int)y(i));
        }
    }
}


TEST(ArrayTests, sum_broadcasted) {
    Array x = Array::ones({2,3,4}, DTYPE_INT32)[Slice(0,2)][Slice(0,3)][Slice(0,4)][Broadcast()];

    for (int reduce_axis = 0; reduce_axis < 3; reduce_axis++) {
        Array y = lazy::sum(x, reduce_axis);
        std::vector<int> expected_shape;
        switch (reduce_axis) {
            case 0:
                expected_shape = {3, 4, 1};
                break;
            case 1:
                expected_shape = {2, 4, 1};
                break;
            case 2:
                expected_shape = {2, 3, 1};
                break;
        }
        EXPECT_EQ(expected_shape, y.shape());

        for (int i = 0; i < y.number_of_elements(); i++) {
            EXPECT_EQ(x.shape()[reduce_axis], (int)y(i));
        }
    }
}

TEST(ArrayTests, sum_broadcasted2) {
    Array x = Array::ones({2,4}, DTYPE_INT32)[Slice(0,2)][Broadcast()][Slice(0,4)];

    for (int reduce_axis = 0; reduce_axis < 3; reduce_axis++) {
        Array y = lazy::sum(x, reduce_axis);
        std::vector<int> expected_bshape;
        switch (reduce_axis) {
            case 0:
                expected_bshape = {-1, 4};
                break;
            case 1:
                expected_bshape = {2, 4};
                break;
            case 2:
                expected_bshape = {2, -1};
                break;
        }
        EXPECT_EQ(expected_bshape, y.bshape());

        for (int i = 0; i < y.number_of_elements(); i++) {
            EXPECT_EQ(x.shape()[reduce_axis], (int)y(i));
        }
    }
}

TEST(ArrayTests, sum_broadcasted_2D) {
    Array x = Array::ones({2,4}, DTYPE_INT32);

    for (int reduce_axis = 0; reduce_axis < 2; reduce_axis++) {
        Array y = lazy::sum(x, reduce_axis);
        std::vector<int> expected_shape;
        switch (reduce_axis) {
            case 0:
                expected_shape = {4};
                break;
            case 1:
                expected_shape = {2};
                break;
        }
        EXPECT_EQ(expected_shape, y.bshape());

        for (int i = 0; i < y.number_of_elements(); i++) {
            EXPECT_EQ(x.shape()[reduce_axis], (int)y(i));
        }
    }
}

TEST(ArrayTests, sum_broadcasted_1D) {
    Array x = Array::ones({6,}, DTYPE_INT32);

    EXPECT_THROW(lazy::sum(x, 1), std::runtime_error);
    EXPECT_THROW(lazy::sum(x, -1), std::runtime_error);

    Array y = lazy::sum(x, 0);

    EXPECT_EQ(0, y.ndim());

    EXPECT_EQ(6, (int)y);
}
