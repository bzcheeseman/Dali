#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/array/op/unary.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/other.h"
#include "dali/array/op/initializer.h"
#include "dali/utils/print_utils.h"
#include "dali/runtime_config.h"

#include "dali/array/lazy/binary.h"

using std::vector;

TEST(ArrayTests, slicing) {
    Array x({12});
    Array y({3,2,2});

    EXPECT_THROW(x[0][0], std::runtime_error);
    EXPECT_THROW(y[3], std::runtime_error);

    EXPECT_EQ(y[0].shape().size(), 2);
    EXPECT_EQ(y[1].shape().size(), 2);
    EXPECT_EQ(y[2].shape().size(), 2);
    EXPECT_EQ(y[2][1].shape().size(), 1);
    EXPECT_EQ(y[2][1][0].shape().size(), 0);

    EXPECT_EQ(x[0].shape().size(), 0);

    EXPECT_EQ(x(0).shape().size(), 0);
    EXPECT_EQ(y(0).shape().size(), 0);
}

TEST(ArrayTests, scalar_value) {
    Array x({12}, DTYPE_INT32);
    x(3) = 42;
    int& x_val = x(3);
    EXPECT_EQ(x_val, 42);
    x[3] = 56;
    EXPECT_EQ(x_val, 56);
}

TEST(ArrayTests, scalar_assign) {
    Array x = Array::zeros({3,2}, DTYPE_INT32);
    x = 13;

    ASSERT_EQ(x.shape(), std::vector<int>({3,2}));
    ASSERT_EQ(x.dtype(), DTYPE_INT32);
    for (int i=0; i < 6; ++i) {
        ASSERT_EQ((int)x(i), 13);
    }

    x = 69.1;
    ASSERT_EQ(x.shape(), std::vector<int>({3,2}));
    ASSERT_EQ(x.dtype(), DTYPE_INT32);
    for (int i=0; i <6; ++i) {
        ASSERT_EQ((int)x(i), 69);
    }
}

TEST(ArrayTests, scalar_construct) {
    auto assignable = initializer::fill((float)3.14);
    Array scalar = assignable;
    ASSERT_EQ(scalar.shape(), std::vector<int>());
    ASSERT_EQ(scalar.dtype(), DTYPE_FLOAT);
    ASSERT_NEAR((float)scalar(0), 3.14, 1e-6);

    Array scalar2;
    scalar2 = initializer::fill((double)3.14);
    ASSERT_EQ(scalar2.shape(), std::vector<int>());
    ASSERT_EQ(scalar2.dtype(), DTYPE_DOUBLE);
    ASSERT_NEAR((double)scalar2(0), 3.14, 1e-6);

    Array scalar3 = initializer::fill(314);
    ASSERT_EQ(scalar3.shape(), std::vector<int>());
    ASSERT_EQ(scalar3.dtype(), DTYPE_INT32);
    ASSERT_EQ((int)scalar3(0), 314);
}


TEST(ArrayTests, spans_entire_memory) {
    // an array is said to span its entire memory
    // if it is not a "view" onto said memory.

    // the following 3D tensor spans its entire memory
    // (in fact it even allocated it!)
    Array x = Array::zeros({3,2,2});
    ASSERT_TRUE(x.spans_entire_memory());

    // however a slice of x may not have the same property:
    auto subx = x[0];
    ASSERT_FALSE(subx.spans_entire_memory());

    // Now let's take a corner case:
    // the leading dimension of the following
    // array is 1, so taking a view at "row" 0
    // makes no difference in terms of underlying
    // memory hence, both it and its subview will
    // "span the entire memory"
    Array y = Array::zeros({1,2,2});
    ASSERT_TRUE(y.spans_entire_memory());

    auto suby = y[0];
    ASSERT_TRUE(suby.spans_entire_memory());
}

TEST(ArrayTests, dim_pluck) {
    Array x({2,3,4}, DTYPE_INT32);
    x = initializer::arange();
    EXPECT_TRUE(x.contiguous_memory());

    auto x_plucked = x.dim_pluck(0, 1);
    EXPECT_EQ(x_plucked.shape(),   vector<int>({3, 4}));
    EXPECT_EQ(x_plucked.offset(),  12    );
    EXPECT_EQ(x_plucked.strides(), vector<int>({}));

    auto x_plucked2 = x.dim_pluck(1, 2);
    EXPECT_EQ(x_plucked2.shape(),   vector<int>({2, 4}));
    EXPECT_EQ(x_plucked2.offset(),   8    );
    EXPECT_EQ(x_plucked2.strides(), vector<int>({3, 1}));

    auto x_plucked3 = x.dim_pluck(2, 1);
    EXPECT_EQ(x_plucked3.shape(),   vector<int>({2, 3}));
    EXPECT_EQ(x_plucked3.offset(),  1);
    EXPECT_EQ(x_plucked3.strides(), vector<int>({1, 4}));
}
