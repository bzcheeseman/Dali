#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/array/op/elementwise.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/other.h"
#include "dali/array/op/initializer.h"
#include "dali/utils/print_utils.h"
#include "dali/runtime_config.h"

#include "dali/array/lazy_op/binary.h"


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
