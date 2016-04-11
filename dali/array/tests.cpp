#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/array/op/elementwise.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/other.h"
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



TEST(ArrayTests, chainable) {
    Array x({3,2,2});
    // sigmoid is run and stored,
    // then relu, then tanh. the operations
    // are not fused, but implicit casting to
    // Array from AssignableArray occurs at
    // every stage.
    Array y = tanh(relu(sigmoid(x)));
}
