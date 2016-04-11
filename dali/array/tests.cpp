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


TEST(ArrayTests, lazy) {
    Array x({12});
    Array y({12});
    Array z({12});

    Array q({12});
    Array q2({16});

    q = lazy_mul(x, lazy_mul(lazy_mul(y,z), 1));
    Array qp = lazy_mul(x, lazy_mul(y,z));
    EXPECT_THROW(q2 = lazy_mul(x, lazy_mul(y,z)), std::runtime_error);
}

TEST(ArrayTests, sigmoid) {
    Array x({3,2,2});
    Array y = sigmoid(x);
}

TEST(ArrayTests, relu) {
    Array x({3,2,2});
    Array y = relu(x);
}

TEST(ArrayTests, log_or_zero) {
    Array x({3,2,2});
    Array w = log_or_zero(x);
}

TEST(ArrayTests, abs) {
    Array x({3,2,2});
    Array w = abs(x);
}

TEST(ArrayTests, sign) {
    Array x({3,2,2});
    Array w = sign(x);
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

template<typename T>
void test_binary_shapes(T op_f) {
    Array x({3,2,2});
    Array y({12});
    // binary op on args of different sizes
    auto args_wrong_size = [&]() {     Array z = op_f(x, y);         };
    ASSERT_THROW(args_wrong_size(), std::runtime_error);

    // binary op on args with the same sized args
    Array z = eltdiv(x.ravel(), y);

    // assigning to preallocated output of wrong shape.
    Array q({12});
    auto output_wrong_size = [&]() {   q = op_f(x, y.reshape(x.shape()));   };
    ASSERT_THROW(output_wrong_size(), std::runtime_error);

    // resetting q to baby array makes it stateless again.
    q.reset() = x / y.reshape(x.shape());
}

// TODO(Jonathan) = when scaffolding is cleaner,
// check for actual outputs of sub, add, etc..
TEST(ArrayTests, add) {
    test_binary_shapes([](const Array& a, const Array& b) { return a + b; });
}

TEST(ArrayTests, sub) {
    test_binary_shapes([](const Array& a, const Array& b) { return a - b; });
}

TEST(ArrayTests, eltmul) {
    test_binary_shapes([](const Array& a, const Array& b) { return a * b; });
}

TEST(ArrayTests, eltdiv) {
    test_binary_shapes([](const Array& a, const Array& b) { return a / b; });
}

TEST(ArrayTests, is_nan) {
    Array x = Array::zeros({4,3,5});
    ASSERT_FALSE(is_nan(x));
    x[2][2][1] = NAN;
    ASSERT_TRUE(is_nan(x));
}
