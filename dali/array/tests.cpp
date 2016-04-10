#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/array/op/elementwise.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/other.h"
#include "dali/utils/print_utils.h"

TEST(ArrayTests, sigmoid) {
    Array x({3,2,2});

    x[2][1][0] = 42;

    x.print();
    Array y(x.shape(),x.dtype());
    y = sigmoid(x);
    y.print();

    double x_val = x(4);

    ASSERT_EQ(1, 1);
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

    // reseting q to baby array makes it stateless again.
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
    test_binary_shapes([](const Array& a, const Array& b) { return a/b; });
}

TEST(ArrayTests, is_nan) {
    Array x = Array::zeros({4,3,5});
    ASSERT_FALSE(is_nan(x));
    x[2][3][1] = NAN;
    ASSERT_TRUE(is_nan(x));
}
