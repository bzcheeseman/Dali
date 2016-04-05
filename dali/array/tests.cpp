#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/array/op/elementwise.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/other.h"

TEST(ArrayTests, sigmoid) {
    Array x({3,2,2});

    x[2][1][0] = 42;

    x.print();
    auto y = sigmoid(x);
    y.print();

    double x_val = x(4);

    ASSERT_EQ(1, 1);
}

TEST(ArrayTests, relu) {
    Array x({3,2,2});
    auto y = relu(x);
}

TEST(ArrayTests, log_or_zero) {
    Array x({3,2,2});
    log_or_zero(x);
}

TEST(ArrayTests, abs) {
    Array x({3,2,2});
    abs(x);
}

TEST(ArrayTests, sign) {
    Array x({3,2,2});
    sign(x);
}


// TODO(Jonathan) = when scaffolding is cleaner,
// check for actual outputs of sub, add, etc..
TEST(ArrayTests, add) {
    Array x({3,2,2});
    Array y({12});

    auto z = add(x, y);

    z = x + y;
}

TEST(ArrayTests, sub) {
    Array x({3,2,2});
    Array y({12});

    auto z = sub(x, y);

    z = x - y;
}

TEST(ArrayTests, eltmul) {
    Array x({3,2,2});
    Array y({12});

    auto z = eltmul(x, y);

    z = x * y;
}

TEST(ArrayTests, eltdiv) {
    Array x({3,2,2});
    Array y({12});

    auto z = eltdiv(x, y);

    z = x / y;
}

TEST(ArrayTests, is_nan) {
    Array x = Array::zeros({4,3,5});
    ASSERT_FALSE(is_nan(x));
    x[2][3][1] = NAN;
    ASSERT_TRUE(is_nan(x));
}
