#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/array/op/elementwise.h"
#include "dali/array/op/binary.h"

TEST(ArrayTests, sigmoid) {
    Array x({3,2,2});
    x.print();
    auto y = sigmoid(x);
    y.print();

    double x_val = x(4);

    std::cout << "x_val = " << x_val << std::endl;

    ASSERT_EQ(1, 1);
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

