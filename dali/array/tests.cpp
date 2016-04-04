#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/array/op/elementwise.h"

TEST(ArrayTests, sigmoid) {
    Array x({3,2,2});
    x.print();
    auto y = sigmoid(x);
    y.print();

    double x_val = x(4);

    std::cout << "x_val = " << x_val << std::endl;

    ASSERT_EQ(1, 1);
}
