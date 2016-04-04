#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/array/op/elementwise.h"

TEST(ArrayTests, sigmoid) {
    // Array x;
    // auto res = sigmoid(x);
    //
    // x = Array(dtype::Double);
    // res = sigmoid(x);
    //
    Array x({3,2,2});
    x.print();

    auto subx = x[0];

    subx.print();

    ASSERT_EQ(1, 1);
}
