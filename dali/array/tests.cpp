#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/array/typed_array.h"
#include "dali/array/op/elementwise.h"

TEST(ArrayTests, sigmoid) {
    Array x(0);
    auto res = sigmoid(x);

    x = Array(1);
    res = sigmoid(x);

    ASSERT_EQ(1, 1);
}
