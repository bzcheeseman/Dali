#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/dot.h"
#include "dali/array/op.h"


TEST(RTCTests, dot) {
    auto a = Array::ones({3, 4});
    auto b = Array::ones({4, 5});

    auto res = Array::zeros({3, 5});

    res = op::dot2(a, b);
}
