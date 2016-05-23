#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#define DALI_USE_LAZY 0
#include "dali/array/op/reshape.h"
#include "dali/array/op/initializer.h"

using namespace op;

TEST(ArrayInitializerTests, uniform_bounds_check) {
    Array x({2,3}, DTYPE_FLOAT);
    EXPECT_THROW(x = initializer::uniform(0.0, 0.0), std::runtime_error);
    EXPECT_THROW(x = initializer::uniform(1.0, -1.0), std::runtime_error);
    x = initializer::uniform(-1.0, 1.0);
}
