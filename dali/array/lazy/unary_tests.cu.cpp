#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"
#include "dali/array/functor.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"


TEST(ArrayLazyOpsTests, elementwise_F) {
    auto x = Array::zeros({2,1});
    auto expr = lazy::F<functor::sigmoid>(x);
    Array y = expr;
    ASSERT_NEAR((float)y(0), 0.5, 1e-4);
    ASSERT_NEAR((float)y(1), 0.5, 1e-4);
}
