#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#define DALI_USE_LAZY 0
#include "dali/array/op.h"
#include "dali/array/op/initializer.h"

using namespace op;


TEST(ArrayReshapeTests, all_close_and_all_equals) {
    Array a({10,3,1,2}, DTYPE_DOUBLE);
    Array b({10,3,1,2}, DTYPE_DOUBLE);
    Array perturb({10,3,1,2}, DTYPE_DOUBLE);
    a       = initializer::uniform(-20.0, 20.0);
    b       = a + 200.0;
    perturb = initializer::uniform(-0.1, 0.1);

    EXPECT_TRUE(Array::equals(a, a));
    EXPECT_TRUE(Array::equals(b, b));
    EXPECT_TRUE(Array::allclose(b, b, 1e-5));
    EXPECT_TRUE(Array::allclose(b, b, 1e-5));

    EXPECT_FALSE(Array::equals(a ,b));
    EXPECT_FALSE(Array::allclose(a, b, 1e-5));


    EXPECT_FALSE(Array::equals(a, a + perturb));
    EXPECT_TRUE(Array::allclose(a, a + perturb, 0.11));
}
