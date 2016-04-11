#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"

#include "dali/runtime_config.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"


TEST(ArrayLazyOpsTests, imports) {
    ASSERT_TRUE(lazy::ops_loaded);
}

TEST(ArrayLazyOpsTests, lazy) {
    Array x({12});
    Array y({12});
    Array z({12});

    Array q({12});
    Array q2({16});

    // q = x * 1 * z * x * y * 1 * 2;
    q = x * (y * z);
    // lazy_mul(x, lazy_mul(lazy_mul(y,z), 1));
    Array qp = x * (y * z);
    EXPECT_THROW(q2 = x * (y * z), std::runtime_error);
}
