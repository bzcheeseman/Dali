#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"

#include "dali/runtime_config.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"


TEST(ArrayLazyOpsTests, imports) {
    ASSERT_TRUE(lazy::ops_loaded);
}

TEST(ArrayLazyOpsTests, lazy_dtype_deduction) {
    Array x({16}, DTYPE_FLOAT);
    Array y({16}, DTYPE_FLOAT);
    Array z({16}, DTYPE_FLOAT);


    // test deduction of dtype of lazy expression
    auto partial = x * (y * z);
    EXPECT_EQ(partial.dtype(), DTYPE_FLOAT);

    // detect wrong dtype during lazy expression construction.
    Array z_wrong({16}, DTYPE_DOUBLE);
    EXPECT_THROW(x * (y * z_wrong), std::runtime_error);

    // assiging to preallocated memory
    Array q({16}, DTYPE_FLOAT);
    q = partial;

    // auto allocate memory
    Array qp = x * (y * z);

    // cannot assign to memory of wrong type
    Array q2({16}, DTYPE_INT32);
    EXPECT_THROW(q2 = x * (y * z), std::runtime_error);
}


TEST(ArrayLazyOpsTests, lazy_shape_deduction) {
    Array x({16});
    Array y({16});
    Array z({16});


    // test deduction of shape of lazy expression
    auto partial = x * (y * z);
    EXPECT_EQ(partial.shape(), std::vector<int>({16}));

    // detect wrong shape during lazy expression construction.
    Array z_wrong({20,4});
    EXPECT_THROW(x * (y * z_wrong), std::runtime_error);

    // assiging to preallocated memory
    Array q({16});
    q = partial;

    // auto allocate memory
    Array qp = x * (y * z);

    // cannot assign to memory of wrong shape
    Array q2({14, 5});
    EXPECT_THROW(q2 = x * (y * z), std::runtime_error);
}
