#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"
#include "dali/array/functor.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"

TEST(ArrayLazyOpsTests, long_chain) {
    Array x({2,1});
    Array y({2,1});
    Array z({2,1});

    int lazy_evaluator_calls = 0;
    auto callback_handle = debug::lazy_evaluation_callback.register_callback([&](const Array&) {
        lazy_evaluator_calls += 1;
    });
    auto partial = (
        lazy::sigmoid(lazy::tanh(x)) * 2 +
        x * y * lazy::sign(z) * 2 +
        1 +
        x +
        lazy::log_or_zero(y)
    );

    ASSERT_EQ(lazy_evaluator_calls, 0);
    Array result = partial;
    ASSERT_EQ(lazy_evaluator_calls, 1);
    debug::lazy_evaluation_callback.deregister_callback(callback_handle);
}
