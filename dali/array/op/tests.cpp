#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"

#include "dali/runtime_config.h"

#define DALI_USE_LAZY 0
#include "dali/array/op.h"

TEST(ArrayOpsTests, imports) {
    ASSERT_FALSE(lazy::ops_loaded);
}

TEST(ArrayOpsTests, sigmoid) {
    auto x = Array::zeros({3,2,2});
    Array y = sigmoid(x);
}

TEST(ArrayOpsTests, relu) {
    auto x = Array::zeros({3,2,2});
    Array y = relu(x);
}

TEST(ArrayOpsTests, log_or_zero) {
    auto x = Array::zeros({3,2,2});
    Array w = log_or_zero(x);
}

TEST(ArrayOpsTests, abs) {
    auto x = Array::zeros({3,2,2});
    Array w = abs(x);
}

TEST(ArrayOpsTests, sign) {
    auto x = Array::zeros({3,2,2});
    Array w = sign(x);
}

template<typename T>
void test_binary_shapes(T op_f) {
    auto x = Array::zeros({3,2,2});
    auto y = Array::zeros({12});
    // binary op on args of different sizes
    auto args_wrong_size = [&]() {     Array z = op_f(x, y);         };
    ASSERT_THROW(args_wrong_size(), std::runtime_error);

    // binary op on args with the same sized args
    Array z = eltdiv(x.ravel(), y);

    // assigning to preallocated output of wrong shape.
    Array q({12});
    auto output_wrong_size = [&]() {   q = op_f(x, y.reshape(x.shape()));   };
    ASSERT_THROW(output_wrong_size(), std::runtime_error);

    // resetting q to baby array makes it stateless again.
    q.reset() = x / y.reshape(x.shape());
}

// TODO(Jonathan) = when scaffolding is cleaner,
// check for actual outputs of sub, add, etc..
TEST(ArrayOpsTests, add) {
    test_binary_shapes([](const Array& a, const Array& b) { return a + b; });
}

TEST(ArrayOpsTests, sub) {
    test_binary_shapes([](const Array& a, const Array& b) { return a - b; });
}

TEST(ArrayOpsTests, eltmul) {
    test_binary_shapes([](const Array& a, const Array& b) { return a * b; });
}

TEST(ArrayOpsTests, eltdiv) {
    test_binary_shapes([](const Array& a, const Array& b) { return a / b; });
}

TEST(ArrayOpsTests, is_nan) {
    Array x = Array::zeros({4,3,5});
    ASSERT_FALSE(is_nan(x));
    x[2][2][1] = NAN;
    ASSERT_TRUE(is_nan(x));
}

TEST(ArrayOpsTests, chainable) {
    Array x({3,2,2});
    // sigmoid is run and stored,
    // then relu, then tanh. the operations
    // are not fused, but implicit casting to
    // Array from AssignableArray occurs at
    // every stage.
    Array y = tanh(relu(sigmoid(x)));
}
