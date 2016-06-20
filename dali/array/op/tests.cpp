#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"

#include "dali/runtime_config.h"
#include "dali/array/function/function.h"
#define DALI_USE_LAZY 0
#include "dali/array/op.h"

using namespace op;

TEST(ArrayOpsTests, imports) {
    ASSERT_FALSE(lazy::ops_loaded);
}

TEST(ArrayOpsTests, sigmoid) {
    auto x = Array::zeros({3,2,2});
    Array y = sigmoid(x);
}

TEST(ArrayOpsTests, relu) {
    auto x = Array::zeros({3,2,2}, DTYPE_DOUBLE);
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


TEST(ArrayOpsTests, log_exp) {
    Array x({50}, DTYPE_DOUBLE);
    x = initializer::uniform(0.1, 20.0);
    Array exp_log_x = op::exp(op::log(x));
    EXPECT_TRUE(Array::allclose(x, exp_log_x, 1e-3));
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
    // Array from Assignable<Array> occurs at
    // every stage.
    Array y = tanh(relu(sigmoid(x)));
}

TEST(ArrayOpsTests, ascontiguousarray) {
    Array x({3,2});
    // create array is contiguous
    EXPECT_EQ(true, x.contiguous_memory());
    // contiguous version of an already
    // contiguous array is just the same array:
    Array x_contig = x.ascontiguousarray();
    EXPECT_EQ(x.memory(), x_contig.memory());

    Array x_T = x.transpose();
    // transpose is not contiguous
    EXPECT_EQ(false, x_T.contiguous_memory());
    // is a view:
    EXPECT_EQ(x.memory(), x_T.memory());

    // we copy memory into a contiguous array
    x_T = x_T.ascontiguousarray();
    // now memory is contiguous:
    EXPECT_EQ(true, x.contiguous_memory());
    EXPECT_NE(x.memory(), x_T.memory());
}

TEST(ArrayOpsTests, add_vector) {
    int lazy_evaluator_calls = 0;
    auto callback_handle = debug::lazy_evaluation_callback.register_callback([&](const Array&) {
        lazy_evaluator_calls += 1;
    });

    Array res = op::add({
        Array::ones({1, 2}),
        Array::ones({2})[Broadcast()],
        Array::ones({1,1, 2})[0],
        Array::ones({1, 2})
    });

    // one single function call performs all the additions above:
    ASSERT_EQ(lazy_evaluator_calls, 1);


    EXPECT_EQ(std::vector<int>({1, 2}), res.shape());
    EXPECT_EQ(4, (int)res(0));
    EXPECT_EQ(4, (int)res(1));

    lazy_evaluator_calls = 0;

    Array res2 = op::add({
        Array::ones({1, 2}),
        Array::ones({2})[Broadcast()],
        Array::ones({1,1, 2})[0],
        Array::ones({1, 2}),
        Array::ones({1, 2}),
        Array::ones({1, 2})
    });

    EXPECT_EQ(6, (int)res2(0));
    EXPECT_EQ(6, (int)res2(1));

    // Current add-vector will eagerly grab up to 5 arrays at a time. If there are more, then
    // operation may perform more than one evaluation:
    ASSERT_EQ(lazy_evaluator_calls, 2);

    debug::lazy_evaluation_callback.deregister_callback(callback_handle);
}


TEST(ArrayOpsTests, arange) {
    Array x_float({2,3,3}, DTYPE_FLOAT);
    x_float = initializer::arange();

    Array x_double({2,3,3}, DTYPE_DOUBLE);
    x_double = initializer::arange();

    Array x_int({2,3,3}, DTYPE_INT32);
    x_int = initializer::arange();

    ASSERT_NEAR((float)(Array)x_float.sum(),  153, 1e-4);
    ASSERT_NEAR((double)(Array)x_double.sum(), 153, 1e-4);
    ASSERT_EQ(  (int)(Array)x_int.sum(),    153);
}
