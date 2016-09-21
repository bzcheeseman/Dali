#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op.h"
#include "dali/array/op2/fused_operation.h"

#define DALI_RTC_UNARY_TEST(funcname)\
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {\
        int size = 10;\
        auto a = Array::arange({5, size}, dtype) + 1;\
        auto dst = Array::zeros({5, size}, dtype);\
        dst = op2::funcname(a);\
        EXPECT_TRUE(Array::allclose(dst, (Array)(op::funcname(a)), 1e-6));\
    }\

TEST(RTCTests, unary) {
    DALI_RTC_UNARY_TEST(softplus);
    DALI_RTC_UNARY_TEST(sigmoid);
    DALI_RTC_UNARY_TEST(tanh);
    DALI_RTC_UNARY_TEST(log);
    DALI_RTC_UNARY_TEST(cube);
    DALI_RTC_UNARY_TEST(sqrt);
    DALI_RTC_UNARY_TEST(rsqrt);
    DALI_RTC_UNARY_TEST(eltinv);
    DALI_RTC_UNARY_TEST(relu);
    DALI_RTC_UNARY_TEST(abs);
    DALI_RTC_UNARY_TEST(sign);
    DALI_RTC_UNARY_TEST(identity);
}

#define DALI_RTC_SCALAR_UNARY_TEST(funcname)\
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {\
        int size = 10;\
        auto a = Array::arange({5, size}, dtype) + 1;\
        auto dst = Array::zeros({5, size}, dtype);\
        dst = op2::funcname(a, 2.5);\
        EXPECT_TRUE(Array::allclose(dst, (Array)(op::funcname(a, 2.5)), 1e-5));\
    }\

TEST(RTCTests, scalar_unary) {
    DALI_RTC_SCALAR_UNARY_TEST(scalar_add);
    DALI_RTC_SCALAR_UNARY_TEST(scalar_sub);
    DALI_RTC_SCALAR_UNARY_TEST(scalar_mul);
    DALI_RTC_SCALAR_UNARY_TEST(scalar_div);
    DALI_RTC_SCALAR_UNARY_TEST(scalar_pow);
}

TEST(RTCTests, circular_conv_unary) {
    int size = 10;
    // TODO: allow placement of scalar at arbitrary location
    // + keep track of shapeless elements in computation graph
    // + ensure that scalars of some type do not cause dtype issues:
    // Fails due to dtype:
    // Array res = op2::circular_convolution(
    //     Array::arange({5, size}, DTYPE_FLOAT) + 1,
    //     op2::relu(2.5)
    // );
    // Fails due to shape:
    // Array res = op2::circular_convolution(
    //     2.5,
    //     Array::arange({5, size}, DTYPE_FLOAT) + 1
    // );
    // Succeeds:
    Array res = op2::circular_convolution(
        2.1,
        2.5
    );
    Array res2 = op2::circular_convolution(
        Array::arange({5, size}, DTYPE_FLOAT) + 1,
        2.5
    );
}
