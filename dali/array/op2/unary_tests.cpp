#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op.h"
#include "dali/array/op2/operation.h"

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

#define DALI_RTC_SCALAR_UNARY_TEST(rtc_funcname, funcname)\
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {\
        int size = 10;\
        auto a = Array::arange({5, size}, dtype) + 1;\
        auto dst = Array::zeros({5, size}, dtype);\
        dst = op2::rtc_funcname(a, 2.5);\
        EXPECT_TRUE(Array::allclose(dst, (Array)(op::funcname(a, 2.5)), 1e-5));\
    }\

TEST(RTCTests, scalar_unary) {
    DALI_RTC_SCALAR_UNARY_TEST(add, scalar_add);
    DALI_RTC_SCALAR_UNARY_TEST(sub, scalar_sub);
    DALI_RTC_SCALAR_UNARY_TEST(eltmul, scalar_mul);
    DALI_RTC_SCALAR_UNARY_TEST(eltdiv, scalar_div);
    DALI_RTC_SCALAR_UNARY_TEST(pow, scalar_pow);
}
