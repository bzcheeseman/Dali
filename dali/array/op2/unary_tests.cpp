#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op.h"
#include "dali/array/op2/operation.h"
#include "dali/array/functor.h"

#define DALI_DEFINE_REFERENCE_UNARY(FUNCNAME, FUNCTOR_NAME)\
    Array reference_ ##FUNCNAME (Array x) {\
        Array out = Array::zeros_like(x);\
        auto raveled_x = x.ravel();\
        auto raveled_out = out.ravel();\
        if (x.dtype() == DTYPE_DOUBLE) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                raveled_out(i) = functor::FUNCTOR_NAME<double>::Map((double)raveled_x(i));\
            }\
        } else if (x.dtype() == DTYPE_FLOAT) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                raveled_out(i) = functor::FUNCTOR_NAME<float>::Map((float)raveled_x(i));\
            }\
        } else {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                raveled_out(i) = functor::FUNCTOR_NAME<int>::Map((int)raveled_x(i));\
            }\
        }\
        return out;\
    }

#define DALI_RTC_UNARY_TEST(funcname)\
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {\
        int size = 10;\
        auto a = Array::arange({5, size}, dtype) + 1;\
        auto dst = Array::zeros({5, size}, dtype);\
        dst = op::funcname(a);\
        EXPECT_TRUE(Array::allclose(dst, reference_ ##funcname(a), 1e-6));\
    }\

DALI_DEFINE_REFERENCE_UNARY(softplus, softplus);
DALI_DEFINE_REFERENCE_UNARY(sigmoid, sigmoid);
DALI_DEFINE_REFERENCE_UNARY(tanh, tanh);
DALI_DEFINE_REFERENCE_UNARY(log, log);
DALI_DEFINE_REFERENCE_UNARY(cube, cube);
DALI_DEFINE_REFERENCE_UNARY(sqrt, sqrt_f);
DALI_DEFINE_REFERENCE_UNARY(rsqrt, rsqrt);
DALI_DEFINE_REFERENCE_UNARY(eltinv, inv);
DALI_DEFINE_REFERENCE_UNARY(relu, relu);
DALI_DEFINE_REFERENCE_UNARY(abs, abs);
DALI_DEFINE_REFERENCE_UNARY(sign, sign);
DALI_DEFINE_REFERENCE_UNARY(identity, identity);


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

#define DALI_DEFINE_REFERENCE_UNARY_SCALAR(FUNCNAME, FUNCTOR_NAME)\
    Array reference_scalar_ ##FUNCNAME (Array x, double scalar) {\
        Array out = Array::zeros_like(x);\
        auto raveled_x = x.ravel();\
        auto raveled_out = out.ravel();\
        if (x.dtype() == DTYPE_DOUBLE) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                raveled_out(i) = functor::FUNCTOR_NAME<double>::Map((double)raveled_x(i), scalar);\
            }\
        } else if (x.dtype() == DTYPE_FLOAT) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                raveled_out(i) = functor::FUNCTOR_NAME<float>::Map((float)raveled_x(i), scalar);\
            }\
        } else {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                raveled_out(i) = functor::FUNCTOR_NAME<int>::Map((int)raveled_x(i), scalar);\
            }\
        }\
        return out;\
    }

#define DALI_RTC_SCALAR_UNARY_TEST(funcname)\
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {\
        int size = 10;\
        auto a = Array::arange({5, size}, dtype) + 1;\
        auto dst = Array::zeros({5, size}, dtype);\
        dst = op::funcname(a, 2.0);\
        EXPECT_TRUE(Array::allclose(dst, reference_scalar_ ##funcname(a, 2.0), 1e-5));\
    }\

DALI_DEFINE_REFERENCE_UNARY_SCALAR(add, add);
DALI_DEFINE_REFERENCE_UNARY_SCALAR(sub, sub);
DALI_DEFINE_REFERENCE_UNARY_SCALAR(eltmul, eltmul);
DALI_DEFINE_REFERENCE_UNARY_SCALAR(eltdiv, eltdiv);
DALI_DEFINE_REFERENCE_UNARY_SCALAR(pow, power);

TEST(RTCTests, scalar_unary) {
    DALI_RTC_SCALAR_UNARY_TEST(add);
    DALI_RTC_SCALAR_UNARY_TEST(sub);
    DALI_RTC_SCALAR_UNARY_TEST(eltmul);
    DALI_RTC_SCALAR_UNARY_TEST(eltdiv);
    DALI_RTC_SCALAR_UNARY_TEST(pow);
}
