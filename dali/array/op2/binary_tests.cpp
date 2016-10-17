#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op.h"
#include "dali/array/op2/operation.h"
#include "dali/array/functor.h"

TEST(RTCTests, add) {
    int size = 10;

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        auto b = Array::arange({size}, dtype);
        Array dst = op::add(a, b);
        EXPECT_TRUE(Array::equals(dst, (Array)(a + b)));
    }

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        auto b = Array::arange({size}, dtype);
        Array dst = Array::arange({size}, dtype) + 2;
        dst = op::add(a, b);
        EXPECT_TRUE(Array::equals(dst, (Array)(a + b)));
    }

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        auto b = Array::arange({size}, dtype);
        Array dst = Array::arange({size}, dtype) + 2;
        dst += op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)(Array::arange({size}, dtype) + 2 + a + b)
            )
        );
    }

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        auto b = Array::arange({size}, dtype);
        Array dst = Array::arange({size}, dtype) + 2;
        dst -= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)(Array::arange({size}, dtype) + 2 - (a + b))
            )
        );
    }

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        auto b = Array::arange({size}, dtype);
        Array dst = Array::arange({size}, dtype) + 2;
        dst *= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)((Array::arange({size}, dtype) + 2) * (a + b))
            )
        );
    }
}

TEST(RTCTests, add_strided) {
    int size = 10;
    // single striding
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        Array b = Array::arange({2 * size}, dtype)[Slice(0, 2*size, 2)];
        Array dst = Array::arange({size}, dtype) + 2;
        dst -= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)(Array::arange({size}, dtype) + 2 - (a + b))
            )
        );
    }

    // double striding:
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        Array a = Array::arange({size}, dtype);
        Array b = Array::arange({2 * size}, dtype)[Slice(0, 2*size, 2)];
        Array dst = ((Array)(Array::arange({2 * size}, dtype) + 2))[Slice(0, 2 * size, 2)];
        dst -= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)(
                    ((Array)(Array::arange({2 * size}, dtype) + 2))[Slice(0, 2 * size, 2)] -
                    (a + b)
                )
            )
        );
    }

    // triple striding:
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        Array a = Array::arange({3 * size}, dtype)[Slice(0, 3 * size, 3)];
        Array b = Array::arange({2 * size}, dtype)[Slice(0, 2 * size, 2)];
        Array dst = ((Array)(Array::arange({2 * size}, dtype) + 2))[Slice(0, 2 * size, 2)];
        dst -= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)(
                    ((Array)(Array::arange({2 * size}, dtype) + 2))[Slice(0, 2 * size, 2)] -
                    (a + b)
                )
            )
        );
    }
}

TEST(RTCTests, add_strided_nd) {
    int size = 10;
    // single striding
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        Array a = Array::arange({5, size}, dtype);
        Array b = Array::arange({5, 2 * size}, dtype)[Slice()][Slice(0, 2*size, 2)];
        Array dst = Array::arange({5, size}, dtype) + 2;
        dst -= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)(Array::arange({5, size}, dtype) + 2 - (a + b))
            )
        );
    }
}


#define DALI_DEFINE_REFERENCE_BINARY_OP(FUNCNAME, FUNCTOR_NAME)\
    Array reference_ ##FUNCNAME (Array x, Array y) {\
        Array out = Array::zeros_like(x);\
        auto raveled_x = x.ravel();\
        auto raveled_y = y.ravel();\
        auto raveled_out = out.ravel();\
        if (x.dtype() == DTYPE_DOUBLE) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                raveled_out(i) = functor::FUNCTOR_NAME<double>::Map((double)raveled_x(i), (double)raveled_y(i));\
            }\
        } else if (x.dtype() == DTYPE_FLOAT) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                raveled_out(i) = functor::FUNCTOR_NAME<float>::Map((float)raveled_x(i), (float)raveled_y(i));\
            }\
        } else {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                raveled_out(i) = functor::FUNCTOR_NAME<int>::Map((int)raveled_x(i), (int)raveled_y(i));\
            }\
        }\
        return out;\
    }

#define DALI_RTC_BINARY_TEST(funcname)\
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {\
        int size = 4;\
        auto a = Array::arange({5, size}, dtype);\
        auto b = Array::arange({5, size}, dtype) + 1;\
        Array dst = Array::arange({5, size}, dtype) + 2;\
        dst = op::funcname(a, b);\
        Array reference = reference_ ##funcname(a, b);\
        EXPECT_TRUE(Array::allclose(dst, reference, 1e-7));\
    }\

DALI_DEFINE_REFERENCE_BINARY_OP(eltmul, eltmul);
DALI_DEFINE_REFERENCE_BINARY_OP(eltdiv, eltdiv);
DALI_DEFINE_REFERENCE_BINARY_OP(prelu, prelu);
DALI_DEFINE_REFERENCE_BINARY_OP(pow, power);
DALI_DEFINE_REFERENCE_BINARY_OP(equals, equals);

TEST(RTCTests, elementwise_binary_ops) {
    DALI_RTC_BINARY_TEST(eltmul);
    DALI_RTC_BINARY_TEST(eltdiv);
    DALI_RTC_BINARY_TEST(prelu);
    DALI_RTC_BINARY_TEST(pow);
    DALI_RTC_BINARY_TEST(equals);
}


TEST(RTCTests, chained_add) {
    int size = 10;
    // single striding
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({5, size}, dtype);
        Array b = Array::arange({5, 2 * size}, dtype)[Slice()][Slice(0, 2*size, 2)];
        Array c = Array::arange({5, 3 * size}, dtype)[Slice()][Slice(0, 3*size, 3)];
        Array dst = Array::arange({5, size}, dtype) + 2;
        // these two additions are a single kernel:
        dst -= op::add(op::add(a, b), c);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)(Array::arange({5, size}, dtype) + 2 - (a + b + c))
            )
        );
    }
}

TEST(RTCTests, cast_binary) {
    // auto casts to the right type before adding:
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        Array res = op::add(
            Array::arange({10}, dtype),
            Array::arange({10}, DTYPE_INT32)
        );
        EXPECT_EQ(dtype, res.dtype());
        EXPECT_TRUE(Array::allclose(Array::arange({10}, dtype) * 2, res, 1e-8));
    }
}
