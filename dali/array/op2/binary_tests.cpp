#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op.h"


TEST(ArrayBinary2, add) {
    int size = 10;

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        auto b = Array::arange({size}, dtype);
        auto dst = op2::add(a, b);
        EXPECT_TRUE(Array::equals(dst, (Array)(a + b)));
    }

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        auto b = Array::arange({size}, dtype);
        Array dst = Array::arange({size}, dtype) + 2;
        dst = op2::add(a, b);
        EXPECT_TRUE(Array::equals(dst, (Array)(a + b)));
    }

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        auto b = Array::arange({size}, dtype);
        Array dst = Array::arange({size}, dtype) + 2;
        dst += op2::add(a, b);
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
        dst -= op2::add(a, b);
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
        dst *= op2::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)((Array::arange({size}, dtype) + 2) * (a + b))
            )
        );
    }
}

TEST(ArrayBinary2, add_strided) {
    int size = 10;
    // single striding
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        auto b = Array::arange({2 * size}, dtype)[Slice(0, 2*size, 2)];
        Array dst = Array::arange({size}, dtype) + 2;
        dst -= op2::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)(Array::arange({size}, dtype) + 2 - (a + b))
            )
        );
    }

    // double striding:
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({size}, dtype);
        auto b = Array::arange({2 * size}, dtype)[Slice(0, 2*size, 2)];
        Array dst = ((Array)(Array::arange({2 * size}, dtype) + 2))[Slice(0, 2 * size, 2)];
        dst -= op2::add(a, b);
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
        auto a = Array::arange({3 * size}, dtype)[Slice(0, 3 * size, 3)];
        auto b = Array::arange({2 * size}, dtype)[Slice(0, 2 * size, 2)];
        Array dst = ((Array)(Array::arange({2 * size}, dtype) + 2))[Slice(0, 2 * size, 2)];
        dst -= op2::add(a, b);
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

TEST(ArrayBinary2, add_strided_nd) {
    int size = 10;
    // single striding
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({5, size}, dtype);
        auto b = Array::arange({5, 2 * size}, dtype)[Slice()][Slice(0, 2*size, 2)];
        Array dst = Array::arange({5, size}, dtype) + 2;
        dst -= op2::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)(Array::arange({5, size}, dtype) + 2 - (a + b))
            )
        );
    }
}

#define DALI_RTC_BINARY_TEST(funcname)\
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {\
        int size = 10;\
        auto a = Array::arange({5, size}, dtype);\
        auto b = Array::arange({5, size}, dtype) + 1;\
        Array dst = Array::arange({5, size}, dtype) + 2;\
        dst = op2::funcname(a, b);\
        EXPECT_TRUE(Array::equals(dst, (Array)(op::funcname(a, b))));\
    }\

TEST(ArrayBinary2, elementwise_binary_ops) {
    DALI_RTC_BINARY_TEST(eltmul);
    DALI_RTC_BINARY_TEST(eltdiv);
    DALI_RTC_BINARY_TEST(prelu);
    DALI_RTC_BINARY_TEST(pow);
    DALI_RTC_BINARY_TEST(equals);
}



