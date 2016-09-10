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
