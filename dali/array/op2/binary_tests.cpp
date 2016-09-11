#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op.h"


TEST(RTCTests, add) {
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

TEST(RTCTests, add_strided) {
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

TEST(RTCTests, add_strided_nd) {
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

TEST(RTCTests, elementwise_binary_ops) {
    DALI_RTC_BINARY_TEST(eltmul);
    DALI_RTC_BINARY_TEST(eltdiv);
    DALI_RTC_BINARY_TEST(prelu);
    DALI_RTC_BINARY_TEST(pow);
    DALI_RTC_BINARY_TEST(equals);
}

namespace {
    void reference_circular_convolution(const Array& content, const Array& shift, Array* dest_ptr) {
        auto& dest = *dest_ptr;
        if (content.ndim() == 1) {
            for (int col = 0; col < content.shape()[0]; ++col) {
                for (int shift_idx = 0; shift_idx < content.shape()[0]; ++shift_idx) {
                    // here we intentionally avoid expensive % operation.
                    int offset = col + shift_idx;
                    if (offset >= content.shape()[0]) {
                        offset -= content.shape()[0];
                    }
                    dest[col] = dest[col] + content[offset] * shift[shift_idx];
                }
            }
        } else {
            for (int i = 0; i < content.shape()[0];i++) {
                Array dest_slice = dest[i];
                reference_circular_convolution(content[i], shift[i], &dest_slice);
            }
        }
    }

    Array reference_circular_convolution(const Array& content, const Array& shift) {
        ASSERT2(content.ndim() == shift.ndim(), "content and shift must have the same ndim");
        std::vector<int> final_bshape;
        std::vector<int> final_shape;
        for (int i = 0; i < content.ndim(); i++) {
            ASSERT2(content.bshape()[i] == -1 ||
                    shift.bshape()[i] == -1 ||
                    content.bshape()[i] == shift.bshape()[i],
                    "content and shift must have same sizes or broadcasted sizes"
            );
            final_bshape.emplace_back(
                std::max(content.bshape()[i], shift.bshape()[i])
            );
            final_shape.emplace_back(
                std::abs(final_bshape.back())
            );
        }
        Array res = Array::zeros(final_shape, content.dtype(), content.preferred_device());
        auto shift_reshaped = shift.reshape_broadcasted(final_bshape);
        auto content_reshaped = content.reshape_broadcasted(final_bshape);
        reference_circular_convolution(content_reshaped, shift_reshaped, &res);
        return res;
    }
}

TEST(RTCTests, circular_convolution) {
    Array x({2, 3, 4}, DTYPE_FLOAT);
    Array shift({2, 3, 4}, DTYPE_FLOAT);

    x     = initializer::uniform(-1.0, 1.0);
    shift = initializer::uniform(-1.0, 1.0);

    Array res = op2::circular_convolution(x, shift);
    Array expected_res = reference_circular_convolution(x, shift);

    EXPECT_TRUE(Array::allclose(res, expected_res, 1e-6));
}


TEST(RTCTests, circular_convolution_broadcast) {
    Array x({2, 3, 4}, DTYPE_FLOAT);
    Array shift = Array({2, 3}, DTYPE_FLOAT)[Slice()][Slice()][Broadcast()];

    x     = initializer::uniform(-1.0, 1.0);
    shift = initializer::uniform(-1.0, 1.0);

    Array res = op2::circular_convolution(x, shift);
    Array expected_res = reference_circular_convolution(x, shift);

    EXPECT_TRUE(Array::allclose(res, expected_res, 1e-6));
}

