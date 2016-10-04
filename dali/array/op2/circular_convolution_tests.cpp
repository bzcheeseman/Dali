#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op2/circular_convolution.h"
#include "dali/array/op.h"
#include "dali/array/op2/operation.h"

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
    Array shift = Array({2, 3}, DTYPE_FLOAT);

    x     = initializer::uniform(-1.0, 1.0);
    shift = initializer::uniform(-1.0, 1.0);

    shift = shift[Slice()][Slice()][Broadcast()];

    Array res = op2::circular_convolution(x, shift);
    Array expected_res = reference_circular_convolution(x, shift);

    EXPECT_TRUE(Array::allclose(res, expected_res, 1e-6));
}

TEST(RTCTests, circular_convolution_1d) {
    Array x({3}, DTYPE_FLOAT);
    Array shift = Array({6}, DTYPE_FLOAT);

    x     = initializer::uniform(-1.0, 1.0);
    shift = initializer::uniform(-1.0, 1.0);

    shift = shift[Slice(0, 6, 2)];

    Array res = op2::circular_convolution(x, shift);
    Array expected_res = reference_circular_convolution(x, shift);

    EXPECT_TRUE(Array::allclose(res, expected_res, 1e-6));
}


TEST(RTCTests, chained_circular_convolution) {
    int size = 10;
    // single striding
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = Array::arange({5, size}, dtype);
        Array b = Array::arange({5, 2 * size}, dtype)[Slice()][Slice(0, 2*size, 2)];
        Array c = Array::arange({5, 3 * size}, dtype)[Slice()][Slice(0, 3*size, 3)];
        Array dst = Array::arange({5, size}, dtype) + 2;
        // the circular convolution and the addition are a single kernel:
        dst -= op2::circular_convolution(op2::add(a, b), c);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (Array)(Array::arange({5, size}, dtype) + 2 - (op::circular_convolution(a + b, c)))
            )
        );
    }
}




TEST(RTCTests, circular_conv_unary) {
    int size = 10;
    {
        Array res = op2::circular_convolution(
            Array::arange({5, size}, DTYPE_FLOAT) + 1,
            op2::relu(2.5)
        );
    }
    {
        Array res = op2::circular_convolution(
            2.5,
            Array::arange({5, size}, DTYPE_FLOAT) + 1
        );
    }
    {
        Array res = op2::circular_convolution(2.1, 2.5);
        EXPECT_NEAR((double)res, 2.1 * 2.5, 1e-9);
    }
    {
        Array res = op2::circular_convolution(
            Array::arange({5, size}, DTYPE_FLOAT) + 1, 2.5
        );
    }
}

