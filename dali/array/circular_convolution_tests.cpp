#include <gtest/gtest.h>

#include "dali/array/op/circular_convolution.h"
#include "dali/array/op/uniform.h"
#include "dali/array/op/arange.h"
#include "dali/array/op/unary.h"
#include "dali/array/expression/assignment.h"


namespace {
    void reference_circular_convolution(const Array& content, const Array& shift, Array* dest_ptr) {
        auto& dest = *dest_ptr;
        if (content.ndim() == 1) {
            for (int col = 0; col < std::max(content.shape()[0], shift.shape()[0]); ++col) {
                for (int shift_idx = 0; shift_idx < content.shape()[0]; ++shift_idx) {
                    // here we intentionally avoid expensive % operation.
                    int offset = col + shift_idx;
                    if (offset >= content.shape()[0]) {
                        offset -= content.shape()[0];
                    }
                    op::assign(dest[col], OPERATOR_T_EQL, dest[col] + content[offset] * shift[shift_idx]).eval();
                }
            }
        } else {
            for (int i = 0; i < std::max(content.shape()[0], shift.shape()[0]);i++) {
                Array dest_slice = dest[i];
                reference_circular_convolution(content[i], shift[i], &dest_slice);
            }
        }
    }

    Array reference_circular_convolution(const Array& content, const Array& shift) {
        ASSERT2(content.ndim() == shift.ndim(), "content and shift must have the same ndim");
        std::vector<int> final_shape;
        for (int i = 0; i < content.ndim(); i++) {
            ASSERT2(content.shape()[i] == 1 ||
                    shift.shape()[i] == 1 ||
                    content.shape()[i] == shift.shape()[i],
                    "content and shift must have same sizes or broadcasted sizes"
            );
            final_shape.emplace_back(
                std::max(content.shape()[i], shift.shape()[i])
            );
        }
        Array res = Array::zeros(final_shape, content.dtype(), content.preferred_device());
        auto shift_reshaped = shift.broadcast_to_shape(final_shape);
        auto content_reshaped = content.broadcast_to_shape(final_shape);
        reference_circular_convolution(content_reshaped, shift_reshaped, &res);
        return res;
    }
}

TEST(JITTests, circular_convolution) {
    auto x     = op::uniform(-1.0f, 1.0f, {2, 3, 4});
    auto shift = op::uniform(-1.0f, 1.0f, {2, 3, 4});

    Array res = op::circular_convolution(x, shift);
    Array expected_res = reference_circular_convolution(x, shift);

    EXPECT_TRUE(Array::allclose(res, expected_res, 1e-6));
}


TEST(JITTests, circular_convolution_broadcast) {
    auto x     = op::arange(2 * 3 * 4).reshape({2, 3, 4}).astype(DTYPE_FLOAT);
    auto shift = op::arange(2 * 3).reshape({2, 3}).astype(DTYPE_FLOAT);

    shift = shift[Slice()][Slice()][Broadcast()];

    Array res = op::circular_convolution(x, shift);
    Array expected_res = reference_circular_convolution(x, shift);

    EXPECT_TRUE(Array::allclose(res, expected_res, 1e-6));
}

TEST(JITTests, circular_convolution_1d) {
    auto x     = op::uniform(-1.0f, 1.0f, {3});
    auto shift = op::uniform(-1.0f, 1.0f, {6});

    shift = shift[Slice(0, 6, 2)];

    Array res = op::circular_convolution(x, shift);
    Array expected_res = reference_circular_convolution(x, shift);

    EXPECT_TRUE(Array::allclose(res, expected_res, 1e-6));
}


TEST(JITTests, chained_circular_convolution) {
    int size = 10;
    // single striding
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = op::arange(5 * size).reshape({5, size}).astype(dtype);
        Array b = op::arange(5 * 2 * size).reshape({5, 2 * size}).astype(dtype)[Slice()][Slice(0, 2*size, 2)];
        Array c = op::arange(5 * 3 * size).reshape({5, 3 * size}).astype(dtype)[Slice()][Slice(0, 3*size, 3)];
        Array dst = op::arange(5 * size).reshape({5, size}).astype(dtype) + 2;
        // the circular convolution and the addition are a single kernel:
        dst -= op::circular_convolution(a + b, c);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (op::arange(5 * size).reshape({5, size}).astype(dtype) + 2) - op::circular_convolution(a + b, c)
            )
        );
    }
}




TEST(JITTests, circular_conv_unary) {
    int size = 10;
    {
        Array res = op::circular_convolution(
            op::arange(5 * size).reshape({5, size}).astype(DTYPE_FLOAT) + 1,
            op::relu(2.5)
        );
    }
    {
        Array res = op::circular_convolution(
            2.5,
            op::arange(5 * size).reshape({5, size}).astype(DTYPE_FLOAT) + 1
        );
    }
    {
        Array res = op::circular_convolution(2.1, 2.5);
        EXPECT_NEAR((double)res, 2.1 * 2.5, 1e-9);
    }
    {
        Array res = op::circular_convolution(
            op::arange(5 * size).reshape({5, size}).astype(DTYPE_FLOAT) + 1, 2.5
        );
    }
}

