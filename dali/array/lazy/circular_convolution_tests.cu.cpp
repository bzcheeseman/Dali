#include <gtest/gtest.h>
#include "dali/array/array.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"
#include "dali/array/functor.h"
#include "dali/array/lazy_op.h"
#include "dali/array/lazy/circular_convolution.h"

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
        for (int i = 0; i < content.ndim(); i++) {
            ASSERT2(content.bshape()[i] == -1 ||
                    shift.bshape()[i] == -1 ||
                    content.bshape()[i] == shift.bshape()[i],
                    "content and shift must have same sizes or broadcasted sizes"
            );
        }
        Array res = Array::zeros_like(content);
        reference_circular_convolution(content, shift, &res);
        return res;
    }
}

TEST(ArrayCircularConvolutionTests, circular_convolution) {
    Array x({2, 3, 4}, DTYPE_FLOAT);
    Array shift({2, 3, 4}, DTYPE_FLOAT);

    x     = initializer::uniform(-1.0, 1.0);
    shift = initializer::uniform(-1.0, 1.0);


    Array res = lazy::circular_convolution(x, shift);
    Array expected_res = reference_circular_convolution(x, shift);

    EXPECT_TRUE(Array::allclose(res, expected_res, 1e-6));
}
