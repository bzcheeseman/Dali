#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/dot.h"
#include "dali/array/op.h"


Array reference_gemm(const Array& a, const Array& b) {
    ASSERT2(a.ndim() == 2 && b.ndim() == 2 &&
            a.shape()[1] == b.shape()[0],
        "incompatible dimensions for gemm");
    Array out = Array::zeros({a.shape()[0], b.shape()[1]}, a.dtype());
    for (int i = 0; i < a.shape()[0]; i++) {
        for (int k = 0; k < b.shape()[1]; k++) {
            double res = 0.0;
            for (int j = 0; j < a.shape()[1]; j++) {
                res += ((double)a[i][j]) * ((double)b[j][k]);
            }
            out[i][k] = res;
        }
    }
    return out;
}

TEST(RTCTests, dot) {
    auto a = Array::ones({3, 4});
    auto b = Array::ones({4, 5});
    auto res = Array::zeros({3, 5});
    res = op::dot2(a, b);
    auto reference_res = reference_gemm(a, b);
    // currently without jit the following test cannot be run:
    // EXPECT_TRUE(Array::allclose(res, reference_gemm(a, b), 1e-6));
}
