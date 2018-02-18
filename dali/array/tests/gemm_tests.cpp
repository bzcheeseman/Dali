#include <gtest/gtest.h>
#include "dali/array/gemm/gemm_utils.h"

TEST(GemmTests, blas_friendly_tensor) {
    Array s1 = Array::ones({3,4}, DTYPE_INT32);
    Array s2 = Array::ones({1,4}, DTYPE_INT32);
    Array s3 = Array::ones({3,1}, DTYPE_INT32);
    Array s4 = Array::ones({1,1}, DTYPE_INT32);

    auto verify_result = [](std::string testname, Array arr, bool expected_tranpose, int expected_stride) {
        SCOPED_TRACE(testname);
        bool tranpose;
        int stride;
        std::tie(tranpose, stride) = gemm_stride_transpose(arr);
        EXPECT_EQ(expected_tranpose, tranpose);
        EXPECT_EQ(expected_stride, stride);
    };

    verify_result("3x4, not transposed", s1,             false, 4);
    verify_result("1x4, not transposed", s2,             false, 4);
    verify_result("3x1, not transposed", s3,             false, 1);
    verify_result("1x1, not transposed", s4,             false, 1);

    verify_result("3x4, transposed",     s1.transpose(), true,  4);
    verify_result("1x4, transposed",     s2.transpose(), true,  4);
    verify_result("3x1, transposed",     s3.transpose(), true,  1);
    verify_result("1x1, transposed",     s4.transpose(), false, 1);
}
