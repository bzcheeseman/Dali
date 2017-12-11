#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/gemm/reference_gemm.h"
#include "dali/array/gemm/gemm_utils.h"
#include "dali/array/op/dot.h"
#include "dali/array/expression/computation.h"

extern "C" {
    #include <cblas.h>
}

// support integers in gemm
void igemm(bool transpose_a, bool transpose_b,
           size_t m, size_t n, size_t k, double alpha, const int* a,
           size_t lda, const int* b, size_t ldb, double beta, int* c, size_t ldc) {
    ReferenceGemm<int>(
        transpose_a, transpose_b, true, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
    );
}

struct CpuGemmImpl : public Computation {
    using Computation::Computation;
    virtual void run() {
        // TODO(szymon): make less brittle (please :)
        Array dst = left_;
        op::MatMul* mm = static_cast<op::MatMul*>(right_.expression().get());
        Array lhs = mm->left_;
        Array rhs = mm->right_;
        auto op_dtype = dst.dtype();
        auto device = memory::Device::cpu();
        double destination_multiplier_ = 0;
        double result_multiplier_ = 1.0;
        void* dst_ptr = destination_multiplier_ == 0 ?
            dst.memory()->overwrite_data(device) : dst.memory()->mutable_data(device);
        const void* rhs_ptr = rhs.memory()->readonly_data(device);
        const void* lhs_ptr = lhs.memory()->readonly_data(device);
        bool rhs_transpose, lhs_transpose, dst_transpose;
        int rhs_stride, lhs_stride, dst_stride;
        std::tie(rhs_transpose, rhs_stride) = gemm_stride_transpose(rhs);
        std::tie(lhs_transpose, lhs_stride) = gemm_stride_transpose(lhs);
        std::tie(dst_transpose, dst_stride) = gemm_stride_transpose(dst);
        // in row major:
        // dst = result_multiplier * left * right + destination_multiplier * dst
        // in col major:
        // dst.T = result_multiplier * right.T * left.T + destination_multiplier * dst.T
        int m = rhs.shape()[1],
            n = lhs.shape()[0],
            k = rhs.shape()[0];

        if (op_dtype == DTYPE_INT32) {
            igemm(
                !rhs_transpose, !lhs_transpose,
                m, n, k,
                /*alpha=*/result_multiplier_,
                (const int*)rhs_ptr, rhs_stride,
                (const int*)lhs_ptr, lhs_stride,
                /*beta=*/destination_multiplier_,
                (int*)dst_ptr, dst_stride
            );
        } else if (op_dtype == DTYPE_FLOAT) {
            cblas_sgemm(CblasColMajor,
                rhs_transpose ? CblasTrans : CblasNoTrans, lhs_transpose ? CblasTrans : CblasNoTrans,
                m, n, k,
                /*alpha=*/result_multiplier_,
                (const float*)rhs_ptr, rhs_stride,
                (const float*)lhs_ptr, lhs_stride,
                /*beta=*/destination_multiplier_,
                (float*)dst_ptr, dst_stride
            );
        } else if (op_dtype == DTYPE_DOUBLE) {
            cblas_dgemm(CblasColMajor,
                rhs_transpose ? CblasTrans : CblasNoTrans, lhs_transpose ? CblasTrans : CblasNoTrans,
                m, n, k,
                /*alpha=*/result_multiplier_,
                (const double*)rhs_ptr, rhs_stride,
                (const double*)lhs_ptr, lhs_stride,
                /*beta=*/destination_multiplier_,
                (double*)dst_ptr, dst_stride
            );
        } else {
            ASSERT2(false, utils::make_message("gemm only supports "
                DALI_ACCEPTABLE_DTYPE_STR " (got dtype=", op_dtype, ")."));
        }
    }
};

int cpu_gemm_impl = register_implementation(
    typeid(op::MatMul).name(),
    [](Array dest, OPERATOR_T operator_t, Array x) -> std::shared_ptr<Computation> {
        if (dest.preferred_device().is_cpu()) {
            return std::make_shared<CpuGemmImpl>(dest, operator_t, x);
        } else {
            return nullptr;
        }
    }
);
