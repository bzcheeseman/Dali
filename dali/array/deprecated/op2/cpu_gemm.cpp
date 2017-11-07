#include "cpu_gemm.h"

#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"
#include "dali/array/mshadow_extension/reference_gemm.h"

extern "C" {
    #include <cblas.h>
}

namespace expression {
// support integers in gemm
void igemm(bool transpose_a, bool transpose_b,
           size_t m, size_t n, size_t k, double alpha, const int* a,
           size_t lda, const int* b, size_t ldb, double beta, int* c, size_t ldc) {
    ReferenceGemm<int>(
        transpose_a, transpose_b, true, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
    );
}

// compute gemm col-major transpose + stride argument
std::tuple<bool, int> gemm_stride_transpose(const Array& array) {
    if (array.strides().size() == 0) {
        return std::make_tuple(false, array.normalized_strides()[0]);
    }
    const std::vector<int>& strides = array.strides();
    if (strides[0] == 1) {
        return std::make_tuple(true, strides[1]);
    } else if (strides[1] == 1) {
        return std::make_tuple(false, strides[0]);
    } else {
        ASSERT2(false, utils::make_message("gemm "
            "only supports arrays with a single stride (array strides: ",
            strides, ")"));
        return std::make_tuple(false, 1);
    }
}

CpuGemmAssignExpressionNode::CpuGemmAssignExpressionNode(
        std::shared_ptr<const expression::ArrayWrapper> dest,
        std::shared_ptr<const expression::Runnable> left,
        std::shared_ptr<const expression::Runnable> right,
        double result_multiplier,
        double destination_multiplier) :
        dest_(dest),
        left_(left),
        right_(right),
        result_multiplier_(result_multiplier),
        destination_multiplier_(destination_multiplier) {
}

std::string CpuGemmAssignExpressionNode::name() const {
    return "gemm";
}

DType CpuGemmAssignExpressionNode::dtype() const {
    return left_->dtype();
}

std::vector<int> CpuGemmAssignExpressionNode::bshape() const {
    return dest_->bshape();
}

int CpuGemmAssignExpressionNode::ndim() const {
    return 2;
}

std::vector<std::shared_ptr<const ExpressionNode>> CpuGemmAssignExpressionNode::arguments() const {
    return {dest_, left_, right_};
}

void CpuGemmAssignExpressionNode::run() const {
    // TODO(szymon): make less brittle (please :)
    Array dst = dest_->array_;
    Array lhs = left_->destination_op()->as_rvalue()->as_array()->array_;
    Array rhs = right_->destination_op()->as_rvalue()->as_array()->array_;
    auto op_dtype = dtype();
    auto device = memory::Device::cpu();
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

std::shared_ptr<const ExpressionNode> CpuGemmAssignExpressionNode::destination_op() const {
    return dest_;
}

}  // namespace expression
