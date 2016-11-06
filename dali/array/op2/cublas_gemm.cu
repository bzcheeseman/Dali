#include "cublas_gemm.h"

#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/expression/array_wrapper.h"
#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"

#include <cublas_v2.h>
namespace {
    // convert cublas error into char* for human readable output
    const char* cublas_get_error_string(cublasStatus_t status) {
        switch (status) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        }
        return "unknown cublas error";
    }

    void check_cuda_status(cublasStatus_t err, const std::string& msg) {
        ASSERT2(err == CUBLAS_STATUS_SUCCESS, utils::make_message("could not ",
            msg, " (error = ", cublas_get_error_string(err), ")"));
    }
}

// dummy wrapper to keep track of blas-handle lifecycle
class CublasHandleHolder {
    private:
        cublasHandle_t blas_handle_;
        bool initialized_ = false;
    public:
        CublasHandleHolder() : initialized_(false) {
        }
        cublasHandle_t get() {
            if (!initialized_) {
                check_cuda_status(cublasCreate(&blas_handle_), "create cublas handle");
                initialized_ = true;
            }
            return blas_handle_;
        }
        ~CublasHandleHolder() {
            if (initialized_) {
                check_cuda_status(cublasDestroy(blas_handle_), "destroy cublas handle");
            }
        }
};

CublasHandleHolder cublas_handler;

void set_cublas_stream(cudaStream_t stream) {
    cublasSetStream(
        cublas_handler.get(),
        stream
    );
}

// inefficient cuda matrix multiply of integers on gpu
template<typename R>
__global__
void gemm_kernel(bool transpose_a, bool transpose_b, bool transpose_c,
                 int m, int n, int k, R alpha, const R *A, int lda,
                 const R *B, int ldb, R beta, R* C, int ldc) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int a_i_stride;
    int a_l_stride;
    if (transpose_a) {
        a_i_stride = 1;
        a_l_stride = lda;
    } else {
        a_i_stride = lda;
        a_l_stride = 1;
    }
    int b_j_stride;
    int b_l_stride;
    if (transpose_b) {
        b_j_stride = ldb;
        b_l_stride = 1;
    } else {
        b_j_stride = 1;
        b_l_stride = ldb;
    }
    int c_i_stride;
    int c_j_stride;
    if (transpose_c) {
        c_i_stride = 1;
        c_j_stride = ldc;
    } else {
        c_i_stride = ldc;
        c_j_stride = 1;
    }

    int l;
    if ((i < m) && (j < n)) {
        C[i * c_i_stride + j * c_j_stride] *= beta;
        for (l=0; l < k; l++) {
            C[i * c_i_stride + j * c_j_stride] += (
                alpha *
                A[i * a_i_stride + l * a_l_stride] *
                B[l * b_l_stride + j * b_j_stride]
            );
        }
    }
}

namespace expression {

CublasGemmAssignExpressionState::CublasGemmAssignExpressionState(
        std::shared_ptr<const expression::ArrayWrapper> dest,
        std::shared_ptr<const expression::Runnable> left,
        std::shared_ptr<const expression::Runnable> right,
        double result_multiplier,
        double destination_multiplier,
        memory::Device device) :
        CpuGemmAssignExpressionState(dest, left, right, result_multiplier, destination_multiplier),
        device_(device) {
}

void CublasGemmAssignExpressionState::run() const {
    Array dst = dest_->array_;
    Array lhs = left_->destination_op()->as_rvalue()->as_array()->array_;
    Array rhs = right_->destination_op()->as_rvalue()->as_array()->array_;
    auto op_dtype = dtype();
    void* dst_ptr = destination_multiplier_ == 0 ?
        dst.memory()->overwrite_data(device_) : dst.memory()->mutable_data(device_);
    const void* rhs_ptr = rhs.memory()->readonly_data(device_);
    const void* lhs_ptr = lhs.memory()->readonly_data(device_);
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

     // use null stream for now
    cudaStream_t stream = NULL;
    // check_cuda_status(cudaStreamCreate(&stream), "create gemm stream");

    if (op_dtype == DTYPE_INT32) {
        dim3 blocks(1,1);
        dim3 threads(16,16);
        gemm_kernel<int><<<blocks, threads, 0, stream>>>(
            !rhs_transpose, !lhs_transpose, true,
            m, n, k,
            /*alpha=*/result_multiplier_,
            (const int*)rhs_ptr, rhs_stride,
            (const int*)lhs_ptr, lhs_stride,
            /*beta=*/destination_multiplier_,
            (int*)dst_ptr, dst_stride
        );
    } else if (op_dtype == DTYPE_FLOAT) {
        set_cublas_stream(stream);
        const float result_multiplier_float = result_multiplier_;
        const float destination_multiplier_float = destination_multiplier_;
        cublasSgemm(
            /*stream=*/cublas_handler.get(),
            rhs_transpose ? CUBLAS_OP_T : CUBLAS_OP_N, lhs_transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
            m, n, k,
            /*alpha=*/&result_multiplier_float,
            (const float*)rhs_ptr, rhs_stride,
            (const float*)lhs_ptr, lhs_stride,
            /*beta=*/&destination_multiplier_float,
            (float*)dst_ptr, dst_stride
        );
    } else if (op_dtype == DTYPE_DOUBLE) {
        set_cublas_stream(stream);
        cublasDgemm(
            /*stream=*/cublas_handler.get(),
            rhs_transpose ? CUBLAS_OP_T : CUBLAS_OP_N, lhs_transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
            m, n, k,
            /*alpha=*/&result_multiplier_,
            (const double*)rhs_ptr, rhs_stride,
            (const double*)lhs_ptr, lhs_stride,
            /*beta=*/&destination_multiplier_,
            (double*)dst_ptr, dst_stride
        );
    } else {
        ASSERT2(false, utils::make_message("gemm only supports "
            DALI_ACCEPTABLE_DTYPE_STR " (got dtype=", op_dtype, ")."));
    }
    // check_cuda_status(cudaStreamSynchronize(stream), "synchronize gemm result");
    // check_cuda_status(cudaStreamDestroy(stream), "destroy gemm stream");
}

}  // namespace expression
