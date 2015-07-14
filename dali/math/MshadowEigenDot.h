#ifndef DALI_MATH_MSHADOW_EIGEN_DOT_H
#define DALI_MATH_MSHADOW_EIGEN_DOT_H
#if MSHADOW_USE_EIGEN_DOT
#include <Eigen/Eigen>
#include <mshadow/tensor.h>
#include <cblas.h>
#include "dali/math/BlasSwitcher.h"

// Eigen Backend for Dot-Product in Mshadow
// Causes Adagrad to be slower.

namespace mshadow {
namespace expr {

struct CBlasEngine {
    inline static CBLAS_TRANSPOSE GetT(bool t) {
    return t ? CblasTrans : CblasNoTrans;
    }
    inline static void SetStream(mshadow::Stream<mshadow::cpu> *stream) {
    }
    inline static void gemm(bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda, const float *B, int ldb,
                          float beta, float *C, int ldc) {
        cblas_sgemm(CblasColMajor, GetT(transa), GetT(transb),
                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    inline static void gemm(bool transa, bool transb,
                          int m, int n, int k, double alpha,
                          const double *A, int lda, const double *B, int ldb,
                          double beta, double *C, int ldc) {
        cblas_dgemm(CblasColMajor, GetT(transa), GetT(transb),
                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    inline static void gemv(bool trans, int m, int n,
                          float alpha, const float *A, int lda,
                          const float *X, int incX,
                          float beta, float *Y, int incY) {
        cblas_sgemv(CblasColMajor, GetT(trans), m, n, alpha,
                    A, lda, X, incX, beta, Y, incY);
    }
    inline static void gemv(bool trans, int m, int n, double alpha,
                          const double *A, int lda,
                          const double *X, int incX,
                          double beta, double *Y, int incY) {
        cblas_dgemv(CblasColMajor, GetT(trans), m, n, alpha,
                    A, lda, X, incX, beta, Y, incY);
    }
    inline static void ger(int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
        cblas_sger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
    }
    inline static void ger(int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda) {
        cblas_dger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
    }
};

template<>
struct BLASEngine<cpu> {
    template<typename R>
    using eigen_mat_t = Eigen::Map<Eigen::Matrix<R, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> >;

    template<typename R>
    using eigen_vector_t = Eigen::Map<Eigen::Matrix<R, Eigen::Dynamic, 1> >;

    inline static void SetStream(Stream<cpu> *stream) {
    }

    inline static CBLAS_TRANSPOSE GetT(bool t) {
        return t ? CblasTrans : CblasNoTrans;
    }

    #define GEMM_EIGEN(dtype) \
        inline static void gemm(bool trans_rhs, bool trans_lhs,\
                                int m, int n, int k, dtype alpha,\
                                const dtype *rhs_ptr, int ldb, const dtype *lhs_ptr, int lda,\
                                dtype beta, dtype *C, int ldc) {\
            if (blas::should_use_cblas()) {\
                CBlasEngine::gemm(trans_rhs, trans_lhs, m, n, k, alpha, rhs_ptr, ldb, lhs_ptr, lda, beta, C, ldc);\
                return;\
            }\
            /* collect sizes from arguments.*/\
            /* Thanks blas!*/\
            const int rhs_size1 = trans_rhs ? k : m;\
            const int rhs_size0 = trans_rhs ? m : k;\
            const int lhs_size1 = trans_lhs ? n : k;\
            const int lhs_size0 = trans_lhs ? k : n;\
            const eigen_mat_t<dtype> lhs(\
                const_cast<dtype*>(lhs_ptr), lhs_size0, lhs_size1\
            );\
            const eigen_mat_t<dtype> rhs(\
                const_cast<dtype*>(rhs_ptr), rhs_size0, rhs_size1\
            );\
            eigen_mat_t<dtype> dst(\
                C,\
                trans_lhs ? lhs_size1 : lhs_size0,\
                trans_rhs ? rhs_size0 : rhs_size1\
            );\
            if (trans_lhs && trans_rhs) {\
                if (beta != 0.0f) {\
                    dst.noalias() = alpha * lhs.transpose() * rhs.transpose() + dst * beta;\
                } else {\
                    dst.noalias() = alpha * lhs.transpose() * rhs.transpose();\
                }\
            } else if (trans_lhs && !trans_rhs) {\
                if (beta != 0.0f) {\
                    dst.noalias() = alpha * lhs.transpose() * rhs + dst * beta;\
                } else {\
                    dst.noalias() = alpha * lhs.transpose() * rhs;\
                }\
            } else if (!trans_lhs && trans_rhs) {\
                if (beta != 0.0f) {\
                    dst.noalias() = alpha * lhs * rhs.transpose() + dst * beta;\
                } else {\
                    dst.noalias() = alpha * lhs * rhs.transpose();\
                }\
            } else if (!trans_lhs && !trans_rhs) {\
                if (beta != 0.0f) {\
                    dst.noalias() = alpha * lhs * rhs + dst * beta;\
                } else {\
                    dst.noalias() = alpha * lhs * rhs;\
                }\
            }\
        }

    #define GEMV_EIGEN(dtype)\
        inline static void gemv(bool trans_rhs, int rhs_size1, int rhs_size0,\
                                dtype alpha,\
                                const dtype *rhs_ptr, int rhs_stride,\
                                const dtype *lhs_ptr, int lhs_stride,\
                                dtype beta,\
                                dtype *dst_ptr, int dst_stride) {\
            if (blas::should_use_cblas()) {\
                CBlasEngine::gemv(trans_rhs, rhs_size1, rhs_size0, alpha, rhs_ptr, rhs_stride, lhs_ptr, lhs_stride, beta, dst_ptr, dst_stride);\
                return;\
            }\
            const eigen_mat_t<dtype> rhs(\
                const_cast<dtype*>(rhs_ptr), rhs_size0, rhs_size1\
            );\
            const eigen_vector_t<dtype> lhs(\
                const_cast<dtype*>(lhs_ptr), rhs_size0\
            );\
            eigen_vector_t<dtype> dst(\
                dst_ptr, rhs_size1\
            );\
            if (trans_rhs) {\
                if (beta != 0.0f) {\
                    dst.noalias() = alpha * lhs * rhs.transpose() + dst * beta;\
                } else {\
                    dst.noalias() = alpha * lhs * rhs.transpose();\
                }\
            } else {\
                if (beta != 0.0f) {\
                    dst.noalias() = alpha * lhs * rhs + dst * beta;\
                } else {\
                    dst.noalias() = alpha * lhs * rhs;\
                }\
            }\
        }

    // outer product
    #define GER_EIGEN(dtype) \
        inline static void ger(int rhs_size0, int lhs_size0, dtype alpha,\
                               const dtype *rhs_ptr, int incY,\
                               const dtype *lhs_ptr, int incX,\
                               dtype *dst_ptr, int lda) {\
            if (blas::should_use_cblas()) {\
                CBlasEngine::ger(rhs_size0, lhs_size0, alpha, rhs_ptr, incY, lhs_ptr, incX, dst_ptr, lda);\
                return;\
            }\
            eigen_mat_t<dtype> dst(\
                dst_ptr, lhs_size0, rhs_size0\
            );\
            const eigen_vector_t<dtype> rhs(\
                const_cast<dtype*>(rhs_ptr), rhs_size0\
            );\
            const eigen_vector_t<dtype> lhs(\
                const_cast<dtype*>(lhs_ptr), lhs_size0\
            );\
            dst.noalias() += alpha * lhs * rhs.transpose();\
        }

    GEMM_EIGEN(float);
    GEMM_EIGEN(double);

    GEMV_EIGEN(float);
    GEMV_EIGEN(double);

    GER_EIGEN(float);
    GER_EIGEN(double);
};
} // namespace expr
} // namespace mshadow
#endif
#endif
