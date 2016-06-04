#ifndef DALI_ARRAY_FUNCTION_ARGS_H
#define DALI_ARRAY_FUNCTION_ARGS_H

#include <mshadow/tensor.h>

// cuda code from (with modifications/clarifications)
// https://github.com/ctuning/ctuning-programs/blob/master/program/polybench-cuda-gemm/gemm.cu

#include "dali/array/mshadow_extension/reference_gemm.h"

#ifdef DALI_USE_CUDA

// inefficient cuda matrix multiply of integers on gpu
template<typename R>
__global__
void gemm(bool transpose_a, bool transpose_b, bool transpose_c, int m, int n, int k, R alpha, const R *A, int lda,
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


namespace mshadow {
namespace expr {

    template<>
    struct BLASEngine<gpu, int> {
        inline static bool GetT(bool t) {
          return t ? true : false;
        }
        inline static void SetStream(Stream<gpu> *stream) {
        }
        inline static void gemv(Stream<gpu> *stream,
                                bool trans, int m, int n,
                                int alpha, const int *A, int lda,
                                const int *X, int incX,
                                int beta, int *Y, int incY) {
          LOG(FATAL) << "Not implmented!";
        }
        inline static void ger(Stream<gpu> *stream,
                               int m, int n, int alpha,
                               const int *X, int incX,
                               const int *Y, int incY, int *A, int lda) {
          LOG(FATAL) << "Not implmented!";
        }
        inline static void dot(Stream<gpu> *stream,
                               int n,
                               const int* X, int incX,
                               const int* Y, int incY,
                               int* ret) {
          LOG(FATAL) << "Not implmented!";
        }

        inline static void gemm(Stream<gpu> *stream_,
                                bool transa, bool transb,
                                int m, int n, int k, int alpha,
                                const int *A, int lda,
                                const int *B, int ldb,
                                int beta, int *C, int ldc) {

            cudaStream_t stream = mshadow::Stream<mshadow::gpu>::GetStream(stream_);
            dim3 blocks(1,1);
            dim3 threads(16,16);

            gemm<int><<<blocks, threads, 0, stream>>>(
                !transa, !transb, true, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
            );
        }
    };
} // namespace expr
} // namespace mshadow

#endif

namespace mshadow {
namespace expr {
    template<>
    struct BLASEngine<cpu, int> {
      inline static bool GetT(bool t) {
        return t ? true : false;
      }
      inline static void SetStream(Stream<cpu> *stream) {
      }
      inline static void gemm(Stream<cpu> *stream,
                              bool transa, bool transb,
                              int m, int n, int k, int alpha,
                              const int *A, int lda, const int *B, int ldb,
                              int beta, int *C, int ldc) {
          ReferenceGemm<int>(
              !transa, !transb, true, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
          );
      }
      inline static void gemv(Stream<cpu> *stream,
                              bool trans, int m, int n,
                              int alpha, const int *A, int lda,
                              const int *X, int incX,
                              int beta, int *Y, int incY) {
        LOG(FATAL) << "Not implmented!";
      }
      inline static void ger(Stream<cpu> *stream,
                             int m, int n, int alpha,
                             const int *X, int incX,
                             const int *Y, int incY, int *A, int lda) {
        LOG(FATAL) << "Not implmented!";
      }
      inline static void dot(Stream<cpu> *stream,
                             int n,
                             const int* X, int incX,
                             const int* Y, int incY,
                             int* ret) {
        LOG(FATAL) << "Not implmented!";
      }
    };
} // namespace expr
} // namespace mshadow






#endif
