#ifndef DALI_ARRAY_FUNCTION_ARGS_H
#define DALI_ARRAY_FUNCTION_ARGS_H

#include <mshadow/tensor.h>

// cuda code from
// https://devtalk.nvidia.com/default/topic/378826/cuda-programming-and-performance/my-speedy-sgemm/

#include "dali/array/function/args/reference_gemm.h"


#ifdef DALI_USE_CUDA

template<typename R>
__device__ void iaxpy( R a, R *b, R *c ) {
    c[0] += a*b[0];

    c[1] += a*b[1];

    c[2] += a*b[2];

    c[3] += a*b[3];

    c[4] += a*b[4];

    c[5] += a*b[5];

    c[6] += a*b[6];

    c[7] += a*b[7];

    c[8] += a*b[8];

    c[9] += a*b[9];

    c[10] += a*b[10];

    c[11] += a*b[11];

    c[12] += a*b[12];

    c[13] += a*b[13];

    c[14] += a*b[14];

    c[15] += a*b[15];
}


template<int x_bits, typename R>
__global__ void igemmNT( const R *A, int lda, const R *B, int ldb, R* C, int ldc, int k, R alpha, R beta ) {
    int inx = threadIdx.x;
    int iny = threadIdx.y;
    int ibx = blockIdx.x * 32;
    int iby = blockIdx.y * 32;

    A += ibx + inx + __mul24( iny, lda );

    B += iby + (inx%16) + __mul24( inx/16 + iny*2, ldb );

    C += ibx + inx + __mul24( iby + iny*16, ldc );

    const R *A1 = A + 2*lda;
    const R *B1 = B + 16;

    R a1 = A[0];
    R a2 = A1[0];
    R b1 = B[0];
    R b2 = B1[0];

    const R *Blast = B + k*ldb;

    A  += 4*lda;
    A1 += 4*lda;
    B  += 4*ldb;
    B1 += 4*ldb;

    __shared__ R a[160], b[128];

    R c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    R *a0 = a + 5*inx;
    R *b0 = b + 32*iny;
    R *_b = b + 64*iny;
    do {
        a0[iny]    = a1;
        a0[iny+2]  = a2;
        b0[inx]    = b1;
        b0[inx+64] = b2;

        __syncthreads();

        a1 = A[0];
        a2 = A1[0];
        b1 = B[0];
        b2 = B1[0];

        iaxpy<R>( a0[0], _b,    c );
        iaxpy<R>( a0[1], _b+16, c );
        iaxpy<R>( a0[2], _b+32, c );
        iaxpy<R>( a0[3], _b+48, c );


        A  += 4*lda;
        A1 += 4*lda;
        B  += 4*ldb;
        B1 += 4*ldb;

        __syncthreads();
    } while( B < Blast );

    a0[iny]    = a1;
    a0[iny+2]  = a2;
    b0[inx]    = b1;
    b0[inx+64] = b2;

    __syncthreads();

    iaxpy<R>( a0[0], _b,    c );
    iaxpy<R>( a0[1], _b+16, c );
    iaxpy<R>( a0[2], _b+32, c );
    iaxpy<R>( a0[3], _b+48, c );


    for( int i = 0; i < 16; i++, C += ldc )
        C[0] = alpha*c[i] + beta*C[0];
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
        inline static void gemm(Stream<gpu> *stream,
                                bool transa, bool transb,
                                int m, int n, int k, int alpha,
                                const int *A, int lda,
                                const int *B, int ldb,
                                int beta, int *C, int ldc) {

            // const int num_threads = mshadow::cuda::kBaseThreadNum;
            // const int thread_bits = mshadow::cuda::kBaseThreadBits;

            // dim3 tiles(dest.size(0));
            // // block size is a matrix column
            // dim3 within_tile(num_threads);
            // cudaStream_t stream = mshadow::Stream<mshadow::gpu>::GetStream(*stream);

            // igemmNT<thread_bits, int>
            //         <<<tiles, within_tile, 0, stream>>>
            //         (
            //             A, lda, B, ldb, C, ldc, num_threads, alpha, beta
            //         );

            // cudaDeviceSynchronize();
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
          !transa, !transb, true,
          m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
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
