#ifndef DALI_ARRAY_FUNCTION_ARGS_REFERENCE_GEMM_H
#define DALI_ARRAY_FUNCTION_ARGS_REFERENCE_GEMM_H
// Tensorflow reference gemm implementation
// source https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantization/kernels/reference_gemm.h
// modified to handle single type, no shifting (quantization), and alpha/beta arguments

/* Copyright 2015 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This is an unoptimized but debuggable implementation of the GEMM matrix
// multiply function, used to compare to faster but more opaque versions, or
// for bit depths or argument combinations that aren't supported by optimized
// code.
// It assumes the row-major convention used by TensorFlow, and implements
// C = A * B, like the standard BLAS GEMM interface. If the tranpose flags are
// true, then the relevant matrix is treated as stored in column-major order.

template <typename R>
void ReferenceGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                   size_t m, size_t n, size_t k, double alpha, const R* a,
                   size_t lda, const R* b, size_t ldb, double beta, R* c, size_t ldc) {

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

    int i, j, l;
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            R total = 0;
            for (l = 0; l < k; l++) {
                const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
                const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
                total += (a[a_index] * b[b_index]);
            }
            const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
            c[c_index] = alpha * total + beta * c[c_index];
        }
    }
}
#endif
