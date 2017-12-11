#ifndef DALI_ARRAY_GEMM_GEMM_UTILS_H
#define DALI_ARRAY_GEMM_GEMM_UTILS_H
#include "dali/array/array.h"
// compute gemm col-major transpose + stride argument
std::tuple<bool, int> gemm_stride_transpose(const Array& array);
#endif  // DALI_ARRAY_GEMM_GEMM_UTILS_H
