#ifndef DALI_TENSOR_OP_DOT_H
#define DALI_TENSOR_OP_DOT_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    Tensor dot(const Tensor&, const Tensor&);

    Tensor vectordot(const Tensor& a, const Tensor& b);
    Tensor matrixdot(const Tensor& a, const Tensor& b);
    Tensor matrix_vector_dot(const Tensor& a, const Tensor& b);
    Tensor tensordot(const Tensor& a, const Tensor& b, const int& axis);
    Tensor tensordot(const Tensor& a, const Tensor& b,
                     const std::vector<int>& a_reduce_axes,
                     const std::vector<int>& b_reduce_axes);
}

#endif
