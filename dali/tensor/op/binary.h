#ifndef DALI_TENSOR_OP_BINARY_H
#define DALI_TENSOR_OP_BINARY_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    Tensor add(const Tensor&, const Tensor&);
    Tensor add(const std::vector<Tensor>& tensors);
    Tensor sub(const Tensor&, const Tensor&);
    Tensor eltmul(const Tensor&, const Tensor&);
    Tensor eltdiv(const Tensor&, const Tensor&);
    Tensor pow(const Tensor&, const Tensor& exponent);
    Tensor circular_convolution(const Tensor& content, const Tensor& shift);
    Tensor prelu(const Tensor& x, const Tensor& weights);
}

#endif
