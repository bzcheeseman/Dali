#ifndef DALI_TENSOR_OP_COST_H
#define DALI_TENSOR_OP_COST_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    Tensor binary_cross_entropy(Tensor, double);
    Tensor binary_cross_entropy(Tensor, Tensor);
}  // namespace tensor_ops

#endif
