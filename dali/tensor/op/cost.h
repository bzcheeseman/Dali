#ifndef DALI_TENSOR_OP_COST_H
#define DALI_TENSOR_OP_COST_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    Tensor binary_cross_entropy(const Tensor&, const double& target);
    Tensor binary_cross_entropy(const Tensor&, const Tensor&);
    Tensor sigmoid_binary_cross_entropy(const Tensor&, const double& target);
    Tensor sigmoid_binary_cross_entropy(const Tensor&, const Tensor&);
    Tensor margin_loss(const Tensor&, const int& target, const double& margin, const int& axis);
    Tensor margin_loss(const Tensor&, const Tensor& target, const double& margin, const int& axis);
}  // namespace tensor_ops

#endif
