#ifndef DALI_TENSOR_OP_OTHER_H
#define DALI_TENSOR_OP_OTHER_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    Tensor reshape(const Tensor& t, const std::vector<int>& new_shape);
    Tensor ravel(const Tensor& t);

    Tensor fill(const Tensor& t, const double& filler);
    Tensor fill(const Tensor& t, const float& filler);
    Tensor fill(const Tensor& t, const int& filler);

    void grad(const Tensor& t);

    Tensor consider_constant_if(const Tensor& t, const bool& condition);
    Tensor consider_constant(const Tensor& t);

}  // namespace tensor_ops


#endif
