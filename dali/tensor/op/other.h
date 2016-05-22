#ifndef DALI_TENSOR_OP_OTHER_H
#define DALI_TENSOR_OP_OTHER_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    Tensor reshape(const Tensor& t, const std::vector<int>& new_shape);
}  // namespace tensor_ops


#endif
