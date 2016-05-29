#ifndef DALI_TENSOR_OP_RESHAPE_H
#define DALI_TENSOR_OP_RESHAPE_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    // Join a sequence of arrays along an existing axis.
    Tensor concatenate(const std::vector<Tensor>& tensors, int axis);
    // Join a sequence of arrays along their last axis.
    Tensor hstack(const std::vector<Tensor>& tensors);
    // Stack arrays in sequence vertically (row wise).
    Tensor vstack(const std::vector<Tensor>& tensors);
}

#endif
