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
    // Given a Tensor and a 1D vector of integer indices:
    //    [i_1, ...., i_n]
    // is creates a new tensor by stacking
    //    [params[i_1], ..., [params[i_n]]]
    // In particular if params is a matrix then this operation
    // corresponds to selecting a set of rows from the matrix.
    Tensor gather(const Tensor& params, const Tensor& indices);
}

#endif
