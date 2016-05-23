#ifndef DALI_TENSOR_OP_SOLVER_UPDATES_SGD_H
#define DALI_TENSOR_OP_SOLVER_UPDATES_SGD_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    void sgd_update(Tensor& t, const double& step_size);
} // namespace tensor_ops

#endif
