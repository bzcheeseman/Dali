#ifndef DALI_TENSOR_OP_SOLVER_UPDATES_ADAM_H
#define DALI_TENSOR_OP_SOLVER_UPDATES_ADAM_H

#include "dali/tensor/tensor.h"
#include "dali/array/array.h"

namespace tensor_ops {
    void adam_update(Tensor& param,
                     Array& m,
                     Array& v,
                     const double& b1,
                     const double& b2,
                     const double& smooth_eps,
                     const double& step_size,
                     unsigned long long epoch);
} // namespace tensor_ops

#endif
