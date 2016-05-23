#ifndef DALI_TENSOR_OP_SOLVER_UPDATES_ADAGRAD_H
#define DALI_TENSOR_OP_SOLVER_UPDATES_ADAGRAD_H

#include "dali/tensor/tensor.h"
#include "dali/array/array.h"

namespace tensor_ops {
    void adagrad_update(Tensor& t,
                        Array& cache,
                        const double& step_size,
                        const double& smooth_eps);
} // namespace tensor_ops

#endif
