#ifndef DALI_TENSOR_OP_SOLVER_UPDATES_RMSPROP_H
#define DALI_TENSOR_OP_SOLVER_UPDATES_RMSPROP_H

#include "dali/tensor/tensor.h"
#include "dali/array/array.h"

namespace tensor_ops {
    void rmsprop_update(Tensor& param,
                        Array& cache,
                        const double& decay_rate,
                        const double& step_size,
                        const double& smooth_eps);

    void rmsprop_momentum_update(Tensor& param,
                                 Array& n_cache,
                                 Array& g_cache,
                                 Array& momentum_cache,
                                 const double& decay_rate,
                                 const double& momentum,
                                 const double& step_size,
                                 const double& smooth_eps);
} // namespace tensor_ops

#endif
