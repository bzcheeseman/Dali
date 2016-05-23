#ifndef DALI_TENSOR_OP_SOLVER_UPDATES_ADADELTA_H
#define DALI_TENSOR_OP_SOLVER_UPDATES_ADADELTA_H

#include "dali/tensor/tensor.h"
#include "dali/array/array.h"

namespace tensor_ops {
    void adadelta_update(Tensor& param,
                         Array& gsum,
                         Array& xsum,
                         const double& rho,
                         const double& smooth_eps);
} // namespace tensor_ops

#endif
