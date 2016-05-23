#ifndef DALI_TENSOR_OP_SOLVER_UPDATES_CLIP_AND_REGULARIZE_H
#define DALI_TENSOR_OP_SOLVER_UPDATES_CLIP_AND_REGULARIZE_H

#include "dali/tensor/tensor.h"
#include "dali/array/array.h"

namespace tensor_ops {
    void clip_and_regularize(const Tensor& param,
                             const double& clip_abs,
                             const double& clip_norm,
                             const double& regc);

    void regularize(const Tensor& param, const double& regc);

    void normalize_gradient(const Tensor& param, const double& norm_threshold);
} // namespace tensor_ops

#endif
