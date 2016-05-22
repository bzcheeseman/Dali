#ifndef DALI_TENSOR_OP_SOLVER_UPDATES_H
#define DALI_TENSOR_OP_SOLVER_UPDATES_H

#include "dali/tensor/tensor.h"
#include "dali/array/array.h"

namespace tensor_ops {
    void clip_and_regularize(const Tensor& param,
                             const double& clip_abs,
                             const double& clip_norm,
                             const double& regc);

    void regularize(const Tensor& param, const double& regc);

    void normalize_gradient(const Tensor& param, const double& norm_threshold);

    void sgd_update(Tensor& t, const double& step_size);

    void adagrad_update(Tensor& t,
                        Array& cache,
                        const double& step_size,
                        const double& smooth_eps);

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


    void adadelta_update(Tensor& param,
                         Array& gsum,
                         Array& xsum,
                         const double& rho,
                         const double& smooth_eps);

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
