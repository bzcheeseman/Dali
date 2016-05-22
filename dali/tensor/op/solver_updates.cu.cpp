#include "solver_updates.h"

#include "dali/array/lazy_op.h"

namespace tensor_ops {
    void clip_and_regularize(const Tensor& param,
                             const double& clip_abs,
                             const double& clip_norm,
                             const double& regc) {
        bool use_regc = regc > 0;
        bool use_abs_clip = clip_abs > 0;

        bool use_norm_clip = clip_norm > 0;

        double norm;
        if (use_norm_clip) {
            // compute norm conditionally
            norm = (double)(Array)param.dw.L2_norm();
            // cancel normalization if norm is below threshold
            if (norm <= clip_norm) {
                use_norm_clip = false;
            }
        }

        if (use_regc) {
            if (!use_abs_clip && !use_norm_clip) {
                param.dw = param.dw + regc * param.w;
            } else if (use_abs_clip && !use_norm_clip) {
                param.dw = lazy::clip(param.dw, clip_abs) + (regc * param.w);
            } else if (!use_abs_clip && use_norm_clip) {
                param.dw = (clip_norm / norm) * param.dw + (regc * param.w);
            } else if (use_abs_clip && use_norm_clip) {
                param.dw = lazy::clip((clip_norm / norm) * param.dw, clip_abs) + (regc * param.w);
            }
        } else {
            if (use_abs_clip && !use_norm_clip) {
                param.dw = lazy::clip(param.dw, clip_abs);
            } else if (!use_abs_clip && use_norm_clip) {
                param.dw = (clip_norm / norm) * param.dw;
            } else if (use_abs_clip && use_norm_clip) {
                param.dw = lazy::clip(param.dw, clip_abs);
            }
        }
    }

    void regularize(const Tensor& param,
                    const double& regc) {
        if (regc > 0) {
            param.dw += (regc * param.w);
        }
    }

    void normalize_gradient(const Tensor& param,
                            const double& norm_threshold) {
        double norm = (double)(Array)param.dw.L2_norm();
        if (norm > norm_threshold) {
            param.dw = (norm_threshold / norm) * param.dw;
        }
    }


    void sgd_update(Tensor& param,
                    const double& step_size) {
        param.w -= step_size * param.dw;
    }

    void adagrad_update(Tensor& param,
                        Array& cache,
                        const double& step_size,
                        const double& smooth_eps) {
        ASSERT2(cache.number_of_elements() == param.number_of_elements(),
            utils::MS() << "cache parameter in adagrad_update has different "
                        << "size than parameter (got " << cache.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        // update gradient cache using decay rule:
        cache += lazy::square(param.dw);
        // clip the gradient to prevent explosions:
        // update gradient using RMSprop rule
        param.w -= step_size * param.dw / (lazy::sqrt(cache) + smooth_eps);
    }

    void rmsprop_update(Tensor& param,
                        Array& cache,
                        const double& decay_rate,
                        const double& step_size,
                        const double& smooth_eps) {
        ASSERT2(cache.number_of_elements() == param.number_of_elements(),
            utils::MS() << "cache parameter in rmsprop_update has different "
                        << "size than parameter (got " << cache.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        cache = (
            decay_rate * cache +
            (1.0 - decay_rate) * lazy::square(param.dw)
        );
        // update gradient using RMSprop rule
        // DEBUG_ASSERT_POSITIVE((s.array() + smooth_eps).matrix());
        param.w -= step_size * param.dw / lazy::sqrt(cache + smooth_eps);
    }

    // Based on the "Generating Sequences With
    // Recurrent Neural Networks" paper:
    //     http://arxiv.org/pdf/1308.0850v5.pdf
    void rmsprop_momentum_update(Tensor& param,
                                 Array& n_cache,
                                 Array& g_cache,
                                 Array& momentum_cache,
                                 const double& decay_rate,       // eq. 42
                                 const double& momentum,         // eq. 43
                                 const double& step_size,        // eq. 44
                                 const double& smooth_eps) {     // eq. 45
        ASSERT2(n_cache.number_of_elements() == param.number_of_elements(),
            utils::MS() << "n_cache parameter in rmsprop_momentum_update has different "
                        << "size than parameter (got " << n_cache.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        ASSERT2(g_cache.number_of_elements() == param.number_of_elements(),
            utils::MS() << "g_cache parameter in rmsprop_momentum_update has different "
                        << "size than parameter (got " << g_cache.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        ASSERT2(momentum_cache.number_of_elements() == param.number_of_elements(),
            utils::MS() << "momentum_cache parameter in rmsprop_momentum_update has different "
                        << "size than parameter (got " << momentum_cache.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        n_cache = (
            decay_rate         * n_cache +
            (1.0 - decay_rate) * lazy::square(param.dw)
        );
        g_cache = (
            decay_rate         * g_cache +
            (1.0 - decay_rate) * param.dw
        );
        momentum_cache = (
            momentum * momentum_cache
            - step_size * param.dw / (
                lazy::square(
                    n_cache -
                    lazy::square(g_cache)
                    + smooth_eps
                )
            )
        );
        param.w += momentum_cache;
    }

    void adadelta_update(Tensor& param,
                         Array& gsum,
                         Array& xsum,
                         const double& rho,
                         const double& smooth_eps) {
        ASSERT2(gsum.number_of_elements() == param.number_of_elements(),
            utils::MS() << "gsum parameter in adadelta_update has different "
                        << "size than parameter (got " << gsum.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        ASSERT2(xsum.number_of_elements() == param.number_of_elements(),
            utils::MS() << "xsum parameter in adadelta_update has different "
                        << "size than parameter (got " << xsum.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        // update gradient cache using decay rule:
        gsum = (
            rho * gsum +
            (1.0 - rho) * lazy::square(param.dw)
        );
        // DEBUG_ASSERT_POSITIVE((MAT(gsum).array()  + this->smooth_eps).matrix());
        // DEBUG_ASSERT_POSITIVE(((xsum.array() + this->smooth_eps) / (MAT(gsum).array() + this->smooth_eps)).matrix());
        Array dparam(
            (lazy::sqrt(xsum + smooth_eps) / lazy::sqrt(gsum + smooth_eps)) * param.dw
        );

        xsum = rho * xsum + (1.0 - rho) * lazy::square(dparam);
        // update gradient using AdaDelta rule
        param.w -= dparam;
    }

    void adam_update(Tensor& param,
                     Array& m,
                     Array& v,
                     const double& b1,
                     const double& b2,
                     const double& smooth_eps,
                     const double& step_size,
                     unsigned long long epoch) {
        ASSERT2(m.number_of_elements() == param.number_of_elements(),
            utils::MS() << "m parameter in adam_update has different "
                        << "size than parameter (got " << m.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        ASSERT2(v.number_of_elements() == param.number_of_elements(),
            utils::MS() << "v parameter in adam_update has different "
                        << "size than parameter (got " << v.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        // this affects the learning rate:
        auto fix1 = 1.0 - std::pow(b1, epoch);
        auto fix2 = 1.0 - std::pow(b2, epoch);
        double lr_t = step_size * sqrt(fix2 / fix1);

        ASSERT2(lr_t == lr_t, "Epoch learning rate is NaN. Try changing b1 or b2.");

        // update m acculumulator
        m = (m * (1.0 - b1)) + b1 * param.dw;
        // update v acculumulator
        v = (v * (1.0 - b2)) + b2 * lazy::square(param.dw);

        param.dw = m / (lazy::sqrt(v) + smooth_eps);

        // take gradient step
        param.dw -= lr_t * param.dw;
    }

}  // namespace tensor_ops
