#include "solver.h"
#include "dali/utils/assert2.h"
#include "dali/tensor/op/solver_updates.h"

inline void* get_param_key(const Tensor& param) {
    return param.w.memory().get();
}

namespace solver {
    bool nan_protection = false;

    void create_cache(const Tensor& param, std::unordered_map<cache_key_t, cache_t>& caches) {
        if (caches.count(get_param_key(param)) == 0) {
            auto new_cache = caches.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(get_param_key(param)),
                std::forward_as_tuple(param.shape(), param.dtype(), param.preferred_device())
            );
            // initialize values for step cache to zero:
            new_cache.first->second.clear();
        }
    }

    void create_cache(const std::vector<Tensor>& params,
            std::unordered_map<cache_key_t, cache_t>& caches) {
        for (auto& param: params) {
            create_cache(param, caches);
        }
    }

    void reset_cache(const Tensor& param, std::unordered_map<cache_key_t, cache_t>& caches) {
        auto& s = caches.at(get_param_key(param));
        s.clear();
    }

    void reset_cache(const std::vector<Tensor>& params,
        std::unordered_map<cache_key_t, cache_t>& caches) {
        for (auto& param: params) {
            reset_cache(param, caches);
        }
    }


    /* Abstract Solver */
    AbstractSolver::AbstractSolver() :
            clip_norm(0),
            clip_abs(0),
            smooth_eps(SMOOTH_DEFAULT),
            regc(0.0),
            method(METHOD_UNINITIALIZED) {
    }

    AbstractSolver::AbstractSolver(const double& _clip_norm,
                                   const double& _smooth_eps,
                                   const double& _regc,
                                   Method _method) :
            clip_norm(_clip_norm),
            clip_abs(0),
            smooth_eps(_smooth_eps),
            regc(_regc),
            method(_method) {
    }

    void AbstractSolver::reset_caches(const std::vector<Tensor>& parameters) {}

    void AbstractSolver::create_gradient_caches(const std::vector<Tensor>& parameters) {}

    /* SGD */
    SGD::SGD(const double& clip_norm, const double& regc) :
            AbstractSolver(clip_norm, 0.0, regc, METHOD_SGD) {
    }

    SGD::SGD(const std::vector<Tensor>& parameters,
             const double& clip_norm,
             const double& regc) : AbstractSolver(clip_norm, 0.0, regc, METHOD_SGD) {};

    void SGD::step(std::vector<Tensor>& parameters, const double& step_size) {
        for (auto& param : parameters) {
            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                tensor_ops::clip_and_regularize(param, this->clip_abs, this->clip_norm, this->regc);
                tensor_ops::sgd_update(param, step_size);
            }
            // reset gradient
            param.dw.clear();
        }
    }

    void SGD::step(std::vector<Tensor>& parameters) {
        return step(parameters, this->step_size);
    }

    /* AdaGrad */
    AdaGrad::AdaGrad(const double& smooth_eps,
                     const double& clip_norm,
                     const double& regc) : AbstractSolver(clip_norm, smooth_eps, regc, METHOD_ADAGRAD) {
    }

    AdaGrad::AdaGrad(
            const std::vector<Tensor>& parameters,
            const double& smooth_eps,
            const double& clip_norm,
            const double& regc) : AbstractSolver(clip_norm, smooth_eps, regc, METHOD_ADAGRAD) {
        create_gradient_caches(parameters);
    }

    void AdaGrad::create_gradient_caches(const std::vector<Tensor>& parameters) {
        create_cache(parameters, gsums);
    }

    void AdaGrad::reset_caches(const std::vector<Tensor>& parameters) {
        reset_cache(parameters, gsums);
    }

    void AdaGrad::step(std::vector<Tensor>& parameters, const double& step_size) {
        for (auto& param : parameters) {
            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                auto& s = gsums.at(get_param_key(param));
                tensor_ops::clip_and_regularize(param, this->clip_abs, this->clip_norm, this->regc);
                tensor_ops::adagrad_update(param, s, step_size, this->smooth_eps);
            }

            // reset gradient
            param.dw.clear();
        }
    }

    void AdaGrad::step(std::vector<Tensor>& parameters) {
        return step(parameters, this->step_size);
    }

    /* RMSProp */
    RMSProp::RMSProp (const double& _decay_rate,
                      const double& smooth_eps,
                      const double& clip_norm,
                      const double& regc) :
            AdaGrad(smooth_eps, clip_norm, regc),
            decay_rate(_decay_rate) {
        this->method = METHOD_RMSPROP;
    }

    RMSProp::RMSProp (const std::vector<Tensor>& parameters,
                      const double& _decay_rate,
                      const double& smooth_eps,
                      const double& clip_norm,
                      const double& regc) :
            AdaGrad(parameters, smooth_eps, clip_norm, regc),
            decay_rate(_decay_rate) {
        this->method = METHOD_RMSPROP;
    }

    void RMSProp::step(std::vector<Tensor>& parameters,
            const double& step_size) {
        for (auto& param : parameters) {

            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                auto& s = this->gsums[get_param_key(param)];

                tensor_ops::clip_and_regularize(param, this->clip_abs, this->clip_norm, this->regc);
                tensor_ops::rmsprop_update(param, s, decay_rate, step_size, this->smooth_eps);
            }

            // reset gradient
            param.dw.clear();
        }
    }

    void RMSProp::step(std::vector<Tensor>& parameters) {
        return step(parameters, step_size);
    }

    /* RMSPropMomentum */
    RMSPropMomentum::RMSPropMomentum(const double& decay_rate,
                                     const double& momentum,
                                     const double& step_size,
                                     const double& smooth_eps,
                                     const double& clip_norm,
                                     const double& regc) :
                AbstractSolver(clip_norm, smooth_eps, regc, METHOD_RMSPROPMOMENTUM),
                decay_rate(decay_rate),
                momentum(momentum),
                step_size(step_size) {}

    RMSPropMomentum::RMSPropMomentum(const std::vector<Tensor>& parameters,
                                     const double& decay_rate,
                                     const double& momentum,
                                     const double& step_size,
                                     const double& smooth_eps,
                                     const double& clip_norm,
                                     const double& regc) :
            AbstractSolver(clip_norm, smooth_eps, regc, METHOD_RMSPROPMOMENTUM),
            decay_rate(decay_rate),
            momentum(momentum),
            step_size(step_size) {
        create_gradient_caches(parameters);
    }

    void RMSPropMomentum::create_gradient_caches(const std::vector<Tensor>& parameters) {
        create_cache(parameters, n_cache);
        create_cache(parameters, g_cache);
        create_cache(parameters, momentum_cache);
    }

    void RMSPropMomentum::reset_caches(const std::vector<Tensor>& parameters) {
        reset_cache(parameters, n_cache);
        reset_cache(parameters, g_cache);
        reset_cache(parameters, momentum_cache);
    }

    void RMSPropMomentum::step(std::vector<Tensor>& parameters, const double& step_size_override) {
        for (auto& param : parameters) {
            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                auto& n = n_cache.at(get_param_key(param));
                auto& g = g_cache.at(get_param_key(param));
                auto& m = momentum_cache.at(get_param_key(param));

                tensor_ops::clip_and_regularize(param, this->clip_abs, this->clip_norm, this->regc);
                tensor_ops::rmsprop_momentum_update(
                        param, n, g, m, decay_rate, momentum,
                        step_size_override, this->smooth_eps);
            }
            // reset gradient
            param.dw.clear();
        }
    }

    void RMSPropMomentum::step(std::vector<Tensor>& parameters) {
        return step(parameters, this->step_size);
    }


    /* AdaDelta */
    AdaDelta::AdaDelta(const double& _rho,
                       const double& smooth_eps,
                       const double& clip_norm,
                       const double& regc) :
            AbstractSolver(clip_norm, smooth_eps, regc, METHOD_ADADELTA),
            rho(_rho) {
    }

    AdaDelta::AdaDelta(const std::vector<Tensor>& parameters,
                       const double& _rho,
                       const double& smooth_eps,
                       const double& clip_norm,
                       const double& regc) :
            AbstractSolver(clip_norm, smooth_eps, regc, METHOD_ADADELTA),
            rho(_rho) {
        create_gradient_caches(parameters);
    }

    void AdaDelta::create_gradient_caches(const std::vector<Tensor>& parameters) {
        create_cache(parameters, gsums);
        create_cache(parameters, xsums);
    }

    void AdaDelta::reset_caches(const std::vector<Tensor>& parameters) {
        reset_cache(parameters, gsums);
        reset_cache(parameters, xsums);
    }

    void AdaDelta::step(std::vector<Tensor>& parameters) {
        for (auto& param : parameters) {
            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                auto& gsum = gsums.at(get_param_key(param));
                auto& xsum = xsums.at(get_param_key(param));

                tensor_ops::clip_and_regularize(param, this->clip_abs, this->clip_norm, this->regc);
                tensor_ops::adadelta_update(param, gsum, xsum, rho, this->smooth_eps);
            }

            // reset gradient
            param.dw.clear();
        }
    }

    /* Adam */
    Adam::Adam(const double& _step_size,
               const double& _b1,
               const double& _b2,
               const double& smooth_eps,
               const double& clip_norm,
               const double& regc) :
            AbstractSolver(clip_norm, smooth_eps, regc, METHOD_ADAM),
            step_size(_step_size), b1(_b1), b2(_b2), epoch(0) {
    }

    Adam::Adam (const std::vector<Tensor>& parameters,
                const double& _step_size,
                const double& _b1,
                const double& _b2,
                const double& smooth_eps,
                const double& clip_norm,
                const double& regc) :
            AbstractSolver(clip_norm, smooth_eps, regc, METHOD_ADAM),
            step_size(_step_size),
            b1(_b1),
            b2(_b2),
            epoch(0) {
        create_gradient_caches(parameters);
    }

    void Adam::create_gradient_caches(const std::vector<Tensor>& parameters) {
        create_cache(parameters, gsums);
        create_cache(parameters, xsums);
    }

    void Adam::reset_caches(const std::vector<Tensor>& parameters) {
        reset_cache(parameters, gsums);
        reset_cache(parameters, xsums);
        epoch = 0;
    }

    void Adam::step(std::vector<Tensor>& parameters, const double& step_size) {
        // increase timesteps:
        epoch += 1;
        for (auto& param : parameters) {
            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                auto& m = gsums.at(get_param_key(param));
                auto& v = xsums.at(get_param_key(param));

                tensor_ops::clip_and_regularize(param, this->clip_abs, this->clip_norm, this->regc);

                tensor_ops::adam_update(param, m, v, b1, b2, this->smooth_eps, step_size, epoch);
            }
            // clear gradient
            param.dw.clear();
        }
    }


    void Adam::step(std::vector<Tensor>& parameters) {
        return step(parameters, step_size);
    }

    std::shared_ptr<AbstractSolver> construct(
            std::string solver_name,
            std::vector<Tensor>& params,
            const double& step_size,
            const double& regc) {

        std::shared_ptr<AbstractSolver> solver;

        std::transform(solver_name.begin(), solver_name.end(), solver_name.begin(), ::tolower);

        if (solver_name        == "adadelta") {
            solver = std::make_shared<AdaDelta>(params, 0.95, 1e-4, 100.0, regc);
        } else if (solver_name == "adam") {
            solver = std::make_shared<Adam>(params, step_size, 0.55, 1e-6, 1e-9, 100.0, regc);
        } else if (solver_name == "sgd") {
            solver = std::make_shared<SGD>(params, 100.0, regc);
            dynamic_cast<SGD*>(solver.get())->step_size = step_size;
        } else if (solver_name == "adagrad") {
            solver = std::make_shared<AdaGrad>(params, 1e-9, 100.0, regc);
            dynamic_cast<AdaGrad*>(solver.get())->step_size = step_size;
        } else if (solver_name == "rmsprop") {
            solver = std::make_shared<RMSProp>(params, 0.999, 1e-9, 100.0, regc);
            dynamic_cast<RMSProp*>(solver.get())->step_size = step_size;
        } else if (solver_name == "rmspropmomentum") {
            solver = std::make_shared<RMSPropMomentum>(params);
            dynamic_cast<RMSPropMomentum*>(solver.get())->step_size = step_size;
            dynamic_cast<RMSPropMomentum*>(solver.get())->regc = regc;
        } else {
            ASSERT2(false,
                utils::MS() << "no solver found for name " << solver_name
                            << "; try 'sgd', 'adadelta', 'adam', 'adagrad', 'rmsprop', "
                            << "'rmspropmomentum' instead."
            );
        }
        return solver;
    }

} // namespace solver
