#include "core/Solver.h"

using std::vector;

#define PARAM_KEY_FOR_LOOKUP_TABLE *param

template<typename T>
Solver::SGD<T>::SGD (T _clipval) :
    clipval(_clipval) {};

template<typename T>
Solver::SGD<T>::SGD (vector<typename Solver::SGD<T>::shared_mat>& parameters, T _clipval) :
    clipval(_clipval) {};

template<typename T>
void Solver::SGD<T>::step (vector<typename Solver::SGD<T>::shared_mat>& parameters,
    T step_size,
    T regc
    ) {
    for (auto& param : parameters) {
        if (param->sparse) {
            for (auto& i : *(param->sparse_row_keys)) {
                if (regc > 0) {
                    param->w.row(i) -= (step_size * param->dw.row(i).array().min(clipval).max(-clipval)).matrix() - (regc * param->w.row(i));
                } else {
                    param->w.row(i) -= (step_size * param->dw.row(i).array().min(clipval).max(-clipval)).matrix();
                }
                // reset gradient
                param->dw.row(i).fill(0);
            }
        } else {
            if (regc > 0) {
                param->w -= (step_size * param->dw.array().min(clipval).max(-clipval)).matrix() - (regc * param->w);
            } else {
                param->w -= (step_size * param->dw.array().min(clipval).max(-clipval)).matrix();
            }
            // reset gradient
            param->dw.fill(0);
        }
        DEBUG_ASSERT_NOT_NAN(param->w);
    }
}

template<typename T>
Solver::AdaDelta<T>::AdaDelta (
    T _rho,
    T _smooth_eps,
    T _clipval) :
        smooth_eps(_smooth_eps),
        rho(_rho),
    clipval(_clipval) {};

template<typename T>
Solver::AdaDelta<T>::AdaDelta (
    vector<typename Solver::AdaDelta<T>::shared_mat>& parameters,
    T _rho,
    T _smooth_eps,
    T _clipval) :
        smooth_eps(_smooth_eps),
        rho(_rho),
    clipval(_clipval) {
    create_gradient_caches(parameters);
};

template<typename T>
void Solver::AdaDelta<T>::create_gradient_caches(
    vector<typename Solver::AdaDelta<T>::shared_mat>& parameters) {
    for (auto& param : parameters) {
        // this operation should be run once unless
        // we expect the parameters of the model
        // to change online (probably not the case)
        if (!(gsums.count(PARAM_KEY_FOR_LOOKUP_TABLE) > 0)) {
            auto new_cache = this->gsums.emplace(
                std::piecewise_construct,
            std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
            std::forward_as_tuple(param->n, param->d));
            // initialize values for step cache to zero:
            new_cache.first->second.fill(0);

            new_cache = this->xsums.emplace(
                std::piecewise_construct,
            std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
            std::forward_as_tuple(param->n, param->d));
            // initialize values for step cache to zero:
            new_cache.first->second.fill(0);
        }
    }
}

template<typename T>
void Solver::AdaDelta<T>::step (vector<typename Solver::AdaDelta<T>::shared_mat>& parameters, T regc) {
    for (auto& param : parameters) {
        auto& gsum = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        auto& xsum = xsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        if (param->sparse) {
            for (auto& i : *(param->sparse_row_keys)) {
                if (regc > 0) {
                    param->dw.row(i) = param->dw.row(i).array().min(clipval).max(-clipval).matrix() + (regc * param->w.row(i));
                } else {
                    param->dw.row(i) = param->dw.row(i).array().min(clipval).max(-clipval).matrix();
                }
                // update gradient cache using decay rule:
                DEBUG_ASSERT_POSITIVE(gsum.row(i).matrix());
                gsum.row(i) = (gsum.row(i) * rho) + ((1.0 - rho) * (param->dw.row(i).array().square()).matrix());

                DEBUG_ASSERT_NOT_NAN(((gsum.row(i).array() + smooth_eps).matrix()));
                DEBUG_ASSERT_POSITIVE(((gsum.row(i).array() + smooth_eps)).matrix());
                DEBUG_ASSERT_POSITIVE(((xsum.row(i).array() + smooth_eps) / (gsum.row(i).array() + smooth_eps)).matrix());
                auto dparam = -(((xsum.row(i).array() + smooth_eps) / (gsum.row(i).array() + smooth_eps)).sqrt() * param->dw.row(i).array()).matrix();

                xsum.row(i) = (xsum.row(i) * rho) + ((1.0 - rho) * (dparam.array().square())).matrix();
                // update gradient using AdaDelta rule
                param->w.row(i) += dparam;
                // reset gradient
                param->dw.row(i).fill(0);
            }
        } else {
            if (regc > 0) {
                param->dw = param->dw.array().min(clipval).max(-clipval).matrix() + (regc * param->w);
            } else {
                param->dw = param->dw.array().min(clipval).max(-clipval).matrix();
            }
            // update gradient cache using decay rule:
            gsum = (gsum * rho) + ((1.0 - rho) * (param->dw.array().square()).matrix());
            DEBUG_ASSERT_POSITIVE((gsum.array() + smooth_eps).matrix());
            DEBUG_ASSERT_POSITIVE(((xsum.array() + smooth_eps) / (gsum.array() + smooth_eps)).matrix());
            auto dparam = -(((xsum.array() + smooth_eps) / (gsum.array() + smooth_eps)).sqrt() * param->dw.array()).matrix();

            xsum = (xsum * rho) + ((1.0 - rho) * (dparam.array().square())).matrix();
            // update gradient using AdaDelta rule
            param->w += dparam;
            // reset gradient
            param->dw.fill(0);
        }
        DEBUG_ASSERT_NOT_NAN(param->w);
    }
}

template<typename T>
Solver::AdaGrad<T>::AdaGrad(T _smooth_eps, T _clipval) : smooth_eps(_smooth_eps), clipval(_clipval) {}

template<typename T>
Solver::AdaGrad<T>::AdaGrad (vector<typename Solver::AdaGrad<T>::shared_mat>& parameters,
    T _smooth_eps,
    T _clipval) : smooth_eps(_smooth_eps), clipval(_clipval) {
    create_gradient_caches(parameters);
}

template<typename T>
void Solver::AdaGrad<T>::step(
    vector<typename Solver::AdaGrad<T>::shared_mat>& parameters,
    T step_size,
    T regc
    ) {
    for (auto& param : parameters) {
    auto& s = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
    if (param->sparse) {
        for (auto& i : *(param->sparse_row_keys)) {
        param->dw.row(i) = param->dw.row(i).array().min(clipval).max(-clipval).matrix() + (regc * param->w.row(i));
        // update gradient cache using decay rule:
        s.row(i) += param->dw.row(i).array().square().matrix();
        // clip the gradient to prevent explosions:
        // update gradient using RMSprop rule
        DEBUG_ASSERT_POSITIVE((s.row(i).array() + smooth_eps).matrix());
        param->w.row(i) -= step_size * (param->dw.row(i).array() / (s.row(i).array() + smooth_eps).sqrt() ).matrix();
        // reset gradient
        param->dw.row(i).fill(0);
        }
    } else {
        param->dw = param->dw.array().min(clipval).max(-clipval).matrix() + (regc * param->w);
        // update gradient cache using decay rule:
        s += param->dw.array().square().matrix();
        // clip the gradient to prevent explosions:
        // update gradient using RMSprop rule
        DEBUG_ASSERT_POSITIVE((s.array() + smooth_eps).matrix());
        param->w -= step_size * (param->dw.array() / (s.array() + smooth_eps).sqrt() ).matrix();
        // reset gradient
        param->dw.fill(0);
    }
    DEBUG_ASSERT_NOT_NAN(param->w);
    }
}

template<typename T>
void Solver::AdaGrad<T>::reset_caches(
    vector<typename Solver::AdaGrad<T>::shared_mat>& parameters) {
    for (auto& param : parameters) {
    auto& s = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
    s.fill(0);
    }
}

template<typename T>
void Solver::AdaGrad<T>::create_gradient_caches(
    vector<typename Solver::AdaGrad<T>::shared_mat>& parameters) {
    for (auto& param : parameters) {
    // this operation should be run once unless
    // we expect the parameters of the model
    // to change online (probably not the case)
    if (!(gsums.count(PARAM_KEY_FOR_LOOKUP_TABLE) > 0)) {
        auto new_cache = this->gsums.emplace(
            std::piecewise_construct,
        std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
        std::forward_as_tuple(param->n, param->d));
        // initialize values for step cache to zero:
        new_cache.first->second.fill(0);
    }
    }
}

template<typename T>
Solver::RMSProp<T>::RMSProp (
    T _decay_rate,
    T _smooth_eps,
    T _clipval) :
    decay_rate(_decay_rate),
    smooth_eps(_smooth_eps),
    clipval(_clipval) {};

template<typename T>
Solver::RMSProp<T>::RMSProp (
    vector<typename Solver::RMSProp<T>::shared_mat>& parameters,
    T _decay_rate,
    T _smooth_eps,
    T _clipval) :
    decay_rate(_decay_rate),
    smooth_eps(_smooth_eps),
    clipval(_clipval)
    {
    create_gradient_caches(parameters);
};

template<typename T>
void Solver::RMSProp<T>::create_gradient_caches(
    vector<typename Solver::RMSProp<T>::shared_mat>& parameters) {
    for (auto& param : parameters) {
    // this operation should be run once unless
    // we expect the parameters of the model
    // to change online (probably not the case)
    if (!(gsums.count(PARAM_KEY_FOR_LOOKUP_TABLE) > 0)) {
        auto new_cache = this->gsums.emplace(
            std::piecewise_construct,
        std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
        std::forward_as_tuple(param->n, param->d));
        // initialize values for step cache to zero:
        new_cache.first->second.fill(0);
    }
    }
}

template<typename T>
void Solver::RMSProp<T>::step(
    vector<typename Solver::RMSProp<T>::shared_mat>& parameters,
    T step_size,
    T regc
    ) {
    for (auto& param : parameters) {
    auto& s = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
    if (param->sparse) {
        for (auto& i : *(param->sparse_row_keys)) {
        s.row(i) = s.row(i) * decay_rate + (1.0 - decay_rate) * param->dw.row(i).array().square().matrix();
        // clip the gradient to prevent explosions:
        param->dw.row(i) = param->dw.row(i).array().min(clipval).max(-clipval).matrix();
        // update gradient using RMSprop rule
        DEBUG_ASSERT_POSITIVE((s.row(i).array() + smooth_eps).matrix());
        param->w.row(i) -= step_size * (param->dw.row(i).array() / (s.row(i).array() + smooth_eps).sqrt() ).matrix()  - (regc * param->w.row(i));
        // reset gradient
        param->dw.row(i).fill(0);
        }
    } else {
        s = s * decay_rate + (1.0 - decay_rate) * param->dw.array().square().matrix();
        // clip the gradient to prevent explosions:
        param->dw = param->dw.array().min(clipval).max(-clipval).matrix();
        // update gradient using RMSprop rule
        DEBUG_ASSERT_POSITIVE((s.array() + smooth_eps).matrix());
        param->w -= step_size * (param->dw.array() / (s.array() + smooth_eps).sqrt() ).matrix()  - (regc * param->w);
        // reset gradient
        param->dw.fill(0);
    }
    DEBUG_ASSERT_NOT_NAN(param->w);
    }

}

template class Solver::SGD<float>;
template class Solver::SGD<double>;

template class Solver::AdaGrad<float>;
template class Solver::AdaGrad<double>;

template class Solver::AdaDelta<float>;
template class Solver::AdaDelta<double>;

template class Solver::RMSProp<float>;
template class Solver::RMSProp<double>;
