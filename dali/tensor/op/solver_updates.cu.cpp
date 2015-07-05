#include "dali/tensor/op/solver_updates.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"


using std::vector;

namespace matops {
    using namespace TensorOps;

    template<typename R>
    void SolverUpdates<R>::clip_and_regularize(Mat<R> param, R clipval, R regc) {
        if (regc > 0) {
            GRAD(param) = F<op::clip<R>>(GRAD(param).wrapper(), clipval) + (regc * MAT(param).wrapper());
        } else {
            GRAD(param) = F<op::clip<R>>(GRAD(param).wrapper(), clipval);
        }
    }

    template<typename R>
    void SolverUpdates<R>::sgd_update(Mat<R> param, R step_size) {
        DEBUG_ASSERT_NOT_NAN(MAT(param));

        MAT(param) -= step_size * GRAD(param).wrapper();

        DEBUG_ASSERT_NOT_NAN(MAT(param));
    }

    template<typename R>
    void SolverUpdates<R>::adagrad_update(Mat<R> param,
                                          TensorInternal<R, 1>& cache,
                                          R step_size,
                                          R smooth_eps) {

        // update gradient cache using decay rule:
        cache += F<op::square<R>>(GRAD(param).ravel().wrapper());
        // clip the gradient to prevent explosions:
        // update gradient using RMSprop rule

        // TODO: REENABLE
        // DEBUG_ASSERT_POSITIVE(cache + smooth_eps);

        MAT(param).ravel() -= step_size * GRAD(param).ravel().wrapper() /
                F<op::sqrt_f<R>>(cache.ravel().wrapper() + smooth_eps);

        DEBUG_ASSERT_NOT_NAN(MAT(param));
    }

    template<typename R>
    void SolverUpdates<R>::rmsprop_update(Mat<R> param, TensorInternal<R,1>& cache,
            R decay_rate, R step_size,  R smooth_eps) {
        cache = (cache * decay_rate) + ((R)1.0 - decay_rate) * F<op::square<R>>(GRAD(param).ravel().wrapper());
        // ELOG("graD");
        // GRAD(param).print();
        // update gradient using RMSprop rule
        // DEBUG_ASSERT_POSITIVE((s.array() + smooth_eps).matrix());
        MAT(param).ravel() -= step_size * GRAD(param).ravel().wrapper() /
                F<op::sqrt_f<R>>(cache.ravel().wrapper() + smooth_eps);
    }

    template<typename R>
    void SolverUpdates<R>::adadelta_update(Mat<R> param,
                                           TensorInternal<R,1>& gsum,
                                           TensorInternal<R,1>& xsum,
                                           R rho,
                                           R smooth_eps) {
        // update gradient cache using decay rule:
        gsum = (gsum.wrapper() * rho) + ((R)((R)1.0 - rho)) * F<op::square<R>>(GRAD(param).ravel().wrapper());
        // DEBUG_ASSERT_POSITIVE((gsum.array()  + this->smooth_eps).matrix());
        // DEBUG_ASSERT_POSITIVE(((xsum.array() + this->smooth_eps) / (gsum.array() + this->smooth_eps)).matrix());
        TensorInternal<R,1> dparam(xsum.shape);
        dparam = (
            F<op::sqrt_f<R>>(xsum.wrapper() + smooth_eps) /
            F<op::sqrt_f<R>>(gsum.wrapper() + smooth_eps)
        ) * GRAD(param).ravel().wrapper();

        xsum = (
            rho * xsum.wrapper() +
            ((R)(1.0 - rho)) * F<op::square<R>>(dparam.wrapper())
        );
        // update gradient using AdaDelta rule
        MAT(param).ravel() -= dparam.wrapper();
    }

    template<typename R>
    void SolverUpdates<R>::adam_update(Mat<R> param,
                                           TensorInternal<R,1>& m,
                                           TensorInternal<R,1>& v,
                                           R b1,
                                           R b2,
                                           R smooth_eps,
                                           R step_size,
                                           unsigned long long epoch) {
        // this affects the learning rate:
        auto fix1 = 1.0 - std::pow(b1, epoch);
        auto fix2 = 1.0 - std::pow(b2, epoch);
        R lr_t = step_size * sqrt(fix2 / fix1);

        assert(lr_t == lr_t);

        // update m acculumulator
        m = m * (R)(1.0 - b1) + b1 * GRAD(param).ravel().wrapper();
        // update v acculumulator
        v = (v * (R)(1.0 - b2)) + b2 * F<op::square<R>>(GRAD(param).ravel().wrapper());

        GRAD(param).ravel() = m.wrapper() / (F<op::sqrt_f<R>>(v.wrapper()) + smooth_eps);

        // take gradient step
        MAT(param) -= lr_t * GRAD(param).wrapper();
    }

    template class SolverUpdates<float>;
    template class SolverUpdates<double>;

}
