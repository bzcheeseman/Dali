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
        cache *= decay_rate;
        cache += ((R)1.0 - decay_rate) * F<op::square<R>>(GRAD(param).ravel().wrapper());
        // ELOG("graD");
        // GRAD(param).print();
        // update gradient using RMSprop rule
        // DEBUG_ASSERT_POSITIVE((s.array() + smooth_eps).matrix());
        MAT(param).ravel() -= step_size * GRAD(param).ravel().wrapper() /
                F<op::sqrt_f<R>>(cache.ravel().wrapper() + smooth_eps);

        DEBUG_ASSERT_NOT_NAN(MAT(param));
    }

    template<typename R>
    void SolverUpdates<R>::adadelta_update(Mat<R> param,
                                           TensorInternal<R,1>& gsum,
                                           TensorInternal<R,1>& xsum,
                                           R rho,
                                           R smooth_eps) {
        // update gradient cache using decay rule:
        gsum *= rho;
        gsum += ((R)1.0 - rho) * F<op::square<R>>(GRAD(param).ravel().wrapper());
        // DEBUG_ASSERT_POSITIVE((gsum.array()  + this->smooth_eps).matrix());
        // DEBUG_ASSERT_POSITIVE(((xsum.array() + this->smooth_eps) / (gsum.array() + this->smooth_eps)).matrix());
        TensorInternal<R,1> dparam(xsum.shape);
        dparam = -(R)1.0 * F<op::sqrt_f<R>>(
                       (xsum.wrapper() + smooth_eps) /
                       (gsum.wrapper() + smooth_eps)
                ) * GRAD(param).ravel().wrapper();

        xsum *= rho;
        xsum += ((R)1.0 - rho) * F<op::square<R>>(dparam.wrapper());
        // update gradient using AdaDelta rule
        MAT(param).ravel() += dparam.wrapper();

        DEBUG_ASSERT_NOT_NAN(MAT(param));
    }


    template class SolverUpdates<float>;
    template class SolverUpdates<double>;

}
