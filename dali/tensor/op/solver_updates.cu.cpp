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
    void SolverUpdates<R>::regularize(Mat<R> param, R regc) {
        if (regc > 0) {
            GRAD(param) += (regc * MAT(param).wrapper());
        }
    }

    template<typename R>
    void SolverUpdates<R>::normalize(Mat<R> param, R norm_threshold) {
        R norm = param.dw().L2_norm();
        if (norm > norm_threshold) {
            GRAD(param) = (norm_threshold / norm) * GRAD(param).wrapper();
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
                                          Mat<R>& cache,
                                          R step_size,
                                          R smooth_eps) {

        // update gradient cache using decay rule:
        MAT(cache) += F<op::square<R>>(GRAD(param).wrapper());
        // clip the gradient to prevent explosions:
        // update gradient using RMSprop rule

        MAT(param) -= step_size * GRAD(param).wrapper() /
                (F<op::sqrt_f<R>>(MAT(cache).wrapper()) + smooth_eps);

        DEBUG_ASSERT_NOT_NAN(MAT(param));
    }

    template<typename R>
    void SolverUpdates<R>::rmsprop_update(Mat<R> param, Mat<R>& cache,
            R decay_rate, R step_size,  R smooth_eps) {
        MAT(cache) = (
            decay_rate            * MAT(cache).wrapper() +
            ((R)1.0 - decay_rate) * F<op::square<R>>(GRAD(param).wrapper())
        );
        // update gradient using RMSprop rule
        // DEBUG_ASSERT_POSITIVE((s.array() + smooth_eps).matrix());
        MAT(param) -= step_size * GRAD(param).wrapper() /
                F<op::sqrt_f<R>>(MAT(cache).wrapper() + smooth_eps);
    }

    template<typename R>
    void SolverUpdates<R>::adadelta_update(Mat<R> param,
                                           Mat<R>& gsum,
                                           Mat<R>& xsum,
                                           R rho,
                                           R smooth_eps) {
        // update gradient cache using decay rule:
        MAT(gsum) = (
            rho                 * MAT(gsum).wrapper() +
            ((R)((R)1.0 - rho)) * F<op::square<R>>(GRAD(param).wrapper())
        );
        // DEBUG_ASSERT_POSITIVE((MAT(gsum).array()  + this->smooth_eps).matrix());
        // DEBUG_ASSERT_POSITIVE(((xsum.array() + this->smooth_eps) / (MAT(gsum).array() + this->smooth_eps)).matrix());
        TensorInternal<R, 2> dparam(MAT(xsum).shape);
        dparam = (
            F<op::sqrt_f<R>>(MAT(xsum).wrapper() + smooth_eps) /
            F<op::sqrt_f<R>>(MAT(gsum).wrapper() + smooth_eps)
        ) * GRAD(param).wrapper();

        MAT(xsum) = (
            rho * MAT(xsum).wrapper() +
            ((R)(1.0 - rho)) * F<op::square<R>>(dparam.wrapper())
        );
        // update gradient using AdaDelta rule
        MAT(param) -= dparam.wrapper();
    }

    template<typename R>
    void SolverUpdates<R>::adam_update(Mat<R> param,
                                           Mat<R>& m,
                                           Mat<R>& v,
                                           R b1,
                                           R b2,
                                           R smooth_eps,
                                           R step_size,
                                           unsigned long long epoch) {
        // this affects the learning rate:
        auto fix1 = 1.0 - std::pow(b1, epoch);
        auto fix2 = 1.0 - std::pow(b2, epoch);
        R lr_t = step_size * sqrt(fix2 / fix1);

        ASSERT2(lr_t == lr_t, "Epoch learning rate is NaN. Try changing b1 or b2.");

        // update m acculumulator
        MAT(m) = (MAT(m).wrapper() * (R)(1.0 - b1)) + b1 * GRAD(param).wrapper();
        // update v acculumulator
        MAT(v) = (MAT(v).wrapper() * (R)(1.0 - b2)) + b2 * F<op::square<R>>(GRAD(param).wrapper());

        GRAD(param) = MAT(m).wrapper() / (F<op::sqrt_f<R>>(MAT(v).wrapper()) + smooth_eps);

        // take gradient step
        MAT(param) -= lr_t * GRAD(param).wrapper();
    }

    template class SolverUpdates<float>;
    template class SolverUpdates<double>;
    template class SolverUpdates<int>;

}
