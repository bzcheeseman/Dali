#include "dali/tensor/op/solver_updates.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"


using std::vector;

namespace matops {
    using namespace TensorOps;

    template<typename R>
    void SolverUpdates<R>::sgd_update(Mat<R> param, R step_size, R clipval, R regc) {
         DEBUG_ASSERT_NOT_NAN(MAT(param));

        if (regc > 0) {
            MAT(param) -= step_size * F<op::clip<R>>(GRAD(param).wrapper(), (R)clipval) - (regc * MAT(param).wrapper());
        } else {
            MAT(param) -= step_size * F<op::clip<R>>(GRAD(param).wrapper(), (R)clipval);
        }

        DEBUG_ASSERT_NOT_NAN(MAT(param));
    }

    template<typename R>
    void SolverUpdates<R>::adagrad_update(Mat<R> param,
                                          TensorInternal<R, 1>& cache,
                                          R step_size,
                                          R clipval,
                                          R regc,
                                          R smooth_eps) {

        if (regc > 0) {
            GRAD(param) = F<op::clip<R>>(GRAD(param).wrapper(), clipval) + (regc * MAT(param).wrapper());
        } else {
            GRAD(param) = F<op::clip<R>>(GRAD(param).wrapper(), clipval);
        }
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

    template class SolverUpdates<float>;
    template class SolverUpdates<double>;

}
