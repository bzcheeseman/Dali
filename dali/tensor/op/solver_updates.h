#ifndef DALI_TENSOR_OP_SOLVER_UPDATES_H
#define DALI_TENSOR_OP_SOLVER_UPDATES_H

#include "dali/tensor/Mat.h"
#include "dali/tensor/Tape.h"
#include "dali/utils.h"

template<typename R> class Mat;

namespace matops {
    template<typename R>
    struct SolverUpdates {
        static void clip_and_regularize(Mat<R> param, R clipval, R regc);


        static void sgd_update(Mat<R> matrix, R step_size);
        static void adagrad_update(Mat<R> matrix,
                                   TensorInternal<R, 1>& cache,
                                   R step_size,
                                   R smooth_eps);
        static void rmsprop_update(Mat<R> param, TensorInternal<R,1>& cache,
                R decay_rate, R step_size, R smooth_eps);
    };
}

#endif
