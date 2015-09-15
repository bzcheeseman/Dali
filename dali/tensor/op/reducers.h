#ifndef DALI_TENSOR_OP_REDUCERS_H
#define DALI_TENSOR_OP_REDUCERS_H

#include "dali/tensor/Mat.h"
#include "dali/tensor/Tape.h"
#include "dali/utils.h"

template<typename R> class Mat;

namespace matops {
    template<typename R>
    struct Reducers {
        static Mat<R> grad_norm(Mat<R>);
        static Mat<R> grad_norm_colwise(Mat<R>);
        static Mat<R> grad_norm_rowwise(Mat<R>);

        static Mat<R> L2_norm(Mat<R>);
        static Mat<R> L2_norm_colwise(Mat<R>);
        static Mat<R> L2_norm_rowwise(Mat<R>);

        static Mat<R> sum(Mat<R>);
        static Mat<R> sum_colwise(Mat<R>);
        static Mat<R> sum_rowwise(Mat<R>);

        static Mat<R> mean(Mat<R>);
        static Mat<R> mean_colwise(Mat<R>);
        static Mat<R> mean_rowwise(Mat<R>);
    };
}

#endif
