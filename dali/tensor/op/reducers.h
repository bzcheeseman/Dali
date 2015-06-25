#ifndef DALI_TENSOR_OP_REDUCERS_H
#define DALI_TENSOR_OP_REDUCERS_H

#include "dali/tensor/Mat.h"
#include "dali/tensor/Tape.h"
#include "dali/utils.h"

template<typename R> class Mat;

namespace matops {
    template<typename R>
    struct Reducers {
        static Mat<R> L2_norm(Mat<R>);
        static Mat<R> sum(Mat<R>);
        static Mat<R> mean(Mat<R>);
    };
}

#endif
