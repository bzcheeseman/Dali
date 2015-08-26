#ifndef DALI_TENSOR_OP_ELEMENTWISE_H
#define DALI_TENSOR_OP_ELEMENTWISE_H

#include "dali/tensor/Mat.h"
#include "dali/tensor/Tape.h"
#include "dali/utils.h"

template<typename R> class Mat;

namespace matops {
    template<typename R>
    struct Elementwise {
        static Mat<R> add(Mat<R>, R);
        static Mat<R> sub_broadcast_reversed(Mat<R>, R);
        static Mat<R> eltmul(Mat<R>, R);
        static Mat<R> eltdivide(Mat<R>, R);

        static Mat<R> max(Mat<R>, R);
        static Mat<R> square(Mat<R>);
        static Mat<R> log(Mat<R>);
        static Mat<R> exp(Mat<R>);
        static Mat<R> sigmoid(Mat<R>);
        static Mat<R> steep_sigmoid(Mat<R>, R aggressiveness = 3.75);
        static Mat<R> tanh(Mat<R>);
        static Mat<R> softplus(Mat<R>);
        static Mat<R> relu(Mat<R>);
        static Mat<R> abs(Mat<R>);
        static Mat<R> pow(Mat<R>, R);
        static Mat<R> sqrt(Mat<R>);
        static Mat<R> elt_inv(Mat<R>);
    };
}

#endif
