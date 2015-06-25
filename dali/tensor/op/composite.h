#ifndef DALI_TENSOR_OP_COMPOSITE_H
#define DALI_TENSOR_OP_COMPOSITE_H

#include "dali/tensor/Mat.h"
#include "dali/tensor/Tape.h"
#include "dali/utils.h"

template<typename R> class Mat;

namespace matops {
    template<typename R>
    struct Composite {
        static Mat<R> mul_with_bias(Mat<R>, Mat<R>, Mat<R>);
        // operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
        static Mat<R> mul_add_mul_with_bias(Mat<R>, Mat<R>, Mat<R>, Mat<R>, Mat<R>);
        static Mat<R> mul_add_mul_with_bias(std::initializer_list<Mat<R>>);
        static Mat<R> mul_add_mul_with_bias(std::vector<Mat<R>>&);
        // operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
        // and with caveat that x is actually a column, and should be broadcasted
        static Mat<R> mul_add_broadcast_mul_with_bias(Mat<R>, Mat<R>, Mat<R>, Mat<R>, Mat<R>);
        static Mat<R> quadratic_form(Mat<R> left, Mat<R> weigths, Mat<R> right);
    };
}

#endif
