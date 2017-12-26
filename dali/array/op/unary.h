#ifndef DALI_ARRAY_OP_UNARY_H
#define DALI_ARRAY_OP_UNARY_H

#include "dali/array/array.h"

namespace op {
    Array identity(Array x);
    Array identity(int x);
    Array identity(float x);
    Array identity(double x);
    Array sqrt(Array x);
    Array square(Array x);
    Array abs(Array x);
    Array sigmoid(Array x);
    Array dsigmoid(Array x);
    Array tanh(Array x);
    Array dtanh(Array x);
    Array relu(Array x);
    Array relu_backward(Array x);
    Array eltinv(Array x);
    Array exp(Array x);
    Array log(Array x);
    Array log_or_zero(Array x);
    Array sign(Array x);
    Array softplus(Array x);
    Array softplus_backward(Array x);
    Array cube(Array x);
    Array rsqrt(Array x);
    Array isnan(Array x);
    Array isinf(Array x);
    Array inverse_tanh(Array x);
}

#endif
