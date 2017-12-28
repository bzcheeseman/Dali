#include "unary.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/array/jit/scalar_view.h"

namespace op {
    Array identity(Array x) {
        return elementwise(x, "functor::identity");
    }

    Array identity(float x) {
        return jit::wrap_scalar(x);
    }

    Array identity(double x) {
        return jit::wrap_scalar(x);
    }

    Array identity(int x) {
        return jit::wrap_scalar(x);
    }

    Array sqrt(Array x) {
        return elementwise(x, "functor::sqrt_f");
    }

    Array square(Array x) {
        return elementwise(x, "functor::square");
    }

    Array abs(Array x) {
        return elementwise(x, "functor::abs");
    }

    Array sigmoid(Array x) {
        return elementwise(x, "functor::sigmoid");
    }

    Array dsigmoid(Array x) {
        return elementwise(x, "functor::dsigmoid");
    }

    Array tanh(Array x) {
        return elementwise(x, "functor::tanh");
    }

    Array dtanh(Array x) {
        return elementwise(x, "functor::dtanh");
    }

    Array exp(Array x) {
        return elementwise(x, "functor::exp");
    }

    Array softplus(Array x) {
        return elementwise(x, "functor::softplus");
    }

    Array softplus_backward(Array x) {
        return elementwise(x, "functor::softplus_backward");
    }

    Array eltinv(Array x) {
        return elementwise(x, "functor::inv");
    }

    Array relu(Array x) {
        return elementwise(x, "functor::relu");
    }

    Array relu_backward(Array x) {
        return elementwise(x, "functor::relu_backward");
    }

    Array log(Array x) {
        return elementwise(x, "functor::log");
    }

    Array log_or_zero(Array x) {
        return elementwise(x, "functor::log_or_zero");
    }

    Array sign(Array x) {
        return elementwise(x, "functor::sign");
    }

    Array cube(Array x) {
        return elementwise(x, "functor::cube");
    }

    Array rsqrt(Array x) {
        return elementwise(x, "functor::rsqrt");
    }

    Array isnan(Array x) {
        return elementwise(x, "functor::isnotanumber");
    }

    Array isinf(Array x) {
        return elementwise(x, "functor::isinfinity");
    }

    Array inverse_tanh(Array x) {
        return elementwise(x, "functor::inverse_tanh");
    }
}
