#include "unary.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/binary.h"

namespace op {
    Expression identity(const Expression& x) {
        return elementwise(x, "functor::identity");
    }

    Expression sigmoid(const Expression& x) {
        return elementwise(x, "functor::sigmoid");
    }

    Expression dsigmoid(const Expression& x) {
        return elementwise(x, "functor::dsigmoid");
    }

    Expression tanh(const Expression& x) {
        return elementwise(x, "functor::tanh");
    }

    Expression dtanh(const Expression& x) {
        return elementwise(x, "functor::dtanh");
    }

    Expression exp(const Expression& x) {
        return elementwise(x, "functor::exp");
    }

    Expression softplus(const Expression& x) {
        return elementwise(x, "functor::softplus");
    }

    Expression softplus_backward(const Expression& x) {
        return elementwise(x, "functor::softplus_backward");
    }

    Expression eltinv(const Expression& x) {
        return elementwise(x, "functor::inv");
    }

    Expression relu(const Expression& x) {
        return elementwise(x, "functor::relu");
    }

    Expression relu_backward(const Expression& x) {
        return elementwise(x, "functor::relu_backward");
    }

    Expression log(const Expression& x) {
        return elementwise(x, "functor::log");
    }

    Expression log_or_zero(const Expression& x) {
        return elementwise(x, "functor::log_or_zero");
    }

    Expression abs(const Expression& x)  {
        return elementwise(x, "functor::abs");
    }

    Expression sign(const Expression& x) {
        return elementwise(x, "functor::sign");
    }

    Expression square(const Expression& x) {
        return elementwise(x, "functor::square");
    }

    Expression cube(const Expression& x) {
        return elementwise(x, "functor::cube");
    }

    Expression sqrt(const Expression& x) {
        return elementwise(x, "functor::sqrt_f");
    }

    Expression rsqrt(const Expression& x) {
        return elementwise(x, "functor::rsqrt");
    }

    Expression isnan(const Expression& x) {
        return elementwise(x, "functor::isnotanumber");
    }

    Expression isinf(const Expression& x) {
        return elementwise(x, "functor::isinfinity");
    }

    Expression inverse_tanh(const Expression& x) {
        return elementwise(x, "functor::inverse_tanh");
    }

}  // namespace op2
