#include "unary.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/binary.h"

namespace op {
    expression::Expression identity(const expression::Expression& x) {
        return elementwise(x, "functor::identity");
    }

    expression::Expression sigmoid(const expression::Expression& x) {
        return elementwise(x, "functor::sigmoid");
    }

    expression::Expression dsigmoid(const expression::Expression& x) {
        return elementwise(x, "functor::dsigmoid");
    }

    expression::Expression tanh(const expression::Expression& x) {
        return elementwise(x, "functor::tanh");
    }

    expression::Expression dtanh(const expression::Expression& x) {
        return elementwise(x, "functor::dtanh");
    }

    expression::Expression exp(const expression::Expression& x) {
        return elementwise(x, "functor::exp");
    }

    expression::Expression softplus(const expression::Expression& x) {
        return elementwise(x, "functor::softplus");
    }

    expression::Expression softplus_backward(const expression::Expression& x) {
        return elementwise(x, "functor::softplus_backward");
    }

    expression::Expression eltinv(const expression::Expression& x) {
        return elementwise(x, "functor::inv");
    }

    expression::Expression relu(const expression::Expression& x) {
        return elementwise(x, "functor::relu");
    }

    expression::Expression relu_backward(const expression::Expression& x) {
        return elementwise(x, "functor::relu_backward");
    }

    expression::Expression log(const expression::Expression& x) {
        return elementwise(x, "functor::log");
    }

    expression::Expression log_or_zero(const expression::Expression& x) {
        return elementwise(x, "functor::log_or_zero");
    }

    expression::Expression abs(const expression::Expression& x)  {
        return elementwise(x, "functor::abs");
    }

    expression::Expression sign(const expression::Expression& x) {
        return elementwise(x, "functor::sign");
    }

    expression::Expression square(const expression::Expression& x) {
        return elementwise(x, "functor::square");
    }

    expression::Expression cube(const expression::Expression& x) {
        return elementwise(x, "functor::cube");
    }

    expression::Expression sqrt(const expression::Expression& x) {
        return elementwise(x, "functor::sqrt_f");
    }

    expression::Expression rsqrt(const expression::Expression& x) {
        return elementwise(x, "functor::rsqrt");
    }

    expression::Expression isnan(const expression::Expression& x) {
        return elementwise(x, "functor::isnotanumber");
    }

    expression::Expression isinf(const expression::Expression& x) {
        return elementwise(x, "functor::isinfinity");
    }

    expression::Expression inverse_tanh(const expression::Expression& x) {
        return elementwise(x, "functor::inverse_tanh");
    }

}  // namespace op2
