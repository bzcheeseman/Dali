#include "unary.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/binary.h"

namespace op {
    Operation identity(const Operation& x) {
        return elementwise(x, "functor::identity");
    }

    Operation sigmoid(const Operation& x) {
        return elementwise(x, "functor::sigmoid");
    }

    Operation dsigmoid(const Operation& x) {
        return elementwise(x, "functor::dsigmoid");
    }

    Operation tanh(const Operation& x) {
        return elementwise(x, "functor::tanh");
    }

    Operation dtanh(const Operation& x) {
        return elementwise(x, "functor::dtanh");
    }

    Operation exp(const Operation& x) {
        return elementwise(x, "functor::exp");
    }

    Operation softplus(const Operation& x) {
        return elementwise(x, "functor::softplus");
    }

    Operation softplus_backward(const Operation& x) {
        return elementwise(x, "functor::softplus_backward");
    }

    Operation eltinv(const Operation& x) {
        return elementwise(x, "functor::inv");
    }

    Operation relu(const Operation& x) {
        return elementwise(x, "functor::relu");
    }

    Operation relu_backward(const Operation& x) {
        return elementwise(x, "functor::relu_backward");
    }

    Operation log(const Operation& x) {
        return elementwise(x, "functor::log");
    }

    Operation log_or_zero(const Operation& x) {
        return elementwise(x, "functor::log_or_zero");
    }

    Operation abs(const Operation& x)  {
        return elementwise(x, "functor::abs");
    }

    Operation sign(const Operation& x) {
        return elementwise(x, "functor::sign");
    }

    Operation square(const Operation& x) {
        return elementwise(x, "functor::square");
    }

    Operation cube(const Operation& x) {
        return elementwise(x, "functor::cube");
    }

    Operation sqrt(const Operation& x) {
        return elementwise(x, "functor::sqrt_f");
    }

    Operation rsqrt(const Operation& x) {
        return elementwise(x, "functor::rsqrt");
    }

    Operation isnan(const Operation& x) {
        return elementwise(x, "functor::isnotanumber");
    }

    Operation isinf(const Operation& x) {
        return elementwise(x, "functor::isinfinity");
    }

    Operation inverse_tanh(const Operation& x) {
        return elementwise(x, "functor::inverse_tanh");
    }

}  // namespace op2
