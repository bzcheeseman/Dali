#include "unary.h"
#include "dali/array/op2/operation.h"
#include "dali/array/op2/elementwise_operation.h"

namespace op2 {
    Operation identity(const Operation& x) {
        return elementwise(x, "functor::identity");
    }

    Operation sigmoid(const Operation& x) {
        return elementwise(x, "functor::sigmoid");
    }

    Operation tanh(const Operation& x) {
        return elementwise(x, "functor::tanh");
    }

    Operation exp(const Operation& x) {
        return elementwise(x, "functor::exp");
    }

    Operation softplus(const Operation& x) {
        return elementwise(x, "functor::softplus");
    }

    Operation eltinv(const Operation& x) {
        return elementwise(x, "functor::inv");
    }

    Operation relu(const Operation& x) {
        return elementwise(x, "functor::relu");
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

}  // namespace op2
