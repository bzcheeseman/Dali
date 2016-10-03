#include "binary.h"
#include "dali/array/op2/operation.h"

namespace op2 {
    Operation add(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::add");
    }

    Operation sub(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::sub");
    }

    Operation eltmul(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::eltmul");
    }

    Operation eltdiv(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::eltdiv");
    }

    Operation pow(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::power");
    }

    Operation equals(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::equals");
    }

    Operation prelu(const Operation& x, const Operation& weights) {
        return elementwise(x, weights, "functor::prelu");
    }

    Operation prelu_backward_weights(const Operation& a, const Operation& grad) {
        return elementwise(a, grad, "functor::prelu_backward_weights");
    }

    Operation prelu_backward_inputs(const Operation& a, const Operation& weights) {
        return elementwise(a, weights, "functor::prelu_backward_inputs");
    }

    Operation lessthanequal(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::lessthanequal");
    }

    Operation greaterthanequal(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::greaterthanequal");
    }

    Operation eltmax(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::max_scalar");
    }

    Operation clip(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::clip");
    }

    Operation eltmin(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::min_scalar");
    }

    Operation binary_cross_entropy(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::binary_cross_entropy");
    }

    Operation binary_cross_entropy_grad(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::binary_cross_entropy_grad");
    }


}
