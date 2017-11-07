#include "binary.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/unary.h"

namespace op {
    expression::ExpressionGraph add(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::add");
    }

    expression::ExpressionGraph add(const std::vector<expression::ExpressionGraph>& arrays) {
        ASSERT2(arrays.size() > 0, "op::add takes requires at least 1 array");
        if (arrays.size() == 1) {
            return identity(arrays[0]);
        } else if (arrays.size() == 2) {
            return add(arrays[0], arrays[1]);
        } else {
            expression::ExpressionGraph res = arrays[0];
            for (int i = 1; i < arrays.size(); i += 4) {
                res = add(res, arrays[i]);
            }
            return res;
        }
    }

    expression::ExpressionGraph sub(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::sub");
    }

    expression::ExpressionGraph eltmul(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::eltmul");
    }

    expression::ExpressionGraph eltdiv(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::eltdiv");
    }

    expression::ExpressionGraph pow(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::power");
    }

    expression::ExpressionGraph equals(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::equals");
    }

    expression::ExpressionGraph prelu(const expression::ExpressionGraph& x, const expression::ExpressionGraph& weights) {
        return elementwise(x, weights, "functor::prelu");
    }

    expression::ExpressionGraph clipped_relu(const expression::ExpressionGraph& x, const expression::ExpressionGraph& upper_bound) {
        return elementwise(x, upper_bound, "functor::clipped_relu");
    }

    expression::ExpressionGraph clipped_relu_backward(const expression::ExpressionGraph& x, const expression::ExpressionGraph& upper_bound) {
        return elementwise(x, upper_bound, "functor::clipped_relu_backward");
    }

    expression::ExpressionGraph steep_sigmoid(const expression::ExpressionGraph& x, const expression::ExpressionGraph& aggressiveness) {
        return elementwise(x, aggressiveness, "functor::steep_sigmoid");
    }

    expression::ExpressionGraph steep_sigmoid_backward(const expression::ExpressionGraph& x, const expression::ExpressionGraph& aggressiveness) {
        return elementwise(x, aggressiveness, "functor::steep_sigmoid_backward");
    }

    expression::ExpressionGraph prelu_backward_weights(const expression::ExpressionGraph& a, const expression::ExpressionGraph& grad) {
        return elementwise(a, grad, "functor::prelu_backward_weights");
    }

    expression::ExpressionGraph prelu_backward_inputs(const expression::ExpressionGraph& a, const expression::ExpressionGraph& weights) {
        return elementwise(a, weights, "functor::prelu_backward_inputs");
    }

    expression::ExpressionGraph lessthanequal(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::lessthanequal");
    }

    expression::ExpressionGraph greaterthanequal(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::greaterthanequal");
    }

    expression::ExpressionGraph eltmax(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::max_scalar");
    }

    expression::ExpressionGraph clip(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::clip");
    }

    expression::ExpressionGraph eltmin(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::min_scalar");
    }

    expression::ExpressionGraph binary_cross_entropy(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::binary_cross_entropy");
    }

    expression::ExpressionGraph binary_cross_entropy_grad(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b) {
        return elementwise(a, b, "functor::binary_cross_entropy_grad");
    }
}
