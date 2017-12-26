#include "binary.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/array/op/unary.h"

namespace op {
    Array all_equals(Array left, Array right) {
        return op::prod(op::equals(left, right));
    }
    Array all_close(Array left, Array right, Array atolerance) {
        return op::prod(op::close(left, right, atolerance));
    }
    Array equals(Array left, Array right) {
        return op::elementwise(left, right, "functor::equals");
    }
    Array close(Array left, Array right, Array atolerance) {
        return op::lessthanequal(op::abs(op::subtract(left, right)), atolerance);
    }
    Array add(Array left, Array right) {
        return op::elementwise(left, right, "functor::add");
    }
    Array subtract(Array left, Array right) {
        return op::elementwise(left, right, "functor::subtract");
    }
    Array eltmul(Array left, Array right) {
        return op::elementwise(left, right, "functor::eltmul");
    }
    Array eltdiv(Array left, Array right) {
        return op::elementwise(left, right, "functor::eltdiv");
    }
    Array pow(Array a, Array b) {
        return elementwise(a, b, "functor::power");
    }
    Array prelu(Array x, Array weights) {
        return elementwise(x, weights, "functor::prelu");
    }
    Array clipped_relu(Array x, Array upper_bound) {
        return elementwise(x, upper_bound, "functor::clipped_relu");
    }
    Array clipped_relu_backward(Array x, Array upper_bound) {
        return elementwise(x, upper_bound, "functor::clipped_relu_backward");
    }
    Array steep_sigmoid(Array x, Array aggressiveness) {
        return elementwise(x, aggressiveness, "functor::steep_sigmoid");
    }
    Array steep_sigmoid_backward(Array x, Array aggressiveness) {
        return elementwise(x, aggressiveness, "functor::steep_sigmoid_backward");
    }
    Array prelu_backward_weights(Array a, Array grad) {
        return elementwise(a, grad, "functor::prelu_backward_weights");
    }
    Array prelu_backward_inputs(Array a, Array weights) {
        return elementwise(a, weights, "functor::prelu_backward_inputs");
    }
    Array lessthanequal(Array a, Array b) {
        return elementwise(a, b, "functor::lessthanequal");
    }
    Array greaterthanequal(Array a, Array b) {
        return elementwise(a, b, "functor::greaterthanequal");
    }
    Array eltmax(Array a, Array b) {
        return elementwise(a, b, "functor::max_scalar");
    }
    Array clip(Array a, Array b) {
        return elementwise(a, b, "functor::clip");
    }
    Array eltmin(Array a, Array b) {
        return elementwise(a, b, "functor::min_scalar");
    }
    Array binary_cross_entropy(Array a, Array b) {
        return elementwise(a, b, "functor::binary_cross_entropy");
    }
    Array binary_cross_entropy_grad(Array a, Array b) {
        return elementwise(a, b, "functor::binary_cross_entropy_grad");
    }
}
