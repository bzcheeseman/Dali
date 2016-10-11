#include "binary.h"
#include "dali/array/op2/operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"

namespace op {
    Operation add(const Operation& a, const Operation& b) {
        return elementwise(a, b, "functor::add");
    }

    Assignable<Array> add(const std::vector<Array>& arrays) {
        ASSERT2(arrays.size() > 0, "op::add takes requires at least 1 array");
        if (arrays.size() == 1) {
            return identity(arrays[0]);
        } else if (arrays.size() == 2) {
            return add(arrays[0], arrays[1]);
        } else {
            return Assignable<Array>([arrays](Array& out, const OPERATOR_T& operator_t) {
                Array res = arrays[0];
                for (int i = 1; i < arrays.size(); i += 4) {
                    Array newres;
                    if (i + 3 < arrays.size()) {
                        // do 4 additions at once
                        newres = add(
                            add(
                                add(
                                    add(
                                        res,
                                        arrays[i]
                                    ),
                                    arrays[i+1]
                                ),
                                arrays[i+2]
                            ),
                            arrays[i+3]
                        );
                    } else if (i + 2 < arrays.size()) {
                        // do 3 additions at once
                        newres = add(
                            add(
                                add(
                                    res,
                                    arrays[i]
                                ),
                                arrays[i+1]
                            ),
                            arrays[i+2]
                        );
                    } else if (i + 1 < arrays.size()) {
                    // do 2 additions at once
                        newres = add(add(res, arrays[i]), arrays[i+1]);
                    } else {
                    // do 1 addition
                        newres = add(res, arrays[i]);
                    }
                    res.reset();
                    res = newres;
                }
                ((Assignable<Array>)op::identity(res)).assign_to(out, operator_t);
            });
        }
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

    Operation clipped_relu(const Operation& x, const Operation& upper_bound) {
        return elementwise(x, upper_bound, "functor::clipped_relu");
    }

    Operation clipped_relu_backward(const Operation& x, const Operation& upper_bound) {
        return elementwise(x, upper_bound, "functor::clipped_relu_backward");
    }

    Operation steep_sigmoid(const Operation& x, const Operation& aggressiveness) {
        return elementwise(x, aggressiveness, "functor::steep_sigmoid");
    }

    Operation steep_sigmoid_backward(const Operation& x, const Operation& aggressiveness) {
        return elementwise(x, aggressiveness, "functor::steep_sigmoid_backward");
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
