#include "binary.h"
#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"

namespace op {
    expression::Expression add(const expression::Expression& a, const expression::Expression& b) {
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

    expression::Expression sub(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::sub");
    }

    expression::Expression eltmul(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::eltmul");
    }

    expression::Expression eltdiv(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::eltdiv");
    }

    expression::Expression pow(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::power");
    }

    expression::Expression equals(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::equals");
    }

    expression::Expression prelu(const expression::Expression& x, const expression::Expression& weights) {
        return elementwise(x, weights, "functor::prelu");
    }

    expression::Expression clipped_relu(const expression::Expression& x, const expression::Expression& upper_bound) {
        return elementwise(x, upper_bound, "functor::clipped_relu");
    }

    expression::Expression clipped_relu_backward(const expression::Expression& x, const expression::Expression& upper_bound) {
        return elementwise(x, upper_bound, "functor::clipped_relu_backward");
    }

    expression::Expression steep_sigmoid(const expression::Expression& x, const expression::Expression& aggressiveness) {
        return elementwise(x, aggressiveness, "functor::steep_sigmoid");
    }

    expression::Expression steep_sigmoid_backward(const expression::Expression& x, const expression::Expression& aggressiveness) {
        return elementwise(x, aggressiveness, "functor::steep_sigmoid_backward");
    }

    expression::Expression prelu_backward_weights(const expression::Expression& a, const expression::Expression& grad) {
        return elementwise(a, grad, "functor::prelu_backward_weights");
    }

    expression::Expression prelu_backward_inputs(const expression::Expression& a, const expression::Expression& weights) {
        return elementwise(a, weights, "functor::prelu_backward_inputs");
    }

    expression::Expression lessthanequal(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::lessthanequal");
    }

    expression::Expression greaterthanequal(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::greaterthanequal");
    }

    expression::Expression eltmax(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::max_scalar");
    }

    expression::Expression clip(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::clip");
    }

    expression::Expression eltmin(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::min_scalar");
    }

    expression::Expression binary_cross_entropy(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::binary_cross_entropy");
    }

    expression::Expression binary_cross_entropy_grad(const expression::Expression& a, const expression::Expression& b) {
        return elementwise(a, b, "functor::binary_cross_entropy_grad");
    }
}
