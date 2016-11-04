#include "binary.h"
#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"

namespace op {
    Expression add(const Expression& a, const Expression& b) {
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

    Expression sub(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::sub");
    }

    Expression eltmul(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::eltmul");
    }

    Expression eltdiv(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::eltdiv");
    }

    Expression pow(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::power");
    }

    Expression equals(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::equals");
    }

    Expression prelu(const Expression& x, const Expression& weights) {
        return elementwise(x, weights, "functor::prelu");
    }

    Expression clipped_relu(const Expression& x, const Expression& upper_bound) {
        return elementwise(x, upper_bound, "functor::clipped_relu");
    }

    Expression clipped_relu_backward(const Expression& x, const Expression& upper_bound) {
        return elementwise(x, upper_bound, "functor::clipped_relu_backward");
    }

    Expression steep_sigmoid(const Expression& x, const Expression& aggressiveness) {
        return elementwise(x, aggressiveness, "functor::steep_sigmoid");
    }

    Expression steep_sigmoid_backward(const Expression& x, const Expression& aggressiveness) {
        return elementwise(x, aggressiveness, "functor::steep_sigmoid_backward");
    }

    Expression prelu_backward_weights(const Expression& a, const Expression& grad) {
        return elementwise(a, grad, "functor::prelu_backward_weights");
    }

    Expression prelu_backward_inputs(const Expression& a, const Expression& weights) {
        return elementwise(a, weights, "functor::prelu_backward_inputs");
    }

    Expression lessthanequal(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::lessthanequal");
    }

    Expression greaterthanequal(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::greaterthanequal");
    }

    Expression eltmax(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::max_scalar");
    }

    Expression clip(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::clip");
    }

    Expression eltmin(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::min_scalar");
    }

    Expression binary_cross_entropy(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::binary_cross_entropy");
    }

    Expression binary_cross_entropy_grad(const Expression& a, const Expression& b) {
        return elementwise(a, b, "functor::binary_cross_entropy_grad");
    }
}
