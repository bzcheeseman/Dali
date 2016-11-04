#ifndef DALI_ARRAY_OP2_BINARY_H
#define DALI_ARRAY_OP2_BINARY_H

#include "dali/array/op2/expression/expression.h"

namespace op {
    Expression add(const Expression& a, const Expression& b);
    Assignable<Array> add(const std::vector<Array>& as);
    Expression sub(const Expression& a, const Expression& b);
    Expression eltmul(const Expression& left, const Expression& right);
    Expression eltdiv(const Expression& left, const Expression& right);
    Expression pow(const Expression& left, const Expression& right);
    Expression equals(const Expression& left, const Expression& right);
    Expression steep_sigmoid(const Expression& x, const Expression& aggressiveness);
    Expression steep_sigmoid_backward(const Expression& x, const Expression& aggressiveness);
    Expression clipped_relu(const Expression& x, const Expression& clipval);
    Expression clipped_relu_backward(const Expression& x, const Expression& clipval);
    Expression prelu(const Expression& x, const Expression& weights);
    Expression prelu_backward_weights(const Expression& a, const Expression& grad);
    Expression prelu_backward_inputs(const Expression& a, const Expression& weights);

    Expression lessthanequal(const Expression& a, const Expression& b);
    Expression greaterthanequal(const Expression& a, const Expression& b);
    Expression eltmax(const Expression& a, const Expression& b);
    Expression clip(const Expression& a, const Expression& b);
    Expression eltmin(const Expression& a, const Expression& b);
    Expression binary_cross_entropy(const Expression& a, const Expression& b);
    Expression binary_cross_entropy_grad(const Expression& a, const Expression& b);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_BINARY_H
