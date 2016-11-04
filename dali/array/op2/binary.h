#ifndef DALI_ARRAY_OP2_BINARY_H
#define DALI_ARRAY_OP2_BINARY_H

#include "dali/array/op2/expression/expression.h"

namespace op {
    expression::Expression add(const expression::Expression& a, const expression::Expression& b);
    Assignable<Array> add(const std::vector<Array>& as);
    expression::Expression sub(const expression::Expression& a, const expression::Expression& b);
    expression::Expression eltmul(const expression::Expression& left, const expression::Expression& right);
    expression::Expression eltdiv(const expression::Expression& left, const expression::Expression& right);
    expression::Expression pow(const expression::Expression& left, const expression::Expression& right);
    expression::Expression equals(const expression::Expression& left, const expression::Expression& right);
    expression::Expression steep_sigmoid(const expression::Expression& x, const expression::Expression& aggressiveness);
    expression::Expression steep_sigmoid_backward(const expression::Expression& x, const expression::Expression& aggressiveness);
    expression::Expression clipped_relu(const expression::Expression& x, const expression::Expression& clipval);
    expression::Expression clipped_relu_backward(const expression::Expression& x, const expression::Expression& clipval);
    expression::Expression prelu(const expression::Expression& x, const expression::Expression& weights);
    expression::Expression prelu_backward_weights(const expression::Expression& a, const expression::Expression& grad);
    expression::Expression prelu_backward_inputs(const expression::Expression& a, const expression::Expression& weights);

    expression::Expression lessthanequal(const expression::Expression& a, const expression::Expression& b);
    expression::Expression greaterthanequal(const expression::Expression& a, const expression::Expression& b);
    expression::Expression eltmax(const expression::Expression& a, const expression::Expression& b);
    expression::Expression clip(const expression::Expression& a, const expression::Expression& b);
    expression::Expression eltmin(const expression::Expression& a, const expression::Expression& b);
    expression::Expression binary_cross_entropy(const expression::Expression& a, const expression::Expression& b);
    expression::Expression binary_cross_entropy_grad(const expression::Expression& a, const expression::Expression& b);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_BINARY_H
