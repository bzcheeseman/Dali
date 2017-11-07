#ifndef DALI_ARRAY_OP2_BINARY_H
#define DALI_ARRAY_OP2_BINARY_H

namespace op {
    expression::ExpressionGraph add(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b);
    expression::ExpressionGraph add(const std::vector<expression::ExpressionGraph>& as);
    expression::ExpressionGraph sub(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b);
    expression::ExpressionGraph eltmul(const expression::ExpressionGraph& left, const expression::ExpressionGraph& right);
    expression::ExpressionGraph eltdiv(const expression::ExpressionGraph& left, const expression::ExpressionGraph& right);
    expression::ExpressionGraph pow(const expression::ExpressionGraph& left, const expression::ExpressionGraph& right);
    expression::ExpressionGraph equals(const expression::ExpressionGraph& left, const expression::ExpressionGraph& right);
    expression::ExpressionGraph steep_sigmoid(const expression::ExpressionGraph& x, const expression::ExpressionGraph& aggressiveness);
    expression::ExpressionGraph steep_sigmoid_backward(const expression::ExpressionGraph& x, const expression::ExpressionGraph& aggressiveness);
    expression::ExpressionGraph clipped_relu(const expression::ExpressionGraph& x, const expression::ExpressionGraph& clipval);
    expression::ExpressionGraph clipped_relu_backward(const expression::ExpressionGraph& x, const expression::ExpressionGraph& clipval);
    expression::ExpressionGraph prelu(const expression::ExpressionGraph& x, const expression::ExpressionGraph& weights);
    expression::ExpressionGraph prelu_backward_weights(const expression::ExpressionGraph& a, const expression::ExpressionGraph& grad);
    expression::ExpressionGraph prelu_backward_inputs(const expression::ExpressionGraph& a, const expression::ExpressionGraph& weights);

    expression::ExpressionGraph lessthanequal(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b);
    expression::ExpressionGraph greaterthanequal(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b);
    expression::ExpressionGraph eltmax(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b);
    expression::ExpressionGraph clip(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b);
    expression::ExpressionGraph eltmin(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b);
    expression::ExpressionGraph binary_cross_entropy(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b);
    expression::ExpressionGraph binary_cross_entropy_grad(const expression::ExpressionGraph& a, const expression::ExpressionGraph& b);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_BINARY_H
