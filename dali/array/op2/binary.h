#ifndef DALI_ARRAY_OP2_BINARY_H
#define DALI_ARRAY_OP2_BINARY_H

class FusedOperation;

namespace op2 {
    FusedOperation add(const FusedOperation& a, const FusedOperation& b);
    FusedOperation sub(const FusedOperation& a, const FusedOperation& b);
    FusedOperation eltmul(const FusedOperation& left, const FusedOperation& right);
    FusedOperation eltdiv(const FusedOperation& left, const FusedOperation& right);
    FusedOperation pow(const FusedOperation& left, const FusedOperation& right);
    FusedOperation equals(const FusedOperation& left, const FusedOperation& right);
    FusedOperation prelu(const FusedOperation& x, const FusedOperation& weights);
    FusedOperation prelu_backward_weights(const FusedOperation& a, const FusedOperation& grad);
    FusedOperation prelu_backward_inputs(const FusedOperation& a, const FusedOperation& weights);

    FusedOperation circular_convolution(const FusedOperation& x, const FusedOperation& weights);
    FusedOperation lessthanequal(const FusedOperation& a, const FusedOperation& b);
    FusedOperation greaterthanequal(const FusedOperation& a, const FusedOperation& b);
    FusedOperation eltmax(const FusedOperation& a, const FusedOperation& b);
    FusedOperation clip(const FusedOperation& a, const FusedOperation& b);
    FusedOperation eltmin(const FusedOperation& a, const FusedOperation& b);
    FusedOperation binary_cross_entropy(const FusedOperation& a, const FusedOperation& b);
    FusedOperation binary_cross_entropy_grad(const FusedOperation& a, const FusedOperation& b);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_BINARY_H
