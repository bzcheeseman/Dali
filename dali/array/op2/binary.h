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
    FusedOperation circular_convolution(const FusedOperation& x, const FusedOperation& weights);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_BINARY_H
