#ifndef DALI_ARRAY_OP2_UNARY_H
#define DALI_ARRAY_OP2_UNARY_H

class FusedOperation;

namespace op2 {
    // Assign one array to another piece of memory as-is
    // this also has the side-effect of a strided memory view
    // non-strided in the output (as it is no longer a view)
    // TODO(jonathan) add support for optional copy if output and
    // input are the same, or if the output is uninitialized
    FusedOperation identity(const FusedOperation& x);
    FusedOperation sigmoid(const FusedOperation& x);
    FusedOperation tanh(const FusedOperation& x);
    FusedOperation relu(const FusedOperation& x);
    FusedOperation eltinv(const FusedOperation& x);
    FusedOperation exp(const FusedOperation& x);
    FusedOperation log(const FusedOperation& x);
    FusedOperation log_or_zero(const FusedOperation& x);
    FusedOperation abs(const FusedOperation& x);
    FusedOperation sign(const FusedOperation& x);
    FusedOperation square(const FusedOperation& x);
    FusedOperation softplus(const FusedOperation& x);
    FusedOperation cube(const FusedOperation& x);
    FusedOperation sqrt(const FusedOperation& x);
    FusedOperation rsqrt(const FusedOperation& x);
} // namespace op2

#endif
