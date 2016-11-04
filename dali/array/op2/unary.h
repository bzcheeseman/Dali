#ifndef DALI_ARRAY_OP2_UNARY_H
#define DALI_ARRAY_OP2_UNARY_H

#include "dali/array/op2/expression/expression.h"

namespace op {
    // Assign one array to another piece of memory as-is
    // this also has the side-effect of a strided memory view
    // non-strided in the output (as it is no longer a view)
    // TODO(jonathan) add support for optional copy if output and
    // input are the same, or if the output is uninitialized
    Expression identity(const Expression& x);
    Expression sigmoid(const Expression& x);
    Expression dsigmoid(const Expression& x);
    Expression tanh(const Expression& x);
    Expression dtanh(const Expression& x);
    Expression relu(const Expression& x);
    Expression relu_backward(const Expression& x);
    Expression eltinv(const Expression& x);
    Expression exp(const Expression& x);
    Expression log(const Expression& x);
    Expression log_or_zero(const Expression& x);
    Expression abs(const Expression& x);
    Expression sign(const Expression& x);
    Expression square(const Expression& x);
    Expression softplus(const Expression& x);
    Expression softplus_backward(const Expression& x);
    Expression cube(const Expression& x);
    Expression sqrt(const Expression& x);
    Expression rsqrt(const Expression& x);
    Expression isnan(const Expression& x);
    Expression isinf(const Expression& x);
    Expression inverse_tanh(const Expression& x);
} // namespace op2

#endif
