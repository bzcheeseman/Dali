#ifndef DALI_ARRAY_OP2_UNARY_H
#define DALI_ARRAY_OP2_UNARY_H

#include "dali/array/op2/expression/expression.h"

namespace op {
    // Assign one array to another piece of memory as-is
    // this also has the side-effect of a strided memory view
    // non-strided in the output (as it is no longer a view)
    // TODO(jonathan) add support for optional copy if output and
    // input are the same, or if the output is uninitialized
    expression::Expression identity(const expression::Expression& x);
    expression::Expression sigmoid(const expression::Expression& x);
    expression::Expression dsigmoid(const expression::Expression& x);
    expression::Expression tanh(const expression::Expression& x);
    expression::Expression dtanh(const expression::Expression& x);
    expression::Expression relu(const expression::Expression& x);
    expression::Expression relu_backward(const expression::Expression& x);
    expression::Expression eltinv(const expression::Expression& x);
    expression::Expression exp(const expression::Expression& x);
    expression::Expression log(const expression::Expression& x);
    expression::Expression log_or_zero(const expression::Expression& x);
    expression::Expression abs(const expression::Expression& x);
    expression::Expression sign(const expression::Expression& x);
    expression::Expression square(const expression::Expression& x);
    expression::Expression softplus(const expression::Expression& x);
    expression::Expression softplus_backward(const expression::Expression& x);
    expression::Expression cube(const expression::Expression& x);
    expression::Expression sqrt(const expression::Expression& x);
    expression::Expression rsqrt(const expression::Expression& x);
    expression::Expression isnan(const expression::Expression& x);
    expression::Expression isinf(const expression::Expression& x);
    expression::Expression inverse_tanh(const expression::Expression& x);
} // namespace op2

#endif
