#ifndef DALI_ARRAY_OP2_UNARY_H
#define DALI_ARRAY_OP2_UNARY_H

#include "dali/array/op2/operation.h"

namespace op {
    // Assign one array to another piece of memory as-is
    // this also has the side-effect of a strided memory view
    // non-strided in the output (as it is no longer a view)
    // TODO(jonathan) add support for optional copy if output and
    // input are the same, or if the output is uninitialized
    Operation identity(const Operation& x);
    Operation sigmoid(const Operation& x);
    Operation dsigmoid(const Operation& x);
    Operation tanh(const Operation& x);
    Operation dtanh(const Operation& x);
    Operation relu(const Operation& x);
    Operation relu_backward(const Operation& x);
    Operation eltinv(const Operation& x);
    Operation exp(const Operation& x);
    Operation log(const Operation& x);
    Operation log_or_zero(const Operation& x);
    Operation abs(const Operation& x);
    Operation sign(const Operation& x);
    Operation square(const Operation& x);
    Operation softplus(const Operation& x);
    Operation softplus_backward(const Operation& x);
    Operation cube(const Operation& x);
    Operation sqrt(const Operation& x);
    Operation rsqrt(const Operation& x);
    Operation isnan(const Operation& x);
    Operation isinf(const Operation& x);
    Operation inverse_tanh(const Operation& x);
} // namespace op2

#endif
