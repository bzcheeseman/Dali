#ifndef DALI_ARRAY_OP2_UNARY_H
#define DALI_ARRAY_OP2_UNARY_H

namespace op {
    // Assign one array to another piece of memory as-is
    // this also has the side-effect of a strided memory view
    // non-strided in the output (as it is no longer a view)
    // TODO(jonathan) add support for optional copy if output and
    // input are the same, or if the output is uninitialized
    expression::ExpressionGraph identity(const expression::ExpressionGraph& x);
    expression::ExpressionGraph sigmoid(const expression::ExpressionGraph& x);
    expression::ExpressionGraph dsigmoid(const expression::ExpressionGraph& x);
    expression::ExpressionGraph tanh(const expression::ExpressionGraph& x);
    expression::ExpressionGraph dtanh(const expression::ExpressionGraph& x);
    expression::ExpressionGraph relu(const expression::ExpressionGraph& x);
    expression::ExpressionGraph relu_backward(const expression::ExpressionGraph& x);
    expression::ExpressionGraph eltinv(const expression::ExpressionGraph& x);
    expression::ExpressionGraph exp(const expression::ExpressionGraph& x);
    expression::ExpressionGraph log(const expression::ExpressionGraph& x);
    expression::ExpressionGraph log_or_zero(const expression::ExpressionGraph& x);
    expression::ExpressionGraph abs(const expression::ExpressionGraph& x);
    expression::ExpressionGraph sign(const expression::ExpressionGraph& x);
    expression::ExpressionGraph square(const expression::ExpressionGraph& x);
    expression::ExpressionGraph softplus(const expression::ExpressionGraph& x);
    expression::ExpressionGraph softplus_backward(const expression::ExpressionGraph& x);
    expression::ExpressionGraph cube(const expression::ExpressionGraph& x);
    expression::ExpressionGraph sqrt(const expression::ExpressionGraph& x);
    expression::ExpressionGraph rsqrt(const expression::ExpressionGraph& x);
    expression::ExpressionGraph isnan(const expression::ExpressionGraph& x);
    expression::ExpressionGraph isinf(const expression::ExpressionGraph& x);
    expression::ExpressionGraph inverse_tanh(const expression::ExpressionGraph& x);
} // namespace op2

#endif
