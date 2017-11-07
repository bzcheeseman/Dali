#include "unary.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/binary.h"

namespace op {
    expression::ExpressionGraph identity(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::identity");
    }

    expression::ExpressionGraph sigmoid(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::sigmoid");
    }

    expression::ExpressionGraph dsigmoid(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::dsigmoid");
    }

    expression::ExpressionGraph tanh(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::tanh");
    }

    expression::ExpressionGraph dtanh(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::dtanh");
    }

    expression::ExpressionGraph exp(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::exp");
    }

    expression::ExpressionGraph softplus(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::softplus");
    }

    expression::ExpressionGraph softplus_backward(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::softplus_backward");
    }

    expression::ExpressionGraph eltinv(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::inv");
    }

    expression::ExpressionGraph relu(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::relu");
    }

    expression::ExpressionGraph relu_backward(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::relu_backward");
    }

    expression::ExpressionGraph log(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::log");
    }

    expression::ExpressionGraph log_or_zero(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::log_or_zero");
    }

    expression::ExpressionGraph abs(const expression::ExpressionGraph& x)  {
        return elementwise(x, "functor::abs");
    }

    expression::ExpressionGraph sign(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::sign");
    }

    expression::ExpressionGraph square(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::square");
    }

    expression::ExpressionGraph cube(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::cube");
    }

    expression::ExpressionGraph sqrt(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::sqrt_f");
    }

    expression::ExpressionGraph rsqrt(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::rsqrt");
    }

    expression::ExpressionGraph isnan(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::isnotanumber");
    }

    expression::ExpressionGraph isinf(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::isinfinity");
    }

    expression::ExpressionGraph inverse_tanh(const expression::ExpressionGraph& x) {
        return elementwise(x, "functor::inverse_tanh");
    }

}  // namespace op2
