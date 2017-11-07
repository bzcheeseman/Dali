#ifndef DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H
#define DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
    expression::ExpressionGraph circular_convolution(const expression::ExpressionGraph& x, const expression::ExpressionGraph& weights);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H
