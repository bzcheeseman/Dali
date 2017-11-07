#ifndef DALI_ARRAY_OP2_GATHER_H
#define DALI_ARRAY_OP2_GATHER_H

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
    expression::ExpressionGraph gather(const expression::ExpressionGraph& source, const expression::ExpressionGraph& indices);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_GATHER_H
