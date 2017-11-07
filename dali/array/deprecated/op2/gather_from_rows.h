#ifndef DALI_ARRAY_OP2_GATHER_FROM_ROWS_H
#define DALI_ARRAY_OP2_GATHER_FROM_ROWS_H

namespace expression {
    struct ExpressionGraph;
}  // namespace expression


namespace op {
    expression::ExpressionGraph gather_from_rows(const expression::ExpressionGraph& source, const expression::ExpressionGraph& indices);
}  // namespace op

#endif  // DALI_ARRAY_OP2_GATHER_FROM_ROWS_H
