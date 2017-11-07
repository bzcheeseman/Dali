#ifndef DALI_ARRAY_OP2_TRANSPOSE_H
#define DALI_ARRAY_OP2_TRANSPOSE_H

#include <vector>

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
    expression::ExpressionGraph transpose(
        const expression::ExpressionGraph& input
    );

    expression::ExpressionGraph transpose(
        const expression::ExpressionGraph& input,
        const std::vector<int>& axes
    );

    expression::ExpressionGraph dimshuffle(
        const expression::ExpressionGraph& input,
        const std::vector<int>& axes
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_TRANSPOSE_H
