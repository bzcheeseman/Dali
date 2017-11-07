#ifndef DALI_ARRAY_OP2_ONE_HOT_H
#define DALI_ARRAY_OP2_ONE_HOT_H

#include <string>
#include <vector>

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
    expression::ExpressionGraph one_hot(
        const expression::ExpressionGraph& indices,
        int depth,
        const expression::ExpressionGraph& on_value,
        const expression::ExpressionGraph& off_value
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_ONE_HOT_H
