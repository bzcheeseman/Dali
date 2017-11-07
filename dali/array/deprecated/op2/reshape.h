#ifndef DALI_ARRAY_OP2_RESHAPE_H
#define DALI_ARRAY_OP2_RESHAPE_H

#include <vector>

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
    expression::ExpressionGraph reshape(
        const expression::ExpressionGraph& input,
        const std::vector<int>& newshape
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_RESHAPE_H
