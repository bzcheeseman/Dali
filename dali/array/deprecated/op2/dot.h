#ifndef DALI_ARRAY_OP2_DOT_H
#define DALI_ARRAY_OP2_DOT_H

#include <string>

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
    expression::ExpressionGraph dot2(
        const expression::ExpressionGraph& left,
        const expression::ExpressionGraph& right
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_DOT_H
