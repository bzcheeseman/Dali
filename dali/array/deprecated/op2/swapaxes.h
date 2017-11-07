#ifndef DALI_ARRAY_OP2_SWAPAXES_H
#define DALI_ARRAY_OP2_SWAPAXES_H

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
    expression::ExpressionGraph swapaxes(
        const expression::ExpressionGraph& input,
        int axis1, int axis2
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_SWAPAXES_H
