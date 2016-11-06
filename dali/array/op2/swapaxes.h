#ifndef DALI_ARRAY_OP2_SWAPAXES_H
#define DALI_ARRAY_OP2_SWAPAXES_H

namespace expression {
    struct Expression;
}  // namespace expression

namespace op {
    expression::Expression swapaxes(
        const expression::Expression& input,
        int axis1, int axis2
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_SWAPAXES_H
