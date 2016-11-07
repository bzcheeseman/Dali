#ifndef DALI_ARRAY_OP2_TRANSPOSE_H
#define DALI_ARRAY_OP2_TRANSPOSE_H

#include <vector>

namespace expression {
    struct Expression;
}  // namespace expression

namespace op {
    expression::Expression transpose(
        const expression::Expression& input
    );

    expression::Expression transpose(
        const expression::Expression& input,
        const std::vector<int>& axes
    );

    expression::Expression dimshuffle(
        const expression::Expression& input,
        const std::vector<int>& axes
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_TRANSPOSE_H
