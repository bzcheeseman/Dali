#ifndef DALI_ARRAY_OP2_RESHAPE_H
#define DALI_ARRAY_OP2_RESHAPE_H

#include <vector>

namespace expression {
    struct Expression;
}  // namespace expression

namespace op {
    expression::Expression reshape(
        const expression::Expression& input,
        const std::vector<int>& newshape
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_RESHAPE_H
