#ifndef DALI_ARRAY_OP2_ONE_HOT_H
#define DALI_ARRAY_OP2_ONE_HOT_H

#include <string>
#include <vector>

namespace expression {
    struct Expression;
}  // namespace expression

namespace op {
    expression::Expression one_hot(
        const expression::Expression& indices,
        int depth,
        const expression::Expression& on_value,
        const expression::Expression& off_value
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_ONE_HOT_H
