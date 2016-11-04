#ifndef DALI_ARRAY_OP2_ONE_HOT_H
#define DALI_ARRAY_OP2_ONE_HOT_H

#include <string>
#include <vector>

struct Expression;

namespace op {
    Expression one_hot(
        const Expression& indices,
        int depth,
        const Expression& on_value,
        const Expression& off_value
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_ONE_HOT_H
