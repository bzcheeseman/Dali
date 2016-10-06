#ifndef DALI_ARRAY_OP2_ONE_HOT_H
#define DALI_ARRAY_OP2_ONE_HOT_H

#include <string>
#include <vector>

struct Operation;

namespace op2 {
    Operation one_hot(
        const Operation& indices,
        int depth,
        const Operation& on_value,
        const Operation& off_value
    );
}  // namespace op2

#endif  // DALI_ARRAY_OP2_ONE_HOT_H
