#ifndef DALI_ARRAY_OP2_DOT_H
#define DALI_ARRAY_OP2_DOT_H

#include <string>

struct Operation;

namespace op {
    Operation dot2(
        const Operation& left,
        const Operation& right
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_DOT_H
