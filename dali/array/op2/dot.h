#ifndef DALI_ARRAY_OP2_DOT_H
#define DALI_ARRAY_OP2_DOT_H

#include <string>

struct Expression;

namespace op {
    Expression dot2(
        const Expression& left,
        const Expression& right
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_DOT_H
