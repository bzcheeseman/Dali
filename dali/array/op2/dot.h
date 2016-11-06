#ifndef DALI_ARRAY_OP2_DOT_H
#define DALI_ARRAY_OP2_DOT_H

#include <string>

namespace expression {
    struct Expression;
}  // namespace expression

namespace op {
    expression::Expression dot2(
        const expression::Expression& left,
        const expression::Expression& right
    );
}  // namespace op

#endif  // DALI_ARRAY_OP2_DOT_H
