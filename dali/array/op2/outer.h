#ifndef DALI_ARRAY_OP2_OUTER_H
#define DALI_ARRAY_OP2_OUTER_H

#include "dali/array/op2/expression/expression.h"

namespace op {
    expression::Expression outer(const expression::Expression& left, const expression::Expression& right);
}  // namespace op

#endif  // DALI_ARRAY_OP2_OUTER_H
