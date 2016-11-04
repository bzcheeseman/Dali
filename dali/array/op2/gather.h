#ifndef DALI_ARRAY_OP2_GATHER_H
#define DALI_ARRAY_OP2_GATHER_H

struct Expression;

namespace op {
    Expression gather(const Expression& source, const Expression& indices);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_GATHER_H
