#ifndef DALI_ARRAY_OP2_GATHER_H
#define DALI_ARRAY_OP2_GATHER_H

namespace expression {
    struct Expression;
}  // namespace expression

namespace op {
    expression::Expression gather(const expression::Expression& source, const expression::Expression& indices);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_GATHER_H
