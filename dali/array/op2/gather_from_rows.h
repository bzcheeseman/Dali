#ifndef DALI_ARRAY_OP2_GATHER_FROM_ROWS_H
#define DALI_ARRAY_OP2_GATHER_FROM_ROWS_H

namespace expression {
    struct Expression;
}  // namespace expression


namespace op {
    expression::Expression gather_from_rows(const expression::Expression& source, const expression::Expression& indices);
}  // namespace op

#endif  // DALI_ARRAY_OP2_GATHER_FROM_ROWS_H
