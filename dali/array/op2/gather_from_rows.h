#ifndef DALI_ARRAY_OP2_GATHER_FROM_ROWS_H
#define DALI_ARRAY_OP2_GATHER_FROM_ROWS_H

struct Expression;

namespace op {
    Expression gather_from_rows(const Expression& source, const Expression& indices);
}  // namespace op

#endif  // DALI_ARRAY_OP2_GATHER_FROM_ROWS_H
