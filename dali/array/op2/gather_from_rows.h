#ifndef DALI_ARRAY_OP2_GATHER_FROM_ROWS_H
#define DALI_ARRAY_OP2_GATHER_FROM_ROWS_H

struct Operation;

namespace op2 {
    Operation gather_from_rows(const Operation& source, const Operation& indices);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_GATHER_FROM_ROWS_H
