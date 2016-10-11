#ifndef DALI_ARRAY_OP2_GATHER_FROM_ROWS_H
#define DALI_ARRAY_OP2_GATHER_FROM_ROWS_H

struct Operation;

namespace op {
    Operation gather_from_rows(const Operation& source, const Operation& indices);
}  // namespace op

#endif  // DALI_ARRAY_OP2_GATHER_FROM_ROWS_H
