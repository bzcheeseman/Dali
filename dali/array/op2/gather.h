#ifndef DALI_ARRAY_OP2_GATHER_H
#define DALI_ARRAY_OP2_GATHER_H

struct Operation;

namespace op2 {
    Operation gather(const Operation& source, const Operation& indices);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_GATHER_H
