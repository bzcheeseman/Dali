#ifndef DALI_ARRAY_OP_GATHER_FROM_ROWS_H
#define DALI_ARRAY_OP_GATHER_FROM_ROWS_H

#include "dali/array/array.h"

namespace op {
Array gather_from_rows(const Array& source, const Array& indices);
}  // namespace op

#endif  // DALI_ARRAY_OP_GATHER_FROM_ROWS_H
