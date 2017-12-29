#ifndef DALI_ARRAY_OP_GATHER_H
#define DALI_ARRAY_OP_GATHER_H

#include "dali/array/array.h"

namespace op {
Array gather(const Array& source, const Array& indices);
}  // namespace op

#endif  // DALI_ARRAY_OP_GATHER_H
