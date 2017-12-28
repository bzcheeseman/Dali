#ifndef DALI_ARRAY_OP_ONE_HOT_H
#define DALI_ARRAY_OP_ONE_HOT_H

#include "dali/array/array.h"

namespace op {
    Array one_hot(Array indices, int depth, Array on_value, Array off_value);
}  // namespace op

#endif  // DALI_ARRAY_OP_ONE_HOT_H
