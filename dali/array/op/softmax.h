#ifndef DALI_ARRAY_OP_SOFTMAX_H
#define DALI_ARRAY_OP_SOFTMAX_H

#include "dali/array/array.h"

namespace op {
    Array softmax(const Array& logits, int axis=-1);
    Array softmax_temperature(const Array& logits, const Array& temperature, int axis=-1);
}  // namespace op

#endif
