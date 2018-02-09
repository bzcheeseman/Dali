#ifndef DALI_ARRAY_OP_CONCATENATE_H
#define DALI_ARRAY_OP_CONCATENATE_H

#include "dali/array/array.h"

namespace op {
// TODO(jonathan): add default argument
Array concatenate(const std::vector<Array>& arrays, int axis);
Array hstack(const std::vector<Array>& arrays);
Array vstack(const std::vector<Array>& arrays);
}

#endif  // DALI_ARRAY_OP_CONCATENATE_H
