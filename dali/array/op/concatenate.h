#ifndef DALI_ARRAY_OP_CONCATENATE_H
#define DALI_ARRAY_OP_CONCATENATE_H

#include "dali/array/array.h"

namespace op {
Array concatenate(const std::vector<Array>& arrays, int axis=0);
Array hstack(const std::vector<Array>& arrays);
Array vstack(const std::vector<Array>& arrays);
}

#endif  // DALI_ARRAY_OP_CONCATENATE_H
