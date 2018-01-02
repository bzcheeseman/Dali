#ifndef DALI_ARRAY_OP_CIRCULAR_CONVOLUTION_H
#define DALI_ARRAY_OP_CIRCULAR_CONVOLUTION_H

#include "dali/array/array.h"

namespace op {
Array circular_convolution(Array x, Array weights);
}  // namespace op

#endif  // DALI_ARRAY_OP_CIRCULAR_CONVOLUTION_H
