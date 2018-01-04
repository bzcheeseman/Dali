#ifndef DALI_ARRAY_OP_ARANGE_H
#define DALI_ARRAY_OP_ARANGE_H

#include "dali/array/array.h"

namespace op {
Array arange(int size);
Array arange(int start, int stop);
Array arange(int start, int stop, int step);
}

#endif  // DALI_ARRAY_OP_ARANGE_H
