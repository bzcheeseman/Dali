#ifndef DALI_ARRAY_OP_DOT_H
#define DALI_ARRAY_OP_DOT_H

#include "dali/array/array.h"
#include <vector>

namespace op {
    Array dot(Array a, Array b);
    Array tensordot_as_dot(Array a, Array b, const std::vector<int>& a_reduce_axes,
    										 const std::vector<int>& b_reduce_axes);
}

#endif
