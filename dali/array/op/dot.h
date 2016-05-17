#ifndef DALI_ARRAY_LAZY_DOT_H
#define DALI_ARRAY_LAZY_DOT_H

#include "dali/array/array.h"

namespace op {
    AssignableArray dot(Array a, Array b);

    AssignableArray _tensordot_as_dot(
        Array a,
        Array b,
        const int& axis,
        int dot_type,
        bool batched);

    AssignableArray _tensordot_as_dot(
        Array a,
        Array b,
        const std::vector<int>& a_reduce_axes,
        const std::vector<int>& b_reduce_axes,
        int dot_type,
        bool batched);
}

#endif  // DALI_ARRAY_LAZY_DOT_H
